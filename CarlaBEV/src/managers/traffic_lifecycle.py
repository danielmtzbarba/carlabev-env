"""Traffic lifecycle: recycle background vehicles that finished their route.

Without recycling, random traffic decays during an episode: every vehicle that
reaches the end of its route stops there forever, so density drops and stopped
cars accumulate on the map. The lifecycle re-dispatches such vehicles onto a new
route sampled from the planning graph, **outside the ego's field of view**, with
the usual low-speed startup ramp, keeping the requested traffic density roughly
constant for the whole episode.

Only *recyclable* actors are touched: procedurally generated traffic without a
scripted behaviour. Scenario actors (lead/rear/adversary with behaviours) and
authored scenes are never recycled.

Determinism: the lifecycle draws from its own RNG derived from the traffic
seed, so cache hits and misses produce the same recycling sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import numpy as np


@dataclass
class LifecycleStats:
    recycled: int = 0
    despawn_deferred: int = 0  # finished but still in view: waited
    respawn_failures: int = 0
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recycled": self.recycled,
            "despawn_deferred": self.despawn_deferred,
            "respawn_failures": self.respawn_failures,
        }


class TrafficLifecycle:
    """Despawn/respawn policy for recyclable background vehicles."""

    def __init__(
        self,
        planners,
        units,
        *,
        seed: int,
        fov_radius_m: float,
        margin_m: float = 10.0,
        min_route_m: float = 30.0,
        max_attempts: int = 25,
        startup_sampler=None,
        enabled: bool = True,
    ):
        self.planners = planners
        self.units = units
        self.metric = units.metric
        self.rng = random.Random(f"traffic-lifecycle:{seed}")
        self.np_rng = np.random.default_rng(abs(hash(("traffic-lifecycle", seed))) % (2**32))
        self.fov_radius_m = float(fov_radius_m)
        self.margin_m = float(margin_m)
        self.min_route_m = float(min_route_m)
        self.max_attempts = int(max_attempts)
        self.startup_sampler = startup_sampler
        self.enabled = enabled
        self.stats = LifecycleStats()

    # --- geometry helpers ---------------------------------------------------
    @property
    def exclusion_radius_m(self) -> float:
        return self.fov_radius_m + self.margin_m

    def _hero_xy_m(self, hero):
        return float(hero.x_m), float(hero.y_m)

    def _distance_m(self, xy_a, xy_b) -> float:
        return math.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1])

    def is_out_of_view(self, actor, hero) -> bool:
        ctrl = actor._controller
        return self._distance_m((ctrl.x_m, ctrl.y_m), self._hero_xy_m(hero)) > self.exclusion_radius_m

    # --- policy ----------------------------------------------------------------
    def step(self, finished_actors, hero, sim_time_s: float) -> int:
        """Recycle finished, recyclable, out-of-view vehicles. Returns the number recycled."""
        if not self.enabled or hero is None:
            return 0
        recycled = 0
        for actor in finished_actors:
            if not getattr(actor, "recyclable", False):
                continue
            if not self.is_out_of_view(actor, hero):
                self.stats.despawn_deferred += 1
                continue
            if self.respawn(actor, hero, sim_time_s):
                recycled += 1
        return recycled

    def respawn(self, actor, hero, sim_time_s: float) -> bool:
        """Move ``actor`` onto a new route whose start lies outside the exclusion radius."""
        from CarlaBEV.src.scenes.utils import find_route, get_random_node

        lane = getattr(actor, "lane", None) or getattr(getattr(actor, "start_node", None), "lane", None) or "R"
        hero_xy = self._hero_xy_m(hero)
        for _ in range(self.max_attempts):
            start = get_random_node(self.planners, "vehicle", lane, rng=self.rng)
            start_m = self.metric.surface_to_meters(start.scaled_pos)
            if self._distance_m(start_m, hero_xy) <= self.exclusion_radius_m:
                continue
            end = get_random_node(self.planners, "vehicle", lane, rng=self.rng)
            if end.id == start.id:
                continue
            actor.start_node, actor.end_node = start, end
            actor.path = []
            actor.rx, actor.ry = [], []
            try:
                _, (rx, ry) = find_route(self.planners, actor, lane)
            except Exception:
                continue
            if len(rx) < 4 or self.metric.route_length_meters(rx, ry) < self.min_route_m:
                continue
            # The first stretch of the route must also be out of view.
            head_m = [self.metric.surface_to_meters((x, y)) for x, y in zip(rx[:4], ry[:4])]
            if any(self._distance_m(p, hero_xy) <= self.exclusion_radius_m for p in head_m):
                continue
            self._redispatch(actor, rx, ry)
            self.stats.recycled += 1
            self.stats.events.append(
                {"t_s": float(sim_time_s), "actor_id": id(actor), "start_m": [float(v) for v in head_m[0]], "route_m": self.metric.route_length_meters(rx, ry)}
            )
            return True
        self.stats.respawn_failures += 1
        return False

    def _redispatch(self, actor, rx, ry) -> None:
        actor.rx, actor.ry = list(rx), list(ry)
        actor._initial_rx, actor._initial_ry = list(rx), list(ry)
        cruise = float(actor.cruise_speed_mps)
        if self.startup_sampler is not None:
            startup = self.startup_sampler(cruise, rng=self.rng)
            actor.set_initial_speed_mps(startup.get("initial_speed_mps", 0.0))
            actor.set_startup_ramp_duration(startup.get("ramp_duration_s", 1.0))
        else:
            actor.set_initial_speed_mps(0.0)
            actor.set_startup_ramp_duration(1.0)
        actor.np_rng = self.np_rng
        actor.reset()
        actor.recycle_count = getattr(actor, "recycle_count", 0) + 1
