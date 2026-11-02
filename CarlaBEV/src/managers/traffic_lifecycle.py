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

Determinism: the lifecycle draws route choices from its own RNG derived from
the traffic seed (``derive_seed``, never ``hash()`` — Python string hashing is
randomized per process), and the respawn jitter from the recycled actor's own
``runtime_seed`` generator, so cache hits, misses and independent processes
produce the same recycling sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import numpy as np

from CarlaBEV.src.randomness import derive_seed


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
        min_spawn_gap_m: float = 0.0,
        reject_uturn_routes: bool = False,
    ):
        self.planners = planners
        self.units = units
        self.metric = units.metric
        self.rng = random.Random(f"traffic-lifecycle:{seed}")
        # Fallback jitter RNG for actors without a runtime seed (scenario/authored traffic).
        self.np_rng = np.random.default_rng(derive_seed(int(seed), "traffic-lifecycle"))
        self.fov_radius_m = float(fov_radius_m)
        self.margin_m = float(margin_m)
        self.min_route_m = float(min_route_m)
        self.max_attempts = int(max_attempts)
        self.startup_sampler = startup_sampler
        self.enabled = enabled
        # Spawn occupancy (generation policy v3): a respawn must keep this gap to every other body.
        self.min_spawn_gap_m = float(min_spawn_gap_m)
        self.reject_uturn_routes = bool(reject_uturn_routes)
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
    def step(self, finished_actors, hero, sim_time_s: float, others=None) -> int:
        """Recycle finished, recyclable, out-of-view vehicles. Returns the number recycled.

        ``others`` (all traffic actors) enables the spawn-occupancy check.
        """
        if not self.enabled or hero is None:
            return 0
        recycled = 0
        for actor in finished_actors:
            if not getattr(actor, "recyclable", False):
                continue
            if not self.is_out_of_view(actor, hero):
                self.stats.despawn_deferred += 1
                continue
            if self.respawn(actor, hero, sim_time_s, others=others):
                recycled += 1
        return recycled

    def _spawn_is_clear(self, actor, rx, ry, hero, others) -> bool:
        if self.min_spawn_gap_m <= 0.0:
            return True
        from CarlaBEV.src.geometry.body import Body
        from CarlaBEV.src.geometry.collision import gap_between

        x0, y0 = self.metric.surface_to_meters((float(rx[0]), float(ry[0])))
        x1, y1 = self.metric.surface_to_meters((float(rx[1]), float(ry[1])))
        body = Body(actor.footprint, float(x0), float(y0), float(math.atan2(y1 - y0, x1 - x0)), 0.0)
        for other in [hero, *(others or ())]:
            if other is None or other is actor or not hasattr(other, "body"):
                continue
            if gap_between(body, other.body()) < self.min_spawn_gap_m:
                return False
        return True

    def respawn(self, actor, hero, sim_time_s: float, others=None) -> bool:
        """Move ``actor`` onto a new route whose start lies outside the exclusion radius (and clear of traffic)."""
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
            if self.reject_uturn_routes:
                from CarlaBEV.src.control.route_geometry import route_has_uturn

                mpp = self.metric.meters_per_pixel
                if route_has_uturn(np.asarray(rx, dtype=float) * mpp, np.asarray(ry, dtype=float) * mpp):
                    continue
            # The first stretch of the route must also be out of view.
            head_m = [self.metric.surface_to_meters((x, y)) for x, y in zip(rx[:4], ry[:4])]
            if any(self._distance_m(p, hero_xy) <= self.exclusion_radius_m for p in head_m):
                continue
            if not self._spawn_is_clear(actor, rx, ry, hero, others):
                continue
            self._redispatch(actor, rx, ry)
            self.stats.recycled += 1
            self.stats.events.append(
                {
                    "t_s": float(sim_time_s),
                    "actor_id": id(actor),
                    "npc_id": getattr(actor, "npc_id", None),
                    "start_m": [float(v) for v in head_m[0]],
                    "route_m": self.metric.route_length_meters(rx, ry),
                }
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
        # A re-dispatched actor draws a fresh spawn jitter from its own runtime
        # RNG (per-NPC seed); the shared lifecycle RNG is only a fallback.
        if getattr(actor, "runtime_seed", None) is None or actor.np_rng is None:
            actor.np_rng = self.np_rng
        actor.clear_spawn_jitter()
        actor.reset()
        actor.recycle_count = getattr(actor, "recycle_count", 0) + 1
