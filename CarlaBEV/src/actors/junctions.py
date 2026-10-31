"""Junction zones for right-of-way decisions.

A junction is modelled as a circular *zone* (centre + radius, metres) around
the crossing point of the road centrelines of the canonical vehicle graph's
junction nodes. Right-of-way is decided on zones, not on lane topology: a
vehicle approaching a zone yields to crossing traffic already inside it and to
crossing traffic arriving earlier from another approach. This is deliberately
simple (no turn-specific conflict matrix) but deterministic, map-driven and
cheap (12 zones in Town01).

Only junctions with at least three arms are zones: the canonical graph also
tags bends and short straights as "junctions" (2 arms) where no crossing
traffic exists. Zone centres are cached per town because the canonical graph
lives in the raw asset frame and is converted through the metric frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
import pickle

import numpy as np

from CarlaBEV.envs.geometry import MetricFrame
from CarlaBEV.envs.utils import asset_path


@dataclass(frozen=True)
class JunctionZone:
    junction_id: str
    x_m: float
    y_m: float
    radius_m: float
    arms: tuple = ()
    """Sides ('N','S','E','W') with an approach road."""

    def distance_to_edge_m(self, x_m: float, y_m: float) -> float:
        """Signed distance from a point to the zone boundary (negative = inside)."""
        return math.hypot(x_m - self.x_m, y_m - self.y_m) - self.radius_m

    def contains(self, x_m: float, y_m: float) -> bool:
        return self.distance_to_edge_m(x_m, y_m) <= 0.0


class JunctionMap:
    """Set of junction zones in metres."""

    def __init__(self, zones: list[JunctionZone]):
        self.zones = list(zones)
        self._xy = np.asarray(
            [[z.x_m, z.y_m] for z in self.zones], dtype=float
        ).reshape(-1, 2)
        self._r = np.asarray([z.radius_m for z in self.zones], dtype=float)

    def __len__(self):
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)

    def nearest(self, x_m: float, y_m: float) -> tuple[JunctionZone | None, float]:
        """Nearest zone and the signed distance to its edge."""
        if not self.zones:
            return None, float("inf")
        d = np.hypot(self._xy[:, 0] - x_m, self._xy[:, 1] - y_m) - self._r
        i = int(np.argmin(d))
        return self.zones[i], float(d[i])

    def zone_ahead(
        self, x_m: float, y_m: float, yaw: float, lookahead_m: float
    ) -> tuple[JunctionZone | None, float]:
        """Zone the heading ray enters within ``lookahead_m`` (or that contains the point).

        Returns ``(zone, distance_to_edge_m)`` with ``distance_to_edge_m <= 0`` when
        inside. The ray test keeps vehicles driving *away* from a junction from
        yielding to it.
        """
        if not self.zones:
            return None, float("inf")
        dx = self._xy[:, 0] - x_m
        dy = self._xy[:, 1] - y_m
        c, s = math.cos(yaw), math.sin(yaw)
        along = dx * c + dy * s
        lateral = -dx * s + dy * c
        dist = np.hypot(dx, dy)
        inside = dist <= self._r
        if np.any(inside):
            i = int(np.argmin(np.where(inside, dist, np.inf)))
            return self.zones[i], float(dist[i] - self._r[i])
        hits = (along > 0.0) & (np.abs(lateral) <= self._r)
        if not np.any(hits):
            return None, float("inf")
        # Entry distance along the ray to the circle boundary.
        entry = along - np.sqrt(np.maximum(self._r**2 - lateral**2, 0.0))
        entry = np.where(hits, entry, np.inf)
        i = int(np.argmin(entry))
        if entry[i] > lookahead_m:
            return None, float("inf")
        return self.zones[i], float(entry[i])

    @classmethod
    def from_canonical_graph(
        cls,
        metric: MetricFrame,
        *,
        town_name: str = "Town01",
        radius_m: float = 7.0,
        min_arms: int = 3,
    ) -> "JunctionMap":
        zones = []
        for junction_id, (rx, ry, arms) in sorted(
            junction_geometry_raw(town_name).items()
        ):
            if len(arms) < min_arms:
                continue
            x_m, y_m = metric.raw_to_meters((rx, ry))
            zones.append(
                JunctionZone(
                    junction_id, float(x_m), float(y_m), float(radius_m), tuple(arms)
                )
            )
        return cls(zones)


def junction_geometry(graph) -> dict[str, tuple[float, float, tuple[str, ...]]]:
    """Per junction id: ``(x_raw, y_raw, arms)``.

    The centre is the crossing of the road centrelines: mean ``x`` of the nodes
    whose lane heading runs north-south and mean ``y`` of the nodes whose lane
    runs east-west (the plain node centroid is biased by ~4 m towards the arm
    with more nodes). Arms are the sides holding nodes further than 2 m from
    the centre.
    """
    grouped: dict[str, list] = {}
    for _, data in graph.nodes(data=True):
        junction_id = data.get("junction_id")
        if junction_id is None:
            continue
        grouped.setdefault(str(junction_id), []).append(
            (tuple(data["pos_raw"]), data.get("heading_rad"))
        )

    out = {}
    for junction_id, items in grouped.items():
        xs_ns = [p[0] for p, h in items if h is not None and abs(math.sin(h)) > 0.7]
        ys_ew = [p[1] for p, h in items if h is not None and abs(math.sin(h)) <= 0.7]
        cx = (
            float(np.mean(xs_ns)) if xs_ns else float(np.mean([p[0] for p, _ in items]))
        )
        cy = (
            float(np.mean(ys_ew)) if ys_ew else float(np.mean([p[1] for p, _ in items]))
        )
        sides = set()
        for (x, y), _ in items:
            dx, dy = x - cx, y - cy
            if math.hypot(dx, dy) < 0.05 * 1024:  # ~2 m in the 1024-raw / 40 m frame
                continue
            sides.add(
                ("E" if dx > 0 else "W")
                if abs(dx) > abs(dy)
                else ("S" if dy > 0 else "N")
            )
        out[junction_id] = (cx, cy, tuple(sorted(sides)))
    return out


@lru_cache(maxsize=4)
def junction_geometry_raw(
    town_name: str,
) -> dict[str, tuple[float, float, tuple[str, ...]]]:
    path = os.path.join(
        asset_path, town_name, f"{town_name.lower()}-canonical-vehicle.pkl"
    )
    with open(path, "rb") as handle:
        graph = pickle.load(handle)
    return junction_geometry(graph)
