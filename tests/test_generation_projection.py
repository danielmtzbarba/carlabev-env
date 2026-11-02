"""Stage 4: the batch point-to-route projection must be bit-identical to the legacy loop."""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from CarlaBEV.src.managers.scene_generator import SceneGenerator  # noqa: E402
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT  # noqa: E402


def legacy_project_point_to_route(metric, point, route_points):
    """Verbatim copy of the pre-Stage-4 implementation (the reference)."""
    if len(route_points) < 2:
        return None
    best = None
    cumulative_surface = 0.0
    point = np.asarray(point, dtype=float)
    for start, end in zip(route_points[:-1], route_points[1:]):
        seg = end - start
        seg_len = np.linalg.norm(seg)
        if seg_len <= 1e-9:
            continue
        t = float(np.clip(np.dot(point - start, seg) / (seg_len ** 2), 0.0, 1.0))
        proj = start + (t * seg)
        dist_surface = float(np.linalg.norm(point - proj))
        tangent = seg / seg_len
        progress_surface = cumulative_surface + (t * seg_len)
        if best is None or dist_surface < best["dist_surface"]:
            best = {
                "dist_surface": dist_surface,
                "dist_m": metric.distance_surface_to_meters(dist_surface),
                "progress_surface": progress_surface,
                "progress_m": metric.distance_surface_to_meters(progress_surface),
                "tangent": tangent,
                "projection": proj,
            }
        cumulative_surface += seg_len
    return best


def _assert_bitwise_equal(test, a, b, label):
    if a is None or b is None:
        test.assertIs(a, b, label)
        return
    for key in ("dist_surface", "dist_m", "progress_surface", "progress_m"):
        test.assertEqual(float(a[key]), float(b[key]), f"{label}: {key}")
    for key in ("tangent", "projection"):
        test.assertTrue(np.array_equal(np.asarray(a[key]), np.asarray(b[key])), f"{label}: {key}")


class BatchProjectionBitIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = SceneGenerator(None, units=DEFAULT_UNIT_CONTRACT)
        cls.metric = DEFAULT_UNIT_CONTRACT.metric
        cls.planner = cls.gen.planners.get("vehicle-canonical")
        cls.node_ids = list(cls.planner.nodes.get("vehicle", []))
        cls.node_pos = np.asarray([cls.planner.get_node_pos_surface(n) for n in cls.node_ids], dtype=float)

    def _routes(self, count, rng):
        """Real graph routes (inexact 2^-19 coordinates) plus synthetic degenerate ones."""
        from CarlaBEV.src.scenes.utils import find_route_in_range

        routes = []
        for _ in range(count):
            _, path, _, _ = find_route_in_range(
                self.gen.planners.all, "agent", "vehicle", 30, 120, rng=rng,
                planner_key="vehicle-canonical", node_cls="vehicle", turn_metric_mode="semantic",
            )
            if path is not None and len(path[0]) > 1:
                routes.append(np.column_stack((np.asarray(path[0], float), np.asarray(path[1], float))))
        # Degenerate: repeated vertex (zero-length segment) and a point exactly on a vertex.
        routes.append(np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 10.0], [20.0, 30.0], [40.0, 30.5]]))
        return routes

    def test_graph_nodes_project_identically(self):
        import random

        rng = random.Random(4)
        for route in self._routes(6, rng):
            batch = self.gen._project_points_to_route(self.node_pos, route)
            self.assertEqual(len(batch), len(self.node_pos))
            for node, got in zip(self.node_pos, batch):
                _assert_bitwise_equal(self, got, legacy_project_point_to_route(self.metric, node, route), f"node {node}")
                _assert_bitwise_equal(self, got, self.gen._project_point_to_route(node, route), f"ref {node}")

    def test_random_inexact_points_project_identically(self):
        import random

        rng = random.Random(5)
        np_rng = np.random.default_rng(5)
        for route in self._routes(4, rng):
            lo, hi = route.min(axis=0) - 50.0, route.max(axis=0) + 50.0
            points = np_rng.uniform(lo, hi, size=(400, 2))
            # Include exact vertices and mid-points (ties between adjacent segments).
            points = np.vstack([points, route, (route[:-1] + route[1:]) / 2.0])
            batch = self.gen._project_points_to_route(points, route)
            for point, got in zip(points, batch):
                _assert_bitwise_equal(self, got, legacy_project_point_to_route(self.metric, point, route), f"pt {point}")

    def test_candidate_summary_and_role_decisions_match_the_legacy_path(self):
        """Role decisions must be identical to the exhaustive legacy projection.

        Far-away candidate points are pruned (``_ROLE_DISTANCE_CUTOFF_M``), which
        may only change ``min_dist_m`` when it is above the cutoff on both sides
        of every threshold; every other summary field must be bit-identical.
        """
        import random

        rng = random.Random(6)
        routes = self._routes(12, rng)
        cutoff = self.gen._ROLE_DISTANCE_CUTOFF_M
        compared = 0
        for ego in routes[:3]:
            for candidate in routes:
                if candidate is ego:
                    continue
                summary = self.gen._summarize_candidate_route(ego, candidate)
                original = self.gen._project_points_to_route
                self.gen._project_points_to_route = lambda pts, rp, skip=None: [
                    legacy_project_point_to_route(self.metric, p, rp) for p in np.asarray(pts, float)
                ]
                try:
                    legacy_summary = self.gen._summarize_candidate_route(ego, candidate)
                    legacy_decisions = {
                        role: self.gen._route_matches_role(role, ego, candidate)
                        for role in ("lead", "rear", "cross_path", "opposite_lane")
                    }
                finally:
                    self.gen._project_points_to_route = original
                decisions = {
                    role: self.gen._route_matches_role(role, ego, candidate)
                    for role in ("lead", "rear", "cross_path", "opposite_lane")
                }
                self.assertEqual(decisions, legacy_decisions)
                for key in ("lead_progress_m", "opposite_progress_m", "cross_progress_m", "rear_longitudinal_m", "start_dot"):
                    self.assertEqual(summary[key], legacy_summary[key], key)
                if legacy_summary["min_dist_m"] is not None and legacy_summary["min_dist_m"] <= cutoff:
                    self.assertEqual(summary["min_dist_m"], legacy_summary["min_dist_m"])
                    self.assertEqual(summary["min_dist_progress_m"], legacy_summary["min_dist_progress_m"])
                else:
                    self.assertTrue(summary["min_dist_m"] is None or summary["min_dist_m"] > 6.0)
                compared += 1
        self.assertGreater(compared, 20)

    def test_far_mask_is_a_sound_lower_bound(self):
        """A point marked far must be farther than the cutoff from every route point (exact check)."""
        import random

        rng = random.Random(8)
        np_rng = np.random.default_rng(8)
        cutoff_m = self.gen._ROLE_DISTANCE_CUTOFF_M
        checked = 0
        for route in self._routes(4, rng):
            lo, hi = route.min(axis=0) - 60.0, route.max(axis=0) + 60.0
            points = np.vstack([np_rng.uniform(lo, hi, size=(300, 2)), route])
            far = self.gen._far_from_route_mask(points, route)
            for point, is_far in zip(points, far):
                exact = legacy_project_point_to_route(self.metric, point, route)
                if is_far:
                    self.assertGreater(exact["dist_m"], 10.0 + 1e-9)  # widest role threshold
                    self.assertGreater(exact["dist_m"], cutoff_m - 1e-6)
                    checked += 1
        self.assertGreater(checked, 100)

    def test_node_projection_cache_is_per_scene_and_role_independent(self):
        import random

        rng = random.Random(7)
        ego = self._routes(1, rng)[0]
        self.gen._scene_projection_cache = {}
        lead = self.gen._candidate_node_ids_for_role(self.planner, ego, "lead")
        self.assertEqual(len(self.gen._scene_projection_cache), 1)
        self.gen._candidate_node_ids_for_role(self.planner, ego, "rear")
        self.gen._candidate_node_ids_for_role(self.planner, ego, "opposite_lane")
        self.assertEqual(len(self.gen._scene_projection_cache), 1)  # vehicle nodes projected once
        self.gen._candidate_node_ids_for_role(self.planner, ego, "cross_path")
        self.assertEqual(len(self.gen._scene_projection_cache), 2)  # + intersection nodes
        # Same answer with a cold cache.
        self.gen._scene_projection_cache = {}
        self.assertEqual(lead, self.gen._candidate_node_ids_for_role(self.planner, ego, "lead"))


if __name__ == "__main__":
    unittest.main()
