"""Stage 7 — traffic realism: spawn occupancy, role pools, feasible NPC route geometry (findings V-01..V-03)."""

from __future__ import annotations

import math
import os
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options  # noqa: E402
from CarlaBEV.envs.carlabev import CarlaBEV  # noqa: E402
from CarlaBEV.src.actors.vehicle import Vehicle  # noqa: E402
from CarlaBEV.src.control.route_geometry import (  # noqa: E402
    curvature_speed_profile,
    fillet_route,
    nudge_from_blocking,
    route_has_uturn,
)
from CarlaBEV.src.geometry.collision import bodies_overlap, gap_between  # noqa: E402
from CarlaBEV.src.managers.scene_generator import GENERATION_POLICIES, generation_policy_for  # noqa: E402
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT, NpcMotion, UnitContract  # noqa: E402


def _env(**overrides):
    kwargs = dict(render_mode="rgb_array", obs_mode="bev_rgb", action_mode="continuous", size=128, scene_library_enabled=False)
    kwargs.update(overrides)
    return CarlaBEV(EnvConfig(**kwargs))


def _options(seed, **overrides):
    kwargs = dict(difficulty_preset_id="medium", scene_seed=seed)
    kwargs.update(overrides)
    return build_random_navigation_options(RandomNavigationReset(**kwargs))


class GenerationPolicyTests(unittest.TestCase):
    def test_policies_are_versioned_and_v3_is_the_default(self):
        v2, v3 = GENERATION_POLICIES["role_traffic_v2"], GENERATION_POLICIES["role_traffic_v3"]
        self.assertFalse(v2.spawn_occupancy or v2.unique_start_nodes or v2.reachable_end_nodes or v2.reject_uturn_routes)
        self.assertEqual(v2.rear_pool_mode, "projection")
        self.assertTrue(v3.spawn_occupancy and v3.unique_start_nodes and v3.reachable_end_nodes and v3.reject_uturn_routes)
        self.assertEqual(v3.min_spawn_gap_m, 2.0)
        self.assertIs(generation_policy_for(None), v3)
        self.assertIs(generation_policy_for("custom_namespace"), v3)
        self.assertIs(generation_policy_for("role_traffic_v2"), v2)
        self.assertEqual(EnvConfig().scene_library_generator_version, "role_traffic_v3")
        env = _env()
        try:
            self.assertEqual(env.scene_generator.policy.version, "role_traffic_v3")
        finally:
            env.close()

    def test_v2_policy_reproduces_the_legacy_generator(self):
        """Same seed under v2 and v3 differ (spawn occupancy); v2 keeps the historical scene."""
        a = _env(scene_library_generator_version="role_traffic_v2")
        b = _env()
        try:
            a.reset(seed=2, options=_options(2))
            b.reset(seed=2, options=_options(2))
            self.assertEqual(a.scene_generator.policy.version, "role_traffic_v2")
            self.assertEqual(a.scene_spec.generator_version, "role_traffic_v2")
            self.assertNotEqual(
                [n.route_x[:3] for n in a.scene_spec.npcs], [n.route_x[:3] for n in b.scene_spec.npcs]
            )
            # The v2 realization is the one the pinned goldens describe: same-node overlaps exist.
            vehicles = a.map.actor_manager.actors["vehicle"]
            same_node = any(
                a.scene_spec.npcs[i].start_node_id == a.scene_spec.npcs[j].start_node_id
                for i in range(len(vehicles)) for j in range(i + 1, len(vehicles))
            )
            self.assertTrue(same_node)
        finally:
            a.close()
            b.close()


class SpawnOccupancyTests(unittest.TestCase):
    def test_no_overlapping_spawns_unique_start_nodes_and_exact_density(self):
        env = _env()
        try:
            for preset, seed in (("medium", 0), ("hard", 1), ("hard", 5)):
                _, info = env.reset(seed=seed, options=_options(seed, difficulty_preset_id=preset))
                vehicles = env.map.actor_manager.actors["vehicle"]
                hero = env.map.hero
                for i in range(len(vehicles)):
                    self.assertFalse(bodies_overlap(hero.body(), vehicles[i].body()))
                    for j in range(i + 1, len(vehicles)):
                        self.assertFalse(bodies_overlap(vehicles[i].body(), vehicles[j].body()), (preset, seed, i, j))
                        # policy gap (2 m) minus the +/-0.3125 m spawn jitter on both bodies
                        self.assertGreaterEqual(gap_between(vehicles[i].body(), vehicles[j].body()), 2.0 - 4 * 0.3125)
                starts = [n.start_node_id for n in env.scene_spec.npcs]
                self.assertEqual(len(starts), len(set(starts)))
                self.assertGreaterEqual(info["scenario"]["spawned_num_vehicles"], info["scenario"]["requested_num_vehicles"] - 1)
                self.assertIn("vehicle_spawn_occupancy_rejections", info["scenario"])
                self.assertEqual(info["scenario"]["scenario_param_generation_policy"], "role_traffic_v3")
        finally:
            env.close()

    def test_spawn_validation_rejects_npc_overlaps_only_via_generation(self):
        env = _env()
        try:
            env.reset(seed=3, options=_options(3))
            self.assertTrue(env.map.spawn_validation_info()["valid"])
        finally:
            env.close()


class RolePoolTests(unittest.TestCase):
    def test_rear_role_spawns_behind_the_ego_in_metres(self):
        env = _env()
        try:
            found = 0
            for seed in range(4):
                env.reset(seed=seed, options=_options(seed, num_vehicles=6, num_vehicles_near_ego=4, guaranteed_candidate_role="rear"))
                rears = [v for v in env.map.actor_manager.actors["vehicle"] if v.traffic_role == "rear"]
                hero = env.map.hero
                for v in rears:
                    rel = np.array([v._controller.x_m - hero.x_m, v._controller.y_m - hero.y_m])
                    longitudinal = float(rel[0] * math.cos(hero.yaw) + rel[1] * math.sin(hero.yaw))
                    self.assertLess(longitudinal, -4.0)  # behind the ego, in metres (legacy window was in pixels)
                    self.assertGreater(longitudinal, -30.0)
                found += bool(rears)
            self.assertGreaterEqual(found, 2)
        finally:
            env.close()

    def test_cross_path_candidates_start_upstream_of_a_junction(self):
        env = _env()
        try:
            found = 0
            for seed in range(6):
                env.reset(seed=seed, options=_options(seed, num_vehicles=6, num_vehicles_near_ego=4, guaranteed_candidate_role="cross_path", intersection_required=True))
                found += any(v.traffic_role == "cross_path" for v in env.map.actor_manager.actors["vehicle"])
            self.assertGreaterEqual(found, 1)
        finally:
            env.close()


class RouteGeometryTests(unittest.TestCase):
    def test_sharp_corner_becomes_a_feasible_arc(self):
        ax = [0, 4, 8, 12, 16, 20, 20, 20, 20, 20, 20]
        ay = [0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20]
        cx, cy, cyaw, ck, s = fillet_route(ax, ay, target_radius_m=6.0, min_radius_m=5.0)
        arc = np.abs(ck) > 0
        self.assertAlmostEqual(1.0 / np.abs(ck[arc]).max(), 6.0, places=6)
        self.assertAlmostEqual(float(s[-1]), 40 - 12 + math.pi / 2 * 6, delta=0.05)  # polygonised arc
        self.assertLess(np.degrees(np.abs(np.diff(np.unwrap(cyaw)))).max(), 12.0)
        self.assertLessEqual(np.hypot(np.diff(cx), np.diff(cy)).max(), 1.0 + 1e-9)

    def test_junction_connectors_are_merged_into_one_corner(self):
        pts = np.array([[263.8, 230.9], [263.8, 104.9], [263.8, 100.2], [260.2, 95.3], [256.7, 95.1], [124.2, 95.1]])
        cx, cy, cyaw, ck, s = fillet_route(pts[:, 0], pts[:, 1], target_radius_m=6.0, min_radius_m=5.02)
        arc = np.abs(ck) > 1e-9
        self.assertAlmostEqual(1.0 / np.abs(ck[arc]).max(), 6.0, places=6)
        self.assertLess(np.degrees(np.abs(np.diff(np.unwrap(cyaw)))).max(), 12.0)

    def test_gentle_curves_are_preserved_and_radius_adapts_to_the_road(self):
        t = np.linspace(0, math.pi / 2, 30)
        cx, cy, _, _, _ = fillet_route(40 * np.cos(t), 40 * np.sin(t), target_radius_m=6.0, min_radius_m=5.0)
        self.assertLess(np.abs(np.hypot(cx, cy) - 40).max(), 0.4)
        ax = [0, 4, 8, 12, 16, 20, 20, 20, 20, 20, 20]
        ay = [0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20]

        def narrow(x, y):
            return (abs(y) <= 1.75 and x <= 21.75) or (abs(x - 20) <= 1.75 and y >= -1.75)

        _, _, _, ck, _ = fillet_route(ax, ay, target_radius_m=6.0, min_radius_m=5.0, is_drivable=narrow, inner_clearance_m=0.9)
        radius = 1.0 / np.abs(ck[np.abs(ck) > 0]).max()
        self.assertGreaterEqual(radius, 5.0)
        self.assertLess(radius, 6.0)

    def test_uturn_detection_and_speed_profile(self):
        self.assertTrue(route_has_uturn([0, 10, 20, 20, 10, 0], [0, 0, 0, 3.2, 3.2, 3.2]))
        self.assertFalse(route_has_uturn([0, 10, 20, 20, 20], [0, 0, 0, 10, 20]))
        ck = np.array([0.0, 0.0, 0.0, 1 / 6.0, 1 / 6.0, 0.0])
        s = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
        cap = curvature_speed_profile(ck, s, lat_accel_mps2=2.5, decel_mps2=2.0, v_max_mps=12.0)
        self.assertAlmostEqual(float(cap[3]), math.sqrt(2.5 * 6.0))
        self.assertAlmostEqual(float(cap[2]), math.sqrt(cap[3] ** 2 + 2 * 2.0 * 5.0))
        self.assertEqual(float(cap[5]), 12.0)

    def test_nudge_moves_the_path_away_from_offroad_pixels(self):
        clearance = np.tile(np.arange(20, dtype=float)[None, :], (20, 1))  # distance grows with x
        cx = np.full(10, 2.0 * 0.3125)  # 2 px from the edge
        cy = np.linspace(2.0, 5.0, 10)
        nx, ny = nudge_from_blocking(cx, cy, clearance, 0.3125, target_clearance_m=1.2, max_shift_m=0.6)
        self.assertTrue(np.all(nx > cx))
        self.assertLessEqual(float((nx - cx).max()), 0.6 + 1e-9)
        self.assertTrue(np.allclose(ny, cy))


class NpcMotionIntegrationTests(unittest.TestCase):
    def test_default_units_use_fillet_geometry_and_legacy_is_selectable(self):
        self.assertEqual(DEFAULT_UNIT_CONTRACT.npc_motion.route_geometry, "fillet_v1")
        legacy = UnitContract.from_env_config(EnvConfig(render_mode="rgb_array", npc_route_geometry="legacy"))
        self.assertEqual(legacy.npc_motion.route_geometry, "legacy")
        with self.assertRaises(ValueError):
            NpcMotion(route_geometry="spline")

    def test_vehicle_slows_for_a_corner_and_pedestrians_keep_legacy_paths(self):
        route_x = [0, 4, 8, 12, 16, 20, 24, 28, 32, 32, 32, 32, 32, 32, 32, 32]  # px: east then south
        route_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 24, 28]
        route_x = [v * 3.2 for v in route_x]  # ~1 m per unit at 0.3125 m/px
        route_y = [v * 3.2 for v in route_y]
        vehicle = Vehicle(map_size=128, routeX=route_x, routeY=route_y, target_speed=12.0)
        vehicle.reset()
        ctrl = vehicle._controller
        self.assertEqual(ctrl.route_smoothing_mode, "fillet_v1")
        self.assertAlmostEqual(ctrl.min_turn_radius_m, 2.9 / math.tan(math.radians(30)), places=6)
        self.assertIsNotNone(ctrl.curve_speed_cap)
        arc = np.abs(ctrl.ck) > 0
        self.assertTrue(arc.any())
        self.assertLessEqual(float(ctrl.curve_speed_cap[arc].min()), math.sqrt(2.5 * 6.0) + 1e-6)
        # Drive the corner: the speed never exceeds the curvature cap by more than the loop tolerance.
        max_ratio = 0.0
        for _ in range(200):
            vehicle.step(0.1, 0.1)
            cap = ctrl.curve_speed_cap_mps()
            if math.isfinite(cap) and cap < 11.0:
                max_ratio = max(max_ratio, ctrl.v_mps / cap)
        self.assertLess(max_ratio, 1.25)
        from CarlaBEV.src.actors.pedestrian import Pedestrian

        ped = Pedestrian(map_size=128, routeX=[0, 5, 10], routeY=[0, 0, 0], target_speed=1.5)
        ped.reset()
        self.assertIn(ped._controller.route_smoothing_mode, {"smoothed", "raw_fallback"})


if __name__ == "__main__":
    unittest.main()
