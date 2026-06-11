import math
import unittest

import numpy as np

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options
from CarlaBEV.envs.carlabev import CarlaBEV


def _make_env(*, anchor_y_frac: float) -> CarlaBEV:
    cfg = EnvConfig(
        map_name="Town01",
        render_mode="rgb_array",
        obs_mode="bev_rgb",
        action_mode="discrete",
        size=128,
        ego_anchor_x_frac=0.5,
        ego_anchor_y_frac=anchor_y_frac,
    )
    return CarlaBEV(cfg)


def _capture_spawn_state(
    *,
    seed: int,
    anchor_y_frac: float,
    route_seed: int | None = None,
    traffic_seed: int | None = None,
    scene_seed: int | None = None,
) -> dict[str, object]:
    env = _make_env(anchor_y_frac=anchor_y_frac)
    options = build_random_navigation_options(
        RandomNavigationReset(
            difficulty_id="rt_medium_v1",
            scene_seed=scene_seed,
            route_seed=route_seed,
            traffic_seed=traffic_seed,
        )
    )
    try:
        _, info = env.reset(seed=seed, options=options)
        hero = env.map.hero
        route_x, route_y = env.map.route
        vehicles = []
        for actor in env.map.actor_manager.actors.get("vehicle", []):
            ax, ay, ayaw, av = actor.state
            vehicles.append(
                (
                    round(float(ax), 4),
                    round(float(ay), 4),
                    round(float(ayaw), 6),
                    round(float(av), 6),
                )
            )
        return {
            "spawn_validation": info.get("spawn_validation"),
            "hero_pose": (
                round(float(hero.x), 4),
                round(float(hero.y), 4),
                round(float(hero.yaw), 6),
            ),
            "hero_speed": round(float(hero.v), 6),
            "route": (
                tuple(int(x) for x in route_x[:16]),
                tuple(int(y) for y in route_y[:16]),
                len(route_x),
            ),
            "vehicles": tuple(vehicles[:16]),
            "num_vehicles": int(env.num_vehicles),
            "route_length": round(float(env.len_ego_route), 6),
            "scene_seed": info.get("scenario", {}).get("scenario_param_scene_seed"),
            "route_seed": info.get("scenario", {}).get("scenario_param_route_seed"),
            "traffic_seed": info.get("scenario", {}).get("scenario_param_traffic_seed"),
        }
    finally:
        env.close()


def _capture_render_alignment(*, seed: int, anchor_y_frac: float) -> dict[str, object]:
    env = _make_env(anchor_y_frac=anchor_y_frac)
    options = build_random_navigation_options(
        RandomNavigationReset(difficulty_id="rt_medium_v1")
    )
    try:
        env.reset(seed=seed, options=options)
        crop_rect = env.map.compute_crop_rect()
        hero_surface = env.map.render_frame.world_to_surface(env.map.hero.position)
        return {
            "crop_size": int(env.map.fov_renderer.crop_size),
            "hero_in_crop": (
                float(hero_surface.x - crop_rect.x),
                float(hero_surface.y - crop_rect.y),
            ),
            "render_shape": tuple(int(v) for v in env.map._render_layers.render_shape),
            "crop_rect": (int(crop_rect.x), int(crop_rect.y), int(crop_rect.w), int(crop_rect.h)),
        }
    finally:
        env.close()


class SeededSceneConsistencyTests(unittest.TestCase):
    def test_same_seed_same_anchor_repeats_identical_spawn_state(self):
        first = _capture_spawn_state(seed=11, anchor_y_frac=0.5)
        second = _capture_spawn_state(seed=11, anchor_y_frac=0.5)

        self.assertEqual(first, second)

    def test_same_seed_across_anchors_preserves_world_spawn_state(self):
        center = _capture_spawn_state(seed=11, anchor_y_frac=0.5)
        lookahead = _capture_spawn_state(seed=11, anchor_y_frac=0.75)

        self.assertEqual(center["hero_pose"], lookahead["hero_pose"])
        self.assertEqual(center["hero_speed"], lookahead["hero_speed"])
        self.assertEqual(center["route"], lookahead["route"])
        self.assertEqual(center["vehicles"], lookahead["vehicles"])
        self.assertEqual(center["num_vehicles"], lookahead["num_vehicles"])
        self.assertTrue(center["spawn_validation"]["valid"])
        self.assertTrue(lookahead["spawn_validation"]["valid"])

    def test_spawn_validation_is_world_validity_not_anchor_specific_spawn(self):
        center = _capture_spawn_state(seed=7, anchor_y_frac=0.5)
        lookahead = _capture_spawn_state(seed=7, anchor_y_frac=0.75)

        self.assertEqual(center["hero_pose"], lookahead["hero_pose"])
        self.assertEqual(center["route"], lookahead["route"])
        self.assertEqual(center["vehicles"], lookahead["vehicles"])
        self.assertEqual(center["spawn_validation"]["reason"], "ok")
        self.assertEqual(lookahead["spawn_validation"]["reason"], "ok")

    def test_padded_render_keeps_ego_centered_in_crop_for_all_anchors(self):
        center = _capture_render_alignment(seed=7, anchor_y_frac=0.5)
        lookahead = _capture_render_alignment(seed=7, anchor_y_frac=0.75)

        for snapshot in (center, lookahead):
            expected = snapshot["crop_size"] / 2.0
            hx, hy = snapshot["hero_in_crop"]
            self.assertTrue(0 < snapshot["crop_rect"][0] < snapshot["render_shape"][0] - snapshot["crop_size"])
            self.assertTrue(0 < snapshot["crop_rect"][1] < snapshot["render_shape"][1] - snapshot["crop_size"])
            self.assertTrue(math.isclose(hx, expected, abs_tol=1.5))
            self.assertTrue(math.isclose(hy, expected, abs_tol=1.5))

    def test_same_seed_sequence_replays_identical_scene_sequence(self):
        first = [
            _capture_spawn_state(seed=seed, anchor_y_frac=0.5)
            for seed in (101, 102, 103)
        ]
        second = [
            _capture_spawn_state(seed=seed, anchor_y_frac=0.5)
            for seed in (101, 102, 103)
        ]

        self.assertEqual(first, second)

    def test_route_seed_changes_route_without_changing_traffic(self):
        first = _capture_spawn_state(
            seed=11,
            anchor_y_frac=0.5,
            route_seed=1001,
            traffic_seed=2001,
        )
        second = _capture_spawn_state(
            seed=11,
            anchor_y_frac=0.5,
            route_seed=1002,
            traffic_seed=2001,
        )

        self.assertNotEqual(first["route"], second["route"])
        self.assertEqual(first["vehicles"], second["vehicles"])

    def test_traffic_seed_changes_traffic_without_changing_route(self):
        first = _capture_spawn_state(
            seed=11,
            anchor_y_frac=0.5,
            route_seed=3001,
            traffic_seed=4001,
        )
        second = _capture_spawn_state(
            seed=11,
            anchor_y_frac=0.5,
            route_seed=3001,
            traffic_seed=4002,
        )

        self.assertEqual(first["route"], second["route"])
        self.assertNotEqual(first["vehicles"], second["vehicles"])


if __name__ == "__main__":
    unittest.main()
