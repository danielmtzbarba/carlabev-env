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


def _capture_spawn_state(*, seed: int, anchor_y_frac: float) -> dict[str, object]:
    env = _make_env(anchor_y_frac=anchor_y_frac)
    options = build_random_navigation_options(
        RandomNavigationReset(difficulty_id="rt_medium_v1")
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


if __name__ == "__main__":
    unittest.main()
