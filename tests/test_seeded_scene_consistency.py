import math
import unittest

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
        scene_library_enabled=False,
    )
    return CarlaBEV(cfg)


def _capture_spawn_state(
    *,
    seed: int,
    anchor_y_frac: float,
    route_seed: int | None = None,
    traffic_seed: int | None = None,
    scene_seed: int | None = None,
    reset_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    env = _make_env(anchor_y_frac=anchor_y_frac)
    reset_kwargs = dict(reset_kwargs or {})
    options = build_random_navigation_options(
        RandomNavigationReset(
            difficulty_preset_id=reset_kwargs.pop("difficulty_preset_id", "medium"),
            scene_seed=scene_seed,
            route_seed=route_seed,
            traffic_seed=traffic_seed,
            **reset_kwargs,
        )
    )
    try:
        _, info = env.reset(seed=seed, options=options)
        hero = env.map.hero
        route_x, route_y = env.map.route
        vehicles = []
        vehicle_initial_speeds_mps = []
        vehicle_cruise_speeds_mps = []
        vehicle_speed_classes = []
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
            vehicle_initial_speeds_mps.append(round(float(actor.initial_speed_mps), 6))
            vehicle_cruise_speeds_mps.append(round(float(actor.cruise_speed_mps), 6))
            vehicle_speed_classes.append(str(actor.speed_class))
        return {
            "spawn_validation": info.get("spawn_validation"),
            "scene_descriptor": info.get("scene_descriptor"),
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
            "vehicle_initial_speeds_mps": tuple(vehicle_initial_speeds_mps[:16]),
            "vehicle_cruise_speeds_mps": tuple(vehicle_cruise_speeds_mps[:16]),
            "vehicle_speed_classes": tuple(vehicle_speed_classes[:16]),
            "num_vehicles": int(env.num_vehicles),
            "route_length": round(float(env.len_ego_route), 6),
            "scene_seed": info.get("scenario", {}).get("scenario_param_scene_seed"),
            "route_seed": info.get("scenario", {}).get("scenario_param_route_seed"),
            "traffic_seed": info.get("scenario", {}).get("scenario_param_traffic_seed"),
            "base_vehicle_target_speed_mps": info.get("scenario", {}).get("scenario_param_vehicle_target_speed"),
            "vehicle_speed_mix": info.get("scenario", {}).get("scenario_param_vehicle_speed_mix"),
            "requested_num_vehicles": info.get("scenario", {}).get("requested_num_vehicles"),
            "spawned_num_vehicles": info.get("scenario", {}).get("spawned_num_vehicles"),
            "vehicle_spawn_shortfall": info.get("scenario", {}).get("vehicle_spawn_shortfall"),
            "traffic_count_exact": info.get("scenario", {}).get("traffic_count_exact"),
            "vehicle_spawn_attempts": info.get("scenario", {}).get("vehicle_spawn_attempts"),
            "vehicle_spawn_retry_limit": info.get("scenario", {}).get("scenario_param_vehicle_spawn_retry_limit"),
        }
    finally:
        env.close()


def _capture_render_alignment(*, seed: int, anchor_y_frac: float) -> dict[str, object]:
    env = _make_env(anchor_y_frac=anchor_y_frac)
    options = build_random_navigation_options(
        RandomNavigationReset(difficulty_preset_id="medium")
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

    def test_route_seed_changes_route_and_can_retarget_nearby_traffic(self):
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
        self.assertEqual(first["traffic_seed"], second["traffic_seed"])
        self.assertEqual(
            first["scene_descriptor"]["traffic_role_profile"],
            second["scene_descriptor"]["traffic_role_profile"],
        )
        self.assertNotEqual(first["vehicles"], second["vehicles"])

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

    def test_random_traffic_vehicles_spawn_below_cruise_speed(self):
        snapshot = _capture_spawn_state(seed=11, anchor_y_frac=0.5)

        self.assertGreater(len(snapshot["vehicle_initial_speeds_mps"]), 0)
        for initial_speed_mps, cruise_speed_mps in zip(
            snapshot["vehicle_initial_speeds_mps"],
            snapshot["vehicle_cruise_speeds_mps"],
        ):
            self.assertGreaterEqual(initial_speed_mps, 0.0)
            self.assertLessEqual(initial_speed_mps, 0.5 * cruise_speed_mps + 1e-6)
            self.assertLessEqual(initial_speed_mps, cruise_speed_mps)

    def test_random_traffic_vehicle_cruise_speeds_follow_sampled_speed_classes(self):
        snapshot = _capture_spawn_state(seed=11, anchor_y_frac=0.5)

        self.assertEqual(
            snapshot["vehicle_speed_mix"],
            {"slow": 0.55, "medium": 0.35, "fast": 0.1},
        )
        base_speed = float(snapshot["base_vehicle_target_speed_mps"])
        class_bounds = {
            "slow": (0.80 * 0.95 * base_speed, 0.80 * 1.05 * base_speed),
            "medium": (1.00 * 0.95 * base_speed, 1.00 * 1.05 * base_speed),
            "fast": (1.15 * 0.95 * base_speed, 1.15 * 1.05 * base_speed),
        }

        self.assertGreater(len(snapshot["vehicle_speed_classes"]), 0)
        for speed_class, cruise_speed_mps in zip(
            snapshot["vehicle_speed_classes"],
            snapshot["vehicle_cruise_speeds_mps"],
        ):
            self.assertIn(speed_class, class_bounds)
            lower, upper = class_bounds[speed_class]
            self.assertGreaterEqual(cruise_speed_mps, lower - 1e-6)
            self.assertLessEqual(cruise_speed_mps, upper + 1e-6)

    def test_random_traffic_records_requested_vs_spawned_vehicle_counts(self):
        snapshot = _capture_spawn_state(seed=11, anchor_y_frac=0.5)

        requested = int(snapshot["requested_num_vehicles"])
        spawned = int(snapshot["spawned_num_vehicles"])
        shortfall = int(snapshot["vehicle_spawn_shortfall"])
        retry_limit = int(snapshot["vehicle_spawn_retry_limit"])
        attempts = int(snapshot["vehicle_spawn_attempts"])

        self.assertEqual(requested, 16)
        self.assertEqual(spawned, snapshot["num_vehicles"])
        self.assertEqual(shortfall, requested - spawned)
        self.assertEqual(snapshot["traffic_count_exact"], spawned == requested)
        self.assertGreaterEqual(attempts, spawned)
        self.assertLessEqual(attempts, requested * retry_limit)

    def test_scene_descriptor_exposes_seeded_role_schema(self):
        snapshot = _capture_spawn_state(seed=11, anchor_y_frac=0.5)

        descriptor = snapshot["scene_descriptor"]
        self.assertIsInstance(descriptor, dict)
        self.assertEqual(descriptor["scene"], "rdm")
        self.assertEqual(descriptor["difficulty_preset_id"], "medium")
        self.assertEqual(descriptor["scene_seed"], snapshot["scene_seed"])
        self.assertEqual(descriptor["route_seed"], snapshot["route_seed"])
        self.assertEqual(descriptor["traffic_seed"], snapshot["traffic_seed"])
        self.assertEqual(descriptor["num_vehicles_near_ego"], 4)
        self.assertEqual(descriptor["traffic_role_profile"], "mix")
        self.assertEqual(descriptor["guaranteed_candidate_role"], "mix")
        self.assertEqual(descriptor["requested_num_vehicles_near_ego"], 4)
        self.assertIn("spawned_role_counts", descriptor)
        self.assertEqual(sum(descriptor["spawned_role_counts"].values()), snapshot["num_vehicles"])
        self.assertEqual(len(descriptor["actor_roles"]), snapshot["num_vehicles"])
        self.assertIn(
            descriptor["main_actor_role"],
            {"lead", "rear", "cross_path", "opposite_lane", "background"},
        )

    def test_scene_descriptor_exposes_explicit_scene_profile_id(self):
        snapshot = _capture_spawn_state(
            seed=11,
            anchor_y_frac=0.5,
            reset_kwargs={
                "scene_benchmark_id": "navigation_medium_v1",
                "scene_profile_id": "difficulty_hard",
                "scene_split": "train",
            },
        )

        descriptor = snapshot["scene_descriptor"]
        self.assertEqual(descriptor["scene_benchmark_id"], "navigation_medium_v1")
        self.assertEqual(descriptor["scene_profile_id"], "difficulty_hard")
        self.assertEqual(descriptor["scene_split"], "train")

    def test_explicit_role_based_reset_spawns_guaranteed_lead_candidate(self):
        snapshot = _capture_spawn_state(
            seed=11,
            anchor_y_frac=0.5,
            reset_kwargs={
                "scene_profile_id": "lead_focus",
                "num_vehicles": 4,
                "num_vehicles_near_ego": 4,
                "traffic_role_profile": "mix",
                "guaranteed_candidate_role": "lead",
            },
        )

        descriptor = snapshot["scene_descriptor"]
        self.assertEqual(descriptor["requested_num_vehicles_near_ego"], 4)
        self.assertGreaterEqual(descriptor["spawned_num_vehicles_near_ego"], 1)
        self.assertGreaterEqual(descriptor["spawned_role_counts"].get("lead", 0), 1)


if __name__ == "__main__":
    unittest.main()
