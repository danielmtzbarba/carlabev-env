import tempfile
import unittest

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.src.managers.scene_library import SceneLibrary
from CarlaBEV.tools.build_scene_library import build_scene_library


class SceneLibraryTests(unittest.TestCase):
    def test_scene_library_supports_metadata_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/scene_library.db"
            library = SceneLibrary(db_path, generator_version="test_v1")

            library.insert_scene(
                scene_key="scene-a",
                scene_descriptor={
                    "scene": "rdm",
                    "difficulty_preset_id": "easy",
                    "route_profile": "mostly_straight",
                    "main_actor_role": "lead",
                    "spawned_num_vehicles": 2,
                    "spawned_num_vehicles_near_ego": 2,
                    "spawned_role_counts": {"lead": 1, "background": 1},
                    "actor_roles": ["lead", "background"],
                    "num_vehicles_near_ego": 2,
                    "traffic_role_profile": "mix",
                    "guaranteed_candidate_role": "lead",
                },
                scenario_context={"scene": "rdm"},
                actors_payload={"agent": None, "vehicle": [], "pedestrian": [], "target": [], "traffic_light": []},
                spawn_validation={"valid": True, "reason": "ok"},
                resolved_options={"scene": "rdm", "difficulty_preset_id": "easy"},
            )
            library.insert_scene(
                scene_key="scene-b",
                scene_descriptor={
                    "scene": "rdm",
                    "difficulty_preset_id": "hard",
                    "route_profile": "single_right",
                    "main_actor_role": "cross_path",
                    "spawned_num_vehicles": 4,
                    "spawned_num_vehicles_near_ego": 3,
                    "spawned_role_counts": {"cross_path": 2, "rear": 1, "background": 1},
                    "actor_roles": ["cross_path", "cross_path", "rear", "background"],
                    "num_vehicles_near_ego": 4,
                    "traffic_role_profile": "mix",
                    "guaranteed_candidate_role": "mix",
                },
                scenario_context={"scene": "rdm"},
                actors_payload={"agent": None, "vehicle": [], "pedestrian": [], "target": [], "traffic_light": []},
                spawn_validation={"valid": True, "reason": "ok"},
                resolved_options={"scene": "rdm", "difficulty_preset_id": "hard"},
            )

            results = library.get_scene_by_filters(
                difficulty_preset_id="hard",
                route_profile="single_right",
                main_actor_role="cross_path",
                required_role_counts={"cross_path": 2},
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["scene_key"], "scene-b")

    def test_env_reset_persists_scene_to_library(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/scene_library.db"
            env = CarlaBEV(
                EnvConfig(
                    map_name="Town01",
                    render_mode="rgb_array",
                    obs_mode="bev_rgb",
                    action_mode="discrete",
                    size=128,
                    scene_library_enabled=True,
                    scene_library_path=db_path,
                    scene_library_generator_version="test_cache_v1",
                )
            )
            options = build_random_navigation_options(
                RandomNavigationReset(
                    difficulty_preset_id="easy",
                    scene_seed=101,
                    route_seed=102,
                    traffic_seed=103,
                )
            )
            try:
                _, info_first = env.reset(seed=11, options=options)
                _, info_second = env.reset(seed=11, options=options)
            finally:
                env.close()

            library = SceneLibrary(db_path, generator_version="test_cache_v1")
            records = library.get_scene_by_filters(difficulty_preset_id="easy", limit=10)

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["scene_descriptor"]["difficulty_preset_id"], "easy")
            self.assertEqual(
                info_first["scene_descriptor"]["scene_seed"],
                info_second["scene_descriptor"]["scene_seed"],
            )
            self.assertFalse(info_first["scene_cache"]["hit"])
            self.assertTrue(info_second["scene_cache"]["hit"])
            self.assertEqual(record["use_count"], 2)

    def test_build_scene_library_generates_without_stepping_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/scene_library.db"

            first = build_scene_library(
                episodes=3,
                difficulty_preset_id="easy",
                seed_offset=500,
                scene_library_path=db_path,
            )
            second = build_scene_library(
                episodes=3,
                difficulty_preset_id="easy",
                seed_offset=500,
                scene_library_path=db_path,
            )

            self.assertEqual(first["misses"], 3)
            self.assertEqual(first["hits"], 0)
            self.assertEqual(first["unique_scene_keys"], 3)
            self.assertEqual(second["hits"], 3)
            self.assertEqual(second["misses"], 0)


if __name__ == "__main__":
    unittest.main()
