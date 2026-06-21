import tempfile
import unittest

from CarlaBEV.cli.main import main as cli_main
from CarlaBEV.src.managers.scene_library import SceneLibrary
from CarlaBEV.tools.analyze_scene_library import (
    collect_scene_library_breakdown,
    collect_scene_library_quality,
    collect_scene_library_query,
    collect_scene_library_stats,
)


class TestCli(unittest.TestCase):
    def test_cli_help_returns_zero(self):
        self.assertEqual(cli_main(["--help"]), 0)

    def test_run_interactive_help_returns_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["run", "interactive", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_analyze_scene_library_reports_core_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/scene_library.db"
            library = SceneLibrary(db_path, generator_version="test_cli_v1")
            library.insert_scene(
                scene_key="scene-a",
                scene_descriptor={
                    "scene": "rdm",
                    "difficulty_preset_id": "easy",
                    "route_profile": "mostly_straight",
                    "main_actor_role": "lead",
                    "spawned_num_vehicles": 2,
                    "spawned_num_vehicles_near_ego": 1,
                    "spawned_role_counts": {"lead": 1, "background": 1},
                },
                scenario_context={},
                actors_payload={},
                spawn_validation={"valid": True},
                resolved_options={"scene": "rdm", "num_vehicles": 3},
            )
            library.insert_scene(
                scene_key="scene-b",
                scene_descriptor={
                    "scene": "rdm",
                    "difficulty_preset_id": "hard",
                    "route_profile": "multi_turn",
                    "main_actor_role": "cross_path",
                    "traffic_role_profile": "mix",
                    "scene_seed": 42,
                    "route_seed": 43,
                    "traffic_seed": 44,
                    "spawned_num_vehicles": 0,
                    "spawned_num_vehicles_near_ego": 0,
                    "spawned_role_counts": {"cross_path": 1},
                },
                scenario_context={},
                actors_payload={},
                spawn_validation={"valid": True},
                resolved_options={"scene": "rdm", "num_vehicles": 4},
            )
            summary = collect_scene_library_stats(db_path)
            quality = collect_scene_library_quality(db_path)
            breakdown = collect_scene_library_breakdown(
                db_path,
                group_by=["difficulty", "route-profile"],
            )
            query = collect_scene_library_query(
                db_path,
                filters={"main_actor_role": "cross_path"},
                scene_seed=42,
                role_mins={"spawned_role_cross_path": 1},
                limit=5,
            )

        self.assertEqual(summary["total_scenes"], 2)
        self.assertEqual(summary["difficulty_counts"]["easy"], 1)
        self.assertEqual(summary["route_profile_counts"]["mostly_straight"], 1)
        self.assertEqual(summary["main_actor_role_counts"]["lead"], 1)
        self.assertAlmostEqual(quality["zero_spawned_rate"], 0.5)
        self.assertEqual(breakdown["breakdowns"]["difficulty"]["hard"], 1)
        self.assertEqual(breakdown["breakdowns"]["route-profile"]["multi_turn"], 1)
        self.assertEqual(query["total_matches"], 1)
        self.assertEqual(query["rows"][0]["scene_seed"], 42)


if __name__ == "__main__":
    unittest.main()
