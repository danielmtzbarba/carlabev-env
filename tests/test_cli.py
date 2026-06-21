import tempfile
import unittest

from CarlaBEV.cli.main import main as cli_main
from CarlaBEV.src.managers.scene_library import SceneLibrary
from CarlaBEV.tools.analyze_scene_library import collect_scene_library_stats


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
            summary = collect_scene_library_stats(db_path)

        self.assertEqual(summary["total_scenes"], 1)
        self.assertEqual(summary["difficulty_counts"]["easy"], 1)
        self.assertEqual(summary["route_profile_counts"]["mostly_straight"], 1)
        self.assertEqual(summary["main_actor_role_counts"]["lead"], 1)


if __name__ == "__main__":
    unittest.main()
