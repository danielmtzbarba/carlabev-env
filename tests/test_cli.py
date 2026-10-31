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

    def test_run_sim_help_returns_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["run", "sim", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_run_sim_manual_requires_human_render(self):
        with self.assertRaises(SystemExit):
            cli_main(["run", "sim", "--control", "manual", "--render", "rgb_array", "--episodes", "1"])

    def test_run_sim_headless_random_rollout_writes_episode_log(self):
        import json
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        with tempfile.TemporaryDirectory() as tmpdir:
            code = cli_main(
                [
                    "run", "sim", "--control", "random", "--episodes", "2", "--no-traffic",
                    "--max-episode-duration-s", "1.0", "--no-scene-library", "--seed", "5",
                    "--log-dir", tmpdir, "--exp-name", "cli-test",
                ]
            )
            self.assertEqual(code, 0)
            log_path = f"{tmpdir}/cli-test/episodes.jsonl"
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(records), 2)
        self.assertEqual({r["termination"] for r in records}, {"max_actions"})
        self.assertTrue(all(abs(r["episode_duration_s"] - 1.0) < 1e-9 for r in records))

    def test_run_sim_scenario_mode_with_custom_physics_dt(self):
        import os

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        with tempfile.TemporaryDirectory() as tmpdir:
            code = cli_main(
                [
                    "run", "sim", "--control", "idle", "--reset-mode", "scenario", "--scenario-id", "lead_brake",
                    "--physics-dt", "0.05", "--episodes", "1", "--max-episode-duration-s", "1.0",
                    "--no-scene-library", "--no-log", "--log-dir", tmpdir,
                ]
            )
        self.assertEqual(code, 0)

    def test_analyze_scene_library_reports_core_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/scene_library.db"
            library = SceneLibrary(db_path, generator_version="test_cli_v1")
            library.insert_scene(
                scene_key="scene-a",
                scene_descriptor={
                    "scene": "rdm",
                    "scene_benchmark_id": "navigation_medium_v1",
                    "scene_profile_id": "easy",
                    "scene_split": "train",
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
                    "scene_benchmark_id": "navigation_medium_v1",
                    "scene_profile_id": "hard",
                    "scene_split": "eval",
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
                group_by=["scene-benchmark", "scene-profile", "scene-split", "difficulty", "route-profile"],
            )
            query = collect_scene_library_query(
                db_path,
                filters={"main_actor_role": "cross_path"},
                scene_seed=42,
                role_mins={"spawned_role_cross_path": 1},
                limit=5,
            )

        self.assertEqual(summary["total_scenes"], 2)
        self.assertEqual(summary["scene_benchmark_counts"]["navigation_medium_v1"], 2)
        self.assertEqual(summary["scene_profile_counts"]["easy"], 1)
        self.assertEqual(summary["scene_split_counts"]["train"], 1)
        self.assertEqual(summary["difficulty_counts"]["easy"], 1)
        self.assertEqual(summary["route_profile_counts"]["mostly_straight"], 1)
        self.assertEqual(summary["main_actor_role_counts"]["lead"], 1)
        self.assertAlmostEqual(quality["zero_spawned_rate"], 0.5)
        self.assertEqual(breakdown["breakdowns"]["scene-benchmark"]["navigation_medium_v1"], 2)
        self.assertEqual(breakdown["breakdowns"]["scene-profile"]["hard"], 1)
        self.assertEqual(breakdown["breakdowns"]["scene-split"]["eval"], 1)
        self.assertEqual(breakdown["breakdowns"]["difficulty"]["hard"], 1)
        self.assertEqual(breakdown["breakdowns"]["route-profile"]["multi_turn"], 1)
        self.assertEqual(query["total_matches"], 1)
        self.assertEqual(query["rows"][0]["scene_seed"], 42)


if __name__ == "__main__":
    unittest.main()
