"""Tests for ``carlabev validate replay`` (tools/validate_replay.py)."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from CarlaBEV.cli.main import main as cli_main  # noqa: E402
from CarlaBEV.tools.validate_replay import (  # noqa: E402
    ReplaySpec,
    build_action_sequence,
    compare_traces,
    compare_variation_independence,
    run_replay,
    run_validation,
)


def _trace(reset=None, steps=None, digest="d"):
    return {
        "reset": reset if reset is not None else {"ego": {"x_m": 1.0, "yaw": math.pi}, "npcs": []},
        "steps": steps if steps is not None else [],
        "digest": digest,
    }


def _step(k, x=0.0, yaw=0.0, npc_x=5.0):
    return {
        "step": k,
        "sim_t": 0.1 * (k + 1),
        "hero": {"x_m": x, "y_m": 0.0, "yaw": yaw, "v_mps": 1.0},
        "npcs": {"vehicle:000": {"x_m": npc_x, "y_m": 0.0, "yaw": 0.0, "v_mps": 0.0, "behavior_state": "idle", "recycle_count": 0}},
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "cause": None,
    }


class CompareTracesTests(unittest.TestCase):
    def test_identical_traces_compare_equal(self):
        a = _trace(steps=[_step(0), _step(1)])
        b = _trace(steps=[_step(0), _step(1)])
        result = compare_traces(a, b)
        self.assertTrue(result["identical"])
        self.assertTrue(result["reset_identical"])
        self.assertIsNone(result["first_divergence"])
        self.assertTrue(result["digest_equal"])

    def test_first_step_divergence_is_reported_with_key_and_delta(self):
        a = _trace(steps=[_step(0), _step(1), _step(2)])
        b = _trace(steps=[_step(0), _step(1, npc_x=5.3125), _step(2, npc_x=5.3125)])
        result = compare_traces(a, b)
        self.assertFalse(result["identical"])
        self.assertTrue(result["reset_identical"])
        first = result["first_divergence"]
        self.assertEqual(first["step"], 1)
        self.assertEqual(first["key"], "npcs.vehicle:000.x_m")
        self.assertAlmostEqual(first["delta"], 0.3125)
        self.assertEqual(result["step_mismatch_count"], 2)
        self.assertAlmostEqual(result["max_abs_delta"], 0.3125)

    def test_reset_divergence_is_reported_before_steps(self):
        a = _trace(reset={"ego": {"x_m": 1.0}, "npcs": [{"npc_id": "vehicle:000", "x_m": 2.0}]})
        b = _trace(reset={"ego": {"x_m": 1.0}, "npcs": [{"npc_id": "vehicle:000", "x_m": 2.5}]})
        result = compare_traces(a, b)
        self.assertFalse(result["identical"])
        self.assertFalse(result["reset_identical"])
        self.assertEqual(result["reset_mismatches"][0]["key"], "reset.npcs[0].x_m")

    def test_headings_are_compared_modulo_two_pi(self):
        a = _trace(reset={"ego": {"yaw": math.pi}}, steps=[_step(0, yaw=math.pi)])
        b = _trace(reset={"ego": {"yaw": -math.pi}}, steps=[_step(0, yaw=-math.pi)])
        self.assertTrue(compare_traces(a, b)["identical"])

    def test_tolerance_is_bitwise_by_default_and_relaxable(self):
        a = _trace(steps=[_step(0, x=1.0)])
        b = _trace(steps=[_step(0, x=1.0 + 1e-12)])
        self.assertFalse(compare_traces(a, b)["identical"])
        self.assertTrue(compare_traces(a, b, atol=1e-9)["identical"])

    def test_episode_length_mismatch_is_a_divergence(self):
        a = _trace(steps=[_step(0), _step(1)])
        b = _trace(steps=[_step(0)])
        result = compare_traces(a, b)
        self.assertFalse(result["identical"])
        self.assertTrue(result["length_mismatch"])
        self.assertEqual(result["first_divergence"]["what"], "episode length differs")

    def test_npc_set_mismatch_is_a_divergence(self):
        a = _trace(steps=[_step(0)])
        extra = _step(0)
        extra["npcs"]["vehicle:001"] = dict(extra["npcs"]["vehicle:000"])
        b = _trace(steps=[extra])
        result = compare_traces(a, b)
        self.assertFalse(result["identical"])
        self.assertEqual(result["first_divergence"]["what"], "keys differ")


class VariationIndependenceTests(unittest.TestCase):
    def _reset(self, seed, x, npc_x):
        return {
            "scene_key": "k", "seeds": {"scene_seed": 1},
            "ego": {"x_m": x, "y_m": 0.0, "yaw": 0.0, "v_mps": 0.0, "route_digest": "r", "route_len": 5, "route_length_m": 40.0},
            "targets": [[1.0, 2.0]], "signals": [],
            "descriptor": {"variation_seed": seed},
            "npcs": [{"npc_id": "vehicle:000", "kind": "vehicle", "traffic_role": "lead", "behavior": None,
                      "cruise_speed_mps": 8.0, "obeys_signals": True, "yields_at_junctions": True,
                      "route_digest": "n", "x_m": npc_x, "y_m": 0.0}],
        }

    def test_variant_must_keep_identity_and_change_state(self):
        ref = {"reset": self._reset(7, 1.0, 5.0), "steps": []}
        good = {"reset": self._reset(8, 1.3, 5.6), "steps": []}
        same_state = {"reset": self._reset(8, 1.0, 5.0), "steps": []}
        broken = {"reset": self._reset(8, 1.3, 5.6), "steps": []}
        broken["reset"]["scene_key"] = "other"
        self.assertTrue(compare_variation_independence(ref, good)["passed"])
        self.assertFalse(compare_variation_independence(ref, same_state)["passed"])
        result = compare_variation_independence(ref, broken)
        self.assertFalse(result["passed"])
        self.assertEqual(result["identity_mismatches"][0]["key"], "reset.scene_key")


class ActionSequenceTests(unittest.TestCase):
    def test_scripted_continuous_sequence_is_explicit_and_deterministic(self):
        a = build_action_sequence("scripted", steps=25, action_mode="continuous", action_seed=0)
        b = build_action_sequence("scripted", steps=25, action_mode="continuous", action_seed=0)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 25)
        self.assertEqual(a[0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(a[10][1], 0.6, places=6)  # left-steer phase (float32 action)

    def test_random_sequence_depends_only_on_the_action_seed(self):
        a = build_action_sequence("random", steps=10, action_mode="discrete", action_seed=7)
        b = build_action_sequence("random", steps=10, action_mode="discrete", action_seed=7)
        c = build_action_sequence("random", steps=10, action_mode="discrete", action_seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(all(isinstance(v, int) for v in a))
        cont = build_action_sequence("random", steps=4, action_mode="continuous", action_seed=7)
        self.assertEqual(len(cont[0]), 3)

    def test_explicit_sequence_is_loaded_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "actions.json"
            path.write_text(json.dumps([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), encoding="utf-8")
            actions = build_action_sequence(
                "explicit", steps=0, action_mode="continuous", action_seed=0, actions_file=str(path)
            )
        self.assertEqual(actions, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            build_action_sequence("explicit", steps=0, action_mode="continuous", action_seed=0)

    def test_spec_round_trips_through_json(self):
        spec = ReplaySpec(env={"size": 128}, seed=3, options={"scene": "rdm"}, actions=[1, 2], include_obs=False)
        restored = ReplaySpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(restored.to_dict(), spec.to_dict())


class ReplayRunTests(unittest.TestCase):
    """Cheap end-to-end runs: no traffic (no NPC RNG) so the contract holds today."""

    _SPEC_ENV = dict(
        map_name="Town01",
        render_mode="rgb_array",
        obs_mode="bev_rgb",
        action_mode="continuous",
        size=128,
        traffic_enabled=False,
        scene_library_enabled=False,
    )

    def _spec(self, steps=15):
        from CarlaBEV.config import RandomNavigationReset, build_random_navigation_options

        options = build_random_navigation_options(
            RandomNavigationReset(difficulty_preset_id="no_traffic", scene_seed=5, num_vehicles=0)
        )
        options = {k: v for k, v in options.items() if k != "reset_mask"}
        actions = build_action_sequence("forward", steps=steps, action_mode="continuous", action_seed=5)
        return ReplaySpec(env=dict(self._SPEC_ENV), seed=5, options=options, actions=actions)

    def test_trace_contains_reset_and_step_fingerprints(self):
        trace = run_replay(self._spec(steps=5))
        self.assertEqual(len(trace["steps"]), 5)
        self.assertIn("scene_key", trace["reset"])
        self.assertIn("route_digest", trace["reset"]["ego"])
        self.assertTrue(trace["reset"]["targets"])
        self.assertEqual(trace["reset"]["npcs"], [])
        step = trace["steps"][0]
        for key in ("hero", "npcs", "signals", "reward", "terminated", "truncated", "cause", "obs_sha256"):
            self.assertIn(key, step)
        self.assertGreater(step["hero"]["v_mps"], 0.0)
        self.assertNotIn("hit", trace["reset"])  # cache bookkeeping is metadata, not state
        self.assertIn("hit", trace["cache"])

    def test_in_process_validation_passes_without_traffic(self):
        report = run_validation(self._spec(steps=10), runs=2, isolation="in-process")
        self.assertTrue(report["passed"], report["results"])
        self.assertTrue(report["all_identical"])
        self.assertIsNone(report["results"][0]["comparison"])
        self.assertTrue(report["results"][1]["comparison"]["identical"])

    def test_process_isolation_varies_python_hash_seed_and_passes_without_traffic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_validation(
                self._spec(steps=10), runs=2, isolation="process", workdir=Path(tmpdir)
            )
        self.assertTrue(report["passed"], report["results"])
        hash_seeds = {r["python_hash_seed"] for r in report["results"]}
        self.assertEqual(len(hash_seeds), 2)
        pids = {r["pid"] for r in report["results"]}
        self.assertEqual(len(pids), 2)
        self.assertNotIn(os.getpid(), pids)

    def test_validation_flags_a_divergent_run(self):
        spec = self._spec(steps=10)
        report = run_validation(spec, runs=2, isolation="in-process")
        self.assertTrue(report["passed"])
        # Tamper with the reference trace to simulate a divergence at step 3.
        traces = report["_traces"]
        traces[1]["steps"][3]["hero"]["x_m"] += 0.5
        comparison = compare_traces(traces[0], traces[1])
        self.assertFalse(comparison["identical"])
        self.assertEqual(comparison["first_divergence"]["step"], 3)
        self.assertEqual(comparison["first_divergence"]["key"], "hero.x_m")


class ReplayCliTests(unittest.TestCase):
    def test_help_returns_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["validate", "replay", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_in_process_no_traffic_passes_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = f"{tmpdir}/replay.json"
            code = cli_main(
                [
                    "validate", "replay", "--seed", "5", "--runs", "2", "--steps", "10",
                    "--isolation", "in-process", "--no-traffic", "--policy", "forward",
                    "--report", report_path, "--save-traces", "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(report["cache_mode"], "off")
        self.assertEqual(len(report["traces"]), 2)
        self.assertEqual(report["spec"]["seed"], 5)
        self.assertEqual(len(report["spec"]["actions"]), 10)

    def test_cli_variation_flags_replay_a_variant_and_check_independence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = f"{tmpdir}/replay.json"
            code = cli_main(
                [
                    "validate", "replay", "--seed", "5", "--runs", "2", "--steps", "5",
                    "--isolation", "in-process", "--no-traffic", "--policy", "forward",
                    "--variation-seed", "3", "--variation-independence",
                    "--report", report_path, "--quiet",
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        self.assertEqual(report["spec"]["options"]["variation_seed"], 3)
        check = report["variation_check"]
        self.assertTrue(check["passed"])
        self.assertTrue(check["identity_preserved"])
        self.assertTrue(check["state_differs"])
        self.assertEqual(check["variant_seed_recorded"], 4)

    def test_cli_hit_only_is_strict_read_only_and_require_hit(self):
        from CarlaBEV.tools.validate_replay import build_spec_from_args, _build_parser

        args = _build_parser().parse_args(["--cache-mode", "hit-only", "--scene-library", "x.db", "--no-traffic"])
        spec = build_spec_from_args(args, scene_library_path="x.db")
        self.assertTrue(spec.env["scene_library_read_only"])
        self.assertTrue(spec.env["scene_library_require_hit"])
        args = _build_parser().parse_args(["--cache-mode", "miss-then-hit", "--no-traffic"])
        spec = build_spec_from_args(args, scene_library_path="y.db")
        self.assertNotIn("scene_library_require_hit", spec.env)

    def test_cli_hit_only_requires_existing_library(self):
        with self.assertRaises(SystemExit):
            cli_main(
                [
                    "validate", "replay", "--cache-mode", "hit-only",
                    "--scene-library", "/nonexistent/library.db", "--quiet",
                ]
            )


if __name__ == "__main__":
    unittest.main()
