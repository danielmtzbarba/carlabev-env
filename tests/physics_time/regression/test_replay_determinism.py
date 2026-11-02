"""REGRESSION / replay determinism guards (findings R-01..R-03, fixed in Stage 2).

Originally the Stage 0 characterization tests of
``CarlaBEV/docs/history/replay_and_scene_cache_plan.md`` (``tests/test_replay_characterization.py``);
promoted here in Stage 5. They pin the fixes:

* **B1** — a scene-library *hit* must be indistinguishable from the miss that
  produced it (NPC spawn offsets are resolved at generation time and persisted).
* **B2** — independent interpreters (subprocesses with different
  ``PYTHONHASHSEED``) must replay identical trajectories through a traffic
  lifecycle recycle event (lifecycle RNGs derive from the traffic seed, never
  from ``hash()``; recycle jitter comes from the NPC's own ``runtime_seed``).
* **B3** — scenario scenes keep their behaviours across resets (they bypass
  the scene library, which only serves random-navigation scenes).

Do not loosen these assertions; they define the replay contract at reset and
step granularity. ``carlabev validate replay`` is the CLI counterpart.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from CarlaBEV.config import (  # noqa: E402
    EnvConfig,
    RandomNavigationReset,
    build_random_navigation_options,
)
from CarlaBEV.envs.carlabev import CarlaBEV  # noqa: E402
from CarlaBEV.src.diagnostics.rollouts import LEAD_BRAKE_OPTIONS, JAYWALK_OPTIONS  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
CHILD_SCRIPT = Path(__file__).with_name("_replay_probe_child.py")

# Reset options shared by the cache tests: small random scene, fixed sub-seeds.
RANDOM_RESET = dict(
    difficulty_preset_id="easy",
    scene_seed=1000,
    route_seed=2000,
    traffic_seed=3000,
    num_vehicles=4,
    route_dist_range=(30, 60),
)


def _make_env(db_path: str) -> CarlaBEV:
    return CarlaBEV(
        EnvConfig(
            map_name="Town01",
            render_mode="rgb_array",
            obs_mode="bev_rgb",
            action_mode="discrete",
            size=128,
            scene_library_enabled=True,
            scene_library_path=db_path,
        )
    )


def _npc_reset_states(env: CarlaBEV) -> list[tuple]:
    """Resolved NPC spawn state right after reset (pose, speed, route head)."""
    states = []
    for kind in ("vehicle", "pedestrian"):
        for actor in env.map.actor_manager.actors.get(kind, []):
            ctrl = actor._controller
            states.append(
                (
                    kind,
                    round(float(ctrl.x_m), 9),
                    round(float(ctrl.y_m), 9),
                    round(float(ctrl.yaw), 9),
                    round(float(ctrl.v_mps), 9),
                    round(float(actor.initial_speed_mps or 0.0), 9),
                    round(float(actor.cruise_speed_mps), 9),
                    tuple(int(x) for x in actor.rx[:8]),
                    tuple(int(y) for y in actor.ry[:8]),
                )
            )
    return states


def _reset_snapshot(env: CarlaBEV, options: dict) -> dict:
    _, info = env.reset(seed=11, options=dict(options))
    hero = env.map.hero
    return {
        "cache_hit": bool(info.get("scene_cache", {}).get("hit", False)),
        "hero": (
            round(float(hero.x_m), 9),
            round(float(hero.y_m), 9),
            round(float(hero.yaw), 9),
        ),
        "target": tuple(round(float(v), 6) for v in env.map.target_position),
        "npcs": _npc_reset_states(env),
    }


def _run_child(out_path: str, *, python_hash_seed: str, num_steps: int = 170) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = python_hash_seed
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    subprocess.run(
        [sys.executable, str(CHILD_SCRIPT), out_path, str(num_steps)],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        timeout=600,
    )
    with open(out_path, encoding="utf-8") as handle:
        return json.load(handle)


def _first_divergent_step(a: dict, b: dict):
    for step_a, step_b in zip(a["steps"], b["steps"]):
        if step_a != step_b:
            return step_a["step"]
    if len(a["steps"]) != len(b["steps"]):
        return min(len(a["steps"]), len(b["steps"]))
    return None


class CacheHitSpawnDeterminismTests(unittest.TestCase):
    """B1 — the scene cache must be transparent: miss == hit == hit."""

    def test_cache_hit_replays_identical_npc_spawn_state(self):
        options = build_random_navigation_options(RandomNavigationReset(**RANDOM_RESET))
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(f"{tmpdir}/scene_library.db")
            try:
                miss = _reset_snapshot(env, options)
                hit_1 = _reset_snapshot(env, options)
                hit_2 = _reset_snapshot(env, options)
            finally:
                env.close()

        self.assertFalse(miss["cache_hit"])
        self.assertTrue(hit_1["cache_hit"])
        self.assertTrue(hit_2["cache_hit"])
        self.assertTrue(miss["npcs"], "characterization scene must contain NPCs")
        # Ego spawn and target are shared state: assert unconditionally.
        self.assertEqual(miss["hero"], hit_1["hero"])
        self.assertEqual(miss["target"], hit_1["target"])
        # The replay contract: a cache hit is indistinguishable from the miss
        # that produced it, and hits are indistinguishable from each other.
        self.assertEqual(miss["npcs"], hit_1["npcs"])
        self.assertEqual(hit_1["npcs"], hit_2["npcs"])


class CrossProcessReplayTests(unittest.TestCase):
    """B2 — independent processes must replay identical trajectories."""

    def test_independent_processes_replay_identical_trajectories_through_recycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_a = _run_child(f"{tmpdir}/a.json", python_hash_seed="101")
            run_b = _run_child(f"{tmpdir}/b.json", python_hash_seed="202")

        # The fragment must actually exercise the lifecycle, otherwise this
        # test proves nothing (first recycle is at step 63 under policy v3 for this seed).
        self.assertGreaterEqual(run_a["recycled"], 1)
        self.assertGreaterEqual(run_b["recycled"], 1)
        divergent_step = _first_divergent_step(run_a, run_b)
        self.assertIsNone(
            divergent_step,
            f"independent runs diverged at step {divergent_step} "
            f"(recycled: {run_a['recycled']}/{run_b['recycled']})",
        )


class CachedScenarioFidelityTests(unittest.TestCase):
    """B3 — a cached scene must keep behaviors and rule flags (or not be cached)."""

    def test_cached_scenario_scene_retains_behaviors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _make_env(f"{tmpdir}/scene_library.db")
            try:
                # lead_brake: the single vehicle must carry LeadBrakeBehavior.
                env.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
                miss_behaviors = [
                    type(v.behavior).__name__ if v.behavior else None
                    for v in env.map.actor_manager.actors["vehicle"]
                ]
                _, info = env.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
                replay_behaviors = [
                    type(v.behavior).__name__ if v.behavior else None
                    for v in env.map.actor_manager.actors["vehicle"]
                ]
                self.assertEqual(miss_behaviors, ["LeadBrakeBehavior"])
                # Either the replayed scene keeps its behaviors, or scenario
                # scenes are not served from the cache at all. Both satisfy
                # the contract; silent degradation does not.
                self.assertEqual(miss_behaviors, replay_behaviors)
                self.assertFalse(info["scene_cache"]["hit"])
                self.assertEqual(info["scene_cache"].get("bypassed"), "non_random_scene")

                # jaywalk: the pedestrian must keep its crossing behavior.
                env.reset(seed=3, options=dict(JAYWALK_OPTIONS))
                miss_ped = [
                    type(p.behavior).__name__ if p.behavior else None
                    for p in env.map.actor_manager.actors["pedestrian"]
                ]
                env.reset(seed=3, options=dict(JAYWALK_OPTIONS))
                replay_ped = [
                    type(p.behavior).__name__ if p.behavior else None
                    for p in env.map.actor_manager.actors["pedestrian"]
                ]
                self.assertEqual(miss_ped, ["StopReturnBehavior"])
                self.assertEqual(miss_ped, replay_ped)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
