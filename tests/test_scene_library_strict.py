"""Stage 6: immutable (read-only) scene libraries and strict require-hit corpus replay."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pydantic import ValidationError  # noqa: E402

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options  # noqa: E402
from CarlaBEV.envs.carlabev import CarlaBEV  # noqa: E402
from CarlaBEV.src.diagnostics.rollouts import LEAD_BRAKE_OPTIONS  # noqa: E402
from CarlaBEV.src.managers.scene_library import SceneLibrary, SceneLibraryError, SceneLibraryMissError  # noqa: E402


def _options(seed=5, **overrides):
    kwargs = dict(difficulty_preset_id="easy", scene_seed=seed, num_vehicles=4, route_dist_range=(30, 60))
    kwargs.update(overrides)
    return build_random_navigation_options(RandomNavigationReset(**kwargs))


def _env(db, **overrides):
    kwargs = dict(render_mode="rgb_array", obs_mode="bev_rgb", action_mode="continuous", size=128,
                  scene_library_enabled=True, scene_library_path=db)
    kwargs.update(overrides)
    return CarlaBEV(EnvConfig(**kwargs))


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spawn_state(env):
    hero = env.map.hero
    return (round(hero.x_m, 9), round(hero.y_m, 9)), [
        (a.npc_id, round(a._controller.x_m, 9), round(a._controller.y_m, 9)) for a in env.map.actor_manager.actors["vehicle"]
    ]


class ReadOnlyLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = f"{self.tmp.name}/corpus.db"
        env = _env(self.db)
        try:
            env.reset(seed=5, options=_options(5))
            self.frozen = _spawn_state(env)
        finally:
            env.close()

    def tearDown(self):
        os.chmod(self.db, 0o644)
        self.tmp.cleanup()

    def test_read_only_library_never_writes(self):
        os.chmod(self.db, 0o444)
        before = _file_digest(self.db)
        with sqlite3.connect(f"file:{self.db}?mode=ro", uri=True) as conn:
            use_count_before = conn.execute("SELECT use_count FROM scenes").fetchone()[0]
        env = _env(self.db, scene_library_read_only=True)
        try:
            for _ in range(2):
                _, info = env.reset(seed=5, options=_options(5))
                self.assertTrue(info["scene_cache"]["hit"])
                self.assertTrue(info["scene_cache"]["read_only"])
                self.assertEqual(_spawn_state(env), self.frozen)
            # A miss in read-only (non-strict) mode generates but does not store.
            _, info = env.reset(seed=6, options=_options(6))
            self.assertFalse(info["scene_cache"]["hit"])
        finally:
            env.close()
        self.assertEqual(_file_digest(self.db), before)
        with sqlite3.connect(f"file:{self.db}?mode=ro", uri=True) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT use_count FROM scenes").fetchone()[0], use_count_before)

    def test_read_only_requires_an_existing_library_with_the_current_schema(self):
        with self.assertRaises(FileNotFoundError):
            SceneLibrary(f"{self.tmp.name}/missing.db", generator_version="role_traffic_v2", read_only=True)
        legacy = f"{self.tmp.name}/legacy.db"
        with sqlite3.connect(legacy) as conn:
            conn.execute("CREATE TABLE scenes (scene_id INTEGER PRIMARY KEY, scene_key TEXT)")
        with self.assertRaises(SceneLibraryError):
            SceneLibrary(legacy, generator_version="role_traffic_v2", read_only=True)
        library = SceneLibrary(self.db, generator_version="role_traffic_v2", read_only=True)
        self.assertIsNone(library.insert_scene(
            scene_key="x", scene_descriptor={}, scenario_context={}, actors_payload={}, spawn_validation=None, resolved_options={},
        ))
        library.delete_scene("x")
        library.touch_scene("x")
        self.assertEqual(len(library.get_scene_by_filters(limit=10)), 1)


class RequireHitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = f"{self.tmp.name}/corpus.db"
        env = _env(self.db)
        try:
            env.reset(seed=5, options=_options(5))
            self.frozen = _spawn_state(env)
        finally:
            env.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_require_hit_replays_hits_and_raises_on_misses(self):
        env = _env(self.db, scene_library_read_only=True, scene_library_require_hit=True)
        try:
            _, info = env.reset(seed=5, options=_options(5))
            self.assertTrue(info["scene_cache"]["hit"])
            self.assertTrue(info["scene_cache"]["require_hit"])
            self.assertEqual(_spawn_state(env), self.frozen)
            # Variants of a cached scene are still hits (variation is not identity).
            _, info = env.reset(seed=5, options=_options(5, variation_seed=3))
            self.assertTrue(info["scene_cache"]["hit"])
            self.assertNotEqual(_spawn_state(env), self.frozen)
            with self.assertRaises(SceneLibraryMissError) as ctx:
                env.reset(seed=6, options=_options(6))
            self.assertEqual(ctx.exception.seeds["scene_seed"], 6)
            self.assertIn("regeneration is disabled", str(ctx.exception))
            # Scenario resets never use the library and remain available.
            _, info = env.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
            self.assertEqual(info["scene_cache"].get("bypassed"), "non_random_scene")
        finally:
            env.close()

    def test_require_hit_refuses_stale_entries_instead_of_regenerating(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE scenes SET actors_json = '{\"schema_version\": 1}'")
        env = _env(self.db, scene_library_require_hit=True)
        try:
            with self.assertRaises(SceneLibraryMissError) as ctx:
                env.reset(seed=5, options=_options(5))
            self.assertIn("not replayable", ctx.exception.reason)
        finally:
            env.close()
        with sqlite3.connect(self.db) as conn:  # not evicted either (strict mode never regenerates)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0], 1)

    def test_require_hit_needs_the_library_enabled(self):
        with self.assertRaises(ValidationError):
            EnvConfig(scene_library_enabled=False, scene_library_require_hit=True)


if __name__ == "__main__":
    unittest.main()
