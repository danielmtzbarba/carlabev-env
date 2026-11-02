"""Scene spec layers (Stage 3): capture/instantiate, NPC table, variation layer."""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options  # noqa: E402
from CarlaBEV.envs.carlabev import CarlaBEV  # noqa: E402
from CarlaBEV.src.diagnostics.rollouts import JAYWALK_OPTIONS, LEAD_BRAKE_OPTIONS  # noqa: E402
from CarlaBEV.src.managers.scene_library import SceneLibrary  # noqa: E402
from CarlaBEV.src.randomness import derive_seed  # noqa: E402
from CarlaBEV.src.scenes.scene_spec import (  # noqa: E402
    VARIATION_PROFILES,
    SceneSpec,
    VariationRequest,
    apply_variation,
    instantiate_scene,
    npc_uid_for,
    scene_identity_key,
)


def _env(**overrides) -> CarlaBEV:
    kwargs = dict(render_mode="rgb_array", obs_mode="bev_rgb", action_mode="continuous", size=128, scene_library_enabled=False)
    kwargs.update(overrides)
    return CarlaBEV(EnvConfig(**kwargs))


def _random_options(**overrides):
    kwargs = dict(difficulty_preset_id="easy", scene_seed=1000, route_seed=2000, traffic_seed=3000, num_vehicles=5, route_dist_range=(30, 60))
    kwargs.update(overrides)
    return build_random_navigation_options(RandomNavigationReset(**kwargs))


def _spawn_state(env):
    hero = env.map.hero
    npcs = [
        (a.npc_id, round(a._controller.x_m, 9), round(a._controller.y_m, 9), round(a._controller.v_mps, 9), len(a.rx))
        for kind in ("vehicle", "pedestrian")
        for a in env.map.actor_manager.actors[kind]
    ]
    return (round(hero.x_m, 9), round(hero.y_m, 9), round(hero.yaw, 9)), npcs


def _reset_states(actors, env):
    """Reset freshly instantiated actors (bound to the map like the ActorManager does) and read their spawn poses."""
    out = []
    for kind in ("vehicle", "pedestrian"):
        for actor in actors[kind]:
            actor.set_units(env.units)
            actor.drivable_mask = env.map._drivable_mask
            actor.blocking_clearance_px = env.map.blocking_clearance_px
            actor.reset()
            out.append((actor.npc_id, round(actor._controller.x_m, 9), round(actor._controller.y_m, 9)))
    return out


class CaptureInstantiateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        _, cls.info = cls.env.reset(seed=11, options=_random_options())
        cls.spec = cls.env.scene_spec

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_env_exposes_the_identity_spec(self):
        spec = self.spec
        self.assertIsInstance(spec, SceneSpec)
        self.assertEqual(spec.scene_key, self.info["scene_descriptor"]["scene_key"])
        self.assertEqual(spec.seeds["traffic_seed"], 3000)
        self.assertEqual(spec.identity["scene"], "rdm")
        self.assertEqual(len(spec.npcs), self.env.num_vehicles)
        for ordinal, npc in enumerate(spec.npcs):
            self.assertEqual(npc.npc_id, f"vehicle:{ordinal:03d}")
            self.assertEqual(npc.npc_uid, npc_uid_for(spec.scene_key, npc.npc_id))
            self.assertEqual(npc.runtime_seed, derive_seed(3000, "npc", ordinal))
            self.assertIsNotNone(npc.spawn_jitter_m)
        self.assertIsNotNone(spec.agent.spawn_jitter_m)

    def test_instantiation_is_deterministic_and_matches_the_live_scene(self):
        units = self.env.units
        a = _reset_states(instantiate_scene(self.spec, units=units), self.env)
        b = _reset_states(instantiate_scene(self.spec, units=units), self.env)
        self.assertEqual(a, b)
        live = [(x.npc_id, round(x._controller.x_m, 9), round(x._controller.y_m, 9)) for x in self.env.map.actor_manager.actors["vehicle"]]
        self.assertEqual(a, live)

    def test_payload_round_trip_is_lossless(self):
        payload = self.spec.to_payload()
        restored = SceneSpec.from_payload(payload)
        self.assertEqual(restored.to_payload(), payload)
        self.assertEqual(restored.digest(), self.spec.digest())
        # Reference-only payload + NPC rows reassembles to the same spec.
        refs = self.spec.to_payload(inline_npcs=False)
        self.assertNotIn("npcs", refs)
        self.assertEqual(refs["npc_uids"], [n.npc_uid for n in self.spec.npcs])
        restored = SceneSpec.from_payload(refs, npcs=[n.to_dict() for n in self.spec.npcs])
        self.assertEqual(restored.digest(), self.spec.digest())

    def test_identity_key_ignores_variation_and_reset_mask(self):
        base = {"scene": "rdm", "difficulty_preset_id": "easy", "scene_seed": 1}
        seeds = {"scene_seed": 1, "route_seed": 2, "traffic_seed": 3, "scenario_seed": 4}
        k1 = scene_identity_key(generator_version="v", surface_size=128, options=base, seeds=seeds)
        k2 = scene_identity_key(
            generator_version="v", surface_size=128,
            options={**base, "variation_seed": 9, "variation_profile": "spawn_v1", "reset_mask": [True]}, seeds=seeds,
        )
        k3 = scene_identity_key(generator_version="v", surface_size=128, options={**base, "num_vehicles": 3}, seeds=seeds)
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)


class ScenarioSpecFidelityTests(unittest.TestCase):
    def test_scenario_actors_keep_behaviours_flags_and_signals_through_the_spec_path(self):
        env = _env()
        try:
            env.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
            lead = env.map.actor_manager.actors["vehicle"][0]
            self.assertEqual(type(lead.behavior).__name__, "LeadBrakeBehavior")
            self.assertEqual(lead.behavior.start_brake_t, LEAD_BRAKE_OPTIONS["brake_delay"])
            self.assertEqual(lead.initial_speed_mps, lead.cruise_speed_mps)  # scripted lead starts at cruise
            self.assertIsNone(env.scene_spec.npcs[0].initial_speed_mps)
            self.assertEqual(env.scene_spec.npcs[0].behavior["type"], "timed_brake")

            env.reset(seed=3, options=dict(JAYWALK_OPTIONS))
            ped = env.map.actor_manager.actors["pedestrian"][0]
            self.assertEqual(type(ped.behavior).__name__, "StopReturnBehavior")
            self.assertFalse(ped.reacts_to_traffic)

            env.reset(seed=3, options={"scene": "red_light_runner", "level": 1})
            actors = env.map.actor_manager.actors
            self.assertTrue(actors["traffic_light"], "red-light scenario must instantiate its signals")
            runner = [v for v in actors["vehicle"] if not v.obeys_signals]
            self.assertEqual(len(runner), 1)
            self.assertEqual(len(env.scene_spec.signals), len(actors["traffic_light"]))
        finally:
            env.close()

    def test_scenario_resets_are_replayable_in_process(self):
        env = _env()
        try:
            env.reset(seed=3, options={"scene": "red_light_runner", "level": 2})
            first = _spawn_state(env)
            env.reset(seed=3, options={"scene": "red_light_runner", "level": 2})
            self.assertEqual(first, _spawn_state(env))
        finally:
            env.close()


class VariationLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _env()
        cls.env.reset(seed=11, options=_random_options())
        cls.spec = cls.env.scene_spec
        cls.metric = cls.env.units.metric
        cls.mask = cls.env.map._drivable_mask

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_variation_is_a_pure_function_of_its_seeds(self):
        a, ra = apply_variation(self.spec, VariationRequest(seed=7), metric=self.metric, drivable_mask=self.mask)
        b, rb = apply_variation(self.spec, VariationRequest(seed=7), metric=self.metric, drivable_mask=self.mask)
        c, _ = apply_variation(self.spec, VariationRequest(seed=8), metric=self.metric, drivable_mask=self.mask)
        d, _ = apply_variation(self.spec, VariationRequest(seed=7, attempt=1), metric=self.metric, drivable_mask=self.mask)
        self.assertEqual(a.to_payload(), b.to_payload())
        self.assertEqual(ra, rb)
        self.assertNotEqual(a.to_payload(), c.to_payload())
        self.assertNotEqual(a.to_payload(), d.to_payload())
        # The identity spec is untouched.
        self.assertEqual(self.spec.scene_key, a.scene_key)
        self.assertEqual([n.npc_id for n in a.npcs], [n.npc_id for n in self.spec.npcs])
        self.assertEqual(a.agent.route_x, self.spec.agent.route_x)

    def test_spawn_v1_only_perturbs_start_offsets(self):
        varied, report = apply_variation(self.spec, VariationRequest(seed=7, profile="spawn_v1"), metric=self.metric)
        self.assertEqual(report["ego"]["progress_m"], 0.0)
        for npc, original in zip(varied.npcs, self.spec.npcs):
            self.assertEqual(npc.route_x, original.route_x)
            self.assertEqual(npc.initial_speed_mps, original.initial_speed_mps)
            self.assertEqual(npc.startup_ramp_duration_s, original.startup_ramp_duration_s)
            for value in npc.spawn_jitter_m:
                self.assertAlmostEqual(abs(value) / 0.3125, round(abs(value) / 0.3125))
                self.assertLessEqual(abs(value), 0.3125 + 1e-12)

    def test_spawn_phase_v1_shifts_phases_speeds_and_mixes_runtime_seeds(self):
        varied, report = apply_variation(self.spec, VariationRequest(seed=7, profile="spawn_phase_v1"), metric=self.metric, drivable_mask=self.mask)
        self.assertGreater(report["ego"]["progress_m"], 0.0)
        trimmed = [report["npcs"][n.npc_id]["trimmed_waypoints"] for n in self.spec.npcs]
        self.assertTrue(any(t > 0 for t in trimmed))
        for npc, original in zip(varied.npcs, self.spec.npcs):
            t = report["npcs"][npc.npc_id]["trimmed_waypoints"]
            self.assertEqual(npc.route_x, original.route_x[t:])
            self.assertGreaterEqual(len(npc.route_x), 4)
            self.assertLessEqual(npc.initial_speed_mps, npc.cruise_speed_mps)
            self.assertEqual(npc.runtime_seed, derive_seed(original.runtime_seed, "variation", 7))
        self.assertEqual(set(VARIATION_PROFILES), {"spawn_v1", "spawn_phase_v1"})
        with self.assertRaises(ValueError):
            apply_variation(self.spec, VariationRequest(seed=1, profile="nope"), metric=self.metric)

    def test_scripted_actors_keep_their_authored_geometry(self):
        env = _env()
        try:
            env.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
            spec = env.scene_spec
            varied, report = apply_variation(spec, VariationRequest(seed=5), metric=env.units.metric, drivable_mask=env.map._drivable_mask)
            lead = varied.npcs[0]
            self.assertEqual(lead.route_x, spec.npcs[0].route_x)
            self.assertEqual(report["npcs"][lead.npc_id]["trimmed_waypoints"], 0)
            self.assertEqual(report["npcs"][lead.npc_id]["lateral_m"], 0.0)
        finally:
            env.close()


class VariationEnvIntegrationTests(unittest.TestCase):
    def test_variation_seed_changes_state_but_not_identity_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _env(scene_library_enabled=True, scene_library_path=f"{tmpdir}/lib.db")
            try:
                _, frozen = env.reset(seed=11, options=_random_options())
                frozen_state = _spawn_state(env)
                _, v7 = env.reset(seed=11, options=_random_options(variation_seed=7))
                v7_state = _spawn_state(env)
                _, v7_again = env.reset(seed=11, options=_random_options(variation_seed=7))
                v7_again_state = _spawn_state(env)
                _, v8 = env.reset(seed=11, options=_random_options(variation_seed=8))
                v8_state = _spawn_state(env)
            finally:
                env.close()
        self.assertFalse(frozen["scene_cache"]["hit"])
        self.assertTrue(v7["scene_cache"]["hit"] and v8["scene_cache"]["hit"])  # variation never changes the key
        self.assertEqual(frozen["scene_descriptor"]["scene_key"], v7["scene_descriptor"]["scene_key"])
        self.assertIsNone(frozen["scene_descriptor"]["variation_seed"])
        self.assertEqual(v7["scene_descriptor"]["variation_seed"], 7)
        self.assertEqual(v7["scene_descriptor"]["variation_profile"], "spawn_phase_v1")
        self.assertTrue(v7["scene_descriptor"]["variation_applied"])
        self.assertIn("variation", v7)
        self.assertEqual(v7["variation"]["variation_seed"], 7)
        self.assertNotEqual(frozen_state, v7_state)
        self.assertNotEqual(v7_state, v8_state)
        self.assertTrue(v7_again["scene_cache"]["hit"])
        self.assertEqual(v7_state, v7_again_state)
        # NPC identity is preserved across variants.
        self.assertEqual([n[0] for n in frozen_state[1]], [n[0] for n in v7_state[1]])

    def test_per_reset_policy_varies_and_records_a_replayable_seed(self):
        env = _env(variation_policy="per_reset", variation_profile="spawn_v1")
        try:
            _, first = env.reset(seed=11, options=_random_options())
            first_state = _spawn_state(env)
            _, second = env.reset(seed=11, options=_random_options())
            second_state = _spawn_state(env)
            recorded = first["scene_descriptor"]["variation_seed"]
            self.assertIsNotNone(recorded)
            self.assertNotEqual(recorded, second["scene_descriptor"]["variation_seed"])
            self.assertNotEqual(first_state, second_state)
            # Replaying the recorded seed explicitly reproduces the episode's spawn state.
            env.reset(seed=11, options=_random_options(variation_seed=recorded, variation_profile="spawn_v1"))
            self.assertEqual(first_state, _spawn_state(env))
        finally:
            env.close()

    def test_invalid_variants_fall_back_to_the_frozen_realization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _env(scene_library_enabled=True, scene_library_path=f"{tmpdir}/lib.db")
            try:
                env.reset(seed=11, options=_random_options())  # miss: generated, validated, stored
                frozen_state = _spawn_state(env)
                original = env.map.spawn_validation_info
                calls = {"n": 0}

                def reject_variants():
                    # The hit path validates only the variants (then the frozen fallback).
                    calls["n"] += 1
                    if calls["n"] <= env.MAX_VARIATION_ATTEMPTS:
                        return {"valid": False, "reason": "forced"}
                    return original()

                env.map.scene.spawn_validation_info = reject_variants
                _, info = env.reset(seed=11, options=_random_options(variation_seed=7))
                fallback_state = _spawn_state(env)
            finally:
                env.close()
        self.assertTrue(info["scene_cache"]["hit"])
        self.assertFalse(info["scene_descriptor"]["variation_applied"])
        self.assertEqual(info["scene_descriptor"]["variation_attempts"], env.MAX_VARIATION_ATTEMPTS)
        self.assertEqual(info["variation"]["variation_applied"], False)
        self.assertEqual(calls["n"], env.MAX_VARIATION_ATTEMPTS + 1)
        self.assertEqual(frozen_state, fallback_state)


class NpcTableTests(unittest.TestCase):
    def test_npcs_are_stored_in_their_own_table_and_reassembled_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = f"{tmpdir}/lib.db"
            env = _env(scene_library_enabled=True, scene_library_path=db)
            try:
                _, info = env.reset(seed=11, options=_random_options())
                spec = env.scene_spec
            finally:
                env.close()
            library = SceneLibrary(db, generator_version="role_traffic_v2")
            key = info["scene_descriptor"]["scene_key"]
            rows = library.get_npcs_for_scene(key)
            self.assertEqual([r["npc_id"] for r in rows], spec.npc_ids)
            self.assertEqual([r["npc_uid"] for r in rows], [n.npc_uid for n in spec.npcs])
            record = library.get_scene_by_key(key)
            self.assertEqual(record["actors_payload"]["npc_uids"], [n.npc_uid for n in spec.npcs])
            self.assertEqual(SceneSpec.from_payload(record["actors_payload"]).digest(), spec.digest())
            self.assertEqual(sum(library.npc_counts(by="kind").values()), len(spec.npcs))
            self.assertTrue(library.get_npcs_by_filters(kind="vehicle", limit=2))
            library.delete_scene(key)
            self.assertEqual(library.get_npcs_for_scene(key), [])

    def test_stale_payloads_are_evicted_and_regenerated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = f"{tmpdir}/lib.db"
            env = _env(scene_library_enabled=True, scene_library_path=db)
            try:
                _, info = env.reset(seed=11, options=_random_options())
                key = info["scene_descriptor"]["scene_key"]
                library = SceneLibrary(db, generator_version="role_traffic_v2")
                import sqlite3

                with sqlite3.connect(db) as conn:
                    conn.execute("UPDATE scenes SET actors_json = '{\"schema_version\": 1}' WHERE scene_key = ?", (key,))
                _, again = env.reset(seed=11, options=_random_options())
                self.assertFalse(again["scene_cache"]["hit"])
                self.assertIn("evicted_invalid_cache", again["scene_cache"])
                self.assertIsNotNone(library.get_scene_by_key(key))  # regenerated and stored again
            finally:
                env.close()


class ActorManagerAdoptionTests(unittest.TestCase):
    def test_spec_instantiated_actors_are_adopted_without_copying(self):
        env = _env()
        try:
            env.reset(seed=11, options=_random_options())
            actors = instantiate_scene(env.scene_spec, units=env.units)
            vehicle = actors["vehicle"][0]
            env.map.actor_manager.load(actors, copy_actors=False)
            self.assertIs(env.map.actor_manager.actors["vehicle"][0], vehicle)
            env.map.actor_manager.load(actors, copy_actors=True)
            self.assertIsNot(env.map.actor_manager.actors["vehicle"][0], vehicle)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
