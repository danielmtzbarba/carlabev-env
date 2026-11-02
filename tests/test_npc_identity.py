"""Replayable NPC identity: npc_id, runtime_seed, resolved spawn offsets, payload schema v2."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options  # noqa: E402
from CarlaBEV.envs.carlabev import CarlaBEV  # noqa: E402
from CarlaBEV.src.actors.vehicle import Vehicle  # noqa: E402
from CarlaBEV.src.control.stanley_controller import Controller  # noqa: E402
from CarlaBEV.src.managers.actor_manager import npc_id_for  # noqa: E402
from CarlaBEV.src.managers.scene_library import (  # noqa: E402
    ACTOR_PAYLOAD_SCHEMA_VERSION,
    deserialize_scene_payload,
    serialize_scene_payload,
)
from CarlaBEV.src.managers.traffic_lifecycle import TrafficLifecycle  # noqa: E402
from CarlaBEV.src.randomness import derive_seed  # noqa: E402
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT  # noqa: E402


def _straight_vehicle(**kwargs) -> Vehicle:
    route_x = list(range(0, 400, 10))
    return Vehicle(map_size=128, routeX=route_x, routeY=[0] * len(route_x), target_speed=8.0, **kwargs)


class SpawnOffsetResolutionTests(unittest.TestCase):
    def test_resolved_spawn_offset_replays_on_every_reset(self):
        vehicle = _straight_vehicle()
        vehicle.resolve_spawn_jitter(np.random.default_rng(5))
        offset = vehicle.spawn_jitter_m
        self.assertIsNotNone(offset)
        poses = []
        for _ in range(3):
            vehicle.reset()
            poses.append((vehicle._controller.x_m, vehicle._controller.y_m))
            self.assertEqual(vehicle._controller.spawn_jitter_m, offset)
        self.assertEqual(len(set(poses)), 1)

    def test_draw_matches_legacy_controller_draw(self):
        # The frozen offset must equal what the controller used to draw itself.
        rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
        frozen = Controller.draw_spawn_jitter_m(rng_a)
        ctrl = Controller(target_speed=10.0, dt=0.1)
        ctrl.set_route([0.0, 10.0, 20.0], [0.0, 0.0, 0.0], jitter_start=True, np_rng=rng_b)
        self.assertEqual(ctrl.spawn_jitter_m, frozen)
        self.assertEqual(ctrl.x_m, ctrl.cx_m[0] + frozen[0])
        self.assertEqual(ctrl.y_m, ctrl.cy_m[0] + frozen[1])

    def test_unresolved_actor_keeps_legacy_seeded_draw(self):
        a = _straight_vehicle(np_rng=np.random.default_rng(3))
        b = _straight_vehicle(np_rng=np.random.default_rng(3))
        a.reset()
        b.reset()
        self.assertEqual((a._controller.x_m, a._controller.y_m), (b._controller.x_m, b._controller.y_m))
        self.assertIsNone(a.spawn_jitter_m)  # legacy path does not freeze the draw

    def test_runtime_seed_binds_a_private_generator(self):
        vehicle = _straight_vehicle()
        vehicle.set_runtime_seed(derive_seed(3000, "npc", 2))
        other = _straight_vehicle()
        other.set_runtime_seed(derive_seed(3000, "npc", 2))
        self.assertEqual(vehicle.np_rng.integers(0, 1 << 30), other.np_rng.integers(0, 1 << 30))


class ScenePayloadSchemaTests(unittest.TestCase):
    def _scene(self):
        vehicle = _straight_vehicle()
        vehicle.npc_id = npc_id_for("vehicle", 0)
        vehicle.set_runtime_seed(123456)
        vehicle.resolve_spawn_jitter(np.random.default_rng(1))
        vehicle.traffic_role = "lead"
        vehicle.obeys_signals = False
        vehicle.set_initial_speed_mps(2.0)
        vehicle.set_startup_ramp_duration(1.0)
        agent = ([10.0, 20.0, 30.0], [5.0, 5.0, 5.0], 0.0, 9.0, (0.3125, -0.3125))
        return {"agent": agent, "vehicle": [vehicle], "pedestrian": [], "target": [], "traffic_light": []}

    def test_round_trip_preserves_identity_offsets_and_flags(self):
        payload = serialize_scene_payload(self._scene(), surface_size=128)
        self.assertEqual(payload["schema_version"], ACTOR_PAYLOAD_SCHEMA_VERSION)
        self.assertEqual(payload["agent"]["spawn_jitter_m"], [0.3125, -0.3125])
        npc = payload["npcs"][0]
        self.assertEqual(npc["npc_id"], "vehicle:000")
        self.assertEqual(npc["kind"], "vehicle")
        self.assertIsNone(npc["behavior"])
        self.assertEqual(npc["runtime_seed"], 123456)
        self.assertFalse(npc["obeys_signals"])
        self.assertEqual(npc["vehicle_class"], "car")

        actors = deserialize_scene_payload(payload, map_size=128, units=DEFAULT_UNIT_CONTRACT)
        self.assertEqual(actors["agent"][4], (0.3125, -0.3125))
        vehicle = actors["vehicle"][0]
        self.assertEqual(vehicle.npc_id, "vehicle:000")
        self.assertEqual(vehicle.runtime_seed, 123456)
        self.assertEqual(vehicle.spawn_jitter_m, tuple(npc["spawn_jitter_m"]))
        self.assertFalse(vehicle.obeys_signals)
        self.assertTrue(vehicle.yields_at_junctions)
        self.assertEqual(vehicle.traffic_role, "lead")
        self.assertEqual(vehicle.initial_speed_mps, 2.0)
        # A deserialized actor resets to the same pose as the original.
        original = self._scene()["vehicle"][0]
        original.reset()
        vehicle.reset()
        self.assertEqual((original._controller.x_m, original._controller.y_m), (vehicle._controller.x_m, vehicle._controller.y_m))

    def test_legacy_v1_payload_is_rejected(self):
        payload = serialize_scene_payload(self._scene(), surface_size=128)
        payload["schema_version"] = 1
        with self.assertRaises(ValueError):
            deserialize_scene_payload(payload, map_size=128, units=DEFAULT_UNIT_CONTRACT)
        payload = serialize_scene_payload(self._scene(), surface_size=128)
        del payload["schema_version"]  # pre-versioning payloads
        with self.assertRaises(ValueError):
            deserialize_scene_payload(payload, map_size=128, units=DEFAULT_UNIT_CONTRACT)

    def test_payload_without_resolved_offset_is_rejected(self):
        payload = serialize_scene_payload(self._scene(), surface_size=128)
        payload["npcs"][0]["spawn_jitter_m"] = None
        with self.assertRaises(ValueError):
            deserialize_scene_payload(payload, map_size=128, units=DEFAULT_UNIT_CONTRACT)


class GeneratedSceneIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = CarlaBEV(
            EnvConfig(render_mode="rgb_array", obs_mode="bev_rgb", action_mode="discrete", size=128, scene_library_enabled=False)
        )
        options = build_random_navigation_options(
            RandomNavigationReset(difficulty_preset_id="easy", scene_seed=1000, route_seed=2000, traffic_seed=3000, num_vehicles=5)
        )
        _, cls.info = cls.env.reset(seed=11, options=options)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_every_generated_npc_has_identity_seed_and_offset(self):
        vehicles = self.env.map.actor_manager.actors["vehicle"]
        self.assertTrue(vehicles)
        self.assertEqual([v.npc_id for v in vehicles], [npc_id_for("vehicle", i) for i in range(len(vehicles))])
        for ordinal, vehicle in enumerate(vehicles):
            self.assertEqual(vehicle.runtime_seed, derive_seed(3000, "npc", ordinal))
            self.assertIsNotNone(vehicle.spawn_jitter_m)
            self.assertEqual(vehicle._controller.spawn_jitter_m, vehicle.spawn_jitter_m)
        self.assertIsNotNone(self.env.map.agent_spawn_jitter_m)
        self.assertEqual(self.env.map.hero.spawn_jitter_m, self.env.map.agent_spawn_jitter_m)

    def test_collision_info_exposes_npc_id_for_nearby_actors(self):
        info = self.env.map.collision_check(min_dist=10_000)  # everything is "nearby"
        self.assertIn("npc_id", info["collision"])
        nearby_ids = {entry["npc_id"] for entry in info["collision"]["actors_state"]}
        self.assertEqual(nearby_ids, {v.npc_id for v in self.env.map.actor_manager.actors["vehicle"]})

    def test_scenario_actors_get_fallback_ids_and_bypass_the_cache(self):
        from CarlaBEV.src.diagnostics.rollouts import LEAD_BRAKE_OPTIONS

        with tempfile.TemporaryDirectory() as tmpdir:
            env = CarlaBEV(
                EnvConfig(
                    render_mode="rgb_array", obs_mode="bev_rgb", action_mode="continuous", size=128,
                    scene_library_enabled=True, scene_library_path=f"{tmpdir}/lib.db",
                )
            )
            try:
                _, info = env.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
                self.assertEqual([v.npc_id for v in env.map.actor_manager.actors["vehicle"]], ["vehicle:000"])
                self.assertEqual(info["scene_cache"], {"enabled": True, "hit": False, "scene_key": None, "bypassed": "non_random_scene"})
            finally:
                env.close()


class LifecycleSeedingTests(unittest.TestCase):
    def test_lifecycle_rngs_derive_from_the_traffic_seed_not_from_hash(self):
        a = TrafficLifecycle({}, DEFAULT_UNIT_CONTRACT, seed=3000, fov_radius_m=20.0)
        b = TrafficLifecycle({}, DEFAULT_UNIT_CONTRACT, seed=3000, fov_radius_m=20.0)
        expected = np.random.default_rng(derive_seed(3000, "traffic-lifecycle"))
        self.assertEqual(a.np_rng.integers(0, 1 << 30), expected.integers(0, 1 << 30))
        self.assertEqual(b.rng.random(), a.rng.random())

    def test_redispatch_uses_the_actor_runtime_rng_and_refreshes_the_offset(self):
        lifecycle = TrafficLifecycle({}, DEFAULT_UNIT_CONTRACT, seed=3000, fov_radius_m=20.0, startup_sampler=None)
        vehicle = _straight_vehicle()
        vehicle.set_runtime_seed(777)
        vehicle.resolve_spawn_jitter(np.random.default_rng(0))
        vehicle.reset()
        private_rng = vehicle.np_rng
        route_x = list(range(0, 300, 10))
        lifecycle._redispatch(vehicle, route_x, [50] * len(route_x))
        self.assertIs(vehicle.np_rng, private_rng)  # not replaced by the shared lifecycle RNG
        self.assertIsNone(vehicle.spawn_jitter_m)  # frozen offset released for the new route
        self.assertEqual(vehicle.recycle_count, 1)
        expected = Controller.draw_spawn_jitter_m(np.random.default_rng(777))
        self.assertEqual(vehicle._controller.spawn_jitter_m, expected)


if __name__ == "__main__":
    unittest.main()
