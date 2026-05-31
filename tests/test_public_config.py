import unittest

import numpy as np
from pydantic import ValidationError

from CarlaBEV.config import (
    AuthoredSceneReset,
    EnvConfig,
    RandomNavigationReset,
    RunConfig,
    ScenarioConfigReset,
    ScenarioPresetReset,
    build_authored_scene_options,
    build_random_navigation_options,
    build_scenario_config_options,
    build_scenario_options_from_config,
    build_scenario_preset_options,
    get_env_capabilities,
    validate_env_config,
    validate_run_config,
)
from CarlaBEV.envs import make_env
from CarlaBEV.tools.debug.cfg import ArgsCarlaBEV, EnvConfig as LegacyEnvConfig, LoggerConfig


class PublicConfigContractTests(unittest.TestCase):
    def test_validate_env_config_normalizes_legacy_fields(self):
        cfg = validate_env_config(
            {
                "map_name": "Town01",
                "obs_space": "bev",
                "masked": False,
                "action_space": "continuous",
                "reward_type": "shaping",
                "render_mode": "rgb_array",
            }
        )

        self.assertEqual(cfg.obs_mode, "bev_rgb")
        self.assertEqual(cfg.action_mode, "continuous")
        self.assertEqual(cfg.reward_mode, "shaping")
        self.assertEqual(cfg.obs_space, "bev")
        self.assertFalse(cfg.masked)

    def test_validate_env_config_rejects_missing_map_assets(self):
        with self.assertRaises(ValidationError):
            validate_env_config({"map_name": "Town99"})

    def test_validate_run_config_rejects_vector_make_env_path(self):
        with self.assertRaisesRegex(ValueError, "obs_mode='vector'"):
            validate_run_config(
                {
                    "env": {
                        "map_name": "Town01",
                        "obs_mode": "vector",
                        "render_mode": "rgb_array",
                    }
                }
            )

    def test_capabilities_report_public_contract(self):
        capabilities = get_env_capabilities()

        self.assertIn("Town01", capabilities["maps"])
        self.assertIn("discrete", capabilities["action_modes"])
        self.assertIn("bev_semantic", capabilities["obs_modes"])
        self.assertIn("carl", capabilities["reward_modes"])
        self.assertFalse(capabilities["supports_vector_make_env"])
        self.assertIn("jaywalk", capabilities["scenario_ids"])
        self.assertIn("jaywalk_debug", capabilities["scenario_preset_ids"])


class ResetBuilderTests(unittest.TestCase):
    def test_random_navigation_options_include_reset_mask(self):
        options = build_random_navigation_options(
            RandomNavigationReset(num_vehicles=7, route_dist_range=(40, 80)),
            reset_mask=[True, False, True],
        )

        self.assertEqual(options["scene"], "rdm")
        self.assertEqual(options["num_vehicles"], 7)
        self.assertEqual(options["route_dist_range"], [40, 80])
        self.assertTrue(np.array_equal(options["reset_mask"], np.array([True, False, True])))

    def test_authored_scene_options_include_variation_seed(self):
        options = build_authored_scene_options(
            AuthoredSceneReset(
                config_file="assets/scenes/jaywalk-01.01.json",
                variation_enabled=True,
                variation_seed=123,
            )
        )

        self.assertEqual(options["config_file"], "assets/scenes/jaywalk-01.01.json")
        self.assertTrue(options["variation_enabled"])
        self.assertEqual(options["variation_seed"], 123)

    def test_scenario_preset_options_expand_runtime_options(self):
        options = build_scenario_preset_options(
            ScenarioPresetReset(
                preset_id="jaywalk_debug",
                overrides={"anchor_x": 11, "anchor_y": 13},
            )
        )

        self.assertEqual(options["scene"], "jaywalk")
        self.assertEqual(options["anchor_x"], 11)
        self.assertEqual(options["anchor_y"], 13)

    def test_scenario_config_options_include_parameters(self):
        options = build_scenario_config_options(
            ScenarioConfigReset(
                scenario_id="lead_brake",
                level=2,
                parameters={"ego_speed": 5.0},
            ),
            reset_mask=[False, True],
        )

        self.assertEqual(options["scene"], "lead_brake")
        self.assertEqual(options["level"], 2)
        self.assertEqual(options["ego_speed"], 5.0)
        self.assertTrue(np.array_equal(options["reset_mask"], np.array([False, True])))

    def test_loaded_scenario_config_options_use_public_builder(self):
        options = build_scenario_options_from_config(
            {
                "scenario_id": "jaywalk",
                "level": 2,
                "anchor": {"x": 10, "y": 12},
                "parameters": {"ego_speed": 8.0},
            },
            overrides={"cross_delay": 1.5, "anchor_x": 14, "reset_mask": [True]},
        )

        self.assertEqual(options["scene"], "jaywalk")
        self.assertEqual(options["level"], 2)
        self.assertEqual(options["anchor_x"], 14)
        self.assertEqual(options["anchor_y"], 12)
        self.assertEqual(options["ego_speed"], 8.0)
        self.assertEqual(options["cross_delay"], 1.5)
        self.assertTrue(np.array_equal(options["reset_mask"], np.array([True])))


class MakeEnvCompatibilityTests(unittest.TestCase):
    def test_make_env_accepts_public_run_config(self):
        envs = make_env(
            RunConfig(
                env=EnvConfig(render_mode="rgb_array", map_name="Town01"),
                num_envs=1,
                capture_video=False,
            )
        )
        try:
            self.assertEqual(envs.single_action_space.n, 9)
        finally:
            envs.close()

    def test_make_env_accepts_legacy_debug_wrapper(self):
        envs = make_env(
            ArgsCarlaBEV(
                env=LegacyEnvConfig(render_mode="rgb_array", map_name="Town01"),
                logging=LoggerConfig(enabled=False),
            )
        )
        try:
            self.assertEqual(envs.single_action_space.n, 9)
        finally:
            envs.close()


if __name__ == "__main__":
    unittest.main()
