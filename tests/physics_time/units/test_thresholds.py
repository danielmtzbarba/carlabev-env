"""UNITS / thresholds: reward and collision thresholds carry explicit SI units.

Expected: speed limit exposed and consumed in m/s; proximity radius exposed in
meters; both configurable through EnvConfig with behaviour-preserving defaults.
"""

from __future__ import annotations

import math

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.src.deeprl.carl_reward_fn import CaRLRewardFn
from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.units import SceneUnits

from physics_time_helpers import LEAD_BRAKE_OPTIONS, brake_only_action, make_env


def _reward_speed_limit(env, max_steps=10):
    """Speed limit as consumed by CaRL on the first fully evaluated (non-checkpoint) step."""
    for k in range(max_steps):
        env.step(brake_only_action(k))
        debug = env.current_info["reward"].get("debug") or {}
        if "speed_limit" in debug:
            return debug["speed_limit"]
    raise AssertionError("reward debug block never populated")


def test_speed_limit_defaults_preserve_legacy_35_kmh():
    units = SceneUnits()
    assert math.isclose(units.speed_limit_mps, 35 / 3.6, rel_tol=1e-9)
    assert math.isclose(units.speed_limit_kmh, 35.0, rel_tol=1e-9)
    assert math.isclose(
        CaRLRewardFn._speed_limit_to_mps(35), 35 / 3.6, rel_tol=1e-9
    )  # legacy fallback


def test_speed_limit_is_exposed_in_mps():
    """U-04 (resolved): explicit SI key consumed by the reward."""
    env = make_env()
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        scene = env.map.scene_info["scene"]
        assert math.isclose(scene["speed_limit_mps"], 35 / 3.6, rel_tol=1e-9)
        assert math.isclose(
            scene["speed_limit"], 35.0, rel_tol=1e-9
        )  # legacy km/h key kept
        assert math.isclose(_reward_speed_limit(env), 35 / 3.6, rel_tol=1e-9)
    finally:
        env.close()


def test_speed_limit_is_configurable_in_mps():
    env = CarlaBEV(
        EnvConfig(
            render_mode="rgb_array",
            obs_mode="bev_rgb",
            action_mode="continuous",
            speed_limit_mps=8.0,
            scene_library_enabled=False,
        )
    )
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        assert env.speed_limit_mps == 8.0
        assert math.isclose(_reward_speed_limit(env), 8.0, rel_tol=1e-9)
    finally:
        env.close()


def test_proximity_radius_default_preserves_35_surface_pixels():
    assert math.isclose(SceneUnits().proximity_radius_m, 35 * 0.3125, rel_tol=1e-9)
    env = make_env()
    try:
        env.reset(seed=3, options={**LEAD_BRAKE_OPTIONS, "lead_gap": 12.0})
        env.step(brake_only_action(0))
        assert env.current_info["collision"]["actors_state"] == []  # 12 m > 10.94 m
    finally:
        env.close()
    env = make_env()
    try:
        env.reset(seed=3, options={**LEAD_BRAKE_OPTIONS, "lead_gap": 9.0})
        env.step(brake_only_action(0))
        assert len(env.current_info["collision"]["actors_state"]) == 1
    finally:
        env.close()


def test_proximity_radius_is_a_metric_parameter():
    """U-05 (resolved): exposed in meters and configurable."""
    env = make_env()
    try:
        assert math.isclose(env.proximity_radius_m, 10.9375, rel_tol=1e-9)
        assert env.map.proximity_radius_m == env.proximity_radius_m
    finally:
        env.close()
    env = CarlaBEV(
        EnvConfig(
            render_mode="rgb_array",
            obs_mode="bev_rgb",
            action_mode="continuous",
            proximity_radius_m=15.0,
            scene_library_enabled=False,
        )
    )
    try:
        env.reset(seed=3, options={**LEAD_BRAKE_OPTIONS, "lead_gap": 12.0})
        env.step(brake_only_action(0))
        assert (
            len(env.current_info["collision"]["actors_state"]) == 1
        )  # 12 m < 15 m now reported
    finally:
        env.close()


def test_shaping_reward_thresholds_are_metric_with_legacy_pixel_aliases():
    default = RewardFn()
    assert math.isclose(default.lat_clip_m, 1.25, rel_tol=1e-9)
    assert math.isclose(default.lat_small_m, 0.25, rel_tol=1e-9)
    assert math.isclose(default.route_dev_start_m, 2.5, rel_tol=1e-9)
    assert math.isclose(default.out_of_bounds_threshold_m, 18.75, rel_tol=1e-9)
    legacy = RewardFn(lat_clip=4.0, lat_small=0.8, route_dev_start=8.0)
    for name in ("lat_clip_m", "lat_small_m", "route_dev_start_m"):
        assert math.isclose(getattr(legacy, name), getattr(default, name), rel_tol=1e-9)
    assert math.isclose(CaRLRewardFn().out_of_bounds_threshold_m, 15.625, rel_tol=1e-9)
