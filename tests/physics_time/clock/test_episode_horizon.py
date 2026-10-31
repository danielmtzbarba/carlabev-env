"""CLOCK / episode horizon (C-08): the task horizon is defined in simulated seconds.

Expected: ``EnvConfig.max_episode_duration_s`` (default derived from the legacy
``max_actions`` x ``physics_dt`` = 500 s) resolves to steps through the clock; the
environment truncates (Gymnasium ``truncated=True``, ``terminated=False``) at that
horizon for every reward family; the horizon is reported on reset and at episode end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.src.sim_time import EpisodeHorizon, TimeConfig
from CarlaBEV.src.units import UnitContract

from physics_time_helpers import LEAD_BRAKE_OPTIONS, brake_only_action

QUIET_OPTIONS = {
    **LEAD_BRAKE_OPTIONS,
    "ego_speed": 3.0,
    "lead_gap": 30.0,
    "brake_delay": 50.0,
}


def _env(**overrides):
    return CarlaBEV(
        EnvConfig(
            render_mode="rgb_array",
            obs_mode="bev_rgb",
            action_mode="continuous",
            scene_library_enabled=False,
            **overrides,
        )
    )


def test_default_horizon_preserves_legacy_5000_steps_at_default_dt():
    units = UnitContract.from_env_config(EnvConfig(render_mode="rgb_array"))
    assert units.horizon.max_episode_duration_s == 500.0
    assert units.max_episode_steps == 5000
    assert units.horizon.offroad_terminate_after_s == 4.0
    assert units.offroad_terminate_after_steps == 40


def test_horizon_in_seconds_is_invariant_to_physics_dt():
    horizon = EpisodeHorizon(max_episode_duration_s=30.0, offroad_terminate_after_s=4.0)
    assert horizon.max_episode_steps(TimeConfig(0.1)) == 300
    assert horizon.max_episode_steps(TimeConfig(0.05)) == 600
    assert horizon.offroad_terminate_after_steps(TimeConfig(0.05)) == 80
    assert horizon.describe(TimeConfig(0.05))["max_episode_steps"] == 600


def test_legacy_max_actions_is_interpreted_at_the_configured_dt():
    units = UnitContract.from_env_config(
        EnvConfig(render_mode="rgb_array", max_actions=200, physics_dt=0.05)
    )
    assert math.isclose(units.horizon.max_episode_duration_s, 10.0)
    assert units.max_episode_steps == 200
    # Explicit seconds win over the legacy alias.
    units = UnitContract.from_env_config(
        EnvConfig(render_mode="rgb_array", max_actions=200, max_episode_duration_s=2.0)
    )
    assert units.max_episode_steps == 20


@pytest.mark.parametrize("reward_mode", ["carl", "shaping"])
def test_env_truncates_at_horizon_for_every_reward_family(reward_mode):
    env = _env(reward_mode=reward_mode, max_episode_duration_s=1.0)
    try:
        _, info = env.reset(seed=3, options=QUIET_OPTIONS)
        assert info["episode_horizon"]["max_episode_steps"] == 10
        assert info["episode_horizon"]["max_episode_duration_s"] == 1.0
        for k in range(50):
            _, _, terminated, truncated, step_info = env.step(brake_only_action(k))
            if terminated or truncated:
                break
        assert truncated and not terminated, (
            "horizon expiry must be a truncation, not a termination"
        )
        assert k == 9
        summary = step_info["episode_info"]
        assert summary["termination"] == "max_actions"
        assert math.isclose(summary["episode_duration_s"], 1.0)
        assert summary["max_episode_steps"] == 10 and summary["length"] == 10
    finally:
        env.close()


def test_horizon_truncation_happens_at_same_sim_time_for_different_dt():
    ends = {}
    for dt in (0.1, 0.05):
        env = _env(reward_mode="carl", max_episode_duration_s=1.5, physics_dt=dt)
        try:
            env.reset(seed=3, options=QUIET_OPTIONS)
            for k in range(100):
                _, _, terminated, truncated, _ = env.step(
                    np.array([0.0, 0.0, 1.0], dtype=np.float32)
                )
                if terminated or truncated:
                    break
            ends[dt] = (env.sim_time_s, env.map.step_index, truncated)
        finally:
            env.close()
    assert ends[0.1] == (1.5, 15, True)
    assert ends[0.05] == (1.5, 30, True)


def test_physical_termination_is_not_a_truncation():
    env = _env(reward_mode="carl")
    try:
        env.reset(seed=3, options={**LEAD_BRAKE_OPTIONS, "lead_gap": 6.0})
        for k in range(100):
            _, _, terminated, truncated, step_info = env.step(
                np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert step_info["episode_info"]["termination"] in {"collision", "success"}
    finally:
        env.close()


def test_shaping_reward_receives_step_resolved_horizons():
    env = _env(
        reward_mode="shaping",
        max_episode_duration_s=12.0,
        offroad_terminate_after_s=2.0,
        physics_dt=0.05,
    )
    try:
        assert env.reward_fn.max_actions == 240
        assert env.reward_fn.offroad_terminate_after == 40
    finally:
        env.close()
