"""Clock and episode information must survive the public Gymnasium API."""

import numpy as np
import pytest

from CarlaBEV.config import EnvConfig, RunConfig
from CarlaBEV.envs import make_env as make_vector_env
from physics_time_helpers import LEAD_BRAKE_OPTIONS, brake_only_action, make_env


def test_reset_and_nonterminal_step_expose_clock_information():
    env = make_env(physics_dt=0.05, action_repeat=2)
    try:
        _, info = env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        assert info["scene"]["sim_time_s"] == 0.0
        assert info["hero"]["step_index"] == 0
        for decision in range(1, 4):
            _, _, terminated, truncated, info = env.step(brake_only_action(0))
            assert not terminated and not truncated
            assert info["scene"]["sim_time_s"] == pytest.approx(decision * 0.1)
            assert info["hero"]["step_index"] == decision * 2
            assert info["hero"]["physics_dt_s"] == 0.05
            assert info["episode_info"] == {}
            assert info is env.current_info
        _, reset_info = env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        assert reset_info["scene"]["sim_time_s"] == 0.0
    finally:
        env.close()


def test_early_truncation_keeps_clock_and_episode_summary():
    env = make_env(physics_dt=0.05, action_repeat=4, max_episode_duration_s=0.15)
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        _, _, terminated, truncated, info = env.step(brake_only_action(0))
        assert truncated and not terminated
        assert info["scene"]["sim_time_s"] == pytest.approx(0.15)
        assert info["hero"]["step_index"] == 3
        assert info["episode_info"]["episode_duration_s"] == pytest.approx(0.15)
        assert info["episode_info"]["termination"] == "max_actions"
    finally:
        env.close()


def test_physical_termination_keeps_clock_and_episode_summary():
    env = make_env(physics_dt=0.025, action_repeat=4)
    try:
        env.reset(
            seed=0,
            options={
                **LEAD_BRAKE_OPTIONS,
                "lead_gap": 6.0,
                "lead_speed": 0.0,
                "brake_delay": 50.0,
                "ego_speed": 12.0,
            },
        )
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(
                np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["episode_info"]["termination"] == "collision"
        assert info["hero"]["step_index"] == info["episode_info"]["physics_steps"]
        assert info["scene"]["sim_time_s"] == info["episode_info"]["episode_duration_s"]
        assert info["scene"]["sim_time_s"] == env.sim_time_s
    finally:
        env.close()


def test_wrapped_vector_environment_preserves_clock_and_episode_info():
    envs = make_vector_env(
        RunConfig(
            env=EnvConfig(
                render_mode="rgb_array",
                action_mode="continuous",
                scene_library_enabled=False,
                max_episode_duration_s=0.2,
            ),
            num_envs=2,
        )
    )
    try:
        _, info = envs.reset(seed=[3, 4], options=LEAD_BRAKE_OPTIONS)
        np.testing.assert_array_equal(info["scene"]["sim_time_s"], [0.0, 0.0])
        actions = np.array([[0.0, 0.0, 1.0]] * 2, dtype=np.float32)
        envs.step(actions)
        _, _, terminated, truncated, info = envs.step(actions)
        assert not terminated.any() and truncated.all()
        np.testing.assert_allclose(info["scene"]["sim_time_s"], [0.2, 0.2])
        np.testing.assert_array_equal(info["episode_info"]["length"], [2, 2])
        assert info["_episode"].all()  # RecordEpisodeStatistics still works.
    finally:
        envs.close()
