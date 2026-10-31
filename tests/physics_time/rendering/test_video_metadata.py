"""RENDERING / video & pacing: frame rates are derived from the physics clock.

Expected: metadata["render_fps"] == 1 / physics_dt so RecordVideo output plays in
simulated real time (one frame per env.step()); the human viewer's wall-clock
pacing defaults to real time and is reported as a real-time factor; episode
summaries expose the simulated duration.
"""

from __future__ import annotations

import math
import tempfile

import gymnasium as gym
import pytest

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.envs.renderer import Renderer
from CarlaBEV.src.sim_time import TimeConfig

from physics_time_helpers import (
    DOCUMENTED_DT_S,
    LEAD_BRAKE_OPTIONS,
    brake_only_action,
    make_env,
)


def test_class_metadata_declares_render_fps():
    assert CarlaBEV.metadata["render_fps"] == TimeConfig().render_fps == 10


def test_record_video_fps_matches_physics_rate():
    """C-06 (resolved): render_fps is derived from the clock."""
    env = make_env()
    try:
        assert math.isclose(
            env.metadata["render_fps"], 1.0 / DOCUMENTED_DT_S, rel_tol=1e-9
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "physics_dt,expected_fps", [(0.1, 10), (0.05, 20), (1.0 / 30.0, 30)]
)
def test_render_fps_follows_physics_dt(physics_dt, expected_fps):
    env = CarlaBEV(
        EnvConfig(
            render_mode="rgb_array",
            obs_mode="bev_rgb",
            physics_dt=physics_dt,
            scene_library_enabled=False,
        )
    )
    try:
        assert env.metadata["render_fps"] == expected_fps
        assert env.renderer.fps == expected_fps  # default pacing is real time
        assert math.isclose(env.renderer.real_time_factor, 1.0, rel_tol=1e-9)
    finally:
        env.close()


def test_record_video_wrapper_uses_clock_rate_and_one_frame_per_step():
    """Video duration == simulated duration: frames_per_sec = 1/dt and one frame per step.

    R-01: gymnasium's RecordVideo requires moviepy, which is not a declared
    dependency of CarlaBEV; the test is skipped where it is unavailable.
    """
    pytest.importorskip(
        "moviepy", reason="R-01: moviepy not installed; RecordVideo unavailable"
    )
    env = make_env()
    with tempfile.TemporaryDirectory() as tmp:
        wrapped = gym.wrappers.RecordVideo(
            env, tmp, episode_trigger=lambda _: True, disable_logger=True
        )
        try:
            assert wrapped.frames_per_sec == 1.0 / DOCUMENTED_DT_S
            wrapped.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
            n = 12
            for k in range(n):
                wrapped.step(brake_only_action(k))
            assert wrapped.recording
            # reset frame + one frame per env.step()
            assert len(wrapped.recorded_frames) == n + 1
            video_duration_s = (
                len(wrapped.recorded_frames) - 1
            ) / wrapped.frames_per_sec
            assert math.isclose(video_duration_s, env.sim_time_s, rel_tol=1e-9)
        finally:
            # Avoid encoding (moviepy is not a declared dependency); just discard frames.
            wrapped.recording = False
            wrapped.recorded_frames = []
            wrapped.close()


def test_human_fps_is_wall_clock_pacing_only():
    """Explicit fps changes the real-time factor, never the simulated step."""
    env = make_env(fps=15)
    try:
        assert env.renderer.fps == 15
        assert math.isclose(env.renderer.real_time_factor, 1.5, rel_tol=1e-9)
        assert env.physics_dt_s == DOCUMENTED_DT_S
        assert (
            env.metadata["render_fps"] == 10
        )  # recording rate unaffected by viewer pacing
    finally:
        env.close()


def test_renderer_defaults_to_real_time():
    renderer = Renderer(128, fps=None, physics_dt_s=0.1)
    assert renderer.fps == 10 and renderer.real_time_factor == 1.0
    renderer = Renderer(128, fps=None, physics_dt_s=0.05)
    assert renderer.fps == 20


def test_episode_summary_reports_simulated_duration():
    env = make_env()
    try:
        env.reset(seed=3, options=dict(LEAD_BRAKE_OPTIONS))
        for k in range(300):
            _, _, terminated, truncated, info = env.step(brake_only_action(k))
            if terminated or truncated:
                break
        summary = info["episode_info"]
        assert summary, "episode did not terminate"
        assert summary["physics_dt_s"] == DOCUMENTED_DT_S
        assert math.isclose(
            summary["episode_duration_s"],
            summary["length"] * DOCUMENTED_DT_S,
            abs_tol=1e-9,
        )
    finally:
        env.close()
