"""RENDERING / independence: fps and render_mode never change simulated state.

Expected: seeded rollouts are bit-identical across fps values and render modes;
fps is consumed only by the Renderer.
"""

from __future__ import annotations

import pytest

from physics_time_helpers import (
    JAYWALK_OPTIONS,
    LEAD_BRAKE_OPTIONS,
    RANDOM_NAV_OPTIONS,
    assert_trajectories_close,
    make_env,
    rollout,
    trajectory_digest,
)


SCENARIOS = {"lead_brake": LEAD_BRAKE_OPTIONS, "jaywalk": JAYWALK_OPTIONS}
RENDER_CONFIGS = [
    (15, "rgb_array"),
    (60, "rgb_array"),
    (1, "rgb_array"),
    (5, "human"),
    (60, "human"),
]


def _reference(options, *, num_steps=50):
    env = make_env(fps=15, render_mode="rgb_array")
    try:
        return rollout(env, options=options, num_steps=num_steps)
    finally:
        env.close()


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("fps,render_mode", RENDER_CONFIGS)
def test_trajectory_is_independent_of_fps_and_render_mode(scenario, fps, render_mode):
    options = SCENARIOS[scenario]
    expected = _reference(options)
    env = make_env(fps=fps, render_mode=render_mode)
    try:
        actual = rollout(env, options=options, num_steps=50)
    finally:
        env.close()
    assert len(actual["steps"]) > 5, "rollout ended too early to be meaningful"
    assert_trajectories_close(actual, expected, atol=0.0)
    assert trajectory_digest(actual) == trajectory_digest(expected)


def test_random_navigation_trajectory_is_independent_of_fps():
    expected = _reference(RANDOM_NAV_OPTIONS, num_steps=40)
    env = make_env(fps=60, render_mode="rgb_array")
    try:
        actual = rollout(env, options=RANDOM_NAV_OPTIONS, num_steps=40)
    finally:
        env.close()
    assert_trajectories_close(actual, expected, atol=0.0)


def test_repeated_rollout_is_bitwise_deterministic():
    assert trajectory_digest(_reference(LEAD_BRAKE_OPTIONS)) == trajectory_digest(
        _reference(LEAD_BRAKE_OPTIONS)
    )


def test_fps_only_reaches_the_renderer():
    env = make_env(fps=7)
    try:
        assert env.renderer.fps == 7
        assert not hasattr(env.map, "fps")
        assert not hasattr(env.map.actor_manager, "fps")
    finally:
        env.close()
