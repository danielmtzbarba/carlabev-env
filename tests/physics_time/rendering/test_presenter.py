"""RENDERING / presenter (Phase 5): interpolated real-time viewer is a pure observer.

Expected: ``display_fps`` shows ``display_fps * control_dt`` frames per control
step interpolated between the sub-step snapshots; it never changes the
trajectory, the observation or simulated time; snapshots interpolate poses
linearly (angles along the short arc); ``render_fps`` metadata stays at the
control rate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from CarlaBEV.envs.presenter import HeroPose, Presenter, WorldSnapshot
from CarlaBEV.src.geometry.body import Body, Footprint

from physics_time_helpers import (
    LEAD_BRAKE_OPTIONS,
    make_env,
    scripted_action,
    trajectory_digest,
    rollout,
)


def _snapshot(t, x, yaw, v=5.0):
    fp = Footprint.car(4.5, 1.8, 2.9)
    return WorldSnapshot(
        t_s=t,
        hero=HeroPose(x, 10.0, yaw, v),
        bodies=((3, Body(fp, x + 20.0, 12.0, yaw, v)),),
        rects=((5, (1.0, 2.0, 3.0, 4.0)),),
    )


def test_snapshot_interpolation_is_linear_with_short_arc_angles():
    a = _snapshot(0.0, 0.0, math.radians(170))
    b = _snapshot(0.1, 1.0, math.radians(-170))  # 20 degrees further, across +-pi
    m = a.interpolate(b, 0.5)
    assert m.t_s == pytest.approx(0.05)
    assert m.hero.x_m == pytest.approx(0.5)
    assert (m.hero.yaw - math.pi + math.pi) % (2 * math.pi) - math.pi == pytest.approx(
        0.0, abs=1e-9
    )
    assert m.bodies[0][1].x_m == pytest.approx(20.5)
    assert m.rects == b.rects
    assert a.interpolate(b, 0.0).hero == a.hero
    end = a.interpolate(b, 1.0).hero
    assert (end.x_m, end.v_mps) == (b.hero.x_m, b.hero.v_mps)
    assert math.cos(end.yaw) == pytest.approx(math.cos(b.hero.yaw)) and math.sin(
        end.yaw
    ) == pytest.approx(math.sin(b.hero.yaw))
    assert len(list(m.drawables())) == 2


def test_presenter_frame_schedule_covers_the_control_step():
    presenter = Presenter(display_fps=30, control_dt_s=0.1)
    assert presenter.frames_per_control_step == 3
    snaps = [
        _snapshot(0.0, 0.0, 0.0),
        _snapshot(0.05, 0.5, 0.0),
        _snapshot(0.1, 1.0, 0.0),
    ]
    times = presenter.frame_times(snaps)
    assert times == pytest.approx([1 / 30, 2 / 30, 0.1])
    frames = [presenter.frame_at(snaps, t) for t in times]
    assert [f.hero.x_m for f in frames] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert (
        Presenter(display_fps=5, control_dt_s=0.1).frames_per_control_step == 1
    )  # never drops below one


def test_presenter_never_changes_trajectory_observation_or_time():
    plain = make_env(physics_dt=0.05, action_repeat=2, render_mode="rgb_array")
    shown = make_env(
        physics_dt=0.05, action_repeat=2, render_mode="human", display_fps=30, fps=None
    )
    try:
        plain.reset(seed=0, options=LEAD_BRAKE_OPTIONS)
        shown.reset(seed=0, options=LEAD_BRAKE_OPTIONS)
        for i in range(15):
            obs_a, r_a, *_ = plain.step(scripted_action(i))
            obs_b, r_b, *_ = shown.step(scripted_action(i))
            assert np.array_equal(obs_a, obs_b)
            assert r_a == r_b
            assert plain.sim_time_s == shown.sim_time_s
        assert shown.presenter.frames_presented == 15 * 3
        assert shown.presenter.last_frame_times[-1] == pytest.approx(shown.sim_time_s)
        assert (
            plain.map.hero.x_m == shown.map.hero.x_m
            and plain.map.hero.yaw == shown.map.hero.yaw
        )
        assert (
            shown.metadata["render_fps"] == 10
        )  # observation stream rate, not the display rate
        assert (
            shown.presenter.renderer.fps == 30
            and shown.presenter.renderer.real_time_factor == pytest.approx(1.0)
        )
    finally:
        plain.close()
        shown.close()


def test_digest_is_identical_with_and_without_presenter():
    digests = []
    for kwargs in (
        {"render_mode": "rgb_array"},
        {"render_mode": "human", "display_fps": 20, "fps": None},
    ):
        env = make_env(physics_dt=0.05, action_repeat=2, **kwargs)
        try:
            traj = rollout(
                env, options=LEAD_BRAKE_OPTIONS, num_steps=20, policy=scripted_action
            )
            digests.append(trajectory_digest(traj))
        finally:
            env.close()
    assert digests[0] == digests[1]
