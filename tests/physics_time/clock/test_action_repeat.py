"""CLOCK / multi-rate (Phase 5): action_repeat decouples physics from decisions.

Expected: one ``env.step()`` holds the action over ``action_repeat`` physics
sub-steps of ``physics_dt``; the trajectory equals running ``action_repeat = 1``
at the same ``physics_dt`` and keeping every k-th observation (bit-identical);
rewards are summed over the sub-steps and normalised to the physics step so the
return per simulated second is step-invariant; termination latches at the first
terminal sub-step; ``render_fps`` follows the control rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from CarlaBEV.config.env import EnvConfig
from CarlaBEV.src.sim_time import SimClock, TimeConfig

from physics_time_helpers import (
    LEAD_BRAKE_OPTIONS,
    gas_only_action,
    make_env,
    scripted_action,
)


# --------------------------------------------------------------------------- contract
def test_time_config_control_step_and_render_fps():
    tc = TimeConfig(physics_dt_s=0.05, action_repeat=2)
    assert tc.control_dt_s == pytest.approx(0.1)
    assert tc.control_rate_hz == pytest.approx(10.0)
    assert tc.render_fps == 10  # one observation per control step
    assert TimeConfig(0.1).control_dt_s == 0.1 and TimeConfig(0.1).action_repeat == 1
    with pytest.raises(ValueError):
        TimeConfig(0.1, action_repeat=0)
    clock = SimClock(tc)
    clock.begin_control_step()
    clock.tick()
    clock.tick()
    assert (clock.control_index, clock.step_index, clock.t_s) == (
        1,
        2,
        pytest.approx(0.1),
    )
    clock.reset()
    assert (clock.control_index, clock.step_index) == (0, 0)


def test_env_config_validates_multi_rate_fields():
    cfg = EnvConfig(render_mode="rgb_array", physics_dt=0.05, action_repeat=2)
    assert cfg.action_repeat == 2
    with pytest.raises(ValueError):
        EnvConfig(render_mode="rgb_array", action_repeat=0)
    with pytest.raises(ValueError):
        EnvConfig(render_mode="human", display_fps=30, fps=15)  # mutually exclusive
    horizon = EnvConfig(
        render_mode="rgb_array",
        physics_dt=0.05,
        action_repeat=2,
        max_episode_duration_s=3.0,
    )
    env = make_env(physics_dt=0.05, action_repeat=2, max_episode_duration_s=3.0)
    try:
        info = env.reset(seed=0, options=LEAD_BRAKE_OPTIONS)[1]
        assert info["episode_horizon"]["control_dt_s"] == pytest.approx(0.1)
        assert info["episode_horizon"]["max_episode_steps"] == 60  # physics steps
        assert env.metadata["render_fps"] == 10
        assert env.control_dt_s == pytest.approx(0.1) and env.action_repeat == 2
    finally:
        env.close()
    del horizon


# --------------------------------------------------------------------------- equivalence
def _rollout(env, policy, n, options=LEAD_BRAKE_OPTIONS):
    env.reset(seed=0, options=options)
    out = []
    for i in range(n):
        obs, r, term, trunc, info = env.step(policy(i))
        hero = env.map.hero
        out.append(
            {
                "t": env.sim_time_s,
                "steps": env.map.clock.step_index,
                "ctrl": env.map.clock.control_index,
                "hero": (
                    float(hero.x_m),
                    float(hero.y_m),
                    float(hero.yaw),
                    float(hero.v_mps),
                ),
                "r": float(r),
                "obs": obs.copy(),
                "done": bool(term or trunc),
                "info": info,
            }
        )
        if term or trunc:
            break
    return out


@pytest.mark.parametrize("k", [2, 3])
def test_sub_stepping_equals_k_physics_steps_at_the_same_dt(k):
    """env(dt, k) step n == env(dt, 1) step k*n: states, observations bit-identical, rewards summed."""
    dt = 0.1 / k
    fine = make_env(physics_dt=dt, action_repeat=1)
    multi = make_env(physics_dt=dt, action_repeat=k)
    try:
        n = 20
        fine_traj = _rollout(fine, lambda i: scripted_action(i // k), n * k)
        multi_traj = _rollout(multi, scripted_action, n)
        assert len(multi_traj) == n and len(fine_traj) == n * k
        for i, step in enumerate(multi_traj):
            ref = fine_traj[(i + 1) * k - 1]
            assert step["hero"] == ref["hero"]
            assert step["t"] == ref["t"] and step["steps"] == ref["steps"]
            assert step["ctrl"] == i + 1 and ref["ctrl"] == (i + 1) * k
            assert np.array_equal(step["obs"], ref["obs"])
            summed = sum(fine_traj[j]["r"] for j in range(i * k, (i + 1) * k))
            assert step["r"] == pytest.approx(summed, abs=1e-12)
        assert multi.current_info["reward"]["substep_rewards"] == [
            pytest.approx(fine_traj[j]["r"]) for j in range((n - 1) * k, n * k)
        ]
    finally:
        fine.close()
        multi.close()


def test_action_repeat_one_is_the_legacy_step():
    """Defaults unchanged: k=1 at dt=0.1 gives exactly one physics step per decision."""
    env = make_env()
    try:
        traj = _rollout(env, scripted_action, 10)
        assert (
            traj[-1]["t"] == pytest.approx(1.0)
            and traj[-1]["steps"] == 10
            and traj[-1]["ctrl"] == 10
        )
        assert env.action_repeat == 1 and env.control_dt_s == env.physics_dt_s == 0.1
    finally:
        env.close()


# --------------------------------------------------------------------------- rewards
@pytest.mark.parametrize("reward_mode", ["carl", "shaping"])
def test_return_per_simulated_second_is_physics_step_invariant(reward_mode):
    """Same 3 s manoeuvre at (0.1, 1), (0.05, 2), (0.025, 4): returns agree within the physics-convergence band."""
    returns = {}
    for dt, k in ((0.1, 1), (0.05, 2), (0.025, 4)):
        env = make_env(physics_dt=dt, action_repeat=k, reward_mode=reward_mode)
        try:
            # The turning manoeuvre crosses 3 checkpoints at dt=.1 but 4 at
            # finer dt: its discrete event bonus is not a rate-invariance
            # oracle. Use a controlled braking approach for shaping; individual
            # progress/turn/acceleration terms are tested analytically in
            # test_shaping_reward_time.py (without a convergence allowance).
            if reward_mode == "shaping":
                options = {**LEAD_BRAKE_OPTIONS, "ego_speed": 3.0,
                           "lead_gap": 30.0, "brake_delay": 50.0}
                traj = _rollout(env, lambda _: np.array([0.0, 0.0, 1.0], dtype=np.float32),
                                30, options=options)
            else:
                traj = _rollout(env, scripted_action, 30)
            assert traj[-1]["t"] == pytest.approx(3.0)
            returns[(dt, k)] = sum(s["r"] for s in traj)
        finally:
            env.close()
    ref = returns[(0.1, 1)]
    assert ref > 0.5
    for key, value in returns.items():
        assert value == pytest.approx(ref, rel=0.2), (key, value, ref)


def test_reward_step_scale_is_exactly_one_at_the_reference_step():
    from CarlaBEV.envs.carlabev import build_reward_fn

    for mode in ("carl", "shaping"):
        fn = build_reward_fn(EnvConfig(render_mode="rgb_array", reward_mode=mode))
        assert fn.step_scale == 1.0
        fn_fine = build_reward_fn(
            EnvConfig(render_mode="rgb_array", reward_mode=mode, physics_dt=0.05)
        )
        assert fn_fine.step_scale == pytest.approx(0.5)


# --------------------------------------------------------------------------- termination
def test_terminal_sub_step_ends_the_control_step_early():
    """A collision in the middle of a control step terminates with sim_time at that sub-step."""
    crash = {
        **LEAD_BRAKE_OPTIONS,
        "lead_gap": 6.0,
        "lead_speed": 0.0,
        "brake_delay": 50.0,
        "ego_speed": 12.0,
    }
    fine = make_env(physics_dt=0.025, action_repeat=1)
    multi = make_env(physics_dt=0.025, action_repeat=4)
    try:
        fine_traj = _rollout(fine, lambda i: gas_only_action(i), 400, options=crash)
        multi_traj = _rollout(multi, lambda i: gas_only_action(i), 100, options=crash)
        assert fine_traj[-1]["done"] and multi_traj[-1]["done"]
        assert fine_traj[-1]["t"] == multi_traj[-1]["t"]  # same physics step
        assert fine_traj[-1]["steps"] == multi_traj[-1]["steps"]
        ep = multi_traj[-1]["info"]["episode_info"]
        assert ep["termination"] == "collision"
        assert ep["physics_steps"] == multi_traj[-1]["steps"]
        assert ep["decisions"] == ep["length"] == len(multi_traj)
        assert ep["control_dt_s"] == pytest.approx(0.1) and ep["action_repeat"] == 4
        # The terminal control step sums its sub-steps up to and including the terminal one.
        first = 4 * (len(multi_traj) - 1)
        assert multi_traj[-1]["r"] == pytest.approx(
            sum(s["r"] for s in fine_traj[first:]), abs=1e-9
        )
        assert fine_traj[-1]["r"] == -1.0
    finally:
        fine.close()
        multi.close()
