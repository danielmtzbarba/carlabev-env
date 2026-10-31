"""CLOCK / dt convergence: the physics step is a numerical parameter, not a behaviour.

Expected: running the same scenario with the same *time-based* open-loop policy at
dt = 0.1 s and dt = 0.05 s gives trajectories that agree at equal simulated time
within a first-order (Euler) tolerance; scenario timers fire at the same simulated
time; halving dt halves the discretisation error (first-order convergence).

This is what distinguishes an explicit physical clock from a renamed constant.
"""

from __future__ import annotations

import math

import numpy as np

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.envs.geometry import speed_mps_to_surface
from CarlaBEV.src.actors.hero import ContinuousAgent
from CarlaBEV.src.sim_time import TimeConfig
from CarlaBEV.src.units import EgoPhysics, UnitContract

from physics_time_helpers import JAYWALK_OPTIONS, LEAD_BRAKE_OPTIONS

LONG_ROUTE = ([0.0, 200.0, 400.0, 600.0, 800.0], [0.0, 0.0, 0.0, 0.0, 0.0])


def _policy_at_time(t_s: float) -> np.ndarray:
    """Piecewise-constant [gas, steer, brake] defined in simulated time."""
    if t_s < 2.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if t_s < 3.5:
        return np.array([0.6, 0.5, 0.0], dtype=np.float32)
    if t_s < 5.0:
        return np.array([0.6, -0.5, 0.0], dtype=np.float32)
    if t_s < 6.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _ego_rollout(dt: float, horizon_s: float = 8.0) -> dict[float, tuple]:
    units = UnitContract(time=TimeConfig(dt))
    ego = ContinuousAgent(
        route=LONG_ROUTE,
        window_size=128,
        target_speed=speed_mps_to_surface(20.0),
        initial_speed=speed_mps_to_surface(3.0),
        units=units,
        # set_route applies a +/-1 px start jitter; seed it so runs start identically.
        np_rng=np.random.default_rng(0),
    )
    out = {}
    for k in range(int(round(horizon_s / dt))):
        ego.step(_policy_at_time(k * dt))
        out[round((k + 1) * dt, 6)] = (ego.x_m, ego.y_m, ego.yaw, ego.v_mps)
    return out


def _max_errors(coarse: dict, fine: dict) -> tuple[float, float]:
    pos, vel = 0.0, 0.0
    for t, (x, y, _, v) in coarse.items():
        if t in fine:
            xf, yf, _, vf = fine[t]
            pos = max(pos, math.hypot(x - xf, y - yf))
            vel = max(vel, abs(v - vf))
    return pos, vel


def test_accel_filter_gain_is_time_constant_based():
    """F-03 / decision 4: alpha = dt / (tau + dt); legacy alpha = 0.5 at dt = 0.1 s."""
    ego = EgoPhysics()
    assert ego.accel_response_tau_s == 0.1
    assert math.isclose(ego.accel_response_alpha(0.1), 0.5, rel_tol=1e-12)
    assert math.isclose(ego.accel_response_alpha(0.05), 1.0 / 3.0, rel_tol=1e-12)


def test_ego_trajectory_converges_with_dt():
    coarse = _ego_rollout(0.1)
    fine = _ego_rollout(0.05)
    pos_err, vel_err = _max_errors(coarse, fine)
    assert vel_err < 0.1, f"speed at equal sim time differs by {vel_err:.3f} m/s"
    assert pos_err < 1.0, f"position at equal sim time differs by {pos_err:.3f} m"
    # The ego did travel far enough for the comparison to be meaningful.
    x_end = coarse[max(coarse)][0]
    assert x_end > 50.0


def test_ego_discretisation_error_is_first_order():
    """First-order Euler: error ~ c*dt. Against a dt=0.025 reference the expected
    ratio err(0.1)/err(0.05) is (0.1-0.025)/(0.05-0.025) = 3."""
    ref = _ego_rollout(0.025)
    err_coarse, _ = _max_errors(_ego_rollout(0.1), ref)
    err_fine, _ = _max_errors(_ego_rollout(0.05), ref)
    assert err_fine < err_coarse, (err_coarse, err_fine)
    ratio = err_coarse / max(err_fine, 1e-9)
    assert 2.0 < ratio < 4.0, (
        f"convergence ratio {ratio:.2f} (expected ~3 for first order)"
    )


def test_ego_speed_response_is_independent_of_dt():
    """The acceleration filter must respond on the same simulated-time scale for any dt."""
    coarse = _ego_rollout(0.1, horizon_s=2.0)
    fine = _ego_rollout(0.05, horizon_s=2.0)
    for t in (0.5, 1.0, 2.0):
        assert math.isclose(coarse[t][3], fine[t][3], abs_tol=0.05), (
            t,
            coarse[t][3],
            fine[t][3],
        )


# ---------------------------------------------------------------------------
# Full environment: NPCs, scenario timers, behaviours
# ---------------------------------------------------------------------------
def _env(dt: float) -> CarlaBEV:
    return CarlaBEV(
        EnvConfig(
            render_mode="rgb_array",
            obs_mode="bev_rgb",
            action_mode="continuous",
            physics_dt=dt,
            scene_library_enabled=False,
        )
    )


def _scenario_rollout(dt: float, options: dict, horizon_s: float, *, ego_action=None):
    env = _env(dt)
    try:
        env.reset(seed=3, options=dict(options))
        npcs = (
            env.map.actor_manager.actors["vehicle"]
            + env.map.actor_manager.actors["pedestrian"]
        )
        record = {}
        for k in range(int(round(horizon_s / dt))):
            action = (
                np.array([0.0, 0.0, 1.0], dtype=np.float32)
                if ego_action is None
                else ego_action(k * dt)
            )
            _, _, terminated, truncated, _ = env.step(action)
            t = round(env.sim_time_s, 6)
            record[t] = {
                "hero": (env.map.hero.x_m, env.map.hero.y_m, env.map.hero.v_mps),
                "npcs": [
                    (
                        a._controller.x_m,
                        a._controller.y_m,
                        a._controller.v_mps,
                        a.behavior_state,
                        bool(getattr(a.behavior, "braking", False)),
                    )
                    for a in npcs
                ],
            }
            if terminated or truncated:
                break
        return record
    finally:
        env.close()


def _npc_errors(coarse, fine, idx=0):
    v_err = max(
        abs(coarse[t]["npcs"][idx][2] - fine[t]["npcs"][idx][2])
        for t in coarse
        if t in fine
    )
    pos_err = max(
        math.hypot(
            coarse[t]["npcs"][idx][0] - fine[t]["npcs"][idx][0],
            coarse[t]["npcs"][idx][1] - fine[t]["npcs"][idx][1],
        )
        for t in coarse
        if t in fine
    )
    return v_err, pos_err


def test_lead_brake_trigger_and_speed_converge_with_dt():
    options = {
        **LEAD_BRAKE_OPTIONS,
        "ego_speed": 4.0,
        "lead_gap": 20.0,
        "brake_delay": 1.0,
    }
    coarse = _scenario_rollout(0.1, options, 4.0)
    fine = _scenario_rollout(0.05, options, 4.0)

    def first_brake_time(rec):
        return min(t for t, r in rec.items() if r["npcs"][0][4])

    # Timer-triggered event: exact at both resolutions.
    assert math.isclose(first_brake_time(coarse), 1.0, abs_tol=1e-9)
    assert math.isclose(first_brake_time(fine), 1.0, abs_tol=1e-9)

    v_err, pos_err = _npc_errors(coarse, fine)
    # The braking target is a ramp applied with end-of-step values, so the coarse
    # run's forcing leads the fine run's by rate*dt/2 = 0.2 m/s: an O(dt) effect.
    assert v_err < 0.3, f"lead speed differs by {v_err:.3f} m/s at equal sim time"
    assert pos_err < 0.5, f"lead position differs by {pos_err:.3f} m at equal sim time"
    # The lead really braked in both runs.
    assert coarse[4.0]["npcs"][0][2] < 0.5 * coarse[0.5]["npcs"][0][2]


def test_lead_brake_speed_error_is_first_order_in_dt():
    options = {
        **LEAD_BRAKE_OPTIONS,
        "ego_speed": 4.0,
        "lead_gap": 20.0,
        "brake_delay": 1.0,
    }
    ref = _scenario_rollout(0.025, options, 4.0)
    err_coarse, _ = _npc_errors(_scenario_rollout(0.1, options, 4.0), ref)
    err_fine, _ = _npc_errors(_scenario_rollout(0.05, options, 4.0), ref)
    assert err_fine < err_coarse
    ratio = err_coarse / max(err_fine, 1e-9)
    assert 2.0 < ratio < 4.0, (
        f"convergence ratio {ratio:.2f} (expected ~3 for first order)"
    )


def test_jaywalk_crossing_converges_with_dt():
    """Continuous motion after a timer trigger: tight agreement at equal sim time."""
    options = {**JAYWALK_OPTIONS, "level": 1, "ego_speed": 1.0, "cross_delay": 1.0}
    coarse = _scenario_rollout(0.1, options, 6.0)
    fine = _scenario_rollout(0.05, options, 6.0)
    v_err, pos_err = _npc_errors(coarse, fine)
    assert v_err < 0.15, f"pedestrian speed differs by {v_err:.3f} m/s"
    assert pos_err < 0.3, f"pedestrian position differs by {pos_err:.3f} m"
    t_end = max(t for t in coarse if t in fine)
    assert coarse[t_end]["npcs"][0][3] == fine[t_end]["npcs"][0][3] == "cleared"


def test_jaywalk_fsm_transitions_happen_at_the_same_sim_time():
    options = {
        **JAYWALK_OPTIONS,
        "level": 3,
        "ego_speed": 1.0,
        "cross_delay": 1.0,
        "yield_duration": 1.0,
    }
    coarse = _scenario_rollout(0.1, options, 6.0)
    fine = _scenario_rollout(0.05, options, 6.0)

    def transitions(rec):
        out, last = {}, None
        for t in sorted(rec):
            state = rec[t]["npcs"][0][3]
            if state != last:
                out[state] = t
                last = state
        return out

    tc, tf = transitions(coarse), transitions(fine)
    # Timer-triggered transition: exact at both resolutions.
    assert tc.get("entering") == 1.0 and tf.get("entering") == 1.0, (tc, tf)
    # Geometry-triggered transitions (reaching a waypoint index) may shift by up
    # to one coarse step, and the yield timer counts from that entry time.
    for state in ("yielding", "retreating"):
        assert state in tc and state in tf, (tc, tf)
        assert abs(tc[state] - tf[state]) <= 0.2 + 1e-9, (state, tc[state], tf[state])
    # The retreat route is rebuilt from the position at the (jittered) event, so
    # its completion is the most event-sensitive transition.
    assert "retreated" in tc and "retreated" in tf, (tc, tf)
    assert abs(tc["retreated"] - tf["retreated"]) <= 0.5 + 1e-9, (
        tc["retreated"],
        tf["retreated"],
    )
    # Event-time jitter produces a transient position difference while the two
    # runs are in different states; both settle at the same rest position.
    t_end = max(t for t in coarse if t in fine)
    rest_err = math.hypot(
        coarse[t_end]["npcs"][0][0] - fine[t_end]["npcs"][0][0],
        coarse[t_end]["npcs"][0][1] - fine[t_end]["npcs"][0][1],
    )
    assert coarse[t_end]["npcs"][0][2] == 0.0 and fine[t_end]["npcs"][0][2] == 0.0
    # 'retreated' fires on a geometric criterion over a short rebuilt route, so the
    # rest point inherits the event jitter (documented FSM property, not a clock error).
    assert rest_err < 1.0, f"pedestrian rest position differs by {rest_err:.3f} m"


def test_env_sim_time_and_video_rate_follow_dt():
    env = _env(0.05)
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        for _ in range(20):
            env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        assert env.sim_time_s == 1.0
        assert env.map.step_index == 20
        assert env.metadata["render_fps"] == 20
        assert (
            env.map.hero.dt
            == 0.05
            == env.map.actor_manager.actors["vehicle"][0]._controller.dt
        )
    finally:
        env.close()
