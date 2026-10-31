"""CLOCK / timers: scenario behaviours and startup ramps count in physical seconds.

Expected (end-of-step convention): during step k the observed time is (k+1)*dt;
a timer of T seconds fires at the first step whose time is >= T; rates given in
/s are applied as rate*dt per step.
"""

from __future__ import annotations

import math

from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior
from CarlaBEV.src.actors.vehicle import Vehicle

from physics_time_helpers import (
    DOCUMENTED_DT_S,
    LEAD_BRAKE_OPTIONS,
    brake_only_action,
    make_env,
)


def _first_braking_step(brake_delay_s: float, num_steps: int = 40) -> tuple[int, float]:
    env = make_env()
    try:
        env.reset(
            seed=3,
            options={
                **LEAD_BRAKE_OPTIONS,
                "brake_delay": brake_delay_s,
                "ego_speed": 6.0,
            },
        )
        lead = env.map.actor_manager.actors["vehicle"][0]
        assert isinstance(lead.behavior, LeadBrakeBehavior)
        for k in range(num_steps):
            env.step(brake_only_action(k))
            if lead.behavior.braking:
                return k, float(env.map._t)
        raise AssertionError("lead vehicle never started braking")
    finally:
        env.close()


def test_lead_brake_timer_fires_within_one_step_of_requested_delay():
    step_idx, sim_t = _first_braking_step(1.0)
    assert 1.0 - 1e-9 <= sim_t <= 1.0 + DOCUMENTED_DT_S + 1e-9
    assert step_idx in (9, 10)


def test_lead_brake_timer_fires_at_first_step_reaching_delay():
    """C-03 (resolved): grid-aligned timers fire at the exact step."""
    step_idx, sim_t = _first_braking_step(1.0)
    assert step_idx == 9, f"fired at step {step_idx}, sim_t={sim_t!r}"


def test_lead_brake_target_speed_decreases_at_dec_rate_times_dt():
    env = make_env()
    try:
        env.reset(
            seed=3, options={**LEAD_BRAKE_OPTIONS, "brake_delay": 0.0, "ego_speed": 6.0}
        )
        lead = env.map.actor_manager.actors["vehicle"][0]
        v_target_0 = lead.target_speed_mps
        env.step(brake_only_action(0))
        expected = v_target_0 - LEAD_BRAKE_OPTIONS["brake_strength"] * DOCUMENTED_DT_S
        assert math.isclose(lead.target_speed_mps, expected, rel_tol=1e-9)
    finally:
        env.close()


def _ramped_vehicle(ramp_s: float = 1.0) -> Vehicle:
    vehicle = Vehicle(
        map_size=128, routeX=list(range(0, 400, 10)), routeY=[0] * 40, target_speed=10.0
    )
    vehicle.set_initial_speed_mps(0.0)
    vehicle.set_startup_ramp_duration(ramp_s)
    vehicle.reset()
    return vehicle


def test_startup_ramp_is_measured_in_seconds_within_one_step():
    vehicle = _ramped_vehicle(1.0)
    steps = int(round(1.0 / DOCUMENTED_DT_S))
    for k in range(steps - 1):
        vehicle.step(t=(k + 1) * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert vehicle._controller._target_speed_mps < vehicle.cruise_speed_mps - 1e-6
    vehicle.step(t=steps * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    vehicle.step(t=(steps + 1) * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert math.isclose(
        vehicle._controller._target_speed_mps, vehicle.cruise_speed_mps, rel_tol=1e-9
    )


def test_startup_ramp_completes_exactly_at_ramp_duration():
    """C-04 (resolved): ramps and behaviours share the end-of-step convention."""
    vehicle = _ramped_vehicle(1.0)
    steps = int(round(1.0 / DOCUMENTED_DT_S))
    for k in range(steps):
        vehicle.step(t=(k + 1) * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert math.isclose(
        vehicle._controller._target_speed_mps, vehicle.cruise_speed_mps, rel_tol=1e-9
    )


def test_jaywalk_cross_delay_fires_at_exact_step():
    """Behaviour timers derived from t (not accumulated): cross_delay=1.0 -> entering at step 9."""
    from CarlaBEV.src.actors.behavior.jaywalk import StopReturnBehavior
    from CarlaBEV.src.actors.pedestrian import Pedestrian

    ped = Pedestrian(
        map_size=128,
        routeX=[10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        routeY=[10.0] * 6,
        target_speed=1.5,
        behavior=StopReturnBehavior(start_delay=1.0, yield_duration=1.0),
    )
    ped.reset()
    states = []
    for k in range(12):
        ped.step(t=(k + 1) * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
        states.append(ped.behavior_state)
    assert states[8] == "waiting" and states[9] == "entering", states
