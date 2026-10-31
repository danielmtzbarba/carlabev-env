"""BEHAVIOR / speed-loop feed-forward (Phase 1.2, finding S-01).

Expected: a behaviour-driven target-speed ramp is realised (deceleration close to
the requested rate, bounded by the actor's own limits) instead of lagging by
rate/Kp; constant targets and capped targets are unaffected.
"""

from __future__ import annotations

import math

from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior

from physics_time_helpers import (
    DOCUMENTED_DT_S,
    LEAD_BRAKE_OPTIONS,
    brake_only_action,
    make_env,
    make_vehicle,
)


def _realised_decel(brake_strength: float) -> float:
    env = make_env()
    try:
        env.reset(
            seed=3,
            options={
                **LEAD_BRAKE_OPTIONS,
                "ego_speed": 4.0,
                "lead_gap": 20.0,
                "brake_delay": 1.0,
                "brake_strength": brake_strength,
            },
        )
        lead = env.map.actor_manager.actors["vehicle"][0]
        speeds, times = [], []
        for k in range(40):
            env.step(brake_only_action(k))
            speeds.append(lead._controller.v_mps)
            times.append(env.sim_time_s)
        i0 = next(i for i, t in enumerate(times) if t >= 1.0)
        i1 = next(i for i, t in enumerate(times) if t >= 2.5)
        return (speeds[i0] - speeds[i1]) / (times[i1] - times[i0])
    finally:
        env.close()


def test_requested_deceleration_is_realised_within_vehicle_limits():
    realised = _realised_decel(2.0)
    assert 1.8 <= realised <= 2.4, realised  # 2 m/s^2 request, resistance adds ~0.2
    realised = _realised_decel(4.0)
    # Request exceeds max_decel (3.5): bounded by the vehicle limit, not by loop lag.
    assert realised >= 0.9 * Vehicle.traffic_max_decel_mps2, realised


def test_constant_target_has_no_feed_forward():
    vehicle = make_vehicle(10.0)
    vehicle.step(t=DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    vehicle.step(t=2 * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert vehicle._controller._target_accel_ff_mps2 == 0.0


def test_ramp_feed_forward_equals_rate_and_is_clipped():
    vehicle = Vehicle(
        map_size=128,
        routeX=list(range(0, 400, 10)),
        routeY=[0] * 40,
        target_speed=10.0,
        behavior=LeadBrakeBehavior(start_brake_t=0.0, dec_rate=2.0),
    )
    vehicle.reset()
    vehicle.step(
        t=DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S
    )  # first step: no previous target
    vehicle.step(t=2 * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert math.isclose(vehicle._controller._target_accel_ff_mps2, -2.0, rel_tol=1e-9)
    vehicle = Vehicle(
        map_size=128,
        routeX=list(range(0, 400, 10)),
        routeY=[0] * 40,
        target_speed=10.0,
        behavior=LeadBrakeBehavior(start_brake_t=0.0, dec_rate=9.0),
    )
    vehicle.reset()
    vehicle.step(t=DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    vehicle.step(t=2 * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert math.isclose(
        vehicle._controller._target_accel_ff_mps2,
        -Vehicle.traffic_max_decel_mps2,
        rel_tol=1e-9,
    )


def test_startup_ramp_cap_receives_no_positive_feed_forward():
    vehicle = make_vehicle(10.0)
    vehicle.set_initial_speed_mps(0.0)
    vehicle.set_startup_ramp_duration(1.0)
    vehicle.reset()
    for k in range(5):
        vehicle.step(t=(k + 1) * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert vehicle._controller._target_accel_ff_mps2 <= 0.0
