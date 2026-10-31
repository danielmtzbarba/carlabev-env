"""BEHAVIOR / declarative framework (Phase 4.1): FSM behaviours on the scene clock.

Expected: behaviours are state machines whose timers derive from simulated
time (not accumulated), whose specs round-trip through JSON, and whose
triggers can observe the world (gap acceptance) rather than only time.
"""

from __future__ import annotations

import json

import pytest

from CarlaBEV.src.actors.behavior.framework import (
    AfterS,
    Always,
    Behavior,
    BehaviorContext,
    GapAccepted,
    SetTargetSpeed,
    State,
)
from CarlaBEV.src.actors.behavior.jaywalk import CrossBehavior, StopReturnBehavior
from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior
from CarlaBEV.src.actors.behavior.registry import (
    build_behavior,
    normalize_behavior_spec,
)
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.sim_time import SimClock, TimeConfig
from CarlaBEV.src.units import UnitContract


def _vehicle(behavior=None, dt=0.1, speed=10.0, route_y_px=0):
    units = UnitContract(time=TimeConfig(physics_dt_s=dt))
    route_x = list(range(0, 800, 10))
    v = Vehicle(
        128,
        routeX=route_x,
        routeY=[route_y_px] * len(route_x),
        target_speed=speed,
        behavior=behavior,
        units=units,
    )
    v.reset()
    return v


def _run(vehicle, seconds, dt):
    clock = SimClock(TimeConfig(physics_dt_s=dt))
    targets = []
    for _ in range(int(round(seconds / dt))):
        t = clock.tick()
        vehicle.step(t, dt)
        targets.append((round(t, 6), vehicle.behavior_state, vehicle.target_speed_mps))
    return targets


# --------------------------------------------------------------------------- spec round trip
def test_lead_brake_spec_round_trips_and_reproduces_the_same_targets():
    original = LeadBrakeBehavior(start_brake_t=1.0, dec_rate=4.0)
    spec = json.loads(json.dumps(original.to_spec()))
    rebuilt = Behavior.from_spec(spec)
    assert rebuilt.to_spec() == original.to_spec()

    a = _run(_vehicle(LeadBrakeBehavior(start_brake_t=1.0, dec_rate=4.0)), 3.0, 0.1)
    b = _run(_vehicle(rebuilt), 3.0, 0.1)
    assert a == b
    # End-of-step convention: the step ending at t = 1.0 s observes the trigger and enters "braking".
    assert a[8][:2] == (0.9, "cruise")
    assert a[9][:2] == (1.0, "braking")


def test_registry_accepts_inline_fsm_specs_and_gap_acceptance_param():
    spec = {
        "type": "fsm",
        "initial": "slow",
        "states": {
            "slow": {
                "speed_mps": 3.0,
                "transitions": [
                    {"trigger": {"type": "after_s", "seconds": 0.5}, "to": "fast"}
                ],
            },
            "fast": {"speed_mps": 9.0},
        },
    }
    behavior, normalized = build_behavior("vehicle", spec)
    assert normalized["type"] == "fsm"
    targets = _run(_vehicle(behavior), 1.0, 0.1)
    assert targets[0][2] == 3.0 and targets[-1][2] == 9.0
    assert [s for _, s, _ in targets][3:5] == [
        "slow",
        "fast",
    ]  # AfterS(0.5) fires in the step ending at t=0.5

    ped, normalized = build_behavior(
        "pedestrian", {"type": "cross", "params": {"gap_accept_s": 2.5}}
    )
    assert isinstance(ped, CrossBehavior) and ped.gap_accept_s == 2.5
    assert (
        normalize_behavior_spec("pedestrian", {"type": "cross"})["params"][
            "gap_accept_s"
        ]
        == 0.0
    )
    ped_off, _ = build_behavior(
        "pedestrian", {"type": "cross", "params": {"gap_accept_s": 0}}
    )
    assert ped_off.gap_accept_s is None


# --------------------------------------------------------------------------- machine semantics
def test_one_transition_per_step_and_entry_actions_run_on_entry():
    states = {
        "a": State("a", speed_mps=1.0, transitions=((Always(), "b"),)),
        "b": State(
            "b", on_enter=(SetTargetSpeed(2.0),), transitions=((Always(), "c"),)
        ),
        "c": State("c", speed_mps=3.0),
    }
    v = _vehicle(Behavior(states, "a"))
    assert v.behavior_state == "a" and v.target_speed_mps == 1.0
    out = _run(v, 0.3, 0.1)
    assert [s for _, s, _ in out] == ["b", "c", "c"]
    assert [round(x, 6) for _, _, x in out] == [2.0, 3.0, 3.0]


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.025])
def test_state_timers_derive_from_simulated_time(dt):
    states = {
        "wait": State("wait", speed_mps=0.0, transitions=((AfterS(1.0), "go"),)),
        "go": State("go", speed_mps=None),
    }
    out = _run(_vehicle(Behavior(states, "wait"), dt=dt), 2.0, dt)
    first_go = next(t for t, s, _ in out if s == "go")
    assert first_go == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- perception triggers
def test_gap_accepted_rejects_an_approaching_vehicle_and_accepts_once_it_passed():
    units = UnitContract()
    ped = Pedestrian(
        128, routeX=[100, 100, 100], routeY=[50, 45, 40], target_speed=1.5, units=units
    )
    ped.reset()
    car = _vehicle(speed=10.0)  # drives along +x at y = 0 px
    ped_x_m, ped_y_m = units.metric.surface_to_meters((100, 50))
    # Put the car 15 m before the pedestrian's x on a path 1 m off its lateral line.
    car._controller.x_m = ped_x_m - 15.0
    car._controller.y_m = ped_y_m + 1.0
    car._controller.yaw = 0.0
    car._controller._v_mps = 10.0

    trigger = GapAccepted(min_ttc_s=3.0, lateral_margin_m=3.0)
    ctx = BehaviorContext(ped, 1.0, 0.1, other_actors=[car])
    assert not trigger.fires(ctx, 0.0)  # 15 m / 10 m/s = 1.5 s < 3 s

    car._controller._v_mps = 2.0
    assert trigger.fires(ctx, 0.0)  # 7.5 s > 3 s

    car._controller._v_mps = 10.0
    car._controller.x_m = ped_x_m + 2.0  # already past
    assert trigger.fires(ctx, 0.0)

    car._controller.x_m = ped_x_m - 15.0
    car._controller.y_m = ped_y_m + 5.0  # not on its path
    assert trigger.fires(ctx, 0.0)


def test_jaywalk_gap_acceptance_delays_the_crossing_until_the_gap_opens():
    units = UnitContract()

    def scene(gap_accept_s):
        ped = Pedestrian(
            128,
            routeX=[100] * 8,
            routeY=list(range(60, 36, -3)),
            target_speed=1.5,
            behavior=StopReturnBehavior(
                start_delay=0.5, yield_duration=1.0, gap_accept_s=gap_accept_s
            ),
            units=units,
        )
        ped.reset()
        car = _vehicle(
            speed=10.0, route_y_px=57
        )  # lane ~1 m beside the curb point, along +x
        ped_x_m = units.metric.surface_to_meters((100, 60))[0]
        car._controller.x_m = ped_x_m - 20.0
        car._controller._v_mps = 10.0
        return ped, car

    clock = TimeConfig(physics_dt_s=0.1)
    results = {}
    for gap in (None, 2.5):
        ped, car = scene(gap)
        sim = SimClock(clock)
        entered_at = None
        for _ in range(60):
            t = sim.tick()
            car.step(t, 0.1, other_actors=[ped])
            ped.step(t, 0.1, other_actors=[car])
            if entered_at is None and ped.behavior_state == "entering":
                entered_at = t
        results[gap] = entered_at
    assert results[None] == pytest.approx(0.5, abs=1e-9)  # jaywalks on the timer
    # With gap acceptance the pedestrian waits until the car (20 m at 10 m/s) has passed.
    assert results[2.5] is not None and results[2.5] >= 1.5
