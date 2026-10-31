"""NPC MOTION / scenario actors: scripted NPCs can physically perform their behaviour.

Expected: the lead_brake lead decelerates after brake_delay; the jaywalk pedestrian
waits at rest, then actually crosses the road.
"""

from __future__ import annotations

from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle

from physics_time_helpers import (
    JAYWALK_OPTIONS,
    LEAD_BRAKE_OPTIONS,
    brake_only_action,
    make_env,
)


def test_lead_brake_lead_vehicle_physically_decelerates():
    """N-02 (resolved): lead route sized from speed x horizon."""
    env = make_env()
    try:
        env.reset(
            seed=3,
            options={
                **LEAD_BRAKE_OPTIONS,
                "ego_speed": 4.0,
                "lead_gap": 20.0,
                "brake_delay": 1.0,
            },
        )
        lead = env.map.actor_manager.actors["vehicle"][0]
        speeds = []
        for k in range(40):
            env.step(brake_only_action(k))
            speeds.append(lead._controller.v_mps)
        assert speeds[-1] < 0.5 * speeds[5], (
            f"lead speed trace: {[round(s, 2) for s in speeds[::5]]}"
        )
    finally:
        env.close()


def test_lead_brake_lead_route_outlasts_the_braking_manoeuvre():
    env = make_env()
    try:
        env.reset(
            seed=3,
            options={
                **LEAD_BRAKE_OPTIONS,
                "lead_speed": 12.0,
                "brake_delay": 2.5,
                "brake_strength": 4.0,
            },
        )
        lead = env.map.actor_manager.actors["vehicle"][0]
        route_m = float(lead._controller.cs_m[-1])
        needed_m = 12.0 * 2.5 + 12.0**2 / (2 * 4.0)
        assert route_m >= needed_m, (
            f"route {route_m:.1f} m < manoeuvre {needed_m:.1f} m"
        )
    finally:
        env.close()


def test_pedestrian_uses_walker_kinematics():
    """N-03 (resolved): pedestrians are not 2.9 m-wheelbase cars."""
    ped = Pedestrian(map_size=128, routeX=[0, 5, 10], routeY=[0, 0, 0])
    ped.reset()
    assert ped.wheelbase_m < 1.0
    assert ped._controller.L == ped.wheelbase_m
    assert Vehicle(map_size=128, routeX=[0, 5, 10], routeY=[0, 0, 0]).wheelbase_m == 2.9


def test_jaywalk_pedestrian_waits_at_rest_then_crosses():
    """N-03/N-04 (resolved): waiting pedestrian spawns at rest and completes the crossing."""
    env = make_env()
    try:
        env.reset(
            seed=3,
            options={
                **JAYWALK_OPTIONS,
                "level": 1,
                "cross_delay": 1.0,
                "ego_speed": 1.0,
            },
        )
        ped = env.map.actor_manager.actors["pedestrian"][0]
        ctrl = ped._controller
        x_start = ctrl.x_m
        assert ctrl.v_mps == 0.0, "waiting pedestrian must spawn at rest"
        for k in range(9):
            env.step(brake_only_action(k))
        assert ped.behavior_state == "waiting"
        assert ctrl.x_m == x_start, "pedestrian moved while waiting"
        for k in range(9, 80):
            env.step(brake_only_action(k))
            if ped.behavior_state == "cleared":
                break
        assert ped.behavior_state == "cleared", ped.behavior_state
        crossed_m = abs(ctrl.x_m - x_start)
        assert crossed_m > 2.5, f"pedestrian only moved {crossed_m:.2f} m"
    finally:
        env.close()
