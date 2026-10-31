"""NPC MOTION / route end: an NPC's reported velocity matches its realised motion.

Expected: motion is integrated every step (displacement == v_{k} * dt), the actor
decelerates physically (<= max_decel) towards the route end and, once at rest,
reports v == 0 with a frozen position.
"""

from __future__ import annotations

import math

from CarlaBEV.src.actors.vehicle import Vehicle

from physics_time_helpers import DOCUMENTED_DT_S


def _run_to_rest(vehicle, max_steps=200):
    ctrl = vehicle._controller
    trace = []
    for k in range(max_steps):
        x0, y0, v0 = ctrl.x_m, ctrl.y_m, ctrl.v_mps
        finished = vehicle.step(t=(k + 1) * DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
        trace.append(
            {
                "k": k,
                "x0": x0,
                "y0": y0,
                "v0": v0,
                "x": ctrl.x_m,
                "y": ctrl.y_m,
                "v": ctrl.v_mps,
            }
        )
        if finished:
            return trace, True
    return trace, False


def test_npc_comes_to_physical_rest_at_route_end():
    """N-01 (resolved): no frozen actor with phantom velocity."""
    vehicle = Vehicle(
        map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0], target_speed=10.0
    )
    vehicle.reset()
    trace, finished = _run_to_rest(vehicle)
    assert finished, "vehicle never reported a finished stop"
    assert vehicle._controller.v_mps == 0.0
    # Every step's displacement is explained by the speed at the start of the step.
    for row in trace[:-1]:
        moved = math.hypot(row["x"] - row["x0"], row["y"] - row["y0"])
        assert math.isclose(
            moved, row["v0"] * DOCUMENTED_DT_S, rel_tol=1e-9, abs_tol=1e-12
        ), row
    # Deceleration never exceeds the configured limit plus passive resistance.
    max_decel = vehicle.traffic_max_decel_mps2 + 0.05 + 0.002 * 10.0**2
    for prev, cur in zip(trace, trace[1:]):
        assert (prev["v"] - cur["v"]) / DOCUMENTED_DT_S <= max_decel + 1e-6


def test_npc_stops_at_route_end_not_far_beyond_it():
    """The stopping profile brings the actor to rest near the last waypoint."""
    route_x = list(range(0, 200, 10))
    vehicle = Vehicle(
        map_size=128, routeX=route_x, routeY=[0] * len(route_x), target_speed=10.0
    )
    vehicle.reset()
    trace, finished = _run_to_rest(vehicle, max_steps=400)
    assert finished
    end_x_m = vehicle._controller.cx_m[-1]
    overshoot_m = vehicle._controller.x_m - end_x_m
    assert -3.0 <= overshoot_m <= 3.0, f"stopped {overshoot_m:.2f} m from route end"


def test_frozen_position_implies_zero_reported_speed():
    vehicle = Vehicle(
        map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0], target_speed=10.0
    )
    vehicle.reset()
    trace, _ = _run_to_rest(vehicle)
    for prev, cur in zip(trace, trace[1:]):
        if (prev["x"], prev["y"]) == (cur["x"], cur["y"]):
            assert cur["v"] == 0.0
