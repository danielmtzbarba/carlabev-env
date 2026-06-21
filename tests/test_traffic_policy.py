import math

from CarlaBEV.src.actors.traffic_policy import (
    compute_safe_target_speed_mps,
    find_lead_actor,
)
from CarlaBEV.src.actors.vehicle import Vehicle


def _make_vehicle(*, x_m=0.0, y_m=0.0, yaw=0.0, speed_mps=0.0, cruise_speed_mps=8.0):
    route_x = list(range(0, 200, 10))
    route_y = [0] * len(route_x)
    vehicle = Vehicle(
        map_size=128,
        routeX=route_x,
        routeY=route_y,
        target_speed=cruise_speed_mps,
    )
    vehicle.set_initial_speed_mps(speed_mps)
    vehicle.set_startup_ramp_duration(0.0)
    vehicle.reset()
    vehicle._controller.x_m = float(x_m)
    vehicle._controller.y_m = float(y_m)
    vehicle._controller.yaw = float(yaw)
    vehicle._controller.v_mps = float(speed_mps)
    vehicle._controller._speed_limit_mps = float(cruise_speed_mps)
    return vehicle


def test_find_lead_actor_ignores_lateral_neighbors():
    actor = _make_vehicle(speed_mps=8.0)
    side_actor = _make_vehicle(x_m=5.0, y_m=4.0, speed_mps=4.0)

    lead_actor, lead_gap_m = find_lead_actor(actor, [side_actor])

    assert lead_actor is None
    assert lead_gap_m is None


def test_compute_safe_target_speed_reduces_speed_for_slower_lead_actor():
    actor = _make_vehicle(speed_mps=8.0, cruise_speed_mps=8.0)
    lead_actor = _make_vehicle(x_m=10.0, speed_mps=2.0, cruise_speed_mps=2.0)

    safe_target_speed_mps, debug = compute_safe_target_speed_mps(
        actor,
        [lead_actor],
        nominal_target_speed_mps=8.0,
    )

    assert safe_target_speed_mps < 8.0
    assert math.isclose(safe_target_speed_mps, 0.0, rel_tol=1e-6)
    assert debug["traffic_braking_active"] is True
    assert debug["lead_gap_m"] is not None


def test_traffic_safe_target_does_not_change_controller_speed_limit():
    actor = _make_vehicle(speed_mps=8.0, cruise_speed_mps=8.0)
    lead_actor = _make_vehicle(x_m=7.0, speed_mps=0.0, cruise_speed_mps=0.0)

    actor.step(t=0.1, dt=0.1, other_actors=[lead_actor])

    assert actor.last_traffic_policy["traffic_braking_active"] is True
    assert actor._controller._target_speed_mps < actor.cruise_speed_mps
    assert math.isclose(actor._controller._speed_limit_mps, actor.cruise_speed_mps, rel_tol=1e-6)
