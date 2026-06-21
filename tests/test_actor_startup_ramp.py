import math

from CarlaBEV.src.actors.vehicle import Vehicle


def _make_vehicle(*, cruise_speed_mps=12.0):
    route_x = list(range(0, 200, 10))
    route_y = [0] * len(route_x)
    vehicle = Vehicle(
        map_size=128,
        routeX=route_x,
        routeY=route_y,
        target_speed=cruise_speed_mps,
    )
    vehicle.set_initial_speed_mps(0.0)
    vehicle.set_startup_ramp_duration(1.0)
    vehicle.reset()
    return vehicle


def test_vehicle_reset_uses_configured_initial_speed():
    vehicle = _make_vehicle()

    assert math.isclose(vehicle.initial_speed_mps, 0.0, abs_tol=1e-9)
    assert math.isclose(vehicle._controller.v_mps, 0.0, abs_tol=1e-9)


def test_vehicle_startup_ramp_caps_target_speed_until_ramp_finishes():
    vehicle = _make_vehicle(cruise_speed_mps=12.0)

    for step_idx in range(5):
        vehicle.step(t=(step_idx + 1) * 0.1, dt=0.1)

    assert 0.0 < vehicle._controller._target_speed_mps < vehicle.cruise_speed_mps

    for step_idx in range(10):
        vehicle.step(t=0.6 + (step_idx * 0.1), dt=0.1)

    assert math.isclose(
        vehicle._controller._target_speed_mps,
        vehicle.cruise_speed_mps,
        rel_tol=1e-6,
    )


def test_vehicle_controller_uses_softened_speed_gain_and_accel_limits():
    vehicle = _make_vehicle(cruise_speed_mps=12.0)

    assert math.isclose(vehicle._controller.speed_kp, 0.5, rel_tol=1e-9)
    assert math.isclose(vehicle._controller.max_accel_mps2, 2.5, rel_tol=1e-9)
    assert math.isclose(vehicle._controller.max_decel_mps2, 3.5, rel_tol=1e-9)

    vehicle._controller._target_speed_mps = 12.0
    vehicle._controller.v_mps = 0.0
    assert math.isclose(vehicle._controller.pid_control(), 2.5, rel_tol=1e-9)

    vehicle._controller._target_speed_mps = 0.0
    vehicle._controller.v_mps = 12.0
    assert math.isclose(vehicle._controller.pid_control(), -3.5, rel_tol=1e-9)
