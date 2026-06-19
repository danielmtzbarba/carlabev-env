import math

from CarlaBEV.envs.geometry import speed_mps_to_surface
from CarlaBEV.src.actors.hero import DiscreteAgent


def _make_agent(initial_speed_mps: float = 0.0) -> DiscreteAgent:
    route = ([0.0, 10.0, 20.0], [0.0, 0.0, 0.0])
    return DiscreteAgent(
        route=route,
        window_size=128,
        initial_speed=speed_mps_to_surface(initial_speed_mps),
        target_speed=speed_mps_to_surface(max(initial_speed_mps, 1.0)),
    )


def test_speed_helpers_report_surface_and_metric_units():
    agent = _make_agent(initial_speed_mps=12.0)

    assert math.isclose(agent.speed_surface, speed_mps_to_surface(12.0), rel_tol=1e-6)
    assert math.isclose(agent.speed_mps, 12.0, rel_tol=1e-6)


def test_steering_authority_uses_metric_speed_and_stays_above_floor_at_city_speed():
    agent = _make_agent(initial_speed_mps=5.0)

    delta = agent.steering(1.0)

    assert math.degrees(delta) > agent.min_action_steer_deg
    assert math.isclose(math.degrees(delta), 12.0, rel_tol=1e-6)


def test_steering_floor_occurs_near_real_driving_speed():
    agent = _make_agent(initial_speed_mps=12.5)

    delta = agent.steering(1.0)

    assert math.isclose(math.degrees(delta), agent.min_action_steer_deg, rel_tol=1e-6)


def test_brake_scaling_uses_metric_speed():
    slow_agent = _make_agent(initial_speed_mps=2.0)
    fast_agent = _make_agent(initial_speed_mps=8.0)

    slow_brake = slow_agent.brake(1.0)
    fast_brake = fast_agent.brake(1.0)

    assert slow_brake < fast_brake
    assert math.isclose(slow_brake, slow_agent.max_brake_mps2 * 0.4, rel_tol=1e-6)
    assert math.isclose(fast_brake, fast_agent.max_brake_mps2, rel_tol=1e-6)
