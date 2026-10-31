"""UNITS / steering: the speed-dependent steering authority matches the tested curve.

Expected: delta_deg = clamp(18 / (1 + 0.10 |v|), 8, 18); zero when stationary.
(Docs currently state k = 0.35 -> finding D-01, documentation fix.)
"""

from __future__ import annotations

import math

import pytest

from physics_time_helpers import make_ego


@pytest.mark.parametrize(
    "speed_mps,expected_deg", [(1.0, 18.0 / 1.1), (5.0, 12.0), (12.5, 8.0), (20.0, 8.0)]
)
def test_steering_authority_curve(speed_mps, expected_deg):
    ego = make_ego(speed_mps)
    assert math.isclose(math.degrees(ego.steering(1.0)), expected_deg, rel_tol=1e-9)


def test_steering_is_zero_when_stationary():
    assert make_ego(0.0).steering(1.0) == 0.0


def test_steering_parameters_are_explicit_attributes():
    ego = make_ego(5.0)
    assert ego.min_action_steer_deg == 8.0
    assert ego.max_action_steer_deg == 18.0
    assert ego.steer_speed_scale_mps == 0.10
