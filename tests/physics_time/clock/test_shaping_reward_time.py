"""Identical physical motion must not change reward when physics is refined."""

import math

import pytest

from CarlaBEV.semantics import SemanticClass
from CarlaBEV.src.deeprl.reward import RewardFn


WEIGHTS = dict.fromkeys(
    (
        "k_lat_quadratic",
        "k_progress",
        "k_flow",
        "k_align_bonus",
        "k_reverse",
        "k_ttc",
        "alive_bias",
        "k_smooth",
        "k_steer_smooth",
        "k_steer_jerk",
        "k_route_dev",
    ),
    0.0,
)


def motion_info(t, dt, *, speed_change=0.0, turn_rate=0.0, yaw_accel=0.0):
    def state(s):
        return [
            5.0 * s + 0.5 * speed_change * s * s,
            0.0,
            turn_rate * s + 0.5 * yaw_accel * s * s,
            5.0 + speed_change * s,
        ]

    current, previous = state(t), state(t - dt)
    return {
        "hero": {
            "state_m": current,
            "last_state_m": previous,
            "set_point_m": [current[0] + 1.0, 0.0, current[2]],
            "next_wps_m": ([0.0, 100.0], [0.0, 0.0], [0.0, 0.0]),
            "dist2wp_m": 0.0,
        },
        "scene": {
            "dist2goal_m": 1000.0 - current[0],
            "dist2goal_t_1_m": 1000.0 - previous[0],
        },
        "collision": {
            "tile_class": int(SemanticClass.DRIVABLE),
            "collided": None,
            "actor_id": None,
            "actors_state": [],
        },
    }


def contribution(dt, weight, value, **motion):
    reward = RewardFn(dt=dt, **{**WEIGHTS, weight: value})
    baseline = RewardFn(dt=dt, **WEIGHTS)
    total = 0.0
    # Prime both histories with the preceding interval; only the next second
    # is measured, avoiding an artificial instantaneous turn onset at t=0.
    reward.step(motion_info(0.0, dt, **motion))
    baseline.step(motion_info(0.0, dt, **motion))
    for k in range(1, round(1.0 / dt) + 1):
        info = motion_info(k * dt, dt, **motion)
        total += (
            reward.step(info)[0] - baseline.step(motion_info(k * dt, dt, **motion))[0]
        )
    return total


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.025])
def test_five_metres_progress_has_the_same_reward_at_every_timestep(dt):
    # At reference dt: ten increments of 0.5 m, k=.06, tanh gain=1.2.
    expected = 10.0 * math.tanh(0.036)
    assert contribution(dt, "k_progress", 0.06) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.025])
@pytest.mark.parametrize(
    "weight,motion,reference_increment",
    [
        ("k_steer_smooth", {"turn_rate": 0.2}, 0.02),
        ("k_smooth", {"speed_change": 2.0}, 0.2),
        ("k_smooth", {"speed_change": -2.0}, 0.2),
        ("k_steer_jerk", {"yaw_accel": 0.4}, 0.004),
    ],
)
def test_motion_difference_penalties_use_reference_interval(
    dt, weight, motion, reference_increment
):
    expected = -10.0 * math.tanh(1.2 * 0.1 * reference_increment)
    assert contribution(dt, weight, 0.1, **motion) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("dt", [0.1, 0.05, 0.025])
@pytest.mark.parametrize(
    "kind,actor_id,expected",
    [
        ("target", "goal", 18.0),
        ("target", "checkpoint", 0.7),
        ("vehicle", 0, -12.0),
        ("pedestrian", 0, -20.0),
    ],
)
def test_event_and_terminal_rewards_are_not_time_scaled(dt, kind, actor_id, expected):
    reward = RewardFn(dt=dt)
    info = motion_info(dt, dt)
    info["collision"].update(collided=kind, actor_id=actor_id)
    result, terminated, _, _ = reward.step(info)
    assert result == expected
    assert terminated == (actor_id != "checkpoint")
