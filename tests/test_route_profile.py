from __future__ import annotations

from CarlaBEV.config.reset import RandomNavigationReset, build_random_navigation_options
from CarlaBEV.src.control.route_profile import (
    compute_route_profile_metrics,
    matches_route_profile,
)


def test_build_random_navigation_options_includes_route_profile_controls():
    request = RandomNavigationReset(
        difficulty_id="rt_medium_v1",
        route_profile="single_left",
        route_profile_mix={"mostly_straight": 0.5, "single_left": 0.5},
        min_turns=1,
        max_turns=2,
        intersection_required=True,
    )

    options = build_random_navigation_options(request)

    assert options["route_profile"] == "single_left"
    assert options["route_profile_mix"] == {
        "mostly_straight": 0.5,
        "single_left": 0.5,
    }
    assert options["min_turns"] == 1
    assert options["max_turns"] == 2
    assert options["intersection_required"] is True


def test_compute_route_profile_metrics_identifies_straight_route():
    rx = [0, 10, 20, 30, 40, 50]
    ry = [0, 0, 0, 0, 0, 0]

    metrics = compute_route_profile_metrics(rx, ry)

    assert metrics["route_profile"] == "mostly_straight"
    assert metrics["turn_count"] == 0
    assert metrics["straight_fraction"] > 0.99
    assert matches_route_profile(metrics, route_profile="mostly_straight")
    assert not matches_route_profile(metrics, route_profile="single_left")


def test_compute_route_profile_metrics_identifies_multi_turn_route():
    rx = [0, 10, 20, 20, 20, 30, 40]
    ry = [0, 0, 0, 10, 20, 20, 20]

    metrics = compute_route_profile_metrics(rx, ry)

    assert metrics["turn_count"] >= 1
    assert metrics["route_profile"] in {"single_left", "multi_turn", "mixed"}
    assert matches_route_profile(metrics, min_turns=1)
