from __future__ import annotations

import math

from CarlaBEV.tools.analyze_ego_route_distribution import collect_ego_route_distribution


def test_collect_ego_route_distribution_smoke():
    summary = collect_ego_route_distribution(
        episodes=8,
        dist_range=(30.0, 80.0),
        ego_route_graph="canonical",
    )

    assert summary["num_samples"] == 8
    assert sum(summary["route_profile_counts"].values()) == 8
    assert summary["mean_turn_count"] >= 0.0
    assert 0.0 <= summary["intersection_like_rate"] <= 1.0
    assert math.isclose(
        summary["mean_straight_fraction"]
        + summary["mean_left_turn_fraction"]
        + summary["mean_right_turn_fraction"],
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def test_collect_ego_route_distribution_respects_min_turns():
    summary = collect_ego_route_distribution(
        episodes=8,
        dist_range=(30.0, 80.0),
        min_turns=1,
        ego_route_graph="canonical",
    )

    assert summary["num_samples"] == 8
    assert summary["mean_turn_count"] >= 1.0
    for sample in summary["samples"]:
        assert int(sample["route_turn_count"]) >= 1
