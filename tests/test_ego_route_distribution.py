from __future__ import annotations

import math
import random
from statistics import mean

from CarlaBEV.src.managers.scene_generator import SceneGenerator


def _collect_route_contexts(
    *,
    episodes: int,
    min_turns: int | None = None,
    max_turns: int | None = None,
):
    generator = SceneGenerator(
        config={
            "map_size": 128,
            "map_name": "Town01",
            "traffic_enabled": False,
        }
    )
    contexts = []
    for seed in range(episodes):
        actors, route_length = generator.generate_random(
            num_cars=0,
            dist_range=(30, 80),
            traffic_enabled=False,
            ego_route_graph="canonical",
            min_turns=min_turns,
            max_turns=max_turns,
            max_retries=30,
            route_rng=random.Random(seed),
        )
        assert actors["agent"] is not None
        assert route_length is not None
        contexts.append(dict(generator.last_scene_context))
    return contexts


def _summarize_route_contexts(contexts):
    route_profile_counts = {}
    for context in contexts:
        profile = str(context["route_profile"])
        route_profile_counts[profile] = route_profile_counts.get(profile, 0) + 1

    mean_straight = mean(float(context["route_straight_fraction"]) for context in contexts)
    mean_left = mean(float(context["route_left_turn_fraction"]) for context in contexts)
    mean_right = mean(float(context["route_right_turn_fraction"]) for context in contexts)
    mean_turn_count = mean(float(context["route_turn_count"]) for context in contexts)

    return {
        "num_samples": len(contexts),
        "route_profile_counts": route_profile_counts,
        "mean_straight_fraction": mean_straight,
        "mean_left_turn_fraction": mean_left,
        "mean_right_turn_fraction": mean_right,
        "mean_turn_count": mean_turn_count,
    }


def test_ego_route_distribution_summary_smoke():
    contexts = _collect_route_contexts(episodes=12)
    summary = _summarize_route_contexts(contexts)

    assert summary["num_samples"] == 12
    assert sum(summary["route_profile_counts"].values()) == 12
    assert math.isclose(
        summary["mean_straight_fraction"]
        + summary["mean_left_turn_fraction"]
        + summary["mean_right_turn_fraction"],
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )

    for context in contexts:
        straight = float(context["route_straight_fraction"])
        left = float(context["route_left_turn_fraction"])
        right = float(context["route_right_turn_fraction"])
        assert 0.0 <= straight <= 1.0
        assert 0.0 <= left <= 1.0
        assert 0.0 <= right <= 1.0
        assert math.isclose(straight + left + right, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_ego_route_distribution_respects_min_turns_across_multiple_episodes():
    contexts = _collect_route_contexts(episodes=12, min_turns=1)
    summary = _summarize_route_contexts(contexts)

    assert summary["num_samples"] == 12
    assert summary["mean_turn_count"] >= 1.0
    assert summary["mean_left_turn_fraction"] + summary["mean_right_turn_fraction"] > 0.0

    for context in contexts:
        assert int(context["route_turn_count"]) >= 1
        assert context["route_profile"] in {"single_left", "single_right", "multi_turn", "mixed"}
