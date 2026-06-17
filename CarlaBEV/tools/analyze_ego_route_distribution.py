from __future__ import annotations

import argparse
import json
import math
import random
from statistics import mean

from rich.console import Console
from rich.table import Table

from CarlaBEV.src.managers.scene_generator import SceneGenerator


def collect_ego_route_distribution(
    *,
    episodes: int,
    dist_range: tuple[float, float] = (30.0, 80.0),
    min_turns: int | None = None,
    max_turns: int | None = None,
    intersection_required: bool | None = None,
    max_retries: int = 30,
    ego_route_graph: str = "canonical",
    seed_offset: int = 0,
) -> dict:
    generator = SceneGenerator(
        config={
            "map_size": 128,
            "map_name": "Town01",
            "traffic_enabled": False,
        }
    )
    contexts = []
    route_profile_counts: dict[str, int] = {}

    for episode in range(episodes):
        route_rng = random.Random(seed_offset + episode)
        actors, route_length = generator.generate_random(
            num_cars=0,
            dist_range=dist_range,
            traffic_enabled=False,
            ego_route_graph=ego_route_graph,
            min_turns=min_turns,
            max_turns=max_turns,
            intersection_required=intersection_required,
            max_retries=max_retries,
            route_rng=route_rng,
        )
        if actors["agent"] is None or route_length is None:
            raise RuntimeError(f"Failed to generate ego route for episode {episode}")
        context = dict(generator.last_scene_context)
        contexts.append(context)
        profile = str(context["route_profile"])
        route_profile_counts[profile] = route_profile_counts.get(profile, 0) + 1

    if not contexts:
        return {
            "num_samples": 0,
            "route_profile_counts": {},
            "mean_turn_count": 0.0,
            "intersection_like_rate": 0.0,
            "mean_straight_fraction": 0.0,
            "mean_left_turn_fraction": 0.0,
            "mean_right_turn_fraction": 0.0,
            "samples": [],
        }

    return {
        "num_samples": len(contexts),
        "route_profile_counts": route_profile_counts,
        "mean_turn_count": mean(float(context["route_turn_count"]) for context in contexts),
        "intersection_like_rate": mean(
            1.0 if bool(context["route_intersection_like"]) else 0.0 for context in contexts
        ),
        "mean_straight_fraction": mean(float(context["route_straight_fraction"]) for context in contexts),
        "mean_left_turn_fraction": mean(float(context["route_left_turn_fraction"]) for context in contexts),
        "mean_right_turn_fraction": mean(float(context["route_right_turn_fraction"]) for context in contexts),
        "samples": contexts,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample ego routes and summarize turn distribution.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--min-distance", type=float, default=30.0)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--min-turns", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--intersection-required", choices=("true", "false"), default=None)
    parser.add_argument("--max-retries", type=int, default=30)
    parser.add_argument("--ego-route-graph", default="canonical")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _render_summary(summary: dict, *, console: Console) -> None:
    table = Table(title="Ego Route Distribution")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Samples", str(summary["num_samples"]))
    table.add_row("Mean Turns", f"{summary['mean_turn_count']:.2f}")
    table.add_row("Intersection Like %", f"{summary['intersection_like_rate'] * 100.0:.1f}")
    table.add_row("Straight %", f"{summary['mean_straight_fraction'] * 100.0:.1f}")
    table.add_row("Left %", f"{summary['mean_left_turn_fraction'] * 100.0:.1f}")
    table.add_row("Right %", f"{summary['mean_right_turn_fraction'] * 100.0:.1f}")
    console.print(table)

    profile_table = Table(title="Route Profiles")
    profile_table.add_column("Profile", justify="left")
    profile_table.add_column("Count", justify="right")
    profile_table.add_column("Share %", justify="right")
    total = max(int(summary["num_samples"]), 1)
    for profile, count in sorted(summary["route_profile_counts"].items()):
        profile_table.add_row(profile, str(count), f"{(count / total) * 100.0:.1f}")
    console.print(profile_table)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    intersection_required = None
    if args.intersection_required == "true":
        intersection_required = True
    elif args.intersection_required == "false":
        intersection_required = False

    summary = collect_ego_route_distribution(
        episodes=args.episodes,
        dist_range=(args.min_distance, args.max_distance),
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        intersection_required=intersection_required,
        max_retries=args.max_retries,
        ego_route_graph=args.ego_route_graph,
        seed_offset=args.seed_offset,
    )

    total_fraction = (
        summary["mean_straight_fraction"]
        + summary["mean_left_turn_fraction"]
        + summary["mean_right_turn_fraction"]
    )
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(
            "Route-direction fractions are inconsistent: "
            f"straight+left+right={total_fraction:.6f}"
        )

    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _render_summary(summary, console=Console())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
