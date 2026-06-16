from __future__ import annotations

import argparse

from CarlaBEV.src.planning.town01_canonical_graph import (
    build_town01_canonical_vehicle_graph,
    save_town01_canonical_vehicle_graph,
    summarize_canonical_graph,
    validate_town01_canonical_vehicle_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical directed Town01 vehicle graph."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Serialize the canonical graph to the default Town01 asset path.",
    )
    parser.add_argument(
        "--lane-changes",
        action="store_true",
        help="Include inferred lane-change candidate edges.",
    )
    args = parser.parse_args()

    graph = build_town01_canonical_vehicle_graph(
        add_lane_change_edges=args.lane_changes,
    )
    summary = summarize_canonical_graph(graph)
    problems = validate_town01_canonical_vehicle_graph(graph)

    print("Town01 canonical graph summary")
    print("=" * 72)
    print(f"nodes: {summary.node_count}")
    print(f"edges: {summary.edge_count}")
    print(f"weak_components: {summary.weak_component_count}")
    print(f"junctions: {summary.junction_count}")
    print(f"junction_connectors: {summary.junction_connector_count}")
    print(f"flow_groups: {summary.flow_groups}")

    if problems:
        print("-" * 72)
        print("validation: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("-" * 72)
    print("validation: PASS")

    if args.save:
        output = save_town01_canonical_vehicle_graph(
            add_lane_change_edges=args.lane_changes,
        )
        print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
