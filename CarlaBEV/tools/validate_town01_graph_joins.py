from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle

from CarlaBEV.envs.utils import asset_path
from CarlaBEV.src.planning.town01_canonical_graph import (
    DEFAULT_GRAPH_NAME,
    DEFAULT_TOWN_NAME,
    build_town01_canonical_vehicle_graph,
    validate_junction_joins,
)


def _load_graph(*, rebuild: bool = False):
    path = Path(asset_path) / DEFAULT_TOWN_NAME / DEFAULT_GRAPH_NAME
    if path.exists() and not rebuild:
        with path.open("rb") as handle:
            return pickle.load(handle)
    return build_town01_canonical_vehicle_graph()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Town01 canonical graph junction joins."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the graph instead of loading the saved canonical asset.",
    )
    parser.add_argument(
        "--junction-id",
        type=str,
        default=None,
        help="Restrict the report to a single junction_id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of suspicious records to print.",
    )
    args = parser.parse_args()

    graph = _load_graph(rebuild=args.rebuild)
    records = validate_junction_joins(graph)
    if args.junction_id is not None:
        records = [record for record in records if record.junction_id == args.junction_id]

    suspicious = [record for record in records if record.is_suspicious]
    reason_counts = Counter(record.reason for record in suspicious)
    junction_counts = Counter(record.junction_id for record in suspicious)

    print("Town01 junction join validation")
    print("=" * 72)
    print(f"checked connectors: {len(records)}")
    print(f"suspicious connectors: {len(suspicious)}")
    print(f"reasons: {dict(reason_counts)}")
    print(f"junction_counts: {dict(junction_counts)}")
    print("-" * 72)

    for record in suspicious[: args.limit]:
        print(
            f"{record.junction_id}: {record.src} -> {record.dst} | "
            f"{record.src_side}->{record.dst_side} | turn={record.turn_class} | "
            f"lateral_error={record.lateral_error:.1f} | reason={record.reason}"
        )

    return 1 if suspicious else 0


if __name__ == "__main__":
    raise SystemExit(main())
