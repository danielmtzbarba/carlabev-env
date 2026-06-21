from __future__ import annotations

import argparse
import json
import sqlite3

from rich.console import Console
from rich.table import Table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the CarlaBEV scene library SQLite database.")
    parser.add_argument(
        "--scene-library-path",
        default="CarlaBEV/assets/scenes/scene_library.db",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _query_counts(conn: sqlite3.Connection, column: str) -> dict[str, int]:
    rows = conn.execute(
        f"""
        SELECT COALESCE(CAST({column} AS TEXT), 'null') AS label, COUNT(*) AS count
        FROM scenes
        GROUP BY label
        ORDER BY count DESC, label ASC
        """
    ).fetchall()
    return {str(row["label"]): int(row["count"]) for row in rows}


def collect_scene_library_stats(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        total_scenes = int(conn.execute("SELECT COUNT(*) AS count FROM scenes").fetchone()["count"])
        if total_scenes == 0:
            return {
                "scene_library_path": db_path,
                "total_scenes": 0,
                "difficulty_counts": {},
                "route_profile_counts": {},
                "main_actor_role_counts": {},
            }

        aggregate = conn.execute(
            """
            SELECT
                MIN(created_at) AS created_at_min,
                MAX(created_at) AS created_at_max,
                AVG(COALESCE(route_length_m, 0.0)) AS mean_route_length_m,
                AVG(COALESCE(num_vehicles, 0.0)) AS mean_requested_vehicles,
                AVG(COALESCE(spawned_num_vehicles, 0.0)) AS mean_spawned_vehicles,
                AVG(COALESCE(spawned_num_vehicles_near_ego, 0.0)) AS mean_spawned_near_ego,
                AVG(CASE WHEN COALESCE(spawned_num_vehicles_near_ego, 0) > 0 THEN 1.0 ELSE 0.0 END) AS near_ego_presence_rate,
                AVG(CASE WHEN COALESCE(main_actor_role, '') != '' THEN 1.0 ELSE 0.0 END) AS main_actor_role_rate,
                AVG(CASE WHEN COALESCE(route_intersection_like, 0) > 0 THEN 1.0 ELSE 0.0 END) AS intersection_like_rate
            FROM scenes
            """
        ).fetchone()

        role_presence = {}
        for role, column in {
            "lead": "spawned_role_lead",
            "rear": "spawned_role_rear",
            "cross_path": "spawned_role_cross_path",
            "opposite_lane": "spawned_role_opposite_lane",
            "background": "spawned_role_background",
        }.items():
            row = conn.execute(
                f"SELECT AVG(CASE WHEN COALESCE({column}, 0) > 0 THEN 1.0 ELSE 0.0 END) AS rate FROM scenes"
            ).fetchone()
            role_presence[role] = float(row["rate"] or 0.0)

        return {
            "scene_library_path": db_path,
            "total_scenes": total_scenes,
            "created_at_min": aggregate["created_at_min"],
            "created_at_max": aggregate["created_at_max"],
            "mean_route_length_m": float(aggregate["mean_route_length_m"] or 0.0),
            "mean_requested_vehicles": float(aggregate["mean_requested_vehicles"] or 0.0),
            "mean_spawned_vehicles": float(aggregate["mean_spawned_vehicles"] or 0.0),
            "mean_spawned_near_ego": float(aggregate["mean_spawned_near_ego"] or 0.0),
            "near_ego_presence_rate": float(aggregate["near_ego_presence_rate"] or 0.0),
            "main_actor_role_rate": float(aggregate["main_actor_role_rate"] or 0.0),
            "intersection_like_rate": float(aggregate["intersection_like_rate"] or 0.0),
            "generator_version_counts": _query_counts(conn, "generator_version"),
            "difficulty_counts": _query_counts(conn, "difficulty_preset_id"),
            "route_profile_counts": _query_counts(conn, "route_profile"),
            "main_actor_role_counts": _query_counts(conn, "main_actor_role"),
            "traffic_role_profile_counts": _query_counts(conn, "traffic_role_profile"),
            "role_presence_rates": role_presence,
        }
    finally:
        conn.close()


def _render_counts(console: Console, title: str, counts: dict[str, int]) -> None:
    table = Table(title=title)
    table.add_column("Label", justify="left")
    table.add_column("Count", justify="right")
    total = max(sum(counts.values()), 1)
    table.add_column("Share %", justify="right")
    for label, count in counts.items():
        table.add_row(label, str(count), f"{(count / total) * 100.0:.1f}")
    console.print(table)


def _render_summary(console: Console, summary: dict) -> None:
    table = Table(title="Scene Library Summary")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Scenes", str(summary["total_scenes"]))
    table.add_row("Created Min", str(summary.get("created_at_min", "n/a")))
    table.add_row("Created Max", str(summary.get("created_at_max", "n/a")))
    table.add_row("Mean Route Length (m)", f"{summary.get('mean_route_length_m', 0.0):.1f}")
    table.add_row("Mean Requested Vehicles", f"{summary.get('mean_requested_vehicles', 0.0):.1f}")
    table.add_row("Mean Spawned Vehicles", f"{summary.get('mean_spawned_vehicles', 0.0):.1f}")
    table.add_row("Mean Near-Ego Vehicles", f"{summary.get('mean_spawned_near_ego', 0.0):.1f}")
    table.add_row("Near-Ego Presence %", f"{summary.get('near_ego_presence_rate', 0.0) * 100.0:.1f}")
    table.add_row("Main Role Present %", f"{summary.get('main_actor_role_rate', 0.0) * 100.0:.1f}")
    table.add_row("Intersection-Like %", f"{summary.get('intersection_like_rate', 0.0) * 100.0:.1f}")
    console.print(table)

    role_presence = summary.get("role_presence_rates", {})
    if role_presence:
        role_table = Table(title="Role Presence Rates")
        role_table.add_column("Role", justify="left")
        role_table.add_column("Presence %", justify="right")
        for role, rate in role_presence.items():
            role_table.add_row(role, f"{rate * 100.0:.1f}")
        console.print(role_table)

    for title, key in (
        ("Generator Versions", "generator_version_counts"),
        ("Difficulty Presets", "difficulty_counts"),
        ("Route Profiles", "route_profile_counts"),
        ("Main Actor Roles", "main_actor_role_counts"),
        ("Traffic Role Profiles", "traffic_role_profile_counts"),
    ):
        _render_counts(console, title, summary.get(key, {}))


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary = collect_scene_library_stats(args.scene_library_path)
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _render_summary(Console(), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
