from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Any

from rich.console import Console
from rich.table import Table


FIELD_ALIASES = {
    "scene-benchmark": "scene_benchmark_id",
    "scene-profile": "scene_profile_id",
    "scene-split": "scene_split",
    "difficulty": "difficulty_preset_id",
    "route-profile": "route_profile",
    "main-role": "main_actor_role",
    "traffic-role-profile": "traffic_role_profile",
    "generator-version": "generator_version",
}

ROLE_MIN_FILTERS = {
    "min-lead": "spawned_role_lead",
    "min-rear": "spawned_role_rear",
    "min-cross-path": "spawned_role_cross_path",
    "min-opposite-lane": "spawned_role_opposite_lane",
    "min-background": "spawned_role_background",
}

QUERY_COLUMNS = [
    "scene_id",
    "scene_benchmark_id",
    "scene_profile_id",
    "scene_split",
    "difficulty_preset_id",
    "route_profile",
    "main_actor_role",
    "traffic_role_profile",
    "spawned_num_vehicles",
    "spawned_num_vehicles_near_ego",
    "spawned_role_lead",
    "spawned_role_rear",
    "spawned_role_cross_path",
    "spawned_role_opposite_lane",
    "spawned_role_background",
    "scene_seed",
    "route_seed",
    "traffic_seed",
]


def _build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--scene-library-path",
        default="CarlaBEV/assets/scenes/scene_library.db",
    )
    shared.add_argument("--json", action="store_true", dest="as_json")
    shared.add_argument("--scene-benchmark", default=None)
    shared.add_argument("--scene-profile", default=None)
    shared.add_argument("--scene-split", default=None)
    shared.add_argument("--difficulty", default=None)
    shared.add_argument("--route-profile", default=None)
    shared.add_argument("--main-role", default=None)
    shared.add_argument("--traffic-role-profile", default=None)
    shared.add_argument("--generator-version", default=None)

    parser = argparse.ArgumentParser(
        description="Analyze the CarlaBEV scene library SQLite database.",
        parents=[shared],
    )

    subparsers = parser.add_subparsers(dest="mode")

    summary_parser = subparsers.add_parser("summary", help="Show one-screen summary metrics.", parents=[shared])
    summary_parser.set_defaults(mode="summary")

    breakdown_parser = subparsers.add_parser("breakdown", help="Show grouped metadata counts.", parents=[shared])
    breakdown_parser.add_argument(
        "--by",
        action="append",
        choices=tuple(FIELD_ALIASES.keys()),
        default=[],
        help="Metadata dimension to group by. Repeat to print multiple one-dimensional breakdowns.",
    )
    breakdown_parser.add_argument(
        "--cross",
        default=None,
        help="Cross-tab two dimensions, for example 'difficulty,route-profile'.",
    )
    breakdown_parser.set_defaults(mode="breakdown")

    quality_parser = subparsers.add_parser("quality", help="Show training usefulness and spawn quality metrics.", parents=[shared])
    quality_parser.set_defaults(mode="quality")

    query_parser = subparsers.add_parser("query", help="List matching scenes by metadata filters.", parents=[shared])
    query_parser.add_argument("--scene-seed", type=int, default=None)
    query_parser.add_argument("--route-seed", type=int, default=None)
    query_parser.add_argument("--traffic-seed", type=int, default=None)
    query_parser.add_argument("--min-near-ego", type=int, default=None)
    for flag in ROLE_MIN_FILTERS:
        query_parser.add_argument(f"--{flag}", type=int, default=None)
    query_parser.add_argument("--limit", type=int, default=20)
    query_parser.set_defaults(mode="query")
    return parser


def _open_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _base_filters(args) -> dict[str, Any]:
    return {
        "scene_profile_id": args.scene_profile,
        "scene_benchmark_id": args.scene_benchmark,
        "scene_split": args.scene_split,
        "difficulty_preset_id": args.difficulty,
        "route_profile": args.route_profile,
        "main_actor_role": args.main_role,
        "traffic_role_profile": args.traffic_role_profile,
        "generator_version": args.generator_version,
    }


def _scene_columns(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(scenes)").fetchall()
    }


def _where_from_filters(
    filters: dict[str, Any],
    *,
    available_columns: set[str] | None = None,
) -> tuple[str, list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    for column, value in filters.items():
        if value is None:
            continue
        if available_columns is not None and column not in available_columns:
            continue
        where.append(f"{column} = ?")
        params.append(value)
    sql = ""
    if where:
        sql = " WHERE " + " AND ".join(where)
    return sql, params


def _count_by(
    conn: sqlite3.Connection,
    filters: dict[str, Any],
    column: str,
    *,
    available_columns: set[str],
) -> dict[str, int]:
    if column not in available_columns:
        return {}
    where_sql, params = _where_from_filters(filters, available_columns=available_columns)
    rows = conn.execute(
        f"""
        SELECT COALESCE(CAST({column} AS TEXT), 'null') AS label, COUNT(*) AS count
        FROM scenes
        {where_sql}
        GROUP BY label
        ORDER BY count DESC, label ASC
        """,
        params,
    ).fetchall()
    return {str(row["label"]): int(row["count"]) for row in rows}


def _cross_breakdown(
    conn: sqlite3.Connection,
    filters: dict[str, Any],
    first: str,
    second: str,
    *,
    available_columns: set[str],
) -> list[dict[str, Any]]:
    if first not in available_columns or second not in available_columns:
        return []
    where_sql, params = _where_from_filters(filters, available_columns=available_columns)
    rows = conn.execute(
        f"""
        SELECT
            COALESCE(CAST({first} AS TEXT), 'null') AS first_label,
            COALESCE(CAST({second} AS TEXT), 'null') AS second_label,
            COUNT(*) AS count
        FROM scenes
        {where_sql}
        GROUP BY first_label, second_label
        ORDER BY first_label ASC, second_label ASC
        """,
        params,
    ).fetchall()
    return [
        {"first": str(row["first_label"]), "second": str(row["second_label"]), "count": int(row["count"])}
        for row in rows
    ]


def collect_scene_library_summary(db_path: str, *, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    filters = filters or {}
    conn = _open_conn(db_path)
    try:
        available_columns = _scene_columns(conn)
        where_sql, params = _where_from_filters(filters, available_columns=available_columns)
        total_scenes = int(conn.execute(f"SELECT COUNT(*) AS count FROM scenes{where_sql}", params).fetchone()["count"])
        result = {
            "mode": "summary",
            "scene_library_path": db_path,
            "filters": filters,
            "total_scenes": total_scenes,
        }
        if total_scenes == 0:
            result.update(
                {
                    "scene_profile_counts": {},
                    "scene_split_counts": {},
                    "scene_benchmark_counts": {},
                    "difficulty_counts": {},
                    "route_profile_counts": {},
                    "main_actor_role_counts": {},
                    "traffic_role_profile_counts": {},
                    "generator_version_counts": {},
                    "role_presence_rates": {},
                }
            )
            return result

        aggregate = conn.execute(
            f"""
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
            {where_sql}
            """,
            params,
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
                f"""
                SELECT AVG(CASE WHEN COALESCE({column}, 0) > 0 THEN 1.0 ELSE 0.0 END) AS rate
                FROM scenes
                {where_sql}
                """,
                params,
            ).fetchone()
            role_presence[role] = float(row["rate"] or 0.0)

        result.update(
            {
                "created_at_min": aggregate["created_at_min"],
                "created_at_max": aggregate["created_at_max"],
                "mean_route_length_m": float(aggregate["mean_route_length_m"] or 0.0),
                "mean_requested_vehicles": float(aggregate["mean_requested_vehicles"] or 0.0),
                "mean_spawned_vehicles": float(aggregate["mean_spawned_vehicles"] or 0.0),
                "mean_spawned_near_ego": float(aggregate["mean_spawned_near_ego"] or 0.0),
                "near_ego_presence_rate": float(aggregate["near_ego_presence_rate"] or 0.0),
                "main_actor_role_rate": float(aggregate["main_actor_role_rate"] or 0.0),
                "intersection_like_rate": float(aggregate["intersection_like_rate"] or 0.0),
                "generator_version_counts": _count_by(
                    conn, filters, "generator_version", available_columns=available_columns
                ),
                "scene_benchmark_counts": _count_by(
                    conn, filters, "scene_benchmark_id", available_columns=available_columns
                ),
                "scene_profile_counts": _count_by(
                    conn, filters, "scene_profile_id", available_columns=available_columns
                ),
                "scene_split_counts": _count_by(
                    conn, filters, "scene_split", available_columns=available_columns
                ),
                "difficulty_counts": _count_by(
                    conn, filters, "difficulty_preset_id", available_columns=available_columns
                ),
                "route_profile_counts": _count_by(
                    conn, filters, "route_profile", available_columns=available_columns
                ),
                "main_actor_role_counts": _count_by(
                    conn, filters, "main_actor_role", available_columns=available_columns
                ),
                "traffic_role_profile_counts": _count_by(
                    conn, filters, "traffic_role_profile", available_columns=available_columns
                ),
                "role_presence_rates": role_presence,
            }
        )
        return result
    finally:
        conn.close()


def collect_scene_library_breakdown(
    db_path: str,
    *,
    filters: dict[str, Any] | None = None,
    group_by: list[str] | None = None,
    cross: tuple[str, str] | None = None,
) -> dict[str, Any]:
    filters = filters or {}
    group_by = group_by or []
    conn = _open_conn(db_path)
    try:
        available_columns = _scene_columns(conn)
        result: dict[str, Any] = {
            "mode": "breakdown",
            "scene_library_path": db_path,
            "filters": filters,
        }
        if cross is not None:
            result["cross"] = list(cross)
            result["rows"] = _cross_breakdown(
                conn,
                filters,
                cross[0],
                cross[1],
                available_columns=available_columns,
            )
            return result

        requested = group_by or ["difficulty"]
        mapped = [FIELD_ALIASES[name] for name in requested]
        result["group_by"] = requested
        result["breakdowns"] = {
            name: _count_by(conn, filters, column, available_columns=available_columns)
            for name, column in zip(requested, mapped, strict=True)
        }
        return result
    finally:
        conn.close()


def collect_scene_library_quality(db_path: str, *, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    filters = filters or {}
    conn = _open_conn(db_path)
    try:
        available_columns = _scene_columns(conn)
        where_sql, params = _where_from_filters(filters, available_columns=available_columns)
        total_scenes = int(conn.execute(f"SELECT COUNT(*) AS count FROM scenes{where_sql}", params).fetchone()["count"])
        result = {
            "mode": "quality",
            "scene_library_path": db_path,
            "filters": filters,
            "total_scenes": total_scenes,
        }
        if total_scenes == 0:
            result.update({"role_presence_rates": {}})
            return result

        row = conn.execute(
            f"""
            SELECT
                AVG(CASE WHEN COALESCE(spawned_num_vehicles, 0) = 0 THEN 1.0 ELSE 0.0 END) AS zero_spawned_rate,
                AVG(CASE WHEN COALESCE(spawned_num_vehicles_near_ego, 0) = 0 THEN 1.0 ELSE 0.0 END) AS zero_near_ego_rate,
                AVG(CASE WHEN COALESCE(main_actor_role, '') = '' THEN 1.0 ELSE 0.0 END) AS missing_main_role_rate,
                AVG(COALESCE(num_vehicles, 0.0) - COALESCE(spawned_num_vehicles, 0.0)) AS mean_spawn_gap,
                AVG(COALESCE(num_vehicles, 0.0)) AS mean_requested_vehicles,
                AVG(COALESCE(spawned_num_vehicles, 0.0)) AS mean_spawned_vehicles,
                AVG(
                    CASE
                        WHEN (
                            COALESCE(spawned_role_lead, 0) +
                            COALESCE(spawned_role_rear, 0) +
                            COALESCE(spawned_role_cross_path, 0) +
                            COALESCE(spawned_role_opposite_lane, 0)
                        ) > 0 THEN 1.0
                        ELSE 0.0
                    END
                ) AS interaction_scene_rate
            FROM scenes
            {where_sql}
            """,
            params,
        ).fetchone()

        role_presence = {}
        for role, column in {
            "lead": "spawned_role_lead",
            "rear": "spawned_role_rear",
            "cross_path": "spawned_role_cross_path",
            "opposite_lane": "spawned_role_opposite_lane",
            "background": "spawned_role_background",
        }.items():
            presence = conn.execute(
                f"""
                SELECT AVG(CASE WHEN COALESCE({column}, 0) > 0 THEN 1.0 ELSE 0.0 END) AS rate
                FROM scenes
                {where_sql}
                """,
                params,
            ).fetchone()
            role_presence[role] = float(presence["rate"] or 0.0)

        mean_requested = float(row["mean_requested_vehicles"] or 0.0)
        mean_spawned = float(row["mean_spawned_vehicles"] or 0.0)
        fulfillment = 0.0 if mean_requested <= 0.0 else mean_spawned / mean_requested
        result.update(
            {
                "zero_spawned_rate": float(row["zero_spawned_rate"] or 0.0),
                "zero_near_ego_rate": float(row["zero_near_ego_rate"] or 0.0),
                "missing_main_role_rate": float(row["missing_main_role_rate"] or 0.0),
                "mean_spawn_gap": float(row["mean_spawn_gap"] or 0.0),
                "mean_requested_vehicles": mean_requested,
                "mean_spawned_vehicles": mean_spawned,
                "requested_fulfillment_ratio": float(fulfillment),
                "interaction_scene_rate": float(row["interaction_scene_rate"] or 0.0),
                "role_presence_rates": role_presence,
            }
        )
        return result
    finally:
        conn.close()


def collect_scene_library_query(
    db_path: str,
    *,
    filters: dict[str, Any] | None = None,
    scene_seed: int | None = None,
    route_seed: int | None = None,
    traffic_seed: int | None = None,
    min_near_ego: int | None = None,
    role_mins: dict[str, int] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    filters = filters or {}
    role_mins = role_mins or {}
    extra_filters = dict(filters)
    if scene_seed is not None:
        extra_filters["scene_seed"] = int(scene_seed)
    if route_seed is not None:
        extra_filters["route_seed"] = int(route_seed)
    if traffic_seed is not None:
        extra_filters["traffic_seed"] = int(traffic_seed)

    conn = _open_conn(db_path)
    try:
        available_columns = _scene_columns(conn)
        where_sql, params = _where_from_filters(extra_filters, available_columns=available_columns)
        extra_where: list[str] = []
        if min_near_ego is not None and "spawned_num_vehicles_near_ego" in available_columns:
            extra_where.append("COALESCE(spawned_num_vehicles_near_ego, 0) >= ?")
            params.append(int(min_near_ego))
        for column, minimum in role_mins.items():
            if column not in available_columns:
                continue
            extra_where.append(f"COALESCE({column}, 0) >= ?")
            params.append(int(minimum))
        if extra_where:
            if where_sql:
                where_sql += " AND " + " AND ".join(extra_where)
            else:
                where_sql = " WHERE " + " AND ".join(extra_where)
        selected_columns = [column for column in QUERY_COLUMNS if column in available_columns]
        rows = conn.execute(
            f"""
            SELECT {", ".join(selected_columns)}
            FROM scenes
            {where_sql}
            ORDER BY created_at DESC, scene_id DESC
            LIMIT ?
            """,
            [*params, int(limit)],
        ).fetchall()
        data_rows = [
            {key: (row[key] if key in selected_columns else None) for key in QUERY_COLUMNS}
            for row in rows
        ]
        return {
            "mode": "query",
            "scene_library_path": db_path,
            "filters": filters,
            "scene_seed": scene_seed,
            "route_seed": route_seed,
            "traffic_seed": traffic_seed,
            "min_near_ego": min_near_ego,
            "role_mins": role_mins,
            "limit": int(limit),
            "total_matches": len(data_rows),
            "rows": data_rows,
        }
    finally:
        conn.close()


def collect_scene_library_stats(db_path: str, *, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    return collect_scene_library_summary(db_path, filters=filters)


def _render_counts(console: Console, title: str, counts: dict[str, int]) -> None:
    table = Table(title=title)
    table.add_column("Label", justify="left")
    table.add_column("Count", justify="right")
    total = max(sum(counts.values()), 1)
    table.add_column("Share %", justify="right")
    for label, count in counts.items():
        table.add_row(label, str(count), f"{(count / total) * 100.0:.1f}")
    console.print(table)


def _render_summary(console: Console, summary: dict[str, Any]) -> None:
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
    for title, key in (
        ("Generator Versions", "generator_version_counts"),
        ("Scene Benchmarks", "scene_benchmark_counts"),
        ("Scene Profiles", "scene_profile_counts"),
        ("Scene Splits", "scene_split_counts"),
        ("Difficulty Presets", "difficulty_counts"),
        ("Route Profiles", "route_profile_counts"),
        ("Main Actor Roles", "main_actor_role_counts"),
        ("Traffic Role Profiles", "traffic_role_profile_counts"),
    ):
        _render_counts(console, title, summary.get(key, {}))


def _render_quality(console: Console, quality: dict[str, Any]) -> None:
    table = Table(title="Scene Quality")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Scenes", str(quality["total_scenes"]))
    table.add_row("Zero Spawned %", f"{quality.get('zero_spawned_rate', 0.0) * 100.0:.1f}")
    table.add_row("Zero Near-Ego %", f"{quality.get('zero_near_ego_rate', 0.0) * 100.0:.1f}")
    table.add_row("Missing Main Role %", f"{quality.get('missing_main_role_rate', 0.0) * 100.0:.1f}")
    table.add_row("Requested Fulfillment %", f"{quality.get('requested_fulfillment_ratio', 0.0) * 100.0:.1f}")
    table.add_row("Mean Spawn Gap", f"{quality.get('mean_spawn_gap', 0.0):.1f}")
    table.add_row("Interaction Scene %", f"{quality.get('interaction_scene_rate', 0.0) * 100.0:.1f}")
    console.print(table)

    role_table = Table(title="Role Presence Rates")
    role_table.add_column("Role", justify="left")
    role_table.add_column("Presence %", justify="right")
    for role, rate in quality.get("role_presence_rates", {}).items():
        role_table.add_row(role, f"{rate * 100.0:.1f}")
    console.print(role_table)


def _render_breakdown(console: Console, breakdown: dict[str, Any]) -> None:
    if "rows" in breakdown:
        title = " x ".join(breakdown["cross"])
        table = Table(title=f"{title.title()} Breakdown")
        table.add_column(breakdown["cross"][0], justify="left")
        table.add_column(breakdown["cross"][1], justify="left")
        table.add_column("Count", justify="right")
        for row in breakdown["rows"]:
            table.add_row(row["first"], row["second"], str(row["count"]))
        console.print(table)
        return

    for label, counts in breakdown.get("breakdowns", {}).items():
        _render_counts(console, f"{label.title()} Breakdown", counts)


def _render_query(console: Console, payload: dict[str, Any]) -> None:
    table = Table(title="Query Results")
    table.add_column("scene_id", justify="right")
    table.add_column("scene_benchmark", justify="left")
    table.add_column("scene_profile", justify="left")
    table.add_column("scene_split", justify="left")
    table.add_column("difficulty", justify="left")
    table.add_column("route_profile", justify="left")
    table.add_column("main_role", justify="left")
    table.add_column("spawned", justify="right")
    table.add_column("near_ego", justify="right")
    table.add_column("lead", justify="right")
    table.add_column("rear", justify="right")
    table.add_column("cross", justify="right")
    table.add_column("opp", justify="right")
    table.add_column("scene_seed", justify="right")
    for row in payload.get("rows", []):
        table.add_row(
            str(row["scene_id"]),
            str(row["scene_benchmark_id"]),
            str(row["scene_profile_id"]),
            str(row["scene_split"]),
            str(row["difficulty_preset_id"]),
            str(row["route_profile"]),
            str(row["main_actor_role"]),
            str(row["spawned_num_vehicles"]),
            str(row["spawned_num_vehicles_near_ego"]),
            str(row["spawned_role_lead"]),
            str(row["spawned_role_rear"]),
            str(row["spawned_role_cross_path"]),
            str(row["spawned_role_opposite_lane"]),
            str(row["scene_seed"]),
        )
    console.print(table)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    mode = args.mode or "summary"
    filters = _base_filters(args)

    if mode == "summary":
        payload = collect_scene_library_summary(args.scene_library_path, filters=filters)
    elif mode == "quality":
        payload = collect_scene_library_quality(args.scene_library_path, filters=filters)
    elif mode == "breakdown":
        cross = None
        if args.cross:
            parts = [part.strip() for part in args.cross.split(",") if part.strip()]
            if len(parts) != 2 or any(part not in FIELD_ALIASES for part in parts):
                raise SystemExit("--cross must be two comma-separated values from: " + ", ".join(FIELD_ALIASES))
            cross = (FIELD_ALIASES[parts[0]], FIELD_ALIASES[parts[1]])
        payload = collect_scene_library_breakdown(
            args.scene_library_path,
            filters=filters,
            group_by=args.by,
            cross=cross,
        )
    elif mode == "query":
        role_mins = {
            column: int(getattr(args, flag.replace("-", "_")))
            for flag, column in ROLE_MIN_FILTERS.items()
            if getattr(args, flag.replace("-", "_")) is not None
        }
        payload = collect_scene_library_query(
            args.scene_library_path,
            filters=filters,
            scene_seed=args.scene_seed,
            route_seed=args.route_seed,
            traffic_seed=args.traffic_seed,
            min_near_ego=args.min_near_ego,
            role_mins=role_mins,
            limit=args.limit,
        )
    else:
        raise SystemExit(f"Unsupported mode: {mode}")

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        console = Console()
        if mode == "summary":
            _render_summary(console, payload)
        elif mode == "quality":
            _render_quality(console, payload)
        elif mode == "breakdown":
            _render_breakdown(console, payload)
        elif mode == "query":
            _render_query(console, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
