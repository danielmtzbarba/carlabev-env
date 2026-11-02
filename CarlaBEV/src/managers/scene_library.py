from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from CarlaBEV.src.scenes.scene_spec import (
    ACTOR_PAYLOAD_SCHEMA_VERSION,
    SceneSpec,
    capture_scene,
    identity_options,
    instantiate_scene,
    npc_uid_for,
    scene_identity_key,
)

__all__ = [
    "ACTOR_PAYLOAD_SCHEMA_VERSION",
    "SceneLibrary",
    "SceneLibraryError",
    "SceneLibraryMissError",
    "deserialize_scene_payload",
    "serialize_scene_payload",
]


ROLE_COLUMN_MAP = {
    "lead": "spawned_role_lead",
    "rear": "spawned_role_rear",
    "cross_path": "spawned_role_cross_path",
    "opposite_lane": "spawned_role_opposite_lane",
    "background": "spawned_role_background",
}

OPTIONAL_SCENE_COLUMNS = {
    "scene_benchmark_id": "TEXT",
    "scene_profile_id": "TEXT",
    "scene_split": "TEXT",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(item) for item in value]
    return value


def serialize_scene_payload(actors: dict[str, Any], *, surface_size: int | None = None) -> dict[str, Any]:
    """Serialize scene actors (spec layer, inline NPCs). Coordinates are surface px of ``surface_size``."""
    return capture_scene(actors, surface_size=int(surface_size) if surface_size is not None else 128).to_payload()


def deserialize_scene_payload(payload: dict[str, Any], *, map_size: int, units=None) -> dict[str, Any]:
    """Rebuild scene actors from a payload expressed at ``map_size`` surface px.

    A payload recorded at a different surface scale or schema is a contract
    violation (the cache key includes both), so it is rejected explicitly.
    """
    from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT

    payload_size = int((payload.get("frame") or {}).get("size", 128))
    if payload_size != int(map_size):
        raise ValueError(
            f"Scene payload recorded at surface size {payload_size} cannot be loaded at size {map_size}."
        )
    spec = SceneSpec.from_payload(payload)
    return instantiate_scene(spec, units=units if units is not None else DEFAULT_UNIT_CONTRACT, map_size=int(map_size))


class SceneLibraryError(RuntimeError):
    """Base class for scene-library contract violations."""


class SceneLibraryMissError(SceneLibraryError):
    """A required cache hit did not happen (``scene_library_require_hit=True``)."""

    def __init__(self, scene_key: str, *, reason: str, seeds: dict[str, Any] | None = None, options=None):
        self.scene_key = scene_key
        self.reason = reason
        self.seeds = dict(seeds or {})
        self.options = dict(options or {})
        super().__init__(
            f"scene_library_require_hit: {reason} for scene {scene_key} "
            f"(seeds={self.seeds}); the library does not contain this scene and regeneration is disabled."
        )


REQUIRED_TABLES = ("scenes", "npcs")


class SceneLibrary:
    """SQLite scene corpus.

    ``read_only=True`` opens the database with SQLite's ``mode=ro`` (no write
    of any kind: no schema migration, no usage counters, no insert/delete) and
    requires the file and its schema to exist already — an immutable corpus.
    """

    def __init__(self, db_path: str, *, generator_version: str, read_only: bool = False) -> None:
        self.db_path = str(db_path)
        self.generator_version = str(generator_version)
        self.read_only = bool(read_only)
        if self.read_only:
            if not Path(self.db_path).is_file():
                raise FileNotFoundError(
                    f"Read-only scene library not found: {self.db_path} (build it first with 'carlabev scene build-library')."
                )
            self.verify_schema()
        else:
            self._ensure_parent_dir()
            self.ensure_schema()

    def _ensure_parent_dir(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{Path(self.db_path).resolve().as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def verify_schema(self) -> None:
        """Read-only check that the corpus carries the current schema (no migration attempted)."""
        with self._connect() as conn:
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = [name for name in REQUIRED_TABLES if name not in tables]
        if missing:
            raise SceneLibraryError(
                f"Scene library {self.db_path} lacks required table(s) {missing}; "
                "it predates the spec-layer schema. Rebuild it (read-only mode never migrates)."
            )

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id INTEGER PRIMARY KEY,
                    scene_key TEXT UNIQUE NOT NULL,
                    generator_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    scene_kind TEXT NOT NULL,
                    scene_benchmark_id TEXT,
                    scene_profile_id TEXT,
                    scene_split TEXT,
                    difficulty_preset_id TEXT,
                    route_extent TEXT,
                    traffic_density TEXT,
                    speed_profile TEXT,
                    scene_seed INTEGER,
                    route_seed INTEGER,
                    traffic_seed INTEGER,
                    scenario_seed INTEGER,
                    num_vehicles INTEGER,
                    num_vehicles_near_ego INTEGER,
                    traffic_role_profile TEXT,
                    guaranteed_candidate_role TEXT,
                    ego_route_graph TEXT,
                    route_profile TEXT,
                    route_turn_count INTEGER,
                    route_intersection_like INTEGER NOT NULL DEFAULT 0,
                    route_length_m REAL,
                    main_actor_role TEXT,
                    spawned_num_vehicles INTEGER NOT NULL DEFAULT 0,
                    spawned_num_vehicles_near_ego INTEGER NOT NULL DEFAULT 0,
                    spawned_role_lead INTEGER NOT NULL DEFAULT 0,
                    spawned_role_rear INTEGER NOT NULL DEFAULT 0,
                    spawned_role_cross_path INTEGER NOT NULL DEFAULT 0,
                    spawned_role_opposite_lane INTEGER NOT NULL DEFAULT 0,
                    spawned_role_background INTEGER NOT NULL DEFAULT 0,
                    scene_descriptor_json TEXT NOT NULL,
                    scenario_context_json TEXT NOT NULL,
                    actors_json TEXT NOT NULL,
                    spawn_validation_json TEXT,
                    resolved_options_json TEXT NOT NULL
                );

                """
            )
            existing_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(scenes)").fetchall()
            }
            for column_name, column_type in OPTIONAL_SCENE_COLUMNS.items():
                if column_name in existing_columns:
                    continue
                conn.execute(f"ALTER TABLE scenes ADD COLUMN {column_name} {column_type}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_route_profile ON scenes(route_profile)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_scene_profile ON scenes(scene_profile_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_main_actor_role ON scenes(main_actor_role)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_difficulty ON scenes(difficulty_preset_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_seed_triplet ON scenes(scene_seed, route_seed, traffic_seed)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scenes_role_route ON scenes(main_actor_role, route_profile)"
            )
            # Layer 1: one row per NPC definition; the scene row references them in order.
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS npcs (
                    npc_uid TEXT PRIMARY KEY,
                    scene_key TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    npc_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    traffic_role TEXT,
                    behavior_type TEXT,
                    cruise_speed_mps REAL,
                    route_len INTEGER,
                    recyclable INTEGER NOT NULL DEFAULT 0,
                    runtime_seed INTEGER,
                    spec_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_npcs_scene ON npcs(scene_key, ordinal);
                CREATE INDEX IF NOT EXISTS idx_npcs_role ON npcs(traffic_role);
                CREATE INDEX IF NOT EXISTS idx_npcs_kind ON npcs(kind);
                """
            )

    def build_scene_key(
        self, resolved_options: dict[str, Any], seeds: dict[str, Any], *, surface_size: int | None = None
    ) -> str:
        return scene_identity_key(
            generator_version=self.generator_version,
            surface_size=int(surface_size) if surface_size is not None else 128,
            options=resolved_options,
            seeds=seeds,
        )

    def get_scene_by_key(self, scene_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scenes WHERE scene_key = ?",
                (scene_key,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_scene_by_filters(
        self,
        *,
        scene_benchmark_id: str | None = None,
        scene_profile_id: str | None = None,
        scene_split: str | None = None,
        difficulty_preset_id: str | None = None,
        route_profile: str | None = None,
        main_actor_role: str | None = None,
        required_role_counts: dict[str, int] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if scene_benchmark_id is not None:
            where.append("scene_benchmark_id = ?")
            params.append(str(scene_benchmark_id))
        if scene_profile_id is not None:
            where.append("scene_profile_id = ?")
            params.append(str(scene_profile_id))
        if scene_split is not None:
            where.append("scene_split = ?")
            params.append(str(scene_split))
        if difficulty_preset_id is not None:
            where.append("difficulty_preset_id = ?")
            params.append(str(difficulty_preset_id))
        if route_profile is not None:
            where.append("route_profile = ?")
            params.append(str(route_profile))
        if main_actor_role is not None:
            where.append("main_actor_role = ?")
            params.append(str(main_actor_role))
        for role, minimum in (required_role_counts or {}).items():
            column = ROLE_COLUMN_MAP.get(str(role))
            if column is None:
                continue
            where.append(f"{column} >= ?")
            params.append(int(minimum))
        sql = "SELECT * FROM scenes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def insert_scene(
        self,
        *,
        scene_key: str,
        scene_descriptor: dict[str, Any],
        scenario_context: dict[str, Any],
        actors_payload: dict[str, Any],
        spawn_validation: dict[str, Any] | None,
        resolved_options: dict[str, Any],
    ) -> int | None:
        if self.read_only:
            return None
        descriptor = dict(scene_descriptor)
        role_counts = descriptor.get("spawned_role_counts", {}) or {}
        now = utc_now_iso()
        actors_payload = dict(actors_payload or {})
        npc_dicts = actors_payload.pop("npcs", None)
        npc_rows = []
        if npc_dicts is not None:
            uids = []
            for ordinal, npc in enumerate(npc_dicts):
                npc = dict(npc)
                npc_uid = npc.get("npc_uid") or npc_uid_for(str(scene_key), str(npc["npc_id"]))
                npc["npc_uid"] = npc_uid
                uids.append(npc_uid)
                behavior = npc.get("behavior") or {}
                npc_rows.append(
                    {
                        "npc_uid": npc_uid,
                        "scene_key": str(scene_key),
                        "ordinal": int(ordinal),
                        "npc_id": str(npc["npc_id"]),
                        "kind": str(npc.get("kind", "vehicle")),
                        "traffic_role": npc.get("traffic_role"),
                        "behavior_type": behavior.get("type") if isinstance(behavior, dict) else None,
                        "cruise_speed_mps": npc.get("cruise_speed_mps"),
                        "route_len": len(npc.get("route_x") or []),
                        "recyclable": int(bool(npc.get("recyclable", False))),
                        "runtime_seed": npc.get("runtime_seed"),
                        "spec_json": _json_dumps(_normalize_jsonable(npc)),
                    }
                )
            actors_payload["npc_uids"] = uids
        fields = {
            "scene_key": str(scene_key),
            "generator_version": self.generator_version,
            "created_at": now,
            "last_used_at": now,
            "use_count": 1,
            "scene_kind": str(descriptor.get("scene", resolved_options.get("scene", "unknown"))),
            "scene_benchmark_id": descriptor.get("scene_benchmark_id"),
            "scene_profile_id": descriptor.get("scene_profile_id"),
            "scene_split": descriptor.get("scene_split"),
            "difficulty_preset_id": descriptor.get("difficulty_preset_id"),
            "route_extent": descriptor.get("route_extent"),
            "traffic_density": descriptor.get("traffic_density"),
            "speed_profile": descriptor.get("speed_profile"),
            "scene_seed": descriptor.get("scene_seed"),
            "route_seed": descriptor.get("route_seed"),
            "traffic_seed": descriptor.get("traffic_seed"),
            "scenario_seed": scenario_context.get("scenario_param_scenario_seed"),
            "num_vehicles": resolved_options.get("num_vehicles"),
            "num_vehicles_near_ego": descriptor.get("num_vehicles_near_ego"),
            "traffic_role_profile": descriptor.get("traffic_role_profile"),
            "guaranteed_candidate_role": descriptor.get("guaranteed_candidate_role"),
            "ego_route_graph": descriptor.get("ego_route_graph"),
            "route_profile": descriptor.get("route_profile"),
            "route_turn_count": descriptor.get("route_turn_count"),
            "route_intersection_like": int(bool(descriptor.get("route_intersection_like", False))),
            "route_length_m": descriptor.get("route_length_m"),
            "main_actor_role": descriptor.get("main_actor_role"),
            "spawned_num_vehicles": descriptor.get("spawned_num_vehicles", 0),
            "spawned_num_vehicles_near_ego": descriptor.get("spawned_num_vehicles_near_ego", 0),
            "spawned_role_lead": int(role_counts.get("lead", 0)),
            "spawned_role_rear": int(role_counts.get("rear", 0)),
            "spawned_role_cross_path": int(role_counts.get("cross_path", 0)),
            "spawned_role_opposite_lane": int(role_counts.get("opposite_lane", 0)),
            "spawned_role_background": int(role_counts.get("background", 0)),
            "scene_descriptor_json": _json_dumps(_normalize_jsonable(scene_descriptor)),
            "scenario_context_json": _json_dumps(_normalize_jsonable(scenario_context)),
            "actors_json": _json_dumps(_normalize_jsonable(actors_payload)),
            "spawn_validation_json": (
                _json_dumps(_normalize_jsonable(spawn_validation)) if spawn_validation is not None else None
            ),
            "resolved_options_json": _json_dumps(_normalize_jsonable(identity_options(resolved_options))),
        }
        columns = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        with self._connect() as conn:
            cursor = conn.execute(
                f"INSERT OR IGNORE INTO scenes ({columns}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            if cursor.rowcount and npc_rows:
                npc_columns = ", ".join(npc_rows[0].keys())
                npc_placeholders = ", ".join("?" for _ in npc_rows[0])
                conn.executemany(
                    f"INSERT OR REPLACE INTO npcs ({npc_columns}) VALUES ({npc_placeholders})",
                    [tuple(row.values()) for row in npc_rows],
                )
            row = conn.execute(
                "SELECT scene_id FROM scenes WHERE scene_key = ?",
                (scene_key,),
            ).fetchone()
            return None if row is None else int(row["scene_id"])

    def get_npcs_for_scene(self, scene_key: str) -> list[dict[str, Any]]:
        """Ordered NPC specs (dicts) of a scene."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT spec_json FROM npcs WHERE scene_key = ? ORDER BY ordinal",
                (str(scene_key),),
            ).fetchall()
        return [json.loads(row["spec_json"]) for row in rows]

    def get_npcs_by_filters(
        self,
        *,
        kind: str | None = None,
        traffic_role: str | None = None,
        behavior_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where, params = [], []
        if kind is not None:
            where.append("kind = ?")
            params.append(str(kind))
        if traffic_role is not None:
            where.append("traffic_role = ?")
            params.append(str(traffic_role))
        if behavior_type is not None:
            where.append("behavior_type = ?")
            params.append(str(behavior_type))
        sql = "SELECT * FROM npcs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY scene_key, ordinal LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["spec"] = json.loads(record.pop("spec_json"))
            records.append(record)
        return records

    def npc_counts(self, *, by: str = "traffic_role") -> dict[str, int]:
        column = {"traffic_role": "traffic_role", "kind": "kind", "behavior_type": "behavior_type"}[by]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT COALESCE(CAST({column} AS TEXT), 'null') AS label, COUNT(*) AS count FROM npcs GROUP BY label ORDER BY count DESC"
            ).fetchall()
        return {str(row["label"]): int(row["count"]) for row in rows}

    def touch_scene(self, scene_key: str) -> None:
        if self.read_only:
            return  # usage counters are writes; an immutable corpus keeps none
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE scenes
                SET use_count = use_count + 1,
                    last_used_at = ?
                WHERE scene_key = ?
                """,
                (utc_now_iso(), scene_key),
            )

    def delete_scene(self, scene_key: str) -> None:
        if self.read_only:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM scenes WHERE scene_key = ?", (scene_key,))
            conn.execute("DELETE FROM npcs WHERE scene_key = ?", (scene_key,))

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["scene_descriptor"] = json.loads(record.pop("scene_descriptor_json"))
        record["scenario_context"] = json.loads(record.pop("scenario_context_json"))
        record["actors_payload"] = json.loads(record.pop("actors_json"))
        payload = record["actors_payload"]
        if isinstance(payload, dict) and "npcs" not in payload and payload.get("npc_uids") is not None:
            payload["npcs"] = self.get_npcs_for_scene(str(record["scene_key"]))
        record["resolved_options"] = json.loads(record.pop("resolved_options_json"))
        spawn_json = record.pop("spawn_validation_json")
        record["spawn_validation"] = json.loads(spawn_json) if spawn_json is not None else None
        return record
