from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle


ROLE_COLUMN_MAP = {
    "lead": "spawned_role_lead",
    "rear": "spawned_role_rear",
    "cross_path": "spawned_role_cross_path",
    "opposite_lane": "spawned_role_opposite_lane",
    "background": "spawned_role_background",
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


def serialize_scene_payload(actors: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agent": None,
        "vehicle": [],
        "pedestrian": [],
        "target": [],
        "traffic_light": [],
    }
    agent = actors.get("agent")
    if agent is not None:
        route_x, route_y, initial_speed_mps, target_speed_mps = agent
        payload["agent"] = {
            "route_x": [float(x) for x in route_x],
            "route_y": [float(y) for y in route_y],
            "initial_speed_mps": float(initial_speed_mps),
            "target_speed_mps": float(target_speed_mps),
        }

    for key, actor_cls in (("vehicle", Vehicle), ("pedestrian", Pedestrian)):
        serialized = []
        for actor in actors.get(key, []):
            serialized.append(
                {
                    "route_x": [float(x) for x in getattr(actor, "rx", [])],
                    "route_y": [float(y) for y in getattr(actor, "ry", [])],
                    "traffic_role": str(getattr(actor, "traffic_role", "unassigned")),
                    "speed_class": str(getattr(actor, "speed_class", "medium")),
                    "speed_multiplier": float(getattr(actor, "speed_multiplier", 1.0)),
                    "speed_jitter": float(getattr(actor, "speed_jitter", 1.0)),
                    "initial_speed_mps": float(getattr(actor, "initial_speed_mps", 0.0) or 0.0),
                    "cruise_speed_mps": float(getattr(actor, "cruise_speed_mps", 0.0) or 0.0),
                    "startup_ramp_duration_s": float(getattr(actor, "startup_ramp_duration_s", 0.0) or 0.0),
                }
            )
        payload[key] = serialized
    return payload


def deserialize_scene_payload(payload: dict[str, Any], *, map_size: int) -> dict[str, Any]:
    actors = {
        "agent": None,
        "vehicle": [],
        "pedestrian": [],
        "target": [],
        "traffic_light": [],
    }
    agent = payload.get("agent")
    if agent is not None:
        actors["agent"] = (
            [float(x) for x in agent["route_x"]],
            [float(y) for y in agent["route_y"]],
            float(agent["initial_speed_mps"]),
            float(agent["target_speed_mps"]),
        )

    for key, actor_cls in (("vehicle", Vehicle), ("pedestrian", Pedestrian)):
        for actor_payload in payload.get(key, []):
            actor = actor_cls(
                map_size=map_size,
                routeX=[float(x) for x in actor_payload["route_x"]],
                routeY=[float(y) for y in actor_payload["route_y"]],
                target_speed=float(actor_payload.get("cruise_speed_mps", 0.0)),
            )
            actor.traffic_role = str(actor_payload.get("traffic_role", "unassigned"))
            actor.speed_class = str(actor_payload.get("speed_class", "medium"))
            actor.speed_multiplier = float(actor_payload.get("speed_multiplier", 1.0))
            actor.speed_jitter = float(actor_payload.get("speed_jitter", 1.0))
            actor.set_initial_speed_mps(float(actor_payload.get("initial_speed_mps", 0.0)))
            actor.set_startup_ramp_duration(float(actor_payload.get("startup_ramp_duration_s", 0.0)))
            actors[key].append(actor)
    return actors


class SceneLibrary:
    def __init__(self, db_path: str, *, generator_version: str, read_only: bool = False) -> None:
        self.db_path = str(db_path)
        self.generator_version = str(generator_version)
        self.read_only = bool(read_only)
        self._ensure_parent_dir()
        self.ensure_schema()

    def _ensure_parent_dir(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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

                CREATE INDEX IF NOT EXISTS idx_scenes_route_profile ON scenes(route_profile);
                CREATE INDEX IF NOT EXISTS idx_scenes_main_actor_role ON scenes(main_actor_role);
                CREATE INDEX IF NOT EXISTS idx_scenes_difficulty ON scenes(difficulty_preset_id);
                CREATE INDEX IF NOT EXISTS idx_scenes_seed_triplet ON scenes(scene_seed, route_seed, traffic_seed);
                CREATE INDEX IF NOT EXISTS idx_scenes_role_route ON scenes(main_actor_role, route_profile);
                """
            )

    def build_scene_key(self, resolved_options: dict[str, Any], seeds: dict[str, Any]) -> str:
        payload = {
            "generator_version": self.generator_version,
            "options": _normalize_jsonable(
                {
                    key: value
                    for key, value in resolved_options.items()
                    if key != "reset_mask"
                }
            ),
            "seeds": _normalize_jsonable(seeds),
        }
        return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()

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
        difficulty_preset_id: str | None = None,
        route_profile: str | None = None,
        main_actor_role: str | None = None,
        required_role_counts: dict[str, int] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
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
        fields = {
            "scene_key": str(scene_key),
            "generator_version": self.generator_version,
            "created_at": now,
            "last_used_at": now,
            "use_count": 1,
            "scene_kind": str(descriptor.get("scene", resolved_options.get("scene", "unknown"))),
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
            "resolved_options_json": _json_dumps(
                _normalize_jsonable(
                    {
                        key: value
                        for key, value in resolved_options.items()
                        if key != "reset_mask"
                    }
                )
            ),
        }
        columns = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO scenes ({columns}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            row = conn.execute(
                "SELECT scene_id FROM scenes WHERE scene_key = ?",
                (scene_key,),
            ).fetchone()
            return None if row is None else int(row["scene_id"])

    def touch_scene(self, scene_key: str) -> None:
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

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["scene_descriptor"] = json.loads(record.pop("scene_descriptor_json"))
        record["scenario_context"] = json.loads(record.pop("scenario_context_json"))
        record["actors_payload"] = json.loads(record.pop("actors_json"))
        record["resolved_options"] = json.loads(record.pop("resolved_options_json"))
        spawn_json = record.pop("spawn_validation_json")
        record["spawn_validation"] = json.loads(spawn_json) if spawn_json is not None else None
        return record
