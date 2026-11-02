"""Versioned inputs to generation and frozen-spawn acceptance.

Asset contents are fingerprinted once when an environment loads its map and
planners. Assets, configuration and unit contracts are fixed for that environment's
lifetime. Presentation, reward, episode duration and per-reset variation do not
determine the frozen scene and are deliberately excluded.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


IDENTITY_VERSION = 2
NON_IDENTITY_OPTION_KEYS = frozenset(
    {"reset_mask", "variation_seed", "variation_profile"}
)


def normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalize_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return normalize_jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


class GenerationSettings(BaseModel):
    """EnvConfig fields read during generation or initial spawn validation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    map_name: str = Field(min_length=1)
    size: int = Field(gt=0)
    geometry_profile: Literal["legacy", "realistic"]
    traffic_enabled: bool
    physics_dt: float = Field(gt=0)
    npc_route_geometry: Literal["legacy", "fillet_v1", "fillet_v2"]
    npc_fillet_target_radius_m: float = Field(gt=0)
    npc_curve_lat_accel_mps2: float = Field(gt=0)
    map_collision_mode: Literal["center", "footprint"]
    map_contact_inset_m: float = Field(ge=0)
    map_offroad_footprint_fraction: float = Field(gt=0, le=1)


class SceneGenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    environment: GenerationSettings
    resolved_units: dict[str, JsonValue]
    generator_defaults: dict[str, JsonValue]
    asset_sha256: dict[str, str]

    @classmethod
    def from_environment(cls, cfg, units, generator, *, assets_root=None):
        from CarlaBEV.config import EnvConfig
        from CarlaBEV.envs.utils import asset_path

        settings = GenerationSettings.model_validate(
            {
                name: getattr(cfg, name, EnvConfig.model_fields[name].default)
                for name in GenerationSettings.model_fields
            }
        )
        root = Path(asset_path if assets_root is None else assets_root)
        town = settings.map_name
        prefix = town.lower()
        # These are the rasters and graphs actually loaded by BaseMap,
        # SceneGenerator and PlannerManager. RGB artwork is presentation only.
        names = (
            f"{town}/{town}-{settings.size}-sem.png",
            f"{town}/{town}-1024-sem.png",
            f"{town}/{prefix}-canonical-vehicle.pkl",
            f"{town}/{prefix}.pkl",
        )
        return cls(
            environment=settings,
            resolved_units=normalize_jsonable(
                {
                    "metric": asdict(units.metric),
                    "footprints": asdict(units.footprints),
                    "npc_motion": asdict(units.npc_motion),
                    "ego_wheelbase_m": units.ego.wheelbase_m,
                }
            ),
            generator_defaults=normalize_jsonable(
                {
                    name: generator.cfg.get(name, default)
                    for name, default in (
                        ("route_dist_range", [30, 100]),
                        ("ego_target_speed", 12.0),
                        ("vehicle_target_speed", 12.0),
                    )
                }
            ),
            asset_sha256={
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in names
            },
        )


class SceneGenerationIdentity(BaseModel):
    """The same validated request is hashed, persisted and checked on cache loads."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    identity_version: Literal[2]
    generator_version: str = Field(min_length=1)
    config: SceneGenerationConfig
    options: dict[str, JsonValue]
    seeds: dict[str, int | None]

    @classmethod
    def from_request(cls, *, generator_version, config, options, seeds):
        return cls(
            identity_version=IDENTITY_VERSION,
            generator_version=generator_version,
            config=config,
            options=normalize_jsonable(
                {k: v for k, v in options.items() if k not in NON_IDENTITY_OPTION_KEYS}
            ),
            seeds=normalize_jsonable(seeds),
        )

    def to_payload(self):
        return self.model_dump(mode="json")

    def key(self) -> str:
        payload = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
