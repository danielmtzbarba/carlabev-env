from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from CarlaBEV.envs.utils import asset_path
from CarlaBEV.src.scenes.scenarios.specs import (
    list_scenario_ids,
    list_scenario_preset_ids,
)


ObsMode = Literal["bev_rgb", "bev_semantic", "vector"]
SemanticMaskCh = Literal["binary", "2-class", "4-class", "5-class", "6-class", "7-class"]
TemporalFusionMode = Literal["stack", "vehicle_temporal", "vehicle_weighted"]
ActionMode = Literal["discrete", "continuous"]
RewardMode = Literal["shaping", "carl"]
RenderMode = Literal["human", "rgb_array"]


class EnvConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    seed: int = 0
    fps: int = 15
    size: int = 128
    env_id: str = "CarlaBEV-v0"
    map_name: str = "Town01"
    obs_size: tuple[int, int] = (96, 96)
    obs_mode: ObsMode = "bev_semantic"
    semantic_mask_ch: SemanticMaskCh = "6-class"
    temporal_fusion_mode: TemporalFusionMode = "stack"
    fov_masked: bool = False
    ego_anchor_x_frac: float = 0.5
    ego_anchor_y_frac: float = 0.5
    frame_stack: int = 4

    action_mode: ActionMode = "discrete"
    render_mode: RenderMode = "human"
    max_actions: int = 5000
    scenes_path: str = "assets/scenes"
    reward_mode: RewardMode = "carl"

    traffic_enabled: bool = True
    max_vehicles: int = 50
    route_direction_metrics_enabled: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: Any):
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        if "obs_mode" not in normalized:
            obs_space = normalized.get("obs_space")
            masked = normalized.get("masked")
            if obs_space == "vector":
                normalized["obs_mode"] = "vector"
            elif masked is False:
                normalized["obs_mode"] = "bev_rgb"
            else:
                normalized["obs_mode"] = "bev_semantic"

        if "action_mode" not in normalized and "action_space" in normalized:
            normalized["action_mode"] = normalized["action_space"]

        if "reward_mode" not in normalized and "reward_type" in normalized:
            normalized["reward_mode"] = (
                "carl" if normalized["reward_type"] == "carl" else "shaping"
            )

        return normalized

    @model_validator(mode="after")
    def _validate_values(self):
        if self.frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        if self.temporal_fusion_mode != "stack":
            if self.obs_mode != "bev_semantic":
                raise ValueError("temporal_fusion_mode requires obs_mode='bev_semantic'")
            if self.frame_stack < 3:
                raise ValueError("temporal_fusion_mode requires frame_stack >= 3")
            if self.semantic_mask_ch not in {"4-class", "5-class", "6-class", "7-class"}:
                raise ValueError(
                    "temporal_fusion_mode requires a semantic_mask_ch with a vehicle channel "
                    "(one of: '4-class', '5-class', '6-class', '7-class')"
                )
        if self.obs_size[0] < 1 or self.obs_size[1] < 1:
            raise ValueError("obs_size dimensions must be >= 1")
        if not 0.0 <= self.ego_anchor_x_frac <= 1.0:
            raise ValueError("ego_anchor_x_frac must be within [0.0, 1.0]")
        if not 0.0 <= self.ego_anchor_y_frac <= 1.0:
            raise ValueError("ego_anchor_y_frac must be within [0.0, 1.0]")

        map_dir = Path(asset_path) / self.map_name
        sem_path = map_dir / f"{self.map_name}-{self.size}-sem.png"
        rgb_path = map_dir / f"{self.map_name}-{self.size}-rgb.png"
        planning_path = map_dir / f"{self.map_name}-1024-sem.png"
        missing = [
            str(path)
            for path in (sem_path, rgb_path, planning_path)
            if not path.exists()
        ]
        if missing:
            raise ValueError(
                f"map_name='{self.map_name}' is missing required assets: {missing}"
            )
        return self

    @computed_field(return_type=str)
    @property
    def obs_space(self) -> str:
        return "vector" if self.obs_mode == "vector" else "bev"

    @computed_field(return_type=bool)
    @property
    def masked(self) -> bool:
        return self.obs_mode == "bev_semantic"

    @computed_field(return_type=str)
    @property
    def action_space(self) -> str:
        return self.action_mode

    @computed_field(return_type=str)
    @property
    def reward_type(self) -> str:
        return "carl" if self.reward_mode == "carl" else "shaping"


class RunConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    env: EnvConfig = Field(default_factory=EnvConfig)
    exp_name: str = "carlabev-run"
    num_envs: int = 1
    seed: int = 1
    capture_video: bool = False
    capture_every: int = 50
    cuda: bool = True
    torch_deterministic: bool = True

    @model_validator(mode="after")
    def _validate_values(self):
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        return self


def _to_mapping(value: Any):
    if isinstance(value, (EnvConfig, RunConfig)):
        return value
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _to_env_mapping(value: Any):
    raw = _to_mapping(value)
    if isinstance(raw, EnvConfig):
        return raw
    if not isinstance(raw, dict):
        return raw
    env = {key: val for key, val in raw.items() if key in {
        "seed",
        "fps",
        "size",
        "env_id",
        "map_name",
        "obs_size",
        "semantic_mask_ch",
        "temporal_fusion_mode",
        "fov_masked",
        "ego_anchor_x_frac",
        "ego_anchor_y_frac",
        "frame_stack",
        "render_mode",
        "max_actions",
        "scenes_path",
        "traffic_enabled",
        "max_vehicles",
        "route_direction_metrics_enabled",
    }}
    if "obs_mode" in raw:
        env["obs_mode"] = raw["obs_mode"]
    elif raw.get("obs_space") == "vector":
        env["obs_mode"] = "vector"
    elif raw.get("masked") is False:
        env["obs_mode"] = "bev_rgb"
    else:
        env["obs_mode"] = "bev_semantic"

    if "action_mode" in raw:
        env["action_mode"] = raw["action_mode"]
    elif "action_space" in raw:
        env["action_mode"] = raw["action_space"]

    if "reward_mode" in raw:
        env["reward_mode"] = raw["reward_mode"]
    elif "reward_type" in raw:
        env["reward_mode"] = "carl" if raw["reward_type"] == "carl" else "shaping"
    return env


def _to_run_mapping(value: Any):
    raw = _to_mapping(value)
    if isinstance(raw, RunConfig):
        return raw
    if not isinstance(raw, dict):
        return raw
    mapping = {}
    if "env" in raw:
        mapping["env"] = _to_env_mapping(raw["env"])
    allowed = {
        "exp_name",
        "num_envs",
        "seed",
        "capture_video",
        "capture_every",
        "cuda",
        "torch_deterministic",
    }
    for key, val in raw.items():
        if key in allowed:
            mapping[key] = val
    return mapping


def validate_env_config(cfg: EnvConfig | dict[str, Any]) -> EnvConfig:
    if isinstance(cfg, EnvConfig):
        return cfg
    return EnvConfig.model_validate(_to_env_mapping(cfg))


def validate_run_config(cfg: RunConfig | dict[str, Any]) -> RunConfig:
    run_cfg = cfg if isinstance(cfg, RunConfig) else RunConfig.model_validate(_to_run_mapping(cfg))
    if run_cfg.env.obs_mode == "vector":
        raise ValueError(
            "obs_mode='vector' is not supported through make_env()/wrap_env() yet. "
            "Construct CarlaBEV(env_cfg) directly for roadmap-only vector usage."
        )
    return run_cfg


def get_env_capabilities() -> dict[str, object]:
    maps = []
    for path in Path(asset_path).iterdir():
        if not path.is_dir():
            continue
        has_semantic = any(path.glob(f"{path.name}-*-sem.png"))
        has_rgb = any(path.glob(f"{path.name}-*-rgb.png"))
        if has_semantic and has_rgb:
            maps.append(path.name)
    return {
        "maps": sorted(maps),
        "action_modes": ["discrete", "continuous"],
        "obs_modes": ["bev_rgb", "bev_semantic", "vector"],
        "semantic_mask_ch": ["binary", "2-class", "4-class", "5-class", "6-class", "7-class"],
        "temporal_fusion_mode": ["stack", "vehicle_temporal", "vehicle_weighted"],
        "reward_modes": ["shaping", "carl"],
        "scenario_ids": list_scenario_ids(),
        "scenario_preset_ids": list_scenario_preset_ids(),
        "supports_vector_make_env": False,
    }
