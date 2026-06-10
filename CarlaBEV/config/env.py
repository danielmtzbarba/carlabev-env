from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from CarlaBEV.config.action_profiles import (
    get_action_profile_spec,
    list_action_profile_ids,
)
from CarlaBEV.config.difficulty import list_difficulty_ids
from CarlaBEV.config.reward_profiles import (
    get_reward_profile_spec,
    list_reward_profile_ids,
)
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


LEGACY_ACTION_PROFILE_IDS: dict[str, str] = {
    "discrete": "discrete9_v1",
    "continuous": "continuous_gsb_v1",
}
LEGACY_REWARD_PROFILE_IDS: dict[str, str] = {
    "carl": "carl_base_v1",
    "shaping": "shaping_base_v1",
}


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
    action_profile_id: str | None = None
    render_mode: RenderMode = "human"
    max_actions: int = 5000
    scenes_path: str = "assets/scenes"
    reward_mode: RewardMode = "carl"
    reward_profile_id: str | None = None

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

        action_profile_id = normalized.get("action_profile_id")
        if action_profile_id is None:
            action_mode = normalized.get("action_mode", "discrete")
            normalized["action_profile_id"] = LEGACY_ACTION_PROFILE_IDS.get(action_mode, "discrete9_v1")

        reward_profile_id = normalized.get("reward_profile_id")
        if reward_profile_id is None:
            reward_mode = normalized.get("reward_mode", "carl")
            normalized["reward_profile_id"] = LEGACY_REWARD_PROFILE_IDS.get(reward_mode, "carl_base_v1")

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

        action_spec = get_action_profile_spec(self.action_profile_id)
        reward_spec = get_reward_profile_spec(self.reward_profile_id)
        if action_spec.action_mode != self.action_mode:
            raise ValueError(
                f"action_profile_id={self.action_profile_id!r} resolves to action_mode={action_spec.action_mode!r}, "
                f"but EnvConfig.action_mode={self.action_mode!r}"
            )
        if reward_spec.family != self.reward_mode:
            raise ValueError(
                f"reward_profile_id={self.reward_profile_id!r} resolves to reward_mode={reward_spec.family!r}, "
                f"but EnvConfig.reward_mode={self.reward_mode!r}"
            )

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
    video_output_dir: str | None = None
    video_episode_indices: list[int] | None = None
    video_name_prefix: str = "rl-video"
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
        "action_profile_id",
        "reward_profile_id",
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
        "video_output_dir",
        "video_episode_indices",
        "video_name_prefix",
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
            "Use CarlaBEV() directly if you need vector observations."
        )
    return run_cfg


def resolve_env_profiles(env_cfg: EnvConfig | dict[str, Any]) -> dict[str, Any]:
    cfg = validate_env_config(env_cfg)
    action_spec = get_action_profile_spec(cfg.action_profile_id)
    reward_spec = get_reward_profile_spec(cfg.reward_profile_id)
    return {
        "action": action_spec.model_dump(mode="python"),
        "reward": reward_spec.model_dump(mode="python"),
    }


def get_env_capabilities() -> dict[str, Any]:
    maps_root = Path(asset_path)
    maps = sorted(
        path.name
        for path in maps_root.iterdir()
        if path.is_dir() and (path / f"{path.name}-1024-sem.png").exists()
    )
    semantic_mask_channels = ["binary", "2-class", "4-class", "5-class", "6-class", "7-class"]
    temporal_fusion_modes = ["stack", "vehicle_temporal", "vehicle_weighted"]
    return {
        "maps": maps,
        "obs_modes": ["bev_rgb", "bev_semantic", "vector"],
        "semantic_mask_channels": semantic_mask_channels,
        "semantic_mask_ch": semantic_mask_channels,
        "temporal_fusion_modes": temporal_fusion_modes,
        "temporal_fusion_mode": temporal_fusion_modes,
        "action_modes": ["discrete", "continuous"],
        "action_profile_ids": list_action_profile_ids(),
        "reward_modes": ["shaping", "carl"],
        "reward_profile_ids": list_reward_profile_ids(),
        "difficulty_ids": list_difficulty_ids(),
        "render_modes": ["human", "rgb_array"],
        "supports_vector_make_env": False,
        "scenario_ids": list_scenario_ids(),
        "scenario_preset_ids": list_scenario_preset_ids(),
    }
