from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RewardFamily = Literal["shaping", "carl"]


class RewardProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    reward_profile_id: str
    family: RewardFamily
    parameters: dict[str, Any] = Field(default_factory=dict)


REWARD_PROFILE_PRESETS: dict[str, RewardProfileSpec] = {
    "carl_base_v1": RewardProfileSpec(
        reward_profile_id="carl_base_v1",
        family="carl",
        parameters={},
    ),
    "carl_safety_v1": RewardProfileSpec(
        reward_profile_id="carl_safety_v1",
        family="carl",
        parameters={
            "lane_center_exponent": 1.5,
            "lane_center_floor": 0.15,
            "off_lane_penalty": 0.05,
            "speed_penalty_scale": 4.0,
            "speed_penalty_floor": 0.05,
            "ttc_threshold": 5.0,
            "ttc_penalty_floor": 0.05,
            "reward_scale": 0.85,
            "comfort_penalty_floor": 0.25,
        },
    ),
    "shaping_base_v1": RewardProfileSpec(
        reward_profile_id="shaping_base_v1",
        family="shaping",
        parameters={},
    ),
}


def get_reward_profile_spec(reward_profile_id: str) -> RewardProfileSpec:
    try:
        return REWARD_PROFILE_PRESETS[reward_profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(REWARD_PROFILE_PRESETS))
        raise KeyError(
            f"Unknown reward_profile_id={reward_profile_id!r}. Available reward profiles: {available}"
        ) from exc


def list_reward_profile_ids() -> list[str]:
    return sorted(REWARD_PROFILE_PRESETS)
