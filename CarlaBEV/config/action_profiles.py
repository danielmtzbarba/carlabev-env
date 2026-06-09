from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ActionMode = Literal["discrete", "continuous"]


class ActionProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    action_profile_id: str
    action_mode: ActionMode
    discrete_actions: list[tuple[float, float, float]] = Field(default_factory=list)
    low: tuple[float, float, float] | None = None
    high: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def _validate_mode_payload(self):
        if self.action_mode == "discrete":
            if not self.discrete_actions:
                raise ValueError("discrete action profiles require discrete_actions")
            if self.low is not None or self.high is not None:
                raise ValueError("discrete action profiles cannot define low/high bounds")
        else:
            if self.low is None or self.high is None:
                raise ValueError("continuous action profiles require low/high bounds")
            if self.discrete_actions:
                raise ValueError("continuous action profiles cannot define discrete_actions")
        return self


ACTION_PROFILE_PRESETS: dict[str, ActionProfileSpec] = {
    "discrete9_v1": ActionProfileSpec(
        action_profile_id="discrete9_v1",
        action_mode="discrete",
        discrete_actions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, -1.0, 1.0),
        ],
    ),
    "discrete13_v1": ActionProfileSpec(
        action_profile_id="discrete13_v1",
        action_mode="discrete",
        discrete_actions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.5, 0.0),
            (1.0, -0.5, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, -1.0, 1.0),
        ],
    ),
    "continuous_gsb_v1": ActionProfileSpec(
        action_profile_id="continuous_gsb_v1",
        action_mode="continuous",
        low=(0.0, -1.0, 0.0),
        high=(1.0, 1.0, 1.0),
    ),
}


def get_action_profile_spec(action_profile_id: str) -> ActionProfileSpec:
    try:
        return ACTION_PROFILE_PRESETS[action_profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(ACTION_PROFILE_PRESETS))
        raise KeyError(
            f"Unknown action_profile_id={action_profile_id!r}. Available action profiles: {available}"
        ) from exc


def list_action_profile_ids() -> list[str]:
    return sorted(ACTION_PROFILE_PRESETS)
