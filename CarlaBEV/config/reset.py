from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from CarlaBEV.src.scenes.scenarios.specs import (
    build_runtime_scenario_options,
    build_scenario_options_from_config as _build_scenario_options_from_config,
)


class RandomNavigationReset(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    num_vehicles: int = 25
    route_dist_range: tuple[int, int] = (30, 130)


class ScenarioPresetReset(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    preset_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class AuthoredSceneReset(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    config_file: str
    variation_enabled: bool = False
    variation_seed: int | None = None


class ScenarioConfigReset(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    scenario_id: str
    level: int = 1
    anchor_x: int | None = None
    anchor_y: int | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


ResetRequest = (
    RandomNavigationReset
    | ScenarioPresetReset
    | AuthoredSceneReset
    | ScenarioConfigReset
)


def _attach_reset_mask(options: dict[str, Any], reset_mask):
    if reset_mask is not None:
        options["reset_mask"] = np.asarray(reset_mask, dtype=bool)
    return options


def build_random_navigation_options(
    request: RandomNavigationReset,
    *,
    reset_mask=None,
) -> dict[str, Any]:
    options = {
        "scene": "rdm",
        "num_vehicles": int(request.num_vehicles),
        "route_dist_range": list(request.route_dist_range),
    }
    return _attach_reset_mask(options, reset_mask)


def build_scenario_preset_options(
    request: ScenarioPresetReset,
    *,
    reset_mask=None,
) -> dict[str, Any]:
    return build_runtime_scenario_options(
        request.preset_id,
        reset_mask=reset_mask,
        overrides=request.overrides,
    )


def build_authored_scene_options(
    request: AuthoredSceneReset,
    *,
    reset_mask=None,
) -> dict[str, Any]:
    options = {
        "config_file": request.config_file,
        "variation_enabled": request.variation_enabled,
    }
    if request.variation_seed is not None:
        options["variation_seed"] = int(request.variation_seed)
    return _attach_reset_mask(options, reset_mask)


def build_scenario_config_options(
    request: ScenarioConfigReset,
    *,
    reset_mask=None,
) -> dict[str, Any]:
    options = dict(request.parameters)
    options["scene"] = request.scenario_id
    options["level"] = int(request.level)
    if request.anchor_x is not None:
        options["anchor_x"] = int(request.anchor_x)
    if request.anchor_y is not None:
        options["anchor_y"] = int(request.anchor_y)
    return _attach_reset_mask(options, reset_mask)


def build_scenario_options_from_config(
    config: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
    reset_mask=None,
) -> dict[str, Any]:
    return _build_scenario_options_from_config(
        config,
        overrides=overrides,
        reset_mask=reset_mask,
    )


def build_reset_options(request: ResetRequest, *, reset_mask=None) -> dict[str, Any]:
    if isinstance(request, RandomNavigationReset):
        return build_random_navigation_options(request, reset_mask=reset_mask)
    if isinstance(request, ScenarioPresetReset):
        return build_scenario_preset_options(request, reset_mask=reset_mask)
    if isinstance(request, AuthoredSceneReset):
        return build_authored_scene_options(request, reset_mask=reset_mask)
    if isinstance(request, ScenarioConfigReset):
        return build_scenario_config_options(request, reset_mask=reset_mask)
    raise TypeError(f"Unsupported reset request type: {type(request)!r}")
