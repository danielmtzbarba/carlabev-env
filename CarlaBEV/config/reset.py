from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from CarlaBEV.config.difficulty import get_difficulty_spec
from CarlaBEV.src.scenes.scenarios.specs import (
    build_runtime_scenario_options,
    build_scenario_options_from_config as _build_scenario_options_from_config,
)


class RandomNavigationReset(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    difficulty_id: str | None = None
    num_vehicles: int = 25
    route_dist_range: tuple[int, int] = (30, 130)
    ego_route_graph: str = "full_vehicle"
    route_profile: str | None = None
    route_profile_mix: dict[str, float] | None = None
    min_turns: int | None = None
    max_turns: int | None = None
    intersection_required: bool | None = None
    max_route_attempts: int | None = None
    scene_seed: int | None = None
    route_seed: int | None = None
    traffic_seed: int | None = None
    scenario_seed: int | None = None


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
        "ego_route_graph": request.ego_route_graph,
    }
    if request.route_profile is not None:
        options["route_profile"] = request.route_profile
    if request.route_profile_mix is not None:
        options["route_profile_mix"] = dict(request.route_profile_mix)
    if request.min_turns is not None:
        options["min_turns"] = int(request.min_turns)
    if request.max_turns is not None:
        options["max_turns"] = int(request.max_turns)
    if request.intersection_required is not None:
        options["intersection_required"] = bool(request.intersection_required)
    if request.max_route_attempts is not None:
        options["max_route_attempts"] = int(request.max_route_attempts)
    if request.scene_seed is not None:
        options["scene_seed"] = int(request.scene_seed)
    if request.route_seed is not None:
        options["route_seed"] = int(request.route_seed)
    if request.traffic_seed is not None:
        options["traffic_seed"] = int(request.traffic_seed)
    if request.scenario_seed is not None:
        options["scenario_seed"] = int(request.scenario_seed)
    if request.difficulty_id is not None:
        spec = get_difficulty_spec(request.difficulty_id)
        options.update(
            {
                "difficulty_id": spec.difficulty_id,
                "traffic_enabled": spec.traffic_enabled,
                "num_vehicles": int(spec.num_vehicles),
                "route_dist_range": list(spec.route_dist_range),
            }
        )
        if spec.ego_target_speed is not None:
            options["ego_target_speed"] = float(spec.ego_target_speed)
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
