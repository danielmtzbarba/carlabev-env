from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from CarlaBEV.config.difficulty import (
    RouteExtentId,
    SpeedProfileId,
    TrafficDensityId,
    get_difficulty_spec,
    resolve_random_navigation_spec,
)
from CarlaBEV.src.scenes.scenarios.specs import (
    build_runtime_scenario_options,
    build_scenario_options_from_config as _build_scenario_options_from_config,
)


class RandomNavigationReset(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    difficulty_preset_id: str | None = None
    route_extent: RouteExtentId | None = None
    traffic_density: TrafficDensityId | None = None
    speed_profile: SpeedProfileId | None = None
    num_vehicles: int | None = None
    route_dist_range: tuple[int, int] | None = None
    ego_target_speed: float | None = None
    vehicle_target_speed: float | None = None
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
    resolved = None
    difficulty_preset_id = request.difficulty_preset_id
    if difficulty_preset_id is not None:
        spec = get_difficulty_spec(difficulty_preset_id)
        resolved = resolve_random_navigation_spec(
            route_extent=(
                request.route_extent if request.route_extent is not None else spec.route_extent
            ),
            traffic_density=(
                request.traffic_density
                if request.traffic_density is not None
                else spec.traffic_density
            ),
            speed_profile=(
                request.speed_profile if request.speed_profile is not None else spec.speed_profile
            ),
        )
    elif (
        request.route_extent is not None
        or request.traffic_density is not None
        or request.speed_profile is not None
    ):
        resolved = resolve_random_navigation_spec(
            route_extent=(
                RouteExtentId.MEDIUM if request.route_extent is None else request.route_extent
            ),
            traffic_density=(
                TrafficDensityId.MEDIUM
                if request.traffic_density is None
                else request.traffic_density
            ),
            speed_profile=(
                SpeedProfileId.MEDIUM if request.speed_profile is None else request.speed_profile
            ),
        )

    options = {
        "scene": "rdm",
        "ego_route_graph": request.ego_route_graph,
    }
    if difficulty_preset_id is not None:
        options["difficulty_preset_id"] = get_difficulty_spec(
            difficulty_preset_id
        ).difficulty_preset_id
    if request.route_extent is not None or resolved is not None:
        options["route_extent"] = str(
            request.route_extent if request.route_extent is not None else resolved.route_extent
        )
    if request.traffic_density is not None or resolved is not None:
        options["traffic_density"] = str(
            request.traffic_density
            if request.traffic_density is not None
            else resolved.traffic_density
        )
    if request.speed_profile is not None or resolved is not None:
        options["speed_profile"] = str(
            request.speed_profile if request.speed_profile is not None else resolved.speed_profile
        )

    route_dist_range = (
        tuple(request.route_dist_range)
        if request.route_dist_range is not None
        else (resolved.route_dist_range if resolved is not None else (30, 130))
    )
    options["route_dist_range"] = list(route_dist_range)

    num_vehicles = (
        int(request.num_vehicles)
        if request.num_vehicles is not None
        else (int(resolved.num_vehicles) if resolved is not None else 16)
    )
    options["num_vehicles"] = num_vehicles

    traffic_enabled = (
        bool(num_vehicles > 0)
        if request.traffic_density is None and resolved is None
        else bool(
            resolved.traffic_enabled if request.traffic_density is None else request.traffic_density != TrafficDensityId.NONE
        )
    )
    options["traffic_enabled"] = traffic_enabled

    ego_target_speed = (
        float(request.ego_target_speed)
        if request.ego_target_speed is not None
        else (float(resolved.ego_target_speed) if resolved is not None else None)
    )
    if ego_target_speed is not None:
        options["ego_target_speed"] = ego_target_speed

    vehicle_target_speed = (
        float(request.vehicle_target_speed)
        if request.vehicle_target_speed is not None
        else (float(resolved.vehicle_target_speed) if resolved is not None else None)
    )
    if vehicle_target_speed is not None:
        options["vehicle_target_speed"] = vehicle_target_speed

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
