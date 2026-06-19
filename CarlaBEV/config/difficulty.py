from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


DifficultyFamily = Literal["random_navigation"]


class RouteExtentId(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LARGE = "large"


class TrafficDensityId(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    DENSE = "dense"


class SpeedProfileId(StrEnum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class RouteExtentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    route_extent: RouteExtentId
    route_dist_range: tuple[int, int]


class TrafficDensitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    traffic_density: TrafficDensityId
    traffic_enabled: bool
    num_vehicles: int


class SpeedProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    speed_profile: SpeedProfileId
    ego_target_speed: float
    vehicle_target_speed: float


class RandomNavigationResolvedSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    route_extent: RouteExtentId
    traffic_density: TrafficDensityId
    speed_profile: SpeedProfileId
    route_dist_range: tuple[int, int]
    traffic_enabled: bool
    num_vehicles: int
    ego_target_speed: float
    vehicle_target_speed: float


class RandomTrafficDifficultySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    difficulty_preset_id: str
    family: DifficultyFamily = "random_navigation"
    route_extent: RouteExtentId
    traffic_density: TrafficDensityId
    speed_profile: SpeedProfileId


ROUTE_EXTENT_PRESETS: dict[RouteExtentId, RouteExtentSpec] = {
    RouteExtentId.SHORT: RouteExtentSpec(
        route_extent=RouteExtentId.SHORT,
        route_dist_range=(30, 80),
    ),
    RouteExtentId.MEDIUM: RouteExtentSpec(
        route_extent=RouteExtentId.MEDIUM,
        route_dist_range=(40, 100),
    ),
    RouteExtentId.LARGE: RouteExtentSpec(
        route_extent=RouteExtentId.LARGE,
        route_dist_range=(50, 130),
    ),
}


TRAFFIC_DENSITY_PRESETS: dict[TrafficDensityId, TrafficDensitySpec] = {
    TrafficDensityId.NONE: TrafficDensitySpec(
        traffic_density=TrafficDensityId.NONE,
        traffic_enabled=False,
        num_vehicles=0,
    ),
    TrafficDensityId.LOW: TrafficDensitySpec(
        traffic_density=TrafficDensityId.LOW,
        traffic_enabled=True,
        num_vehicles=8,
    ),
    TrafficDensityId.MEDIUM: TrafficDensitySpec(
        traffic_density=TrafficDensityId.MEDIUM,
        traffic_enabled=True,
        num_vehicles=16,
    ),
    TrafficDensityId.DENSE: TrafficDensitySpec(
        traffic_density=TrafficDensityId.DENSE,
        traffic_enabled=True,
        num_vehicles=25,
    ),
}


SPEED_PROFILE_PRESETS: dict[SpeedProfileId, SpeedProfileSpec] = {
    SpeedProfileId.SLOW: SpeedProfileSpec(
        speed_profile=SpeedProfileId.SLOW,
        ego_target_speed=8.0,
        vehicle_target_speed=8.0,
    ),
    SpeedProfileId.MEDIUM: SpeedProfileSpec(
        speed_profile=SpeedProfileId.MEDIUM,
        ego_target_speed=10.0,
        vehicle_target_speed=10.0,
    ),
    SpeedProfileId.FAST: SpeedProfileSpec(
        speed_profile=SpeedProfileId.FAST,
        ego_target_speed=12.0,
        vehicle_target_speed=12.0,
    ),
}


DIFFICULTY_PRESETS: dict[str, RandomTrafficDifficultySpec] = {
    "no_traffic": RandomTrafficDifficultySpec(
        difficulty_preset_id="no_traffic",
        route_extent=RouteExtentId.SHORT,
        traffic_density=TrafficDensityId.NONE,
        speed_profile=SpeedProfileId.MEDIUM,
    ),
    "easy": RandomTrafficDifficultySpec(
        difficulty_preset_id="easy",
        route_extent=RouteExtentId.SHORT,
        traffic_density=TrafficDensityId.LOW,
        speed_profile=SpeedProfileId.SLOW,
    ),
    "medium": RandomTrafficDifficultySpec(
        difficulty_preset_id="medium",
        route_extent=RouteExtentId.MEDIUM,
        traffic_density=TrafficDensityId.MEDIUM,
        speed_profile=SpeedProfileId.MEDIUM,
    ),
    "hard": RandomTrafficDifficultySpec(
        difficulty_preset_id="hard",
        route_extent=RouteExtentId.LARGE,
        traffic_density=TrafficDensityId.DENSE,
        speed_profile=SpeedProfileId.FAST,
    ),
}


def get_route_extent_spec(route_extent: RouteExtentId | str) -> RouteExtentSpec:
    try:
        return ROUTE_EXTENT_PRESETS[RouteExtentId(route_extent)]
    except (KeyError, ValueError) as exc:
        available = ", ".join(sorted(spec.value for spec in RouteExtentId))
        raise KeyError(
            f"Unknown route_extent={route_extent!r}. Available route extents: {available}"
        ) from exc


def list_route_extent_ids() -> list[str]:
    return sorted(preset.value for preset in RouteExtentId)


def get_traffic_density_spec(traffic_density: TrafficDensityId | str) -> TrafficDensitySpec:
    try:
        return TRAFFIC_DENSITY_PRESETS[TrafficDensityId(traffic_density)]
    except (KeyError, ValueError) as exc:
        available = ", ".join(sorted(spec.value for spec in TrafficDensityId))
        raise KeyError(
            f"Unknown traffic_density={traffic_density!r}. Available traffic densities: {available}"
        ) from exc


def list_traffic_density_ids() -> list[str]:
    return sorted(preset.value for preset in TrafficDensityId)


def get_speed_profile_spec(speed_profile: SpeedProfileId | str) -> SpeedProfileSpec:
    try:
        return SPEED_PROFILE_PRESETS[SpeedProfileId(speed_profile)]
    except (KeyError, ValueError) as exc:
        available = ", ".join(sorted(spec.value for spec in SpeedProfileId))
        raise KeyError(
            f"Unknown speed_profile={speed_profile!r}. Available speed profiles: {available}"
        ) from exc


def list_speed_profile_ids() -> list[str]:
    return sorted(preset.value for preset in SpeedProfileId)


def resolve_random_navigation_spec(
    *,
    route_extent: RouteExtentId | str,
    traffic_density: TrafficDensityId | str,
    speed_profile: SpeedProfileId | str,
) -> RandomNavigationResolvedSpec:
    route_extent_spec = get_route_extent_spec(route_extent)
    traffic_density_spec = get_traffic_density_spec(traffic_density)
    speed_profile_spec = get_speed_profile_spec(speed_profile)
    return RandomNavigationResolvedSpec(
        route_extent=route_extent_spec.route_extent,
        traffic_density=traffic_density_spec.traffic_density,
        speed_profile=speed_profile_spec.speed_profile,
        route_dist_range=route_extent_spec.route_dist_range,
        traffic_enabled=traffic_density_spec.traffic_enabled,
        num_vehicles=traffic_density_spec.num_vehicles,
        ego_target_speed=speed_profile_spec.ego_target_speed,
        vehicle_target_speed=speed_profile_spec.vehicle_target_speed,
    )


def get_difficulty_spec(difficulty_preset_id: str) -> RandomTrafficDifficultySpec:
    try:
        return DIFFICULTY_PRESETS[difficulty_preset_id]
    except KeyError as exc:
        available = ", ".join(sorted(DIFFICULTY_PRESETS))
        raise KeyError(
            f"Unknown difficulty_preset_id={difficulty_preset_id!r}. "
            f"Available difficulty presets: {available}"
        ) from exc


def list_difficulty_preset_ids() -> list[str]:
    return sorted(DIFFICULTY_PRESETS)
