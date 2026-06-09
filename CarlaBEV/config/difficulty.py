from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


DifficultyFamily = Literal["random_navigation"]


class RandomTrafficDifficultySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    difficulty_id: str
    family: DifficultyFamily = "random_navigation"
    traffic_enabled: bool = True
    num_vehicles: int = 25
    route_dist_range: tuple[int, int] = (30, 130)
    ego_target_speed: float | None = None


DIFFICULTY_PRESETS: dict[str, RandomTrafficDifficultySpec] = {
    "rt_no_traffic_v1": RandomTrafficDifficultySpec(
        difficulty_id="rt_no_traffic_v1",
        traffic_enabled=False,
        num_vehicles=0,
        route_dist_range=(30, 80),
    ),
    "rt_easy_v1": RandomTrafficDifficultySpec(
        difficulty_id="rt_easy_v1",
        traffic_enabled=True,
        num_vehicles=8,
        route_dist_range=(30, 80),
    ),
    "rt_medium_v1": RandomTrafficDifficultySpec(
        difficulty_id="rt_medium_v1",
        traffic_enabled=True,
        num_vehicles=16,
        route_dist_range=(40, 100),
    ),
    "rt_hard_v1": RandomTrafficDifficultySpec(
        difficulty_id="rt_hard_v1",
        traffic_enabled=True,
        num_vehicles=25,
        route_dist_range=(50, 130),
    ),
}


def get_difficulty_spec(difficulty_id: str) -> RandomTrafficDifficultySpec:
    try:
        return DIFFICULTY_PRESETS[difficulty_id]
    except KeyError as exc:
        available = ", ".join(sorted(DIFFICULTY_PRESETS))
        raise KeyError(
            f"Unknown difficulty_id={difficulty_id!r}. Available difficulty presets: {available}"
        ) from exc


def list_difficulty_ids() -> list[str]:
    return sorted(DIFFICULTY_PRESETS)
