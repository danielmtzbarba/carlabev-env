from __future__ import annotations

from pathlib import Path

from CarlaBEV.envs.utils import asset_path
from CarlaBEV.src.scenes.scenarios.specs import (
    get_scenario_preset,
    get_scenario_spec,
    list_scenario_ids,
    list_scenario_preset_ids,
)


AUTHORED_SCENE_FAMILIES = {
    "jaywalk": "jaywalk-*.json",
    "lead_brake": "leadbrake-*.json",
    "red_light_runner": "redlightrunner-*.json",
}


def list_authored_scene_families() -> list[str]:
    return sorted(AUTHORED_SCENE_FAMILIES)


def list_authored_scene_paths(family: str | None = None) -> list[str]:
    scenes_dir = Path(asset_path) / "scenes"
    if family is None:
        patterns = AUTHORED_SCENE_FAMILIES.values()
    else:
        try:
            patterns = [AUTHORED_SCENE_FAMILIES[family]]
        except KeyError as exc:
            available = ", ".join(sorted(AUTHORED_SCENE_FAMILIES))
            raise KeyError(
                f"Unknown authored scene family '{family}'. Available families: {available}"
            ) from exc

    paths: list[str] = []
    for pattern in patterns:
        paths.extend(str(path) for path in scenes_dir.glob(pattern))
    return sorted(paths)


__all__ = [
    "AUTHORED_SCENE_FAMILIES",
    "get_scenario_preset",
    "get_scenario_spec",
    "list_authored_scene_families",
    "list_authored_scene_paths",
    "list_scenario_ids",
    "list_scenario_preset_ids",
]
