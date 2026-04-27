from __future__ import annotations

import json
from dataclasses import dataclass
from copy import deepcopy

import numpy as np


@dataclass(frozen=True)
class ScenarioField:
    key: str
    label: str
    default: float | int
    cast: type = float
    help_text: str = ""

    def parse(self, value):
        if value in (None, ""):
            return self.cast(self.default)
        return self.cast(value)

    def default_text(self) -> str:
        return str(self.default)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    display_name: str
    description: str
    levels: tuple[int, ...]
    fields: tuple[ScenarioField, ...]

    def level_options(self) -> list[str]:
        return [f"Level {level}" for level in self.levels]


@dataclass(frozen=True)
class ScenarioPreset:
    preset_id: str
    scene: str
    description: str
    options: dict


SCENARIO_SPECS = {
    "jaywalk": ScenarioSpec(
        scenario_id="jaywalk",
        display_name="Jaywalk",
        description="Pedestrian crosses ahead of the ego vehicle.",
        levels=(1, 2, 3, 4),
        fields=(
            ScenarioField("ego_speed", "Ego Speed (m/s)", 12.0),
            ScenarioField("cross_delay", "Cross Delay (s)", 1.5),
            ScenarioField("pedestrian_speed", "Ped Speed (m/s)", 1.6),
            ScenarioField("cross_offset", "Cross Offset (m)", 0.0),
            ScenarioField("yield_duration", "Yield Duration (s)", 1.2),
            ScenarioField("rear_gap", "Rear Gap (m)", 5.0),
            ScenarioField("rear_speed", "Rear Speed (m/s)", 10.0),
        ),
    ),
    "lead_brake": ScenarioSpec(
        scenario_id="lead_brake",
        display_name="Lead Brake",
        description="Lead vehicle brakes hard in front of ego.",
        levels=(1, 2, 3),
        fields=(
            ScenarioField("ego_speed", "Ego Speed (m/s)", 12.0),
            ScenarioField("lead_gap", "Lead Gap (m)", 7.5),
            ScenarioField("lead_speed", "Lead Speed (m/s)", 12.0),
            ScenarioField("brake_delay", "Brake Delay (s)", 2.5),
            ScenarioField("brake_strength", "Brake Strength (m/s^2)", 4.0),
            ScenarioField("left_speed", "Left Lane Speed (m/s)", 14.0),
            ScenarioField("rear_gap", "Rear Gap (m)", 5.0),
            ScenarioField("rear_speed", "Rear Speed (m/s)", 10.0),
            ScenarioField("rear_brake_delay", "Rear Brake Delay (s)", 3.0),
        ),
    ),
    "red_light_runner": ScenarioSpec(
        scenario_id="red_light_runner",
        display_name="Red Light Runner",
        description="Perpendicular adversary runs a red light.",
        levels=(1,),
        fields=(
            ScenarioField("ego_speed", "Ego Speed (m/s)", 10.0),
            ScenarioField("adv_speed", "Adversary Speed (m/s)", 16.0),
            ScenarioField("intersection_index", "Intersection Index", 11, int),
        ),
    ),
}


SCENARIO_PRESETS = {
    "jaywalk_debug": ScenarioPreset(
        preset_id="jaywalk_debug",
        scene="jaywalk",
        description="Debug-friendly jaywalk preset with explicit semantic parameters.",
        options={
            "scene": "jaywalk",
            "level": 3,
            "ego_speed": 10.0,
            "cross_delay": 1.2,
            "pedestrian_speed": 1.6,
            "yield_duration": 1.2,
        },
    ),
    "lead_brake_debug": ScenarioPreset(
        preset_id="lead_brake_debug",
        scene="lead_brake",
        description="Lead-brake preset for interactive debugging.",
        options={
            "scene": "lead_brake",
            "level": 2,
            "ego_speed": 12.0,
            "lead_gap": 8.0,
            "lead_speed": 11.0,
            "brake_delay": 2.0,
            "brake_strength": 4.0,
        },
    ),
    "red_light_debug": ScenarioPreset(
        preset_id="red_light_debug",
        scene="red_light_runner",
        description="Graph-backed signalized intersection conflict preset.",
        options={
            "scene": "red_light_runner",
            "intersection_index": 11,
            "ego_speed": 10.0,
            "adv_speed": 16.0,
        },
    ),
    "rdm_navigation": ScenarioPreset(
        preset_id="rdm_navigation",
        scene="rdm",
        description="Random background-traffic navigation preset.",
        options={
            "scene": "rdm",
            "num_vehicles": 25,
            "route_dist_range": [30, 130],
        },
    ),
}


def list_scenario_ids() -> list[str]:
    return list(SCENARIO_SPECS.keys())


def list_scenario_preset_ids() -> list[str]:
    return list(SCENARIO_PRESETS.keys())


def get_scenario_spec(scenario_id: str) -> ScenarioSpec:
    if scenario_id not in SCENARIO_SPECS:
        raise KeyError(f"Unknown scenario '{scenario_id}'")
    return SCENARIO_SPECS[scenario_id]


def get_scenario_preset(preset_id: str) -> ScenarioPreset:
    if preset_id not in SCENARIO_PRESETS:
        raise KeyError(f"Unknown scenario preset '{preset_id}'")
    return SCENARIO_PRESETS[preset_id]


def build_runtime_scenario_options(
    preset_id: str,
    *,
    reset_mask=None,
    overrides: dict | None = None,
) -> dict:
    preset = get_scenario_preset(preset_id)
    options = deepcopy(preset.options)
    options["scenario_preset_id"] = preset.preset_id
    options["scenario_preset_scene"] = preset.scene
    options["scenario_preset_description"] = preset.description
    for key, value in (overrides or {}).items():
        if value is not None:
            options[key] = value
    if reset_mask is not None:
        options["reset_mask"] = np.asarray(reset_mask, dtype=bool)
    return options


def coerce_parameters(scenario_id: str, raw_values: dict | None) -> dict:
    spec = get_scenario_spec(scenario_id)
    raw_values = raw_values or {}
    params = {}
    for field in spec.fields:
        params[field.key] = field.parse(raw_values.get(field.key))
    return params


def build_scenario_config(
    scene_id: str,
    scenario_id: str,
    level: int,
    anchor: dict | None,
    parameters: dict | None,
) -> dict:
    anchor = anchor or {}
    return {
        "version": 1,
        "type": "scenario_config",
        "scene_id": scene_id,
        "scenario_id": scenario_id,
        "level": int(level),
        "anchor": {
            "x": None if anchor.get("x") is None else int(anchor.get("x")),
            "y": None if anchor.get("y") is None else int(anchor.get("y")),
        },
        "parameters": coerce_parameters(scenario_id, parameters),
    }


def normalize_scenario_config(data: dict) -> dict:
    if data.get("type") == "scenario_config" or "scenario_id" in data:
        scenario_id = data.get("scenario_id")
        level = int(data.get("level", 1))
        anchor = data.get("anchor", {}) or {}
        parameters = data.get("parameters", {}) or {}
    elif "scenario" in data and "kwargs" in data:
        kwargs = dict(data.get("kwargs", {}))
        scenario_id = data.get("scenario")
        level = int(kwargs.pop("level", 1))
        anchor = {
            "x": kwargs.pop("anchor_x", None),
            "y": kwargs.pop("anchor_y", None),
        }
        kwargs.pop("scene", None)
        parameters = kwargs
    else:
        raise ValueError("Unsupported scenario config format.")

    return build_scenario_config(
        scene_id=data.get("scene_id", scenario_id),
        scenario_id=scenario_id,
        level=level,
        anchor=anchor,
        parameters=parameters,
    )


def load_scenario_config_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return normalize_scenario_config(json.load(handle))


def scenario_config_to_options(config: dict, overrides: dict | None = None) -> dict:
    options = dict(config.get("parameters", {}))
    anchor = config.get("anchor", {}) or {}
    if anchor.get("x") is not None:
        options["anchor_x"] = anchor["x"]
    if anchor.get("y") is not None:
        options["anchor_y"] = anchor["y"]
    options["level"] = int(config.get("level", 1))
    options["scene"] = config["scenario_id"]

    for key, value in (overrides or {}).items():
        if key in {"config_file", "scene"} or value is None:
            continue
        options[key] = value
    return options
