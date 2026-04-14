from __future__ import annotations

import json
from dataclasses import dataclass


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


SCENARIO_SPECS = {
    "jaywalk": ScenarioSpec(
        scenario_id="jaywalk",
        display_name="Jaywalk",
        description="Pedestrian crosses ahead of the ego vehicle.",
        levels=(1, 2, 3, 4),
        fields=(
            ScenarioField("ego_speed", "Ego Speed (m/s)", 45.0),
            ScenarioField("cross_delay", "Cross Delay (s)", 3.0),
            ScenarioField("pedestrian_speed", "Ped Speed (m/s)", 2.5),
            ScenarioField("cross_offset", "Cross Offset (px)", 0.0),
            ScenarioField("rear_gap", "Rear Gap (px)", 16, int),
            ScenarioField("rear_speed", "Rear Speed (m/s)", 38.0),
        ),
    ),
    "lead_brake": ScenarioSpec(
        scenario_id="lead_brake",
        display_name="Lead Brake",
        description="Lead vehicle brakes hard in front of ego.",
        levels=(1, 2, 3),
        fields=(
            ScenarioField("ego_speed", "Ego Speed (m/s)", 45.0),
            ScenarioField("lead_gap", "Lead Gap (px)", 24, int),
            ScenarioField("lead_speed", "Lead Speed (m/s)", 45.0),
            ScenarioField("brake_delay", "Brake Delay (s)", 4.0),
            ScenarioField("brake_strength", "Brake Strength", 3.5),
            ScenarioField("left_speed", "Left Lane Speed (m/s)", 52.0),
            ScenarioField("rear_gap", "Rear Gap (px)", 16, int),
            ScenarioField("rear_speed", "Rear Speed (m/s)", 38.0),
            ScenarioField("rear_brake_delay", "Rear Brake Delay (s)", 6.0),
        ),
    ),
    "red_light_runner": ScenarioSpec(
        scenario_id="red_light_runner",
        display_name="Red Light Runner",
        description="Perpendicular adversary runs a red light.",
        levels=(1,),
        fields=(
            ScenarioField("ego_speed", "Ego Speed (m/s)", 50.0),
            ScenarioField("adv_speed", "Adversary Speed (m/s)", 60.0),
            ScenarioField("intersection_index", "Intersection Index", 2, int),
        ),
    ),
}


def list_scenario_ids() -> list[str]:
    return list(SCENARIO_SPECS.keys())


def get_scenario_spec(scenario_id: str) -> ScenarioSpec:
    if scenario_id not in SCENARIO_SPECS:
        raise KeyError(f"Unknown scenario '{scenario_id}'")
    return SCENARIO_SPECS[scenario_id]


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
