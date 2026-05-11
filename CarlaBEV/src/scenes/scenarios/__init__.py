import numpy as np
import random

from CarlaBEV.src.scenes.scenarios.specs import (
    load_scenario_config_file,
    scenario_config_to_options,
)
from CarlaBEV.src.actors.behavior.registry import build_behavior

class Scenario:
    def __init__(self, name, map_size=128):
        self.name = name
        self.map_size = map_size

    def sample(self, **kwargs):
        """
        Return a dict:
        {
            "agent": (rx, ry, target_speed),
            "vehicle": [ Vehicle(...) ],
            "pedestrian": [ Pedestrian(...) ],
            "target": [...]
        }
        """
        config_file = kwargs.pop("config_file", None)
        if config_file:
            return self.load_config(config_file, **kwargs)
        raise NotImplementedError

    def load_config(self, filepath, **overrides):
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "actors" not in data:
            config = load_scenario_config_file(filepath)
            scenario_id = config["scenario_id"]
            if scenario_id != self.name:
                raise ValueError(
                    f"Config scenario '{scenario_id}' does not match loader '{self.name}'."
                )
            return self.sample(**scenario_config_to_options(config, overrides))

        scene_dict = {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }

        # We need to construct actors
        from CarlaBEV.src.actors.vehicle import Vehicle
        from CarlaBEV.src.actors.pedestrian import Pedestrian

        for actor_data in data["actors"]:
            atype = actor_data["type"]
            rx = actor_data["rx"]
            ry = actor_data["ry"]
            speed = actor_data.get(
                "cruise_speed",
                actor_data.get("initial_speed", actor_data.get("speed", 2.0)),
            )

            if atype == "agent":
                scene_dict["agent"] = (rx, ry, speed)
            elif atype == "vehicle":
                behavior, _ = build_behavior("vehicle", actor_data.get("behavior", "constant_speed"))

                v = Vehicle(
                    self.map_size,
                    routeX=rx,
                    routeY=ry,
                    target_speed=speed,
                    behavior=behavior,
                )
                scene_dict["vehicle"].append(v)

            elif atype == "pedestrian":
                behavior, _ = build_behavior("pedestrian", actor_data.get("behavior", "cross"))

                p = Pedestrian(
                    self.map_size,
                    routeX=rx,
                    routeY=ry,
                    target_speed=speed,
                    behavior=behavior,
                )
                scene_dict["pedestrian"].append(p)

        # len_route calculation can just use agent
        if scene_dict.get("agent"):
            rx, ry, _ = scene_dict["agent"]
            from CarlaBEV.src.scenes.utils import compute_total_dist_px

            len_route = compute_total_dist_px([rx, ry])
        else:
            len_route = 0

        return scene_dict, len_route
