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
        from CarlaBEV.src.actors.traffic_light import TrafficLight, TrafficLightState

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
            elif atype == "traffic_light":
                start = actor_data.get("start")
                goal = actor_data.get("goal")
                if start is None and rx and ry:
                    start = {"x": rx[0], "y": ry[0]}
                if goal is None and rx and ry:
                    goal = {"x": rx[-1], "y": ry[-1]}
                if start is None or goal is None:
                    continue
                dx = float(goal["x"]) - float(start["x"])
                dy = float(goal["y"]) - float(start["y"])
                center_x = 0.5 * (float(start["x"]) + float(goal["x"]))
                center_y = 0.5 * (float(start["y"]) + float(goal["y"]))
                orientation = actor_data.get(
                    "orientation",
                    "horizontal" if abs(dx) >= abs(dy) else "vertical",
                )
                state_map = {
                    "red": TrafficLightState.RED,
                    "yellow": TrafficLightState.YELLOW,
                    "green": TrafficLightState.GREEN,
                }
                scene_dict["traffic_light"].append(
                    TrafficLight(
                        pos_x=center_x,
                        pos_y=center_y,
                        map_size=self.map_size,
                        orientation=orientation,
                        signal_state=state_map.get(actor_data.get("signal_state", "red"), TrafficLightState.RED),
                        length=actor_data.get("length"),
                        width=actor_data.get("width"),
                    )
                )

        # len_route calculation can just use agent
        if scene_dict.get("agent"):
            rx, ry, _ = scene_dict["agent"]
            from CarlaBEV.src.scenes.utils import compute_total_dist_px

            len_route = compute_total_dist_px([rx, ry])
        else:
            len_route = 0

        return scene_dict, len_route
