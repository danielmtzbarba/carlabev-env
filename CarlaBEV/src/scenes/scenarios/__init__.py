import numpy as np
import random
from copy import deepcopy

from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_options_from_config,
    load_scenario_config_file,
)
from CarlaBEV.src.actors.behavior.registry import build_behavior


def _build_linear_route(start, end, step_px=8):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(abs(dx), abs(dy))
    num_points = max(2, int(length / max(1, step_px)) + 1)
    rx = np.linspace(start[0], end[0], num_points).round().astype(int).tolist()
    ry = np.linspace(start[1], end[1], num_points).round().astype(int).tolist()
    return rx, ry


def _build_route_from_waypoints(waypoints, step_px=8):
    if len(waypoints) < 2:
        return [], []
    route_x = []
    route_y = []
    for idx in range(len(waypoints) - 1):
        seg_x, seg_y = _build_linear_route(waypoints[idx], waypoints[idx + 1], step_px=step_px)
        if idx > 0:
            seg_x = seg_x[1:]
            seg_y = seg_y[1:]
        route_x.extend(seg_x)
        route_y.extend(seg_y)
    return route_x, route_y


def _round_scalar(value, digits=4):
    return round(float(value), digits)


def _sample_variation_value(spec, rng, fallback=None):
    if spec is None:
        return fallback
    if not isinstance(spec, dict):
        return spec
    mode = spec.get("mode", "fixed")
    if mode == "fixed":
        return spec.get("value", fallback)
    if mode == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if mode == "normal":
        value = rng.normalvariate(float(spec["mean"]), float(spec["std"]))
        clip = spec.get("clip")
        if clip is not None and len(clip) == 2:
            value = max(float(clip[0]), min(float(clip[1]), value))
        return value
    if mode == "choice":
        values = spec.get("values", [])
        if not values:
            return fallback
        return rng.choice(list(values))
    return fallback


def _normalize_waypoints(actor_data):
    if actor_data.get("waypoints"):
        return [
            [int(round(point[0])), int(round(point[1]))]
            for point in actor_data["waypoints"]
        ]
    start = actor_data.get("start")
    goal = actor_data.get("goal")
    rx = actor_data.get("rx", [])
    ry = actor_data.get("ry", [])
    if start is None and rx and ry:
        start = {"x": rx[0], "y": ry[0]}
    if goal is None and rx and ry:
        goal = {"x": rx[-1], "y": ry[-1]}
    if start is None or goal is None:
        return []
    return [
        [int(round(start["x"])), int(round(start["y"]))],
        [int(round(goal["x"])), int(round(goal["y"]))],
    ]


def _resolve_variation_settings(data, overrides):
    variation = deepcopy(data.get("variation") or {})
    enabled = overrides.get("variation_enabled")
    if enabled is None:
        enabled = bool(variation.get("enabled", False))
    else:
        enabled = bool(enabled)
    if not enabled:
        return {
            "enabled": False,
            "seed": None,
            "spec": variation,
        }
    seed = overrides.get("variation_seed")
    if seed is None:
        seed = variation.get("default_seed")
    if seed is None:
        seed = 0
    return {
        "enabled": True,
        "seed": int(seed),
        "spec": variation,
    }


def _apply_actor_variation(actor_data, scene_variation, actor_index):
    actor = deepcopy(actor_data)
    actor_variation = deepcopy(actor.get("variation") or {})
    if not scene_variation["enabled"] or not actor_variation.get("enabled", False):
        return actor, None

    actor_seed = scene_variation["seed"] + int(actor_variation.get("seed_offset", actor_index))
    rng = random.Random(actor_seed)
    realized = {
        "type": actor.get("type"),
        "role": actor.get("role"),
        "seed": actor_seed,
    }

    global_spec = scene_variation["spec"].get("global", {}) or {}
    waypoints = _normalize_waypoints(actor)
    constraints = actor_variation.get("constraints", {}) or {}
    lock_endpoints = constraints.get("lock_endpoints", True)
    waypoint_jitter = actor_variation.get("waypoint_jitter_px", global_spec.get("waypoint_jitter_px"))
    if waypoint_jitter and waypoints:
        jitter_radius = float(waypoint_jitter)
        varied_waypoints = []
        for idx, point in enumerate(waypoints):
            if lock_endpoints and idx in {0, len(waypoints) - 1}:
                varied_waypoints.append(list(point))
                continue
            varied_waypoints.append([
                int(round(point[0] + rng.uniform(-jitter_radius, jitter_radius))),
                int(round(point[1] + rng.uniform(-jitter_radius, jitter_radius))),
            ])
        actor["waypoints"] = varied_waypoints
        actor["start"] = {"x": varied_waypoints[0][0], "y": varied_waypoints[0][1]}
        actor["goal"] = {"x": varied_waypoints[-1][0], "y": varied_waypoints[-1][1]}
        realized["waypoint_jitter_px"] = jitter_radius
        realized["waypoints"] = varied_waypoints

    speed = float(
        actor.get(
            "cruise_speed",
            actor.get("initial_speed", actor.get("speed", 0.0)),
        )
    )
    speed_scale = _sample_variation_value(global_spec.get("speed_scale"), rng, fallback=1.0)
    speed_spec = actor_variation.get("speed")
    if speed_spec is not None:
        speed = float(_sample_variation_value(speed_spec, rng, fallback=speed))
    else:
        speed = speed * float(speed_scale)
    speed = max(0.0, speed)
    actor["speed"] = speed
    actor["initial_speed"] = speed
    actor["cruise_speed"] = speed
    realized["speed"] = _round_scalar(speed)

    behavior = deepcopy(actor.get("behavior") or {})
    params = deepcopy(behavior.get("params") or {})
    behavior_param_specs = actor_variation.get("behavior_params", {}) or {}
    realized_behavior = {}
    for key, spec in behavior_param_specs.items():
        if key in params:
            params[key] = _sample_variation_value(spec, rng, fallback=params[key])
            realized_behavior[key] = _round_scalar(params[key])
    if realized_behavior:
        behavior["params"] = params
        actor["behavior"] = behavior
        realized["behavior_params"] = realized_behavior

    if actor.get("type") == "traffic_light" and actor_variation.get("signal_state"):
        actor["signal_state"] = _sample_variation_value(
            actor_variation.get("signal_state"),
            rng,
            fallback=actor.get("signal_state", "red"),
        )
        realized["signal_state"] = actor["signal_state"]

    return actor, realized

class Scenario:
    def __init__(self, name, map_size=128):
        self.name = name
        self.map_size = map_size
        self.last_loaded_context = {}

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
        self.last_loaded_context = {}

        if "actors" not in data:
            config = load_scenario_config_file(filepath)
            scenario_id = config["scenario_id"]
            if scenario_id != self.name:
                raise ValueError(
                    f"Config scenario '{scenario_id}' does not match loader '{self.name}'."
                )
            return self.sample(
                **build_scenario_options_from_config(config, overrides=overrides)
            )

        scene_variation = _resolve_variation_settings(data, overrides)
        realized_variations = []

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

        for actor_idx, actor_data in enumerate(data["actors"]):
            varied_actor, realized = _apply_actor_variation(actor_data, scene_variation, actor_idx)
            if realized is not None:
                realized_variations.append(realized)
            atype = actor_data["type"]
            rx = varied_actor.get("rx")
            ry = varied_actor.get("ry")
            if (not rx or not ry) and varied_actor.get("waypoints"):
                rx, ry = _build_route_from_waypoints(varied_actor["waypoints"])
            rx = rx or []
            ry = ry or []
            speed = varied_actor.get(
                "cruise_speed",
                varied_actor.get("initial_speed", varied_actor.get("speed", 2.0)),
            )

            if atype == "agent":
                scene_dict["agent"] = (rx, ry, speed, speed)
            elif atype == "vehicle":
                behavior, _ = build_behavior("vehicle", varied_actor.get("behavior", "constant_speed"))

                v = Vehicle(
                    self.map_size,
                    routeX=rx,
                    routeY=ry,
                    target_speed=speed,
                    behavior=behavior,
                )
                scene_dict["vehicle"].append(v)

            elif atype == "pedestrian":
                behavior, _ = build_behavior("pedestrian", varied_actor.get("behavior", "cross"))

                p = Pedestrian(
                    self.map_size,
                    routeX=rx,
                    routeY=ry,
                    target_speed=speed,
                    behavior=behavior,
                )
                scene_dict["pedestrian"].append(p)
            elif atype == "traffic_light":
                start = varied_actor.get("start")
                goal = varied_actor.get("goal")
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
                orientation = varied_actor.get(
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
                        signal_state=state_map.get(varied_actor.get("signal_state", "red"), TrafficLightState.RED),
                        length=varied_actor.get("length"),
                        width=varied_actor.get("width"),
                    )
                )

        # len_route calculation can just use agent
        if scene_dict.get("agent"):
            agent = scene_dict["agent"]
            rx, ry = agent[0], agent[1]
            from CarlaBEV.src.scenes.utils import compute_total_dist_px

            len_route = compute_total_dist_px([rx, ry])
        else:
            len_route = 0

        self.last_loaded_context = {
            "scene_id": data.get("scene_id"),
            "authored_scene": True,
            "variation_enabled": scene_variation["enabled"],
            "variation_seed": scene_variation["seed"],
            "variation_actor_count": len(realized_variations),
            "variation_realized": realized_variations,
        }

        return scene_dict, len_route
