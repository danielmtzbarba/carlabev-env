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


AUTHORED_SCENE_FRAME_KIND = "surface"


def authored_frame_size(data: dict, default_size: int) -> int:
    """Surface raster size the coordinates of an authored/scenario file refer to.

    v3 files declare ``"frame": {"kind": "surface", "size": N}``; earlier files
    (v1/v2) carry no tag and are expressed in the legacy 128-px frame.
    """
    frame = data.get("frame") or {}
    kind = str(frame.get("kind", AUTHORED_SCENE_FRAME_KIND))
    if kind != AUTHORED_SCENE_FRAME_KIND:
        raise ValueError(f"Unsupported authored scene frame kind {kind!r}; expected 'surface'.")
    size = frame.get("size")
    return int(default_size if size is None else size)


def rescale_authored_actor(actor: dict, factor: float) -> dict:
    """Rescale every surface-px coordinate of an authored actor by ``factor``."""
    if factor == 1.0:
        return actor
    scaled = deepcopy(actor)
    for key in ("rx", "ry"):
        if scaled.get(key):
            scaled[key] = [float(v) * factor for v in scaled[key]]
    if scaled.get("waypoints"):
        scaled["waypoints"] = [[float(p[0]) * factor, float(p[1]) * factor] for p in scaled["waypoints"]]
    for key in ("start", "goal"):
        point = scaled.get(key)
        if isinstance(point, dict) and "x" in point and "y" in point:
            scaled[key] = {**point, "x": float(point["x"]) * factor, "y": float(point["y"]) * factor}
    for key in ("length", "width"):
        if scaled.get(key) is not None:
            scaled[key] = float(scaled[key]) * factor
    variation = scaled.get("variation")
    if isinstance(variation, dict) and variation.get("waypoint_jitter_px") is not None:
        variation["waypoint_jitter_px"] = float(variation["waypoint_jitter_px"]) * factor
    return scaled


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
    def __init__(self, name, map_size=128, units=None):
        from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT

        self.name = name
        # Unit contract: surface scale and all metric conversions used by the
        # scenario geometry come from here (authored/legacy anchors are at the
        # surface scale of the active metric frame).
        self.units = units if units is not None else DEFAULT_UNIT_CONTRACT
        self.metric = self.units.metric
        self.map_size = map_size if map_size is not None else self.metric.surface_size
        self.last_loaded_context = {}

    def m_to_px(self, meters):
        """Meters to surface px through the active metric frame."""
        return self.metric.distance_meters_to_surface(meters)

    def resolve_anchor(self, kwargs, default_raw_xy):
        """Return the scenario anchor ``(x, y)`` in the active surface frame.

        ``anchor_x`` / ``anchor_y`` supplied through reset options are surface
        coordinates of the raster declared by ``anchor_frame_size`` (default:
        the legacy 128-px frame every authored asset is expressed in). When a
        coordinate is omitted the raw-frame default is used.
        """
        frame_size = int(kwargs.get("anchor_frame_size", self.metric.LEGACY_SURFACE_SIZE))
        default_surface = self.metric.raw_to_surface(default_raw_xy)
        anchor = []
        for key, default_value in (("anchor_x", default_surface[0]), ("anchor_y", default_surface[1])):
            value = kwargs.get(key)
            if value is None:
                anchor.append(float(default_value))
            else:
                anchor.append(float(self.metric.rescale_surface_from(float(value), frame_size)))
        return anchor[0], anchor[1]

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

        # Authored coordinates are surface px of the raster declared by the file's
        # frame tag (legacy files: 128 px). Convert them into the active frame.
        file_frame_size = authored_frame_size(data, self.metric.LEGACY_SURFACE_SIZE)
        frame_factor = self.metric.surface_scale_from(file_frame_size)
        self.last_loaded_context = {
            "authored_frame_size": file_frame_size,
            "active_frame_size": self.metric.surface_size,
        }
        if frame_factor != 1.0:
            data = deepcopy(data)
            data["actors"] = [rescale_authored_actor(actor, frame_factor) for actor in data["actors"]]
            global_variation = (data.get("variation") or {}).get("global") or {}
            if global_variation.get("waypoint_jitter_px") is not None:
                global_variation["waypoint_jitter_px"] = float(global_variation["waypoint_jitter_px"]) * frame_factor

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
                    units=self.units,
                )
                # Rule-compliance flags (adversaries): {"obeys_signals": false, ...}
                v.obeys_signals = bool(varied_actor.get("obeys_signals", True))
                v.yields_at_junctions = bool(varied_actor.get("yields_at_junctions", True))
                scene_dict["vehicle"].append(v)

            elif atype == "pedestrian":
                behavior, _ = build_behavior("pedestrian", varied_actor.get("behavior", "cross"))

                p = Pedestrian(
                    self.map_size,
                    routeX=rx,
                    routeY=ry,
                    target_speed=speed,
                    behavior=behavior,
                    units=self.units,
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
                        units=self.units,
                        controls_heading=varied_actor.get("controls_heading"),
                        cycle=varied_actor.get("cycle"),
                    )
                )

        # len_route calculation can just use agent
        if scene_dict.get("agent"):
            agent = scene_dict["agent"]
            rx, ry = agent[0], agent[1]
            # Route length is reported in meters everywhere (U-07).
            len_route = self.metric.route_length_meters(rx, ry)
        else:
            len_route = 0.0

        self.last_loaded_context = {
            **self.last_loaded_context,
            "scene_id": data.get("scene_id"),
            "authored_scene": True,
            "variation_enabled": scene_variation["enabled"],
            "variation_seed": scene_variation["seed"],
            "variation_actor_count": len(realized_variations),
            "variation_realized": realized_variations,
        }

        return scene_dict, len_route
