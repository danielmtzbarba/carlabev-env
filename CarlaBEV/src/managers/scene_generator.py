import json
import os
import warnings
from random import choice
from CarlaBEV.src.scenes.utils import get_random_node, find_route, find_route_in_range
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.envs.utils import asset_path
from  CarlaBEV.src.scenes.scenarios.lead_brake import LeadBrakeScenario
from  CarlaBEV.src.scenes.scenarios.jaywalk import JaywalkScenario 
from  CarlaBEV.src.scenes.scenarios.red_light_running import RedLightRunningScenario
from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_options_from_config,
    load_scenario_config_file,
)



class PlannerManager:
    """Centralized manager for map-specific route planners."""

    def __init__(self, town_name: str = "Town01"):
        self.town_name = town_name
        base_path = os.path.join(asset_path, town_name)
        planner_prefix = town_name.lower()

        # --- Load all graph planners ---
        self.graphs = {
            "pedestrian": GraphPlanner(os.path.join(base_path, f"{planner_prefix}.pkl")),
            "vehicle": GraphPlanner(
                os.path.join(base_path, f"{planner_prefix}-vehicles-2lanes-100.pkl")
            ),
            "vehicle-R": GraphPlanner(
                os.path.join(base_path, f"{planner_prefix}-vehicles-right-100.pkl")
            ),
            "vehicle-L": GraphPlanner(
                os.path.join(base_path, f"{planner_prefix}-vehicles-left-100.pkl")
            ),
        }

    def get(self, key: str):
        """Return a specific planner or None if missing."""
        return self.graphs.get(key)

    @property
    def all(self):
        return self.graphs

class SceneGenerator:
    """
    Procedural and curriculum-based scene generator.

    Handles:
      - Random traffic scenes with configurable growth
      - Predefined critical scenarios (catalogue)
    """

    def __init__(self, config=None):
        if hasattr(config, "model_dump"):
            self.cfg = config.model_dump(mode="python")
        else:
            self.cfg = getattr(config, "__dict__", {}) or {}
        self.size = self.cfg.get("map_size", 128)
        self.map_name = self.cfg.get("map_name", "Town01")
        self.traffic_enabled = self.cfg.get("traffic_enabled", True)
        self.planners = PlannerManager(self.map_name)
    
        self.scenarios = {
            "lead_brake": LeadBrakeScenario(map_size=128),
            "jaywalk": JaywalkScenario(map_size=128),
            "red_light_runner": RedLightRunningScenario(
                map_size=128, map_name=self.map_name
            ),
        }
        self.last_scene_context = {}

    def build_scene(self, options):
        self.last_scene_context = {}
        scene = options.get("scene", "rdm")
        config_file = options.get("config_file")
        num_vehicles = options.get("num_vehicles", self.cfg.get("max_vehicles", 25))
        dist_range = options.get("route_dist_range", self.cfg.get("route_dist_range", [30, 100]))

        if isinstance(scene, str) and scene.endswith(".json") and os.path.exists(scene):
            config_file = scene

        if config_file:
            with open(config_file, "r", encoding="utf-8") as handle:
                raw_config = json.load(handle)

            if "actors" in raw_config:
                scenario_id = raw_config.get("scenario_id") or raw_config.get("scenario")
                if scenario_id not in self.scenarios:
                    raise KeyError(
                        f"Unknown scenario '{scenario_id}' in authored config '{config_file}'"
                    )
                authored_options = dict(options)
                authored_options["config_file"] = config_file
                actors, len_route = self.scenarios[scenario_id].sample(**authored_options)
                self.last_scene_context = dict(getattr(self.scenarios[scenario_id], "last_loaded_context", {}) or {})
                self.last_scene_context.setdefault("scene", scenario_id)
                self.last_scene_context.setdefault("config_file", config_file)
                return actors, len_route

            config = load_scenario_config_file(config_file)
            scenario_id = config["scenario_id"]
            if scenario_id not in self.scenarios:
                raise KeyError(f"Unknown scenario '{scenario_id}' in config '{config_file}'")
            return self.scenarios[scenario_id].sample(
                **build_scenario_options_from_config(config, overrides=options)
            )

        # --- Case 1: Random scene generation ---
        if isinstance(scene, str) and scene == "rdm":
            return self.generate_random(num_vehicles, dist_range)

        # --- Case 2: Predefined scenario ---
        elif isinstance(scene, str) and scene in self.scenarios:
            scenario_options = dict(options)
            scenario_options.setdefault("level", choice([1, 2, 3, 4]))
            return self.scenarios[scene].sample(**scenario_options)
            
        # --- Default Fallback: Empty Scene ---
        return {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }, 0

    # =========================================================
    # --- Randomized Curriculum Scene ---
    # =========================================================
    def generate_random(self, num_cars, dist_range, max_retries=20):
        """
        Generates a randomized traffic scene with configurable curriculum.
        Returns actor dictionary compatible with Scene.reset().
        """
        num_cars = num_cars if self.traffic_enabled else 0
        ego_target_speed = float(self.cfg.get("ego_target_speed", 12.0))
        self.last_scene_context = {
            "scene": "rdm",
            "scenario_param_num_vehicles": int(num_cars),
            "scenario_param_route_dist_range": list(dist_range),
        }

        actors = {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }

        # 1️⃣ Agent route
        for attempt in range(max_retries):
            _, path, len_route = find_route_in_range(self.planners.all, "agent", 'R',
                                          dist_range[0], dist_range[1])
            if path is not None and len(path[0]) > 1:
                actors["agent"] = (path[0], path[1], 0.0, ego_target_speed)
                break
        if actors["agent"] is None:
            raise RuntimeError(
                f"Failed to generate a valid ego route in range {dist_range} after {max_retries} attempts."
            )

        # 2️⃣ Background vehicles
        for _ in range(num_cars):
            lane = choice(["L", "R"])
            veh, _ = get_actor("vehicle", lane, self.planners.all)
            if veh is None:
                continue
            actors["vehicle"].append(veh)

        # (Optional future) pedestrians, traffic lights, etc.
        return actors, len_route


def get_actor(actor_type, lane, planners):
    try:
        n1 = get_random_node(planners, actor_type, lane)
        n2 = get_random_node(planners, actor_type, lane)
        veh = Vehicle(start_node=n1, end_node=n2, map_size=128)
        veh, path = find_route(planners, veh, lane=lane)
        if len(path[0]) > 5:
            return veh, path
    except Exception as exc:
        warnings.warn(
            f"Route generation failed for actor_type={actor_type}, lane={lane}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return None, None
