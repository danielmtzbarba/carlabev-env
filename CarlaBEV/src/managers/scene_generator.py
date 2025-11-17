import os
import numpy as np
from random import choice
from CarlaBEV.src.scenes.utils import get_random_node, find_route, find_route_in_range
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.envs.utils import asset_path
from  CarlaBEV.src.scenes.scenarios.lead_brake import LeadBrakeScenario
from  CarlaBEV.src.scenes.scenarios.jaywalk import JaywalkScenario 



class PlannerManager:
    """Centralized manager for map-specific route planners."""

    def __init__(self, town_name: str = "Town01"):
        self.town_name = town_name
        base_path = os.path.join(asset_path, town_name)

        # --- Load all graph planners ---
        self.graphs = {
            "pedestrian": GraphPlanner(os.path.join(base_path, "town01.pkl")),
            "vehicle": GraphPlanner(
                os.path.join(base_path, "town01-vehicles-2lanes-100.pkl")
            ),
            "vehicle-R": GraphPlanner(
                os.path.join(base_path, "town01-vehicles-right-100.pkl")
            ),
            "vehicle-L": GraphPlanner(
                os.path.join(base_path, "town01-vehicles-left-100.pkl")
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
        self.cfg = config.__dict__ or {}
        self.size = self.cfg.get("map_size", 128)
        self.town_name = self.cfg.get("town_name", "Town01")
        self.traffic_enabled = self.cfg.get("traffic_enabled", True)
        self.planners = PlannerManager(self.town_name)
    
        self.scenarios = {
            "lead_brake": LeadBrakeScenario(map_size=128),
            "jaywalk": JaywalkScenario(map_size=128)
        }

    def build_scene(self, options):
        scene = options.get("scene", "rdm")
        num_vehicles = options.get("num_vehicles", self.cfg.get("max_vehicles", 25))
        dist_range = options.get("route_dist_range", self.cfg.get("route_dist_range", [30, 100]))

        # --- Case 1: Random scene generation ---
        if isinstance(scene, str) and scene == "rdm":
            return self.generate_random(num_vehicles, dist_range)

        # --- Case 2: Predefined scenario ---
        elif isinstance(scene, str):
            level = choice([1, 2, 3, 4])
            return self.scenarios[scene].sample(level)

    # =========================================================
    # --- Randomized Curriculum Scene ---
    # =========================================================
    def generate_random(self, num_cars, dist_range, max_retries=20):
        """
        Generates a randomized traffic scene with configurable curriculum.
        Returns actor dictionary compatible with Scene.reset().
        """
        num_cars = num_cars if self.traffic_enabled else 0

        actors = {"agent": None, "vehicle": [], "pedestrian": [], "target": []}

        # 1️⃣ Agent route
        for attempt in range(max_retries):
            
            _, path, len_route = find_route_in_range(self.planners.all, "agent", 'R',
                                          dist_range[0], dist_range[1])
            if path is not None:
                actors["agent"] = (path[0], path[1], 0.0)
                break

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
    except Exception:
        # route generation failed
       pass 
    return None, None
