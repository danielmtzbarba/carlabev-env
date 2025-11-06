import numpy as np
from random import choice
from CarlaBEV.src.scenes.utils import get_random_node, find_route
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian


class SceneGenerator:
    """
    Procedural and curriculum-based scene generator.

    Handles:
      - Random traffic scenes with configurable growth
      - Predefined critical scenarios (catalogue)
    """

    def __init__(self, planner_manager, config=None):
        self.planners = planner_manager

        # --- Curriculum configuration ---
        cfg = config.__dict__ or {}
        self.size = cfg.get("map_size", 128)
        self.traffic_enabled = cfg.get("traffic_enabled", True)
        self.curriculum_enabled = cfg.get("curriculum_enabled", True)
        self.start_ep = cfg.get("start_ep", 1)
        self.max_v = cfg.get("max_vehicles", 50)
        self.mid = cfg.get("midpoint", 10)
        self.growth_rate = cfg.get("growth_rate", 0.01)

    # =========================================================
    # --- Randomized Curriculum Scene ---
    # =========================================================
    def generate_random(self, episode, max_retries=20):
        """
        Generates a randomized traffic scene with configurable curriculum.
        Returns actor dictionary compatible with Scene.reset().
        """
        if self.traffic_enabled:
            if self.curriculum_enabled:
                num_cars = self._vehicle_schedule(episode)
            else:
                num_cars = self.max_v

        actors = {"agent": None, "vehicle": [], "pedestrian": [], "target": []}

        # 1️⃣ Agent route
        for attempt in range(max_retries):
            try:
                n1 = get_random_node(self.planners.all, "agent", "R")
                n2 = get_random_node(self.planners.all, "agent", "R")
                agent = Vehicle(start_node=n1, end_node=n2, map_size=self.size)
                agent, path = find_route(self.planners.all, agent, lane="R")
                if len(path[0]) > 5:
                    actors["agent"] = path
                    break
            except Exception as e:
                print(e)
                continue

        # 2️⃣ Background vehicles
        for _ in range(num_cars):
            lane = choice(["L", "R"])
            try:
                n1 = get_random_node(self.planners.all, "vehicle", lane)
                n2 = get_random_node(self.planners.all, "vehicle", lane)
                veh = Vehicle(start_node=n1, end_node=n2, map_size=self.size)
                veh, path = find_route(self.planners.all, veh, lane=lane)
                if len(path[0]) > 5:
                    actors["vehicle"].append(veh)
            except Exception:
                continue

        # (Optional future) pedestrians, traffic lights, etc.
        return actors

    # =========================================================
    # --- Traffic Scheduling ---
    # =========================================================
    def _vehicle_schedule(self, episode):
        """Compute number of vehicles based on curriculum config."""
        if episode < self.start_ep:
            return 0

        # Logistic curve growth
        num = int(self.max_v / (1 + np.exp(-self.growth_rate * (episode - self.mid))))
        # Add stochastic jitter
        return int(np.clip(num + np.random.randint(-3, 4), 0, self.max_v))
