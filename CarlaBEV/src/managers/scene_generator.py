import numpy as np
from random import choice
from CarlaBEV.src.scenes.utils import get_random_node, find_route
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian


class SceneGenerator:
    """Procedural and curriculum-based scene generator."""

    def __init__(self, planner_manager, map_size):
        self.planners = planner_manager
        self.size = map_size

    # =========================================================
    # --- Randomized Curriculum ---
    # =========================================================
    def generate_random(self, episode, max_retries=20):
        """
        Generates a randomized scene with controlled traffic growth.
        Returns actor dictionary compatible with Scene.reset().
        """
        num_cars = self._vehicle_schedule(episode)
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
                continue

        # 2️⃣ Add background vehicles
        for _ in range(num_cars):
            lane = choice(["L", "R"])
            try:
                n1 = get_random_node(self.planners.all, "vehicle", lane)
                n2 = get_random_node(self.planners.all, "vehicle", lane)
                veh = Vehicle(start_node=n1, end_node=n2, map_size=self.size)
                find_route(self.planners.all, veh, lane=lane)
                actors["vehicle"].append(veh)
            except Exception:
                continue

        # 3️⃣ Pedestrians (optional future)
        return actors

    # =========================================================
    # --- Specific Scenarios (Catalog / Unit tests) ---
    # =========================================================
    def generate_lead_brake(self):
        """Example critical scenario: lead vehicle brakes suddenly."""
        actors = {"agent": None, "vehicle": [], "pedestrian": [], "target": []}

        # Agent follows path northbound
        rx = [850, 850, 850, 850, 850, 850]
        ry = [1080, 1060, 1040, 1020, 1000, 980]
        actors["agent"] = (rx, ry)

        # Lead vehicle ahead braking
        rx_v = [850, 850, 850, 850, 850]
        ry_v = [940, 930, 920, 915, 912]
        v = Vehicle(
            start_node=[850, 940],
            end_node=[850, 912],
            map_size=self.size,
            routeX=rx_v,
            routeY=ry_v,
        )
        actors["vehicle"].append(v)

        return actors

    # =========================================================
    # --- Traffic scheduling ---
    # =========================================================
    def _vehicle_schedule(self, episode):
        """Smooth logistic traffic growth after ep=1000."""
        if episode < 1000:
            return 0
        max_vehicles = 50
        growth_rate = 0.01
        midpoint = 2500
        num = int(max_vehicles / (1 + np.exp(-growth_rate * (episode - midpoint))))
        return int(np.clip(num + np.random.randint(-3, 4), 0, max_vehicles))
