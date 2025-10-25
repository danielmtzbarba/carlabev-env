import os
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.envs.utils import asset_path


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
