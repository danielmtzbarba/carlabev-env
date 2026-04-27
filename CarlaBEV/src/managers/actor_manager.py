import copy
from CarlaBEV.envs.geometry import route_length_meters
from CarlaBEV.envs.geometry import speed_mps_to_surface
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.hero import DiscreteAgent, ContinuousAgent


def compute_route_length(route):
    """Compute route length in meters from surface/world route coordinates."""
    rx, ry = route
    return route_length_meters(rx, ry)

class ActorManager:
    """Handles creation, lifecycle, and updates of all scene actors."""

    def __init__(self, size: int, action_space: str = "discrete"):
        self.size = size
        self.action_space = action_space
        self.clear()

    def clear(self):
        """Reset actor registry to empty."""
        self.route_length = 0.0
        self.actors = {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }

    # ======================================================
    # --- Creation ---
    # ======================================================
    def spawn_hero(self, route, initial_speed_mps, target_speed_mps):
        """Spawn the main agent (hero vehicle)."""
        initial_speed = speed_mps_to_surface(initial_speed_mps)
        target_speed = speed_mps_to_surface(target_speed_mps)
        if self.action_space == "continuous":
             hero = ContinuousAgent(
                window_size=self.size,
                route=route,
                initial_speed=initial_speed,
                color=(0, 0, 0),
                target_speed=target_speed,
                car_size=32,
            )
        else:
            hero = DiscreteAgent(
                window_size=self.size,
                route=route,
                initial_speed=initial_speed,
                color=(0, 0, 0),
                target_speed=target_speed,
                car_size=32,
            )
        hero.initial_speed_mps = float(initial_speed_mps)
        hero.target_speed_mps = float(target_speed_mps)
        self.route_length = compute_route_length(route)
        self.actors["agent"] = hero
        return hero

    def add_vehicle(self, start_node, end_node, routeX=None, routeY=None):
        v = Vehicle(
            start_node=start_node,
            end_node=end_node,
            map_size=self.size,
            routeX=routeX,
            routeY=routeY,
        )
        self.actors["vehicle"].append(v)
        return v

    def add_pedestrian(self, start_node, end_node, routeX=None, routeY=None):
        p = Pedestrian(
            start_node=start_node,
            end_node=end_node,
            map_size=self.size,
            routeX=routeX,
            routeY=routeY,
        )
        self.actors["pedestrian"].append(p)
        return p

    def load(self, actors_dict):
        """Load actors from a prebuilt dictionary (used when loading from CSV)."""
        self.clear()
        for key, value in copy.deepcopy(actors_dict).items():
            if key in {"vehicle", "pedestrian"}:
                self.actors[key] = [actor for actor in value if self._has_valid_route(actor)]
            else:
                self.actors[key] = value

    # ======================================================
    # --- Simulation ---
    # ======================================================
    def reset_all(self):
        """Reset all actor controllers."""
        for k, v in self.actors.items():
            if k == "agent" or not v:
                continue
            for actor in v:
                if k in {"vehicle", "pedestrian"} and not self._has_valid_route(actor):
                    continue
                if hasattr(actor, "reset"):
                    actor.reset()

    def step_all(self, t=0.0, dt=0.05):
        """Step all non-hero actors."""
        for k, v in self.actors.items():
            if k == "agent":
                continue
            for actor in v:
                if k in {"vehicle", "pedestrian"} and not self._has_valid_route(actor):
                    continue
                actor.step(t, dt)

    def draw_all(self, surface, frame):
        """Draw all actors on given pygame surface."""
        for k, v in self.actors.items():
            if not v:
                continue
            if k == "agent":
                continue
            else:
                for actor in v:
                    if k in {"vehicle", "pedestrian"} and not self._has_valid_route(actor):
                        continue
                    actor.draw(surface, frame)

    @property
    def num_vehicles(self):
        return len(self.actors["vehicle"])

    @staticmethod
    def _has_valid_route(actor):
        rx = getattr(actor, "rx", None)
        ry = getattr(actor, "ry", None)
        return rx is not None and ry is not None and len(rx) >= 2 and len(ry) >= 2
