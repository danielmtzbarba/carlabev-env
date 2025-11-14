import copy
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.hero import DiscreteAgent


def compute_route_length(route):
    """Compute full geometric path length from route (rx, ry)."""
    rx, ry = route
    assert len(rx) == len(ry), "Route coordinates mismatch!"

    dist = 0.0
    for i in range(1, len(rx)):
        dx = rx[i] - rx[i - 1]
        dy = ry[i] - ry[i - 1]
        dist += (dx*dx + dy*dy) ** 0.5

    return dist

class ActorManager:
    """Handles creation, lifecycle, and updates of all scene actors."""

    def __init__(self, size: int):
        self.size = size
        self.clear()

    def clear(self):
        """Reset actor registry to empty."""
        self.route_length = 0.0
        self.actors = {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
        }

    # ======================================================
    # --- Creation ---
    # ======================================================
    def spawn_hero(self, route, initial_speed, scale):
        """Spawn the main agent (hero vehicle)."""
        hero = DiscreteAgent(
            window_size=self.size,
            route=route,
            initial_speed=initial_speed,
            color=(0, 0, 0),
            target_speed=int(50 / scale),
            car_size=32,
        )
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
        self.actors = copy.deepcopy(actors_dict)

    # ======================================================
    # --- Simulation ---
    # ======================================================
    def reset_all(self):
        """Reset all actor controllers."""
        for k, v in self.actors.items():
            if k == "agent" or not v:
                continue
            for actor in v:
                if hasattr(actor, "reset"):
                    actor.reset()

    def step_all(self, t=0.0, dt=0.05):
        """Step all non-hero actors."""
        for k, v in self.actors.items():
            if k == "agent":
                continue
            for actor in v:
                actor.step(t, dt)

    def draw_all(self, surface):
        """Draw all actors on given pygame surface."""
        for k, v in self.actors.items():
            if not v:
                continue
            if k == "agent":
                continue
            else:
                for actor in v:
                    actor.draw(surface)

    @property
    def num_vehicles(self):
        return len(self.actors["vehicle"])
