import copy
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.hero import DiscreteAgent, ContinuousAgent
from CarlaBEV.src.sim_time import TimeConfig
from CarlaBEV.src.units import UnitContract


def compute_route_length(route, metric=None):
    """Compute route length in meters from surface/world route coordinates."""
    from CarlaBEV.envs.geometry import DEFAULT_METRIC_FRAME

    metric = metric if metric is not None else DEFAULT_METRIC_FRAME
    rx, ry = route
    return metric.route_length_meters(rx, ry)

class ActorManager:
    """Handles creation, lifecycle, and updates of all scene actors."""

    def __init__(
        self,
        size: int,
        action_space: str = "discrete",
        time_config: TimeConfig | None = None,
        units: UnitContract | None = None,
    ):
        self.size = size
        self.action_space = action_space
        # Unit contract (metric frame, clock, physics, footprints) shared by every actor.
        if units is None:
            units = UnitContract(time=time_config) if time_config is not None else UnitContract()
        self.units = units
        self.time_config = units.time
        self.clear()

    @property
    def physics_dt_s(self) -> float:
        return self.time_config.physics_dt_s

    def clear(self):
        """Reset actor registry to empty."""
        self.route_length = 0.0
        self.drivable_mask = None
        self.junctions = getattr(self, "junctions", None)
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
    def spawn_hero(self, route, initial_speed_mps, target_speed_mps, np_rng=None):
        """Spawn the main agent (hero vehicle)."""
        initial_speed = self.units.metric.speed_mps_to_surface(initial_speed_mps)
        target_speed = self.units.metric.speed_mps_to_surface(target_speed_mps)
        if self.action_space == "continuous":
             hero = ContinuousAgent(
                window_size=self.size,
                route=route,
                initial_speed=initial_speed,
                target_speed=target_speed,
                np_rng=np_rng,
                drivable_mask=self.drivable_mask,
                units=self.units,
            )
        else:
            hero = DiscreteAgent(
                window_size=self.size,
                route=route,
                initial_speed=initial_speed,
                target_speed=target_speed,
                np_rng=np_rng,
                drivable_mask=self.drivable_mask,
                units=self.units,
            )
        hero.initial_speed_mps = float(initial_speed_mps)
        hero.target_speed_mps = float(target_speed_mps)
        self.route_length = compute_route_length(route, self.units.metric)
        self.actors["agent"] = hero
        return hero

    def add_vehicle(self, start_node, end_node, routeX=None, routeY=None):
        v = Vehicle(
            start_node=start_node,
            end_node=end_node,
            map_size=self.size,
            routeX=routeX,
            routeY=routeY,
            units=self.units,
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
            units=self.units,
        )
        self.actors["pedestrian"].append(p)
        return p

    def load(self, actors_dict):
        """Load actors from a prebuilt dictionary (used when loading from CSV)."""
        drivable_mask = self.drivable_mask
        self.clear()
        for key, value in copy.deepcopy(actors_dict).items():
            if key in {"vehicle", "pedestrian"}:
                self.actors[key] = [actor for actor in value if self._has_valid_route(actor)]
            else:
                self.actors[key] = value
        self.set_drivable_mask(drivable_mask)
        self._bind_time_config()
        self.set_junctions(self.junctions)

    def _bind_time_config(self):
        """Propagate the shared unit contract (incl. clock) to every actor."""
        for key in ("vehicle", "pedestrian", "traffic_light"):
            for actor in self.actors.get(key, []):
                if hasattr(actor, "set_units"):
                    actor.set_units(self.units)
                elif hasattr(actor, "set_time_config"):
                    actor.set_time_config(self.time_config)

    def set_junctions(self, junctions):
        """Bind the junction zones (right-of-way) to every traffic actor."""
        self.junctions = junctions
        for key in ("vehicle", "pedestrian"):
            for actor in self.actors.get(key, []):
                actor.junctions = junctions

    def set_drivable_mask(self, drivable_mask):
        self.drivable_mask = drivable_mask
        for key in ("vehicle", "pedestrian"):
            for actor in self.actors.get(key, []):
                actor.drivable_mask = drivable_mask
        if self.actors.get("agent") is not None and hasattr(self.actors["agent"], "drivable_mask"):
            self.actors["agent"].drivable_mask = drivable_mask

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

    def step_all(self, t, dt):
        """Step all non-hero actors with the scene clock's (t, dt).

        Background actors perceive every other traffic actor *and the ego*, so
        that following/yielding behaviour applies to the agent as well.
        """
        traffic_actors = [
            actor
            for key in ("vehicle", "pedestrian")
            for actor in self.actors.get(key, [])
            if self._has_valid_route(actor)
        ]
        hero = self.actors.get("agent")
        perceived_actors = ([hero] if hero is not None else []) + traffic_actors
        signals = list(self.actors.get("traffic_light", []))
        finished = []
        # Signals first so vehicles observe the phase valid at end-of-step time t.
        for light in signals:
            light.step(t, dt)
        for k, v in self.actors.items():
            if k in {"agent", "traffic_light"}:
                continue
            for actor in v:
                if k in {"vehicle", "pedestrian"} and not self._has_valid_route(actor):
                    continue
                if k in {"vehicle", "pedestrian"}:
                    other_actors = [other for other in perceived_actors if other is not actor]
                    if actor.step(t, dt, other_actors=other_actors, signals=signals):
                        finished.append(actor)
                else:
                    actor.step(t, dt)
        return finished

    @property
    def num_vehicles(self):
        return len(self.actors["vehicle"])

    @staticmethod
    def _has_valid_route(actor):
        rx = getattr(actor, "rx", None)
        ry = getattr(actor, "ry", None)
        return rx is not None and ry is not None and len(rx) >= 2 and len(ry) >= 2
