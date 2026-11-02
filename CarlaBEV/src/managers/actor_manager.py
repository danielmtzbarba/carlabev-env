import copy
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.hero import DiscreteAgent, ContinuousAgent
from CarlaBEV.src.sim_time import TimeConfig
from CarlaBEV.src.units import UnitContract


def npc_id_for(kind: str, ordinal: int) -> str:
    """Scene-scoped NPC id: ``"<kind>:<ordinal:03d>"``, ordinal over vehicles then pedestrians."""
    return f"{kind}:{int(ordinal):03d}"


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
        self.blocking_clearance_px = getattr(self, "blocking_clearance_px", None)
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
    def spawn_hero(self, route, initial_speed_mps, target_speed_mps, np_rng=None, spawn_jitter_m=None):
        """Spawn the main agent (hero vehicle).

        ``spawn_jitter_m`` replays a resolved start offset; ``np_rng`` draws one
        otherwise.
        """
        initial_speed = self.units.metric.speed_mps_to_surface(initial_speed_mps)
        target_speed = self.units.metric.speed_mps_to_surface(target_speed_mps)
        agent_cls = ContinuousAgent if self.action_space == "continuous" else DiscreteAgent
        hero = agent_cls(
            window_size=self.size,
            route=route,
            initial_speed=initial_speed,
            target_speed=target_speed,
            np_rng=np_rng,
            drivable_mask=self.drivable_mask,
            units=self.units,
            spawn_jitter_m=spawn_jitter_m,
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

    def load(self, actors_dict, *, copy_actors=True):
        """Load actors from a prebuilt dictionary.

        ``copy_actors=False`` adopts the objects as they are (the environment
        instantiates fresh actors from a scene spec on every reset, so copying
        is pure overhead there); the default deep copy protects callers that
        keep and reuse their own actor objects.
        """
        drivable_mask = self.drivable_mask
        self.clear()
        source = copy.deepcopy(actors_dict) if copy_actors else actors_dict
        for key, value in source.items():
            if key in {"vehicle", "pedestrian"}:
                self.actors[key] = [actor for actor in value if self._has_valid_route(actor)]
            else:
                self.actors[key] = value
        self._assign_missing_npc_ids()
        self.set_drivable_mask(drivable_mask)
        self._bind_time_config()
        self.set_junctions(self.junctions)

    def _assign_missing_npc_ids(self):
        """Give every traffic actor a stable, scene-scoped id (``"vehicle:003"``).

        Generated scenes assign ids at generation time (and persist them); this
        fallback covers scenario/authored actors so telemetry and replay
        validation can always address an NPC.
        """
        ordinal = 0
        for key in ("vehicle", "pedestrian"):
            for actor in self.actors.get(key, []):
                if getattr(actor, "npc_id", None) is None:
                    actor.npc_id = npc_id_for(key, ordinal)
                ordinal += 1

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

    def set_drivable_mask(self, drivable_mask, blocking_clearance_px=None):
        self.drivable_mask = drivable_mask
        if blocking_clearance_px is not None:
            self.blocking_clearance_px = blocking_clearance_px
        clearance = getattr(self, "blocking_clearance_px", None)
        for key in ("vehicle", "pedestrian"):
            for actor in self.actors.get(key, []):
                actor.drivable_mask = drivable_mask
                actor.blocking_clearance_px = clearance
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
