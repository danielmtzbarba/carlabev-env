import numpy as np

from CarlaBEV.envs.camera import Camera, Follow
from CarlaBEV.envs.transforms import SurfaceFrame
from CarlaBEV.src.managers.actor_manager import ActorManager
from CarlaBEV.src.scenes.utils import set_targets
from CarlaBEV.src.sim_time import SimClock
from CarlaBEV.src.units import UnitContract
from CarlaBEV.src.diagnostics import profiling
from CarlaBEV.src.geometry.collision import CollisionSystem, bodies_overlap


class Scene:
    """Main scene orchestrator connecting planners and actors."""

    cols = ["scene_id", "class", "start", "goal", "rx", "ry"]

    def __init__(
        self,
        size,
        screen=None,
        town_name="Town01",
        action_space="discrete",
        time_config=None,
        units=None,
        render_frame=None,
    ):
        self.size = size
        self.screen = screen  # legacy, unused: drawing is owned by the renderer
        self.town_name = town_name
        self.action_space = action_space
        self.surface_frame = SurfaceFrame()
        self.drivable_mask = None

        # --- Unit contract & simulation clock (single source of truth) ---
        if units is None:
            units = UnitContract(time=time_config) if time_config is not None else UnitContract()
        self.units = units
        self.time_config = units.time
        self.clock = SimClock(self.time_config)

        # --- Managers ---
        self.actor_manager = ActorManager(size, action_space=self.action_space, units=self.units)
        self._idx = 0
        # Junction zones (right-of-way) from the canonical vehicle graph, in metres.
        self.junctions = None
        if self.units.rules.right_of_way_enabled:
            from CarlaBEV.src.actors.junctions import JunctionMap

            self.junctions = JunctionMap.from_canonical_graph(
                self.units.metric, town_name=self.town_name, radius_m=self.units.rules.junction_radius_m
            )
        self.actor_manager.set_junctions(self.junctions)

        # --- Traffic lifecycle (set by the environment at reset; None = off) ---
        self.traffic_lifecycle = None

        # --- hero agent placeholder ---
        self.hero = None
        self._last_contacts = []
        # Ego signal compliance (observational: the agent is never constrained).
        self._signal_prev_gap_m = None
        self._red_light_violations = 0
        self.render_frame = render_frame if render_frame is not None else self.surface_frame
        # Semantic tile queries are answered by the map (set by BaseMap).
        self.tile_provider = None

    @property
    def agent_tile(self):
        if self.tile_provider is None or self.hero is None:
            return np.array([0, 0, 0], dtype=np.uint8)
        return self.tile_provider.semantic_tile_at(self.hero.position)

    @property
    def agent_tile_class(self):
        if self.tile_provider is None or self.hero is None:
            return None
        return self.tile_provider.semantic_class_at(self.hero.position)

    def is_obstacle_tile(self, tile):
        if self.tile_provider is None:
            return False
        return self.tile_provider.is_obstacle_tile(tile)

    def _render_frame(self):
        return getattr(self, "render_frame", self.surface_frame)

    # =====================================================
    # --- Rendering interface (observer side)
    # =====================================================
    def drawables(self):
        """Yield what the renderer must draw this frame (targets, lights, actors)."""
        from CarlaBEV.envs.world_renderer import Drawable

        actors = self.actor_manager.actors
        for target in actors.get("target", []):
            if target.visible:
                yield Drawable(class_id=int(target.semantic_class), rect_px=target.rect_px())
        for light in actors.get("traffic_light", []):
            yield Drawable(class_id=int(light.semantic_class), rect_px=light.rect_px())
        for key in ("vehicle", "pedestrian"):
            for actor in actors.get(key, []):
                if not self.actor_manager._has_valid_route(actor):
                    continue
                yield Drawable(class_id=int(actor.semantic_class), body=actor.body())

    # =====================================================
    # --- Scene lifecycle
    # =====================================================
    def reset_scene(self, actors=None, hero_np_rng=None):
        """Reset the scene with optional actor data."""
        self._idx = 0
        self.clock.reset()
        self._signal_prev_gap_m = None
        self._signal_state = None
        self._signal_gap_m = None
        self._red_light_violations = 0
        if actors:
            self.load_scene(actors, hero_np_rng=hero_np_rng)
        else:
            self.actor_manager.reset_all()

        # Metrics
        if self.hero is not None:
            self._dist2goal_t0 = self.dist2goal()
            self._dist2goal_t_1 = self.dist2goal()
            self._dist2goal = self.dist2goal()
            self._dist2wp_1 = self.hero.dist2wp
        else:
            self._dist2goal_t0 = 0.0
            self._dist2goal_t_1 = 0.0
            self._dist2goal = 0.0
            self._dist2wp_1 = 0.0

    def load_scene(self, actors, hero_np_rng=None):
        if self.drivable_mask is not None:
            self.actor_manager.set_drivable_mask(self.drivable_mask)
        self.actor_manager.load(actors)
        if self.actor_manager.actors.get("agent"):
            cx, cy, initial_speed_mps, target_speed_mps = self.agent_route
            self.route = (cx, cy)
            self.hero = self.actor_manager.spawn_hero(
                route=(cx, cy),
                initial_speed_mps=initial_speed_mps,
                target_speed_mps=target_speed_mps,
                np_rng=hero_np_rng,
            )
            self._actors = set_targets(
                self.actor_manager.actors, self.hero.cx, self.hero.cy, units=self.units
            )
            self.actor_manager.reset_all()

            # Camera
            self.camera = Camera(
                self.hero,
                resolution=(self.size, self.size),
                frame=self._render_frame(),
                crop_resolution=getattr(self, "crop_resolution", (self.size, self.size)),
            )
            follow = Follow(self.camera, self.hero)
            self.camera.setmethod(follow)
            self.camera.scroll()
        return True

    def _scene_step(self, action):
        # Advance the clock first: every consumer in this step observes the
        # end-of-step time t = (k + 1) * dt.
        t = self.clock.tick()
        with profiling.phase("scene.hero_step"):
            self.hero_step(action)
        with profiling.phase("scene.npc_step"):
            finished = self.actor_manager.step_all(t, self.clock.dt_s)
        if self.traffic_lifecycle is not None and finished:
            with profiling.phase("scene.traffic_lifecycle"):
                self.traffic_lifecycle.step(finished, self.hero, t)

        self._track_ego_signal_compliance()

        self._dist2goal_t_1 = self._dist2goal
        self._dist2goal = self.dist2goal()

    @property
    def red_light_violations(self) -> int:
        return int(getattr(self, "_red_light_violations", 0))

    def _track_ego_signal_compliance(self):
        """Count ego crossings of a red stop line that controls its heading (info only)."""
        from CarlaBEV.src.actors.traffic_policy import stop_line_ahead

        lights = self.actor_manager.actors.get("traffic_light", [])
        if self.hero is None or not lights:
            self._signal_state = None
            self._signal_gap_m = None
            return
        light, gap_m = stop_line_ahead(self.hero, lights, self.units.rules.signal_lookahead_m)
        self._signal_state = None if light is None else light.state_name
        self._signal_gap_m = None if gap_m is None else float(gap_m)
        crossed = (
            light is not None
            and self._signal_prev_gap_m is not None
            and self._signal_prev_gap_m >= 0.0 > gap_m
            and light.state_name == "red"
        )
        if crossed:
            self._red_light_violations += 1
        self._signal_prev_gap_m = None if gap_m is None else float(gap_m)

    def hero_step(self, action):
        """Advance hero agent one step and update the camera."""
        self.hero.step(action)
        self._theta = self.hero.yaw
        self.camera.scroll()

    # =====================================================
    # --- Collision detection
    # =====================================================
    @property
    def proximity_radius_m(self) -> float:
        return self.units.scene.proximity_radius_m

    def collision_check(self, min_dist=None):
        """Detect collisions and nearby actors on metric bodies.

        Vehicles and pedestrians are tested with the :class:`CollisionSystem`
        (oriented boxes / circles in meters); route targets keep their
        rectangle trigger semantics. ``min_dist`` (surface px) overrides the
        proximity radius for compatibility; by default the unit contract's
        ``proximity_radius_m`` (centre distance) is used.
        """
        proximity_m = (
            self.units.scene.proximity_radius_m
            if min_dist is None
            else self.units.metric.distance_surface_to_meters(min_dist)
        )
        collision_system = CollisionSystem(proximity_radius_m=proximity_m)
        result = None
        coll_id = None
        collision_class = None
        actors_state = []
        contacts = []
        info = self.scene_info
        ego_body = self.hero.body()
        metric = self.units.metric
        hero_rect = self.hero.surface_rect()
        for id_type, actor_list in self.actor_manager.actors.items():
            if id_type in ["agent", "traffic_light"]:
                continue
            for actor in actor_list:
                if id_type in ["vehicle", "pedestrian"]:
                    if not hasattr(actor, "body"):
                        continue
                    contact = collision_system.contact(ego_body, actor.body(), id_type, owner=actor)
                    contacts.append(contact)
                    if collision_system.within_proximity(contact):
                        ax, ay, ayaw, av = actor.state
                        avx = av * np.cos(ayaw)
                        avy = av * np.sin(ayaw)
                        pos_m = metric.surface_to_meters((ax, ay))
                        vel_m = (
                            metric.distance_surface_to_meters(avx),
                            metric.distance_surface_to_meters(avy),
                        )
                        entry = {
                            "pos": (ax, ay),
                            "vel": (avx, avy),
                            "pos_m": (float(pos_m[0]), float(pos_m[1])),
                            "vel_m": (float(vel_m[0]), float(vel_m[1])),
                        }
                        entry.update(contact.to_info())
                        actors_state.append(entry)
                    if contact.overlapping and result is None:
                        result = id_type
                        coll_id = actor.id
                        collision_class = contact.classification
                else:
                    actor_id, collision = actor.trigger(hero_rect)
                    if collision and result is None:
                        result = id_type
                        coll_id = actor_id

        self._last_contacts = contacts
        info["collision"]["collided"] = result
        info["collision"]["actor_id"] = coll_id
        info["collision"]["classification"] = collision_class
        info["collision"]["actors_state"] = actors_state
        info["collision"]["min_gap_m"] = min((c.gap_m for c in contacts), default=None)
        info["collision"]["min_ttc_s"] = min((c.ttc_s for c in contacts), default=None)

        return info

    def spawn_validation_info(self):
        if self.hero is None:
            return {"valid": False, "reason": "missing_hero"}

        hero_tile = self.agent_tile

        if hasattr(self, "is_obstacle_tile") and self.is_obstacle_tile(hero_tile):
            return {
                "valid": False,
                "reason": "hero_on_obstacle",
                "tile": hero_tile.tolist(),
            }

        ego_body = self.hero.body()
        for id_type, actor_list in self.actor_manager.actors.items():
            if id_type in ["agent", "traffic_light", "target"]:
                continue
            for actor in actor_list:
                if not hasattr(actor, "body"):
                    continue
                if bodies_overlap(ego_body, actor.body()):
                    return {
                        "valid": False,
                        "reason": "hero_overlaps_actor",
                        "actor_type": id_type,
                        "actor_id": getattr(actor, "id", None),
                    }

        return {"valid": True, "reason": "ok", "tile": hero_tile.tolist()}

    # =====================================================
    # --- Utilities
    # =====================================================
    def dist2goal(self):
        """Euclidean distance to target (surface px)."""
        return float(np.linalg.norm(np.asarray(self.hero.position, dtype=float) - np.asarray(self.target_position, dtype=float)))

    @property
    def agent_route(self):
        """Return hero route as (cx, cy)."""
        agent = self.actor_manager.actors["agent"]
        if len(agent) == 3:
            cx, cy, speed = agent
            initial_speed = speed
            target_speed = speed
        elif len(agent) == 4:
            cx, cy, initial_speed, target_speed = agent
        else:
            raise ValueError(f"Unsupported agent route format: {agent}")
        return (
            np.array(cx, dtype=np.int32),
            np.array(cy, dtype=np.int32),
            float(initial_speed),
            float(target_speed),
        )

    @property
    def curr_actors(self):
        return self.actor_manager.actors

    @property
    def _t(self):
        """Simulated time [s] observed during the last step (compatibility alias)."""
        return self.clock.t_s

    @property
    def _dt(self):
        """Physics step [s] (compatibility alias)."""
        return self.clock.dt_s

    @property
    def sim_time_s(self):
        return self.clock.t_s

    @property
    def step_index(self):
        return self.clock.step_index

    @property
    def target_position(self):
        return np.asarray(self.actor_manager.actors["target"][-1].position, dtype=float)

    @property
    def scene_info(self):
        hero_info = self.hero.controller_info
        hero_info["sim_time_s"] = self.clock.t_s
        hero_info["step_index"] = self.clock.step_index
        hero_info["physics_dt_s"] = self.clock.dt_s
        return {
            "hero": hero_info,
            "scene": {
                "sim_time_s": self.clock.t_s,
                "step_index": self.clock.step_index,
                "physics_dt_s": self.clock.dt_s,
                "dist2goal": self._dist2goal,
                "dist2goal_t_1": self._dist2goal_t_1,
                "dist2goal_m": self.units.metric.distance_surface_to_meters(self._dist2goal),
                "dist2goal_t_1_m": self.units.metric.distance_surface_to_meters(self._dist2goal_t_1),
                "num_vehicles": self.actor_manager.num_vehicles,
                "route_length": self.actor_manager.route_length,
                "speed_limit_mps": self.units.scene.speed_limit_mps,
                # Legacy key (km/h); consumers should use speed_limit_mps.
                "speed_limit": self.units.scene.speed_limit_kmh,
                "proximity_radius_m": self.units.scene.proximity_radius_m,
            },
            "signals": {
                "state": getattr(self, "_signal_state", None),
                "gap_m": getattr(self, "_signal_gap_m", None),
                "red_light_violations": int(getattr(self, "_red_light_violations", 0)),
            },
            "collision": {
                "tile": self.agent_tile,
                "tile_class": (
                    int(self.agent_tile_class)
                    if self.agent_tile_class is not None
                    else None
                ),
            },
        }
