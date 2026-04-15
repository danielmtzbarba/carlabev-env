import numpy as np

from CarlaBEV.src.scenes.utils import *

from CarlaBEV.envs.camera import Camera, Follow
from CarlaBEV.envs.transforms import SurfaceFrame
from CarlaBEV.src.managers.actor_manager import ActorManager
from CarlaBEV.src.scenes.utils import set_targets

from CarlaBEV.src.managers.scene_serializer import SceneSerializer


class Scene:
    """Main scene orchestrator connecting planners and actors."""

    cols = ["scene_id", "class", "start", "goal", "rx", "ry"]

    def __init__(self, size, screen, town_name="Town01", action_space="discrete"):
        self.size = size
        self.screen = screen
        self.town_name = town_name
        self._scale = int(1024 / size)
        self.action_space = action_space
        #        self._const = int(size / 4) + 1
        self._const = 35
        self.surface_frame = SurfaceFrame()

        # --- Managers ---
        self.actor_manager = ActorManager(size, action_space=self.action_space)
        self.serializer = SceneSerializer()
        self._idx = 0

        # --- hero agent placeholder ---
        self._t = 0.0
        self._dt = 0.1
        self.hero = None

    # =====================================================
    # --- Scene lifecycle
    # =====================================================
    def reset_scene(self, actors=None):
        """Reset the scene with optional actor data."""
        self._idx = 0
        self._t = 0.0
        if actors:
            self.load_scene(actors)
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

    def load_scene(self, actors):
        self.actor_manager.load(actors)
        if self.actor_manager.actors.get("agent"):
            cx, cy, v = self.agent_route
            self.route = (cx, cy)
            self.hero = self.actor_manager.spawn_hero(
                route=(cx, cy),
                initial_speed=v,
                scale=self._scale,
            )
            self._actors = set_targets(
                self.actor_manager.actors, self.hero.cx, self.hero.cy
            )
            self.actor_manager.reset_all()

            # Camera
            self.hero.sync_rect(self.surface_frame)
            self.camera = Camera(
                self.hero,
                resolution=(self.size, self.size),
                frame=self.surface_frame,
                crop_resolution=getattr(self, "crop_resolution", (self.size, self.size)),
            )
            follow = Follow(self.camera, self.hero)
            self.camera.setmethod(follow)
            self.camera.scroll()
        return True

    def _scene_step(self, action):
        self._t += self._dt
        self.hero_step(action)
        self._scene.blit(self._map_img, (0, 0))
        self.actor_manager.step_all(self._t, self._dt)
        self.actor_manager.draw_all(self.map_surface, self.surface_frame)

        self._dist2goal_t_1 = self._dist2goal
        self._dist2goal = self.dist2goal()

    def hero_step(self, action):
        """Advance hero agent one step and update heading."""
        self.hero.step(action)
        self.hero.sync_rect(self.surface_frame)
        self._theta = self.hero.yaw
        self.camera.scroll()

    # =====================================================
    # --- Collision detection
    # =====================================================
    def collision_check(self, min_dist=20.0):
        """Detect collisions and nearby actors."""
        result = None
        coll_id = None
        actors_state = []
        info = self.scene_info
        self.hero.sync_rect(self.surface_frame)
        for id_type, actor_list in self.actor_manager.actors.items():
            if id_type in ["agent", "traffic_light"]:
                continue
            for actor in actor_list:
                if hasattr(actor, "sync_rect"):
                    actor.sync_rect(self.surface_frame)
                actor_id, collision, distance = actor.isCollided(self.hero, self._const)
                if id_type in ["vehicle", "pedestrian"]:
                    if abs(distance) < min_dist:
                        ax, ay, ayaw, av = actor.state
                        avx = av * np.cos(ayaw)
                        avy = av * np.sin(ayaw)
                        actors_state.append(
                            {"pos": (ax, ay), "vel": (avx, avy), "type": id_type}
                        )
                if collision:
                    result = id_type
                    coll_id = actor_id

        info["collision"]["collided"] = result
        info["collision"]["actor_id"] = coll_id
        info["collision"]["actors_state"] = actors_state

        return info

    def spawn_validation_info(self):
        if self.hero is None:
            return {"valid": False, "reason": "missing_hero"}

        self.hero.sync_rect(self.surface_frame)
        hero_tile = self.agent_tile

        if hasattr(self, "is_obstacle_tile") and self.is_obstacle_tile(hero_tile):
            return {
                "valid": False,
                "reason": "hero_on_obstacle",
                "tile": hero_tile.tolist(),
            }

        for id_type, actor_list in self.actor_manager.actors.items():
            if id_type in ["agent", "traffic_light", "target"]:
                continue
            for actor in actor_list:
                if hasattr(actor, "sync_rect"):
                    actor.sync_rect(self.surface_frame)
                if self.hero.rect.colliderect(actor.rect):
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
        """Euclidean distance to target."""
        return np.linalg.norm(self.hero.position - self.target_position)

    @property
    def agent_route(self):
        """Return hero route as (cx, cy)."""
        cx, cy, v = self.actor_manager.actors["agent"]
        return np.array(cx, dtype=np.int32), np.array(cy, dtype=np.int32), v

    @property
    def curr_actors(self):
        return self.actor_manager.actors

    @property
    def target_position(self):
        return self.actor_manager.actors["target"][-1].position

    @property
    def scene_info(self):
        return {
            "hero": self.hero.controller_info,
            "scene": {
                "dist2goal": self._dist2goal,
                "dist2goal_t_1": self._dist2goal_t_1,
                "num_vehicles": self.actor_manager.num_vehicles,
                "route_length": self.actor_manager.route_length,
                "speed_limit": 35,
            },
            "collision": {"tile": self.agent_tile},
        }
