import numpy as np

from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.scenes.utils import *


from CarlaBEV.src.managers.planner_manager import PlannerManager
from CarlaBEV.src.managers.actor_manager import ActorManager
from CarlaBEV.src.scenes.utils import set_targets

from CarlaBEV.src.managers.scene_generator import SceneGenerator
from CarlaBEV.src.managers.scene_serializer import SceneSerializer


class Scene:
    """Main scene orchestrator connecting planners and actors."""

    cols = ["scene_id", "class", "start", "goal", "rx", "ry"]

    def __init__(self, size, screen, town_name="Town01"):
        self.size = size
        self.screen = screen
        self.town_name = town_name
        self._scale = int(1024 / size)
        self._const = int(size / 4) + 1

        # --- Managers ---
        self.planners = PlannerManager(town_name)
        self.actor_manager = ActorManager(size)
        self.generator = SceneGenerator(self.planners, map_size=self.size)
        self.serializer = SceneSerializer()
        self._idx = 0

        # --- hero agent placeholder ---
        self.hero = None

    # =====================================================
    # --- Scene lifecycle
    # =====================================================
    def reset_scene(self, episode, actors=None):
        """Reset the scene with optional actor data."""
        self._idx = 0

        if actors:
            self.actor_manager.load(actors)
            if self.actor_manager.actors.get("agent"):
                self.hero = self.actor_manager.spawn_hero(
                    route=self.agent_route,
                    scale=self._scale,
                )
                self._actors = set_targets(
                    self.actor_manager.actors, self.hero.cx, self.hero.cy
                )
                self.actor_manager.reset_all()
        else:
            self.actor_manager.clear()
            actors = self.generator.generate_random(episode)
            self.reset_scene(episode, actors)

    def _scene_step(self):
        self._scene.blit(self._map_img, (0, 0))
        self.actor_manager.step_all()
        self.actor_manager.draw_all(self.map_surface)

    # =====================================================
    # --- Collision detection
    # =====================================================
    def collision_check(self, min_dist=20.0):
        """Detect collisions and nearby actors."""
        result = None
        coll_id = None
        actors_state = []

        for id_type, actor_list in self.actor_manager.actors.items():
            if id_type == "agent":
                continue
            for actor in actor_list:
                actor_id, collision, distance = actor.isCollided(self.hero, self._const)
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

        return coll_id, result, actors_state

    # =====================================================
    # --- Utilities
    # =====================================================
    @property
    def agent_route(self):
        """Return hero route as (cx, cy)."""
        cx, cy = self.actor_manager.actors["agent"]
        return np.array(cx, dtype=np.int32), np.array(cy, dtype=np.int32)

    @property
    def curr_actors(self):
        return self.actor_manager.actors

    @property
    def target_position(self):
        return self.actor_manager.actors["target"][-1].position
