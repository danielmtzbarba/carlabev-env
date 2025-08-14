import numpy as np
from CarlaBEV.envs import utils
import pygame

class Scene(object):
    def __init__(self, size) -> None:
        self._scale = int(1024 / size)
        self._const = size / 4

    def reset(self, actors):
        self.actors = actors
        for id in self.actors.keys():
            if id == "agent":
                self.hero = self.Agent(
                    route=self.agent_route,
                    window_size=self.size,
                    color=(0, 0, 0),
                    target_speed=int(200 / self._scale),
                    car_size=32,
                )
                continue
            for actor in self.actors[id]:
                actor.reset()

    def _scene_step(self, course):
        self._scene.blit(self._map_img, (0, 0))
        cx, cy, cyaw = course
        for x, y in zip(cx, cy):
            pygame.draw.circle(self._scene, color=(0, 255, 0), center=(x, y), radius=1)
        for id in self.actors.keys():
            if id == "agent":
                continue
            for actor in self.actors[id]:
                actor.step()
                actor.draw(self._scene)

    def collision_check(self):
        result = None
        coll_id = None
        for id in self.actors.keys():
            if id == "agent":
                continue
            for actor in self.actors[id]:
                actor_id, collision = actor.isCollided(self.hero, self._const)
                if collision:
                    result = id
                    coll_id = actor_id
        return coll_id, result

    @property
    def agent_route(self):
        cx, cy = self.actors["agent"]
        cx = np.array(cx, dtype=np.int32)
        cy = np.array(cy, dtype=np.int32)
        return (cx, cy)

    @property
    def num_targets(self):
        return len(self.actors["target"]) - 1

    @property
    def target_position(self):
        return self.actors["target"][self.num_targets].position
