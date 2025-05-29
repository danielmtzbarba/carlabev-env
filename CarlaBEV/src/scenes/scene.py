from random import choice
import numpy as np
from CarlaBEV.envs import utils
from CarlaBEV.src.scenes.target import Target
import pygame

from CarlaBEV.src.scenes import SceneBuilder

SCENE_IDS = [f"scene-{i}" for i in range(10)]
#SCENE_IDS = ["scene-debug"]


class Scene(object):
    def __init__(self, map_surface, size) -> None:
        self._target_id = 0
        self._target_locations = []
        self._map_arr, self._map_img, semap_8 = utils.load_map(size)
        self._scene_ids = SCENE_IDS
        self._buider = SceneBuilder(self._scene_ids, size, semap_8)
        self._map = map_surface
        self._size = size
        self._scale = int(1024 / size)
        self._const = size / 4
        self.reset()

    def reset(self):
        #
        self._target_id = 0
        self._target_locations = []
        #
        rdm_id = choice(self._scene_ids)
        self.actors = self._buider.get_scene_actors(rdm_id)

        for id in self.actors.keys():
            if id == "agent" or id == "target":
                continue
            for actor in self.actors[id]:
                actor.reset()

        rx, ry = self.actors["agent"]
        self._target_locations = [(x, y) for x, y in zip(rx, ry)]
        #
        self.next_target(self._target_id)

    def next_target(self, target_id):
        self._target_id = target_id
        self.actors["target"].clear()
        self.target = Target(self._target_locations[self._target_id], scale=self._scale)
        self.actors["target"].append(self.target)
        return self.target

    def step(self, course):
        self._map.blit(self._map_img, (0, 0))
        cx, cy, cyaw = course
        for x, y in zip(cx, cy):
            pygame.draw.circle(self._map, color=(0, 255, 0), center=(x, y), radius=1)
        for id in self.actors.keys():
            if id == "agent":
                continue
            for actor in self.actors[id]:
                actor.step()
                actor.draw(self._map)

    def collision_check(self, hero):
        result = None
        for id in self.actors.keys():
            if id == "agent":
                continue
            for actor in self.actors[id]:
                collision = actor.isCollided(hero, self._const)
                if collision:
                    result = id
        return result

    @property
    def agent_route(self):
        cx, cy = self.actors["agent"]
        cx = np.array(cx, dtype=np.int32)
        cy = np.array(cy, dtype=np.int32)
        return (cx, cy)

    @property
    def num_targets(self):
        return len(self._target_locations) - 1

    @property
    def target_position(self):
        return self.target.position

    @property
    def current_ckpt(self):
        return self.target.position

    @property
    def final_target(self):
        final_target = Target(self._target_locations[-1], scale=self._scale)
        return final_target.position
