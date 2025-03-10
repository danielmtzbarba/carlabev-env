from random import choice
from CarlaBEV.envs import utils
from CarlaBEV.src.scenes.target import Target, target_locations

from CarlaBEV.src.scenes import SceneBuilder

SCENE_IDS = ["scene_1-1", "scene_1-2", "scene_1-3"]


class Scene(object):
    def __init__(self, map_surface, size) -> None:
        self._target_id = 0
        self._map_arr, self._map_img = utils.load_map(size)
        self._scene_ids = SCENE_IDS
        self._buider = SceneBuilder(self._scene_ids, size)
        self._map = map_surface
        self._size = size
        self._scale = int(1024 / size)
        self._const = size / 4
        self.reset()

    def reset(self):
        self._target_id = 0
        rdm_id = choice(self._scene_ids)
        self.actors = self._buider.get_scene_actors(rdm_id)
        self.next_target(self._target_id)
        for id in self.actors.keys():
            for actor in self.actors[id]:
                actor.reset()

    def next_target(self, target_id):
        self._target_id = target_id
        self.actors["target"].clear()
        self.target = Target(self._target_id, scale=self._scale)
        self.actors["target"].append(self.target)
        return self.target

    def step(self):
        self._map.blit(self._map_img, (0, 0))
        for id in self.actors.keys():
            for actor in self.actors[id]:
                actor.step()
                actor.draw(self._map)

    def collision_check(self, hero):
        result = None
        for id in self.actors.keys():
            for actor in self.actors[id]:
                collision = actor.isCollided(hero, self._const)
                if collision:
                    result = id
        return result

    @property
    def num_targets(self):
        return len(target_locations) - 1

    @property
    def target_position(self):
        return self.target.position

    @property
    def target_pose(self):
        return self.target.pose
