from CarlaBEV.envs import utils
from CarlaBEV.src.scenes.target import Target, target_locations

from CarlaBEV.src.scenes import build_scene


class Scene(object):
    actors = {"vehicles": [], "pedestrians": [], "target": []}

    def __init__(self, map_surface, size) -> None:
        self._map_arr, self._map_img = utils.load_map(size)
        self._map = map_surface
        self._size = size
        self._scale = int(1024 / size)
        self._const = size / 4
        self._scene_setup()

    def _scene_setup(self, scene_id=1):
        build_scene(scene_id, self.actors, self._size)
        self.next_target(0)

    def reset(self):
        for id in self.actors.keys():
            for actor in self.actors[id]:
                actor.reset()

    def next_target(self, target_id):
        self.actors["target"].clear()
        self.target = Target(target_id, scale=self._scale)
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
