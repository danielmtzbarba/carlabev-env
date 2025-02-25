from CarlaBEV.envs import utils
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.target import Target, target_locations
from CarlaBEV.src.actors.pedestrian import Pedestrian


class Scene(object):
    def __init__(self, map_surface, size) -> None:
        self._map_arr, self._map_img = utils.load_map(size)
        self._map = map_surface
        self._curr_goal_id = 0
        self._size = size
        self._scale = int(1024 / size)
        self._const = size / 4
        self._scene_setup(target_id=self._curr_goal_id)

    def _scene_setup(self, target_id):
        start = (8704, 3650)
        goal = (8704, 6650)
        self._pedestrian = Pedestrian(start, goal, map_size=self._size)
        start = (8720, 2000)
        goal = (8720, 6000)
        self._vehicle = Vehicle(start, goal, map_size=self._size)
        self.next_target(target_id)

    def next_target(self, target_id):
        self._target = Target(target_id, scale=self._scale)

    def draw(self):
        self._map.blit(self._map_img, (0, 0))
        self._target.draw(self._map)
        self._pedestrian.draw(self._map)
        self._vehicle.draw(self._map)

    def step(self):
        self._pedestrian.step()
        self._vehicle.step()
        self.draw()

    def collision_check(self, hero):
        result = None
        if self._pedestrian.isCollided(hero, self._const):
            result = "ped"
        if self._vehicle.isCollided(hero, self._const):
            result = "car"
        if self._target.isCollided(hero, self._const):
            result = "target"
        return result

    @property
    def num_targets(self):
        return len(target_locations) - 1

    @property
    def target_position(self):
        return self._target.position

    @property
    def target_pose(self):
        return self._target.pose
