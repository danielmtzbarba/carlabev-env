import random
import pygame

from CarlaBEV.envs import utils
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.target import Target
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
        start = utils.scale_coords(start, 8)
        goal = utils.scale_coords(goal, 8)
        self._pedestrian = Pedestrian(start, goal, map_size=self._size)
        start = (8720, 6000)
        goal = (8720, 2000)
        start = utils.scale_coords(start, 8)
        goal = utils.scale_coords(goal, 8)
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

    def got_target(self, hero):
        offsetx = self._const - hero.rect.w / 2
        offsety = self._const - hero.rect.w / 2
        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx,
            hero.rect.y + offsety,
            hero.rect.w + 1,
            hero.rect.w + 1,
        )
        result = dummy_rect.colliderect(self._target.rect)
        return result

    def hit_pedestrian(self, hero):
        offsetx = self._const - hero.rect.w / 2
        offsety = self._const - hero.rect.w / 2
        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx, hero.rect.y + offsety, hero.rect.w, hero.rect.w
        )
        result = dummy_rect.colliderect(self._pedestrian.rect)
        return result

    @property
    def target_position(self):
        return self._target.position

    @property
    def target_pose(self):
        return self._target.pose
