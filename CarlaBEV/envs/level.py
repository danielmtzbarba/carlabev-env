from numpy import random
import pygame

from .utils import scale_coords, target_locations, load_map, pedestrian_locations


class Scene(object):
    def __init__(self, map_surface, size) -> None:
        self._map_arr, self._map_img = load_map(size)
        self._map = map_surface
        self._curr_goal_id = 0
        self._size = size
        self._scale = int(1024 / size)
        self._const = size / 4
        self.trigger = random.randint(0, 5)
        self._scene_setup(target_id=self._curr_goal_id)

    def _scene_setup(self, target_id):
        self._pedestrian = Pedestrian(scale=self._scale)
        self.next_target(target_id)

    def next_target(self, target_id):
        if target_id == self.trigger:
            self._pedestrian.trigger()
        self._target = Target(target_id, scale=self._scale)

    def draw(self):
        self._map.blit(self._map_img, (0, 0))
        self._target.draw(self._map)
        self._pedestrian.draw(self._map)

    def step(self):
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
