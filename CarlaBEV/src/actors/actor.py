import pygame

from CarlaBEV.src.planning.planner import Planner
from CarlaBEV.src.control.stanley_controller import Controller


class Actor(pygame.sprite.Sprite):
    def __init__(
        self,
        start,
        goal,
        id=0,
        actor_size=1,
        resolution=1.0,
    ):
        pygame.sprite.Sprite.__init__(self)
        #
        self._planner = Planner(id=id, actor_size=actor_size, resolution=resolution)
        self._controller = Controller()
        rx, ry = self._planner.find_global_path(start, goal, self._map_size)
        self._controller.set_route(rx, ry, ds=resolution)
        #
        state = self._controller.state
        self._x0, self._y0 = state[0], state[1]
        self.rect = pygame.Rect((state[0], state[1], self._size, self._size))

    def reset(self):
        self.rect = pygame.Rect((self._x0, self._y0, self._size, self._size))

    def step(self):
        self._controller.control_step(target_speed=20)
        self.rect.x = self._controller.x
        self.rect.y = self._controller.y

    def draw(self, map):
        self.rect = pygame.draw.rect(map, self._color, self.rect)

    def isCollided(self, hero, offset):
        offsetx = offset - hero.rect.w / 2
        offsety = offset - hero.rect.w / 2
        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx, hero.rect.y + offsety, hero.rect.w, hero.rect.w
        )
        result = dummy_rect.colliderect(self.rect)
        return result
