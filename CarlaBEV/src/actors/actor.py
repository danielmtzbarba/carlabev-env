import pygame

from CarlaBEV.src.planning.planner import Planner
from CarlaBEV.src.control.control import Controller


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
        self._controller.set_route(rx, ry, resolution=resolution)
        #
        state = self._controller.state
        self._x0, self._y0 = state.x, state.y
        self.rect = pygame.Rect((state.x, state.y, self._size, self._size))

    def reset(self):
        self.rect = pygame.Rect((self._x0, self._y0, self._size, self._size))

    def step(self):
        state = self._controller.control_step()
        self.rect.x = state.x
        self.rect.y = state.y

    def draw(self, map):
        self.rect = pygame.draw.rect(map, self._color, self.rect)
