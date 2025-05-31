import pygame

from CarlaBEV.src.planning.planner import Planner, scale_route
from CarlaBEV.src.control.stanley_controller import Controller


class Actor(pygame.sprite.Sprite):
    def __init__(
        self,
        start=None,
        goal=None,
        id=0,
        actor_size=1,
        resolution=1.0,
        routeX=None,
        routeY=None,
    ):
        pygame.sprite.Sprite.__init__(self)
        self.ds = resolution
        self.id = id

        if routeX and routeY:
            self.rx = scale_route(routeX, factor=8)
            self.ry = scale_route(routeY, factor=8)
        else:
            self._planner = Planner(id=id, actor_size=actor_size, resolution=resolution)
            self.rx, self.ry = self._planner.find_global_path(
                start, goal, self._map_size
            )

        self._x0, self._y0 = self.rx[0], self.ry[1]
        self.reset()

    def reset(self):
        self._controller = Controller(self._target_speed)
        self._controller.set_route(self.rx, self.ry, ds=self.ds)
        self.rect = pygame.Rect((self._x0, self._y0, self._size, self._size))
        #

    def step(self):
        self._controller.control_step()
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
        return self.id, result
