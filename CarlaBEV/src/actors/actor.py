import pygame
import numpy as np

from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.planning.planner import Planner, scale_route
from CarlaBEV.src.control.stanley_controller import Controller

class Node(object):
    def __init__(self, id, position, lane=None):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        self.draw_x = self._x + cfg.offx
        self.draw_y = self._y + cfg.offy
        self.btn = pygame.Rect(self.draw_x, self.draw_y,  3, 3)
        self.color = None 
    
    def reset(self):
        self.color = None 
    
    def render(self, screen, color=None):
        if color is not None:
            self.color = color

        if self.color is not None:
            pygame.draw.rect(screen, self.color, self.btn)
    
    def clicked(self, event):
        if self.btn.collidepoint(event.pos):
            self.color = cfg.red 
            return True
    
    @property
    def scaled_pos(self):
        return [self._x, self._y]

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]


class Actor(pygame.sprite.Sprite):
    def __init__(
        self,
        start_node=None,
        end_node=None,
        id=0,
        actor_size=1,
        resolution=1.0,
        routeX=None,
        routeY=None,
    ):
        pygame.sprite.Sprite.__init__(self)
        self.id = id
        self.ds = resolution

        self.start_node = start_node
        self.end_node = end_node
        #
        if routeX and routeY:
            self.rx = scale_route(routeX, factor=8)
            self.ry = scale_route(routeY, factor=8)
        else:
            self.rx, self.ry = [], [] 
        #
        self.path = []
        self.selected = False


    def set_route_wp(self, node_id, x, y):
        self.rx.append(x)
        self.ry.append(y)
        pos = np.array([x, y])
        self.path.append(Node(node_id, pos))

    def reset(self):
        self._controller = Controller(self._target_speed)
        self._controller.set_route(self.rx, self.ry, ds=self.ds)
        self.rect = pygame.Rect((self.rx[0], self.ry[0], self._size, self._size))

    def step(self):
        self._controller.control_step()
        self.rect.x = self._controller.x
        self.rect.y = self._controller.y

    def draw(self, screen):
        if self.selected:
            self.start_node.render(screen, cfg.green)
            self.end_node.render(screen, cfg.red)
            for node in self.path:
                node.render(screen, cfg.blue)

        self.rect = pygame.draw.rect(screen, self._color, self.rect)

    def isCollided(self, hero, offset):
        offsetx = offset - hero.rect.w / 2
        offsety = offset - hero.rect.w / 2
        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx, hero.rect.y + offsety, hero.rect.w, hero.rect.w
        )
        result = dummy_rect.colliderect(self.rect)
        return self.id, result

    @property
    def data(self):
        return [None, self.id, self.start_node.scaled_pos, self.end_node.scaled_pos, self.rx, self.ry]
