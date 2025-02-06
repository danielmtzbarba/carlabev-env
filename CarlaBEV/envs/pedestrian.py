import numpy as np
import math
import pygame

from .utils import scale_coords, target_locations


class Pedestrian(pygame.sprite.Sprite):
    def __init__(self, target_id, color=(255, 0, 0), scale=1):
        pygame.sprite.Sprite.__init__(self)
        target_location = scale_coords(target_locations[target_id], scale)
        x, y = target_location[0], target_location[1]
        self.position = pygame.math.Vector2(x, y)
        size = int(128 / scale)
        self.color = color
        self.rect = (x, y, size, size)

    def draw(self, map):
        self.rect = pygame.draw.rect(map, self.color, self.rect)

    @property
    def pose(self):
        return pygame.math.Vector3(self.position.x, self.position.y, 0)
