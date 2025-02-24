import pygame
from CarlaBEV.envs.utils import scale_coords

target_locations = [
    (8704, 2000),
    (8704, 2250),
    (8704, 2500),
    (8704, 2750),
]

"""
(8704, 3000),
(8704, 3250),
(8704, 3500),
(8704, 3750),
(8704, 4000),
(8704, 4250),
(8704, 4500),
(8704, 4750),
(8704, 5000),
(8704, 5250),
(8704, 5500),
(8704, 5750),
(8704, 6000),
(8704, 6250),
(8704, 6500),
(8704, 6650),
#
(8650, 6800),
(8500, 6800),
(8400, 6800),
(8250, 6800),
(8050, 6800),
(7850, 6800),
(7650, 6800),
(7450, 6800),
(7250, 6700),
(7250, 6500),
(7250, 6300),
(7250, 6100),
(7250, 5900),
(7250, 5700),
(7250, 5500),
(7250, 5300),
(7250, 5100),
(7250, 4900),
(7250, 4700),
"""


class Target(pygame.sprite.Sprite):
    lenght, width = 130, 2

    def __init__(self, target_id, color=(0, 255, 0), scale=1):
        pygame.sprite.Sprite.__init__(self)
        target_location = scale_coords(target_locations[target_id], scale)
        x, y = target_location[0], target_location[1]
        self.position = pygame.math.Vector2(x, y)
        self.lenght = int(self.lenght / scale)
        self.color = color
        self.rect = (
            int(x - self.width / 2),
            int(y - self.lenght / 2),
            self.width,
            self.lenght,
        )

    def draw(self, map):
        self.rect = pygame.draw.rect(map, self.color, self.rect)

    @property
    def pose(self):
        return pygame.math.Vector3(self.position.x, self.position.y, 0)
