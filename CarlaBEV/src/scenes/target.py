import pygame
from CarlaBEV.envs.utils import scale_coords

class Target(pygame.sprite.Sprite):
    lenght, width = 5, 5 

    def __init__(self, target_pos, color=(0, 255, 0), scale=1):
        pygame.sprite.Sprite.__init__(self)
        self.color = color
        self.scale = scale
        self.reset(target_pos)

    def reset(self, target_location):
        x, y = target_location[0], target_location[1]
        self.position = pygame.math.Vector2(x, y)
        self.rect = pygame.Rect(
            int(x - self.width / 2),
            int(y - self.lenght / 2),
            self.width,
            self.lenght,
        )

    def step(self):
        pass

    def isCollided(self, hero, offset):
        offsetx = offset - hero.rect.w / 2
        offsety = offset - hero.rect.w / 2

        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx,
            hero.rect.y + offsety,
            hero.rect.w + 1,
            hero.rect.w + 1,
        )
        result = dummy_rect.colliderect(self.rect)
        return result

    def draw(self, map):
        self.rect = pygame.draw.rect(map, self.color, self.rect)

    @property
    def pose(self):
        return pygame.math.Vector3(self.position.x, self.position.y, 0)