import pygame


class Target(pygame.sprite.Sprite):
    def __init__(self, id, target_pos, color=(255, 128, 0), size=1, scale=1):
        pygame.sprite.Sprite.__init__(self)
        self.id = id
        self.color = color
        self.lenght = size
        self.width = size
        self.scale = scale
        self._visible = False
        self.x0, self.y0 = target_pos[0], target_pos[1]

    def reset(self):
        self._visible = True
        self.position = pygame.math.Vector2(self.x0, self.y0)
        self.rect = pygame.Rect(
            int(self.x0 - self.width / 2),
            int(self.y0 - self.lenght / 2),
            self.width,
            self.lenght,
        )

    def step(self):
        pass

    def isCollided(self, hero, offset):
        result = None
        if self._visible:
            offsetx = offset - hero.rect.w / 2
            offsety = offset - hero.rect.w / 2

            dummy_rect = pygame.Rect(
                hero.rect.x + offsetx,
                hero.rect.y + offsety,
                hero.rect.w + 1,
                hero.rect.w + 1,
            )
            result = dummy_rect.colliderect(self.rect)
            if result:
                self._visible = False

        return self.id, result

    def draw(self, map):
        if self.visible:
            self.rect = pygame.draw.rect(map, self.color, self.rect)

    @property
    def visible(self):
        return self._visible

    @property
    def pose(self):
        return pygame.math.Vector3(self.position.x, self.position.y, 0)
