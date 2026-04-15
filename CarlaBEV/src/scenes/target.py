import pygame


class Target(pygame.sprite.Sprite):
    def __init__(self, id, target_pos, color=(0, 255, 128), size=1, scale=1):
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
        self.rect = pygame.Rect(0, 0, self.width, self.lenght)

    def sync_rect(self, frame):
        self.rect = frame.rect_from_world_center(
            (self.position.x, self.position.y), (self.width, self.lenght)
        )

    def step(self, t, dt):
        pass

    def isCollided(self, hero, offset=None):
        result = None
        if self._visible:
            result = hero.rect.colliderect(self.rect)
            if result:
                self._visible = False

        return self.id, result, 9999

    def draw(self, map, frame=None):
        if self.visible:
            if frame is not None:
                self.sync_rect(frame)
            self.rect = pygame.draw.rect(map, self.color, self.rect)

    @property
    def visible(self):
        return self._visible

    @property
    def pose(self):
        return pygame.math.Vector3(self.position.x, self.position.y, 0)
