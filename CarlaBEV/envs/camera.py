from abc import ABC, abstractmethod
import pygame

vec = pygame.math.Vector2


class Camera:
    def __init__(self, player, resolution):
        self.player = player
        self.offset = vec(0, 0)
        self.DISPLAY_W, self.DISPLAY_H = resolution[0], resolution[1]
        self.CONST = vec(-self.DISPLAY_W / 2 + player.rect.w / 2)

    def setmethod(self, method):
        self.method = method

    def scroll(self):
        self.method.scroll()


class CamScroll(ABC):
    def __init__(self, camera, player):
        self.camera = camera
        self.player = player

    @abstractmethod
    def scroll(self):
        pass


class Follow(CamScroll):
    def __init__(self, camera, player):
        CamScroll.__init__(self, camera, player)

    def scroll(self):
        self.camera.offset.x = int(self.player.rect.x + self.camera.CONST.x)
        self.camera.offset.y = int(self.player.rect.y + self.camera.CONST.y)
