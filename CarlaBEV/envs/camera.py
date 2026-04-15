from abc import ABC, abstractmethod
import pygame

vec = pygame.math.Vector2


class Camera:
    def __init__(self, player, resolution, frame=None, crop_resolution=None):
        self.player = player
        self.frame = frame
        self.offset = vec(0, 0)
        self.DISPLAY_W, self.DISPLAY_H = resolution[0], resolution[1]
        self.CROP_W, self.CROP_H = (
            crop_resolution if crop_resolution is not None else resolution
        )
        self.CONST = vec(-self.CROP_W / 2, -self.CROP_H / 2)

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
        player_surface_pos = self.camera.frame.world_to_surface(self.player.position)
        self.camera.offset.x = int(player_surface_pos.x + self.camera.CONST.x)
        self.camera.offset.y = int(player_surface_pos.y + self.camera.CONST.y)
