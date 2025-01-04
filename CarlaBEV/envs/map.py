import numpy as np
import math
import pygame

from PIL import Image

# home
map_path = "/home/danielmtz/Data/datasets/CarlaBEV/maps/Town01/Town01-padded.jpg"

# msi
# map_path = "/home/dan/Data/datasets/CarlaBEV/Town01-1024-RGB.jpg"


class Target(pygame.sprite.Sprite):
    _img = pygame.image.load("rectangle-16.png")

    def __init__(self, target_location):
        pygame.sprite.Sprite.__init__(self)
        self._win_size = 512
        self.rect = self._img.get_rect()
        x, y = target_location[0], target_location[1]
        self.rect.center = (x, y)
        self.position = pygame.math.Vector2(x, y)

        self.box = pygame.Rect(self.rect.x, self.rect.y, self.rect.w * 2, self.rect.h)
        self.box.center = self.rect.center

    def distance_from_agent(self, agent_pos):
        x = int(self.position.x - agent_pos.x)
        y = int(self.position.y - agent_pos.y)
        print(f"agent: {agent_pos.x}, target: {self.position.x} x={x}")
        return x, y

    def draw(self, canvas, agent_pos):
        x, y = self.distance_from_agent(agent_pos)
        canvas.blit(self._img, self.rect)


class Town01(object):
    _img = pygame.image.load(map_path)

    def __init__(self, window_size, target_location) -> None:
        self._map_arr = np.array(Image.open(map_path))
        self._target = Target(target_location)
        self._win_size = window_size[0]
        self._Y, self._X, _ = self._map_arr.shape
        self._origin = ((self._Y / 2), int(self._X / 2))
        self._pad_rotation = 500
        self._theta = 0

    def crop_fov(self, topleft):
        self._xmin = np.clip(
            int(topleft.x), 0, self._X - self._win_size - self._pad_rotation - 1
        )
        self._ymin = np.clip(
            int(topleft.y), 0, self._Y - self._win_size - self._pad_rotation - 1
        )
        #
        self._fov = self._img.subsurface(
            (
                self._xmin,
                self._ymin,
                self._win_size + self._pad_rotation,
                self._win_size + self._pad_rotation,
            )
        )
        return self._fov

    def draw_target(self, canvas, agent_pos):
        self._target.draw(canvas, agent_pos)

    def rotate_fov(self, pos, originPos=(512, 512)):
        # get a rotated image
        if np.math.degrees(self._theta) > 90:
            pass

        rotated_image = pygame.transform.rotate(self._fov, np.math.degrees(self._theta))
        rotated_image_rect = rotated_image.get_rect(center=(512, 512))

        return rotated_image, rotated_image_rect

    def blit(self, display, agent_pos):
        display.blit(self._img, agent_pos)

    def blitRotate(self, display, topleft, pos=(0, 0), originPos=(0, 0)):
        self.crop_fov(topleft)
        rotated_image, rotated_image_rect = self.rotate_fov(pos, originPos)

        display.blit(rotated_image, rotated_image_rect)
        self._agent_tile = display.get_at((512, 512))

        return display

    def set_theta(self, theta):
        self._theta = theta

    @property
    def target_rect(self):
        return self._target.rect

    @property
    def agent_tile(self):
        return self._agent_tile

    @property
    def origin(self):
        return self._origin
