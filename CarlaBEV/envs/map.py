import numpy as np
import math
import pygame
import pygame.surfarray as surfarray


from .utils import load_map, scale_coords, target_locations


class Target(pygame.sprite.Sprite):
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


class Town01(object):
    def __init__(self, target_id, size=1024, scale=1) -> None:
        self._map_arr, self._map_img = load_map(size)
        self._Y, self._X, _ = self._map_arr.shape
        self.size = size  # The size of the square grid
        self.center = (int(self.size / 2), int(self.size / 2))
        self._map_surface = pygame.Surface((self._X, self._Y))
        self._fov_surface = pygame.Surface((self.size, self.size))

        self._target = Target(target_id, scale=scale)

        self.draw_map()

        self._pad_rotation = self.center[0]
        self._theta = 0

    def draw_map(self):
        self._map_surface.blit(self._map_img, (0, 0))
        self._target.draw(self._map_surface)

    def crop_fov(self, topleft):
        self._xmin = np.clip(
            int(topleft.x), 0, self._X - self.size - self._pad_rotation - 1
        )
        self._ymin = np.clip(
            int(topleft.y), 0, self._Y - self.size - self._pad_rotation - 1
        )
        #
        self._fov = self._map_surface.subsurface(
            (
                self._xmin,
                self._ymin,
                self.size + self._pad_rotation,
                self.size + self._pad_rotation,
            )
        )
        return self._fov

    def rotate_fov(self):
        # get a rotated image
        if math.degrees(self._theta) > 90:
            pass

        rotated_image = pygame.transform.rotate(
            self._fov, math.degrees(self._theta) + 90
        )
        rotated_image_rect = rotated_image.get_rect(center=self.center)

        return rotated_image, rotated_image_rect

    def step(self, topleft):
        self.crop_fov(topleft)
        rotated_image, rotated_image_rect = self.rotate_fov()
        self._fov_surface.blit(rotated_image, rotated_image_rect)
        self._agent_tile = self._fov_surface.get_at(self.center)

    def got_target(self, hero):
        const = self.size / 4
        offsetx = const - hero.rect.w / 2
        offsety = const - hero.rect.w / 2
        dummy_rect = pygame.Rect(
            hero.rect.x + offsetx, hero.rect.y + offsety, hero.rect.w, hero.rect.w
        )
        result = dummy_rect.colliderect(self._target.rect)
        return result

    def has_collided(self, vehicle_rect, class_color):
        pixels = surfarray.pixels3d(
            self._fov_surface.subsurface(
                pygame.Rect(*self.center, vehicle_rect[2], vehicle_rect[3])
            )
        )
        if class_color in pixels:
            return True
        return False

    def set_theta(self, theta):
        self._theta = theta

    @property
    def canvas(self):
        return self._fov_surface

    @property
    def agent_tile(self):
        return self._agent_tile

    @property
    def target_pose(self):
        return self._target.pose

    @property
    def target_position(self):
        return self._target.position
