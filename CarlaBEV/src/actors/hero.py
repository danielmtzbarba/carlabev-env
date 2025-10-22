import numpy as np
import pygame
import math

from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.gui.settings import Settings as cfg


class Hero(pygame.sprite.Sprite):
    def __init__(self, window_size, color=(0, 7, 175), car_size=32):
        pygame.sprite.Sprite.__init__(self)
        self.window_center = int(window_size / 2)
        self.scale = int(1024 / window_size)
        self.color = color
        #
        self.w = int(car_size / self.scale)
        self.l = 2 * self.w
        # Visual Offset Bug
        self._offx = -2 * 16
        self._offy = -2 * 16

    def _setup(self):
        center = self.window_center
        self.fov_rect = pygame.Rect((center, center, self.w, self.l))
        self.rect = pygame.Rect((self.x0, self.y0, self.w, self.l))
        # movement
        self.x = self.rect.x + cfg.offx
        self.y = self.rect.y + cfg.offy

    def draw(self, display_fov, display):
        pygame.draw.rect(display_fov, self.color, self.fov_rect)

    #        pygame.draw.rect(display, self.color, self.rect)

    @property
    def position(self):
        return pygame.math.Vector2(self.x, self.y)

    @property
    def pose(self):
        return pygame.math.Vector3(self.x, self.y, self.yaw)


class DiscreteAgent(Controller, Hero):
    dt = 0.1

    def __init__(
        self,
        route,
        window_size,
        target_speed=0.0,
        color=(0, 7, 175),
        car_size=32,
    ):
        Controller.__init__(self, target_speed=target_speed)
        Hero.__init__(self, window_size, color, car_size)

        xs, ys = route[0], route[1]
        self.x0 = int(xs[0])
        self.y0 = int(ys[0])
        self.acc = 0.0
        self._setup()

        self.set_route(xs, ys)
        _, self.target_idx = self.stanley_control()

    def step(self, action):
        """Sprite update function, calculates any new position"""

        _, self.target_idx = self.stanley_control()

        # === Compute acceleration and braking ===
        acc_val = self.accelerate(action[0])
        brake_val = self.brake(action[2])

        # Total longitudinal acceleration
        target_acc = acc_val - brake_val

        # === Smooth acceleration (low-pass filter) ===
        alpha = 0.2
        self.acc = (1 - alpha) * self.acc + alpha * target_acc

        # === Apply steering ===
        self.turn(action[1])

        # === Update physics ===
        self.update(self.acc, 0)

        # === Friction & velocity stabilization ===
        self.v *= 0.98
        if abs(self.v) < 0.05:
            self.v = 0.0

        # === Render update ===
        self.rect.center = (round(self.x) + self._offx, round(self.y) + self._offy)
        # natural drag proportional to speed
        self.v *= 0.985

    def accelerate(self, amount):
        """Return positive forward acceleration"""
        return amount * 1.0 * self.scale  # tweak 1.0 to 0.8–1.2 for feel

    def brake(self, amount):
        speed_factor = np.clip(abs(self.v) / 5.0, 0.3, 1.0)
        return amount * 0.6 * self.scale * speed_factor

    def turn(self, angle_degrees):
        """Turn realistically, depending on speed."""
        if abs(self.v) < 0.1:
            return  # no turn if not moving

        min_turn_scale = 0.4
        max_turn_scale = 1.2
        turn_scale = np.clip(self.v / 2.0, min_turn_scale, max_turn_scale)
        self.yaw += math.radians(angle_degrees * turn_scale)


class ContinuousAgent(Controller, Hero):
    dt = 0.1

    def __init__(
        self,
        route,
        window_size,
        target_speed=0.0,
        color=(0, 7, 175),
        car_size=32,
    ):
        Controller.__init__(self, target_speed=target_speed)
        Hero.__init__(self, window_size, color, car_size)
        #
        self.x0 = int(route[0][0] - self.l / 2)
        self.y0 = int(route[1][0] - self.w / 2)
        self._setup()
        #
        self.set_route(route[0], route[1])
        _, self.target_idx = self.stanley_control()

    def step(self, action):
        """Sprite update function, calcualtes any new position"""
        d, self.target_idx = self.stanley_control()
        acc = self.accelerate(action[1]) - self.brake(action[2]) - 0.05 * self.vx
        delta = action[0]
        self.update(acc, delta)

        self.rect.center = (round(self.x) - 2 * 16, round(self.y) - 2 * 12)

    def accelerate(self, amount):
        """Increase the speed either forward or reverse"""
        return amount * self.scale / 2

    def brake(self, amount):
        return amount * self.scale * (0.5 + 0.5 * (abs(self.vx) / self.max_speed))
