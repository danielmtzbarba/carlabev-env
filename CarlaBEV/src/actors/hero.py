import numpy as np
import pygame
import math


from CarlaBEV.src.control.stanley_controller import Controller


class Hero(pygame.sprite.Sprite):
    def __init__(self, window_size, color=(0, 7, 175), car_size=32):
        pygame.sprite.Sprite.__init__(self)
        self.window_center = int(window_size / 2)
        self.scale = int(1024 / window_size)
        self.color = color
        #
        self.w = int(car_size / self.scale)
        self.l = self.w
        # Visual Offset Bug
        self._offx = -2 * 16
        self._offy = -2 * 16

    def _setup(self):
        center = self.window_center
        self.render_rect = pygame.Rect((center, center, self.w, self.l))
        self.rect = pygame.Rect((self.x0, self.y0, self.w, self.l))
        # movement
        self.x = self.rect.x
        self.y = self.rect.y

    def draw(self, display):
        pygame.draw.rect(display, self.color, self.render_rect)

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
        #
        self.x0 = int(route[0][0])
        self.y0 = int(route[1][0])
        self.acc = 0.0
        self._setup()
        #
        self.set_route(route[0], route[1])
        _, self.target_idx = self.stanley_control()

    def step(self, action):
        """Sprite update function, calculates any new position"""

        _, self.target_idx = self.stanley_control()

        # === Combine acceleration and braking ===
        if action[2] > 0:  # brake
            target_acc = self.brake(action[2])
        else:
            target_acc = self.accelerate(action[0])

        # === Apply low-pass filter to smooth acceleration ===
        alpha = 0.2  # smoothing factor (0 = no update, 1 = instant)
        self.acc = (1 - alpha) * self.acc + alpha * target_acc

        # === Steering remains discrete ===
        self.turn(action[1])

        # === Apply smoothed acceleration ===
        self.update(self.acc, 0)

        # === Update rendering ===
        self.rect.center = (round(self.x) + self._offx, round(self.y) + self._offy)

    def accelerate(self, amount):
        """Return target acceleration based on gas input"""
        return amount * 1.0 * self.scale  # positive accel

    def brake(self, amount):
        """Return target acceleration based on brake input"""
        return -amount * 2.0 * self.scale  # negative accel

    def turn(self, angle_degrees):
        """
        Adjust the angle the car is heading only if it's moving.
        Turn the car realistically — allow turning at low speed, but reduced.
        """
        if abs(self.v) < 0.1:
            return  # No turning allowed when car is stationary

        min_turn_scale = 0.4  # allow some turning even when nearly stopped
        max_turn_scale = 1.2  # allow a slight boost to turning at higher speeds
        turn_scale = np.clip(self.v / 2.0, min_turn_scale, max_turn_scale)
        self.yaw += math.radians(angle_degrees * turn_scale)
        return


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
        acc = self.accelerate(action[1]) - self.brake(action[2])
        delta = action[0]
        self.update(acc, delta)

        self.rect.center = (round(self.x) - 2 * 16, round(self.y) - 2 * 12)

    def accelerate(self, amount):
        """Increase the speed either forward or reverse"""
        return amount * self.scale / 2

    def brake(self, amount):
        """Slow the car by half"""
        return -amount * self.scale / 4
