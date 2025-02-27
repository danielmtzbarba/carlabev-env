import numpy as np
import pygame
import math


from CarlaBEV.src.control.state import State


class Hero(State, pygame.sprite.Sprite):
    dt = 0.1

    def __init__(self, start, window_size, color=(0, 7, 175), car_size=32):
        pygame.sprite.Sprite.__init__(self)
        State.__init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0)
        self.window_center = int(window_size / 2)
        self.scale = int(1024 / window_size)
        self.color = color
        self.w = int(car_size / self.scale)
        self.l = 2 * self.w
        #
        self.x0 = int(start[0] - self.l / 2)
        self.y0 = int(start[1] - self.w / 2)
        #
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        self.L = 2.9  # [m] Wheel base of vehicle
        self._target_speed = int(300 / self.scale)
        #
        self._setup()

    def _setup(self):
        center = self.window_center - int(self.l / 2)
        self.render_rect = pygame.Rect((center, center, self.w, self.l))
        self.rect = pygame.Rect((self.x0, self.y0, self.w, self.l))

        # movement
        self.x = self.rect.x
        self.y = self.rect.y

    def step(self, action):
        """Sprite update function, calcualtes any new position"""
        if action[2] > 0:
            acc = self.brake(action[2])
        else:
            acc = self.accelerate(action[0])

        delta = self.turn(action[1])
        self.update(acc, 0)

        self.rect.center = (round(self.x), round(self.y))

    def accelerate(self, amount):
        """Increase the speed either forward or reverse"""
        return amount * 20 * self.scale

    def brake(self, amount):
        """Slow the car by half"""
        return -amount * 10 * self.scale

    def turn(self, angle_degrees):
        """Adjust the angle the car is heading"""
        self.yaw += math.radians(angle_degrees * 13)
        return 0

    def draw(self, display):
        pygame.draw.rect(display, self.color, self.render_rect)

    @property
    def position(self):
        return pygame.math.Vector2(self.x, self.y)

    @property
    def pose(self):
        return pygame.math.Vector3(self.x, self.y, self.yaw)
