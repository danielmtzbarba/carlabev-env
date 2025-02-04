import numpy as np
import pygame
import math


class Car(pygame.sprite.Sprite):
    def __init__(self, start, window_center, size, color=(0, 7, 175), car_size=32):
        pygame.sprite.Sprite.__init__(self)
        self.color = color
        self.size = size
        self.scale = int((1024 / window_center[0]) / 2)
        self._w = int(car_size / self.scale)
        self._l = 2 * self._w
        self._spawn_location = (start[0], start[1])
        self._window_center = window_center
        self._setup()

    def _setup(self):
        self._draw_rect = pygame.Rect(
            self._window_center[0] - int(self._l / 2),
            self._window_center[1] - int(self._l / 2),
            self._w,
            self._l,
        )
        self.rect = pygame.Rect(
            self._spawn_location[0], self._spawn_location[1], self._w, self._l
        )

        # movement
        self._phi, self._theta = 0, 0
        self._u1, self._u2 = 0, 0
        self._velocity = pygame.math.Vector2(0, 0)
        self._position = pygame.math.Vector2(self.rect.x, self.rect.y)
        self._reversing = False

    def step(self, action):
        """Sprite update function, calcualtes any new position"""
        self.accelerate(action[0])
        self.turn(action[1])

        if action[2] > 0:
            self.brake()

        self._velocity.x = self._u1 * math.cos(self._theta)
        self._velocity.y = self._u1 * math.sin(self._theta)

        self._position += self._velocity
        self.rect.center = (round(self._position[0]), round(self._position[1]))

    def accelerate(self, amount):
        """Increase the speed either forward or reverse"""
        if not self._reversing:
            self._u1 += amount
        else:
            self._u1 -= amount

    def brake(self):
        """Slow the car by half"""
        self._u1 /= 2
        if abs(self._u1) < 0.1:
            self._u1 = 0

    def turn(self, angle_degrees):
        """Adjust the angle the car is heading"""
        #    self._phi = np.math.radians(angle_degrees)
        #        self._theta += (1 / self._lenght) * math.tan(self._phi) * self._u1
        self._theta += math.radians(angle_degrees)

    def reverse(self):
        """Change forward/reverse, reset any speed to 0"""
        self._u1 = 0
        self._reversing = not self._reversing

    def draw(self, display):
        pygame.draw.rect(
            display,
            self.color,
            self._draw_rect,
        )

    @property
    def theta(self):
        return self._theta

    @property
    def position(self):
        return self._position

    @property
    def pose(self):
        return pygame.math.Vector3(self._position.x, self._position.y, self.theta)
