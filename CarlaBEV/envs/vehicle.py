import numpy as np
import pygame
import math

from .utils import load_car_sprite


class Car(pygame.sprite.Sprite):
    def __init__(self, start, window_center, size, length=1):
        pygame.sprite.Sprite.__init__(self)
        self.size = size
        self._spawn_location = (start[0], start[1])
        self._window_center = window_center
        self._lenght = length
        self._setup()

    def _setup(self, rotations=360):
        car_image = load_car_sprite(self.size)
        self.img = pygame.transform.rotozoom(car_image, 0, 1)
        self._centered_rect = self.img.get_rect(center=self._window_center)
        self.rect = self.img.get_rect()
        self.rect.center = self._spawn_location

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

        if action[1] > 0:
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
        self._u1 /= 1.2
        if abs(self._u1) < 0.1:
            self._u1 = 0

    def turn(self, angle_degrees):
        """Adjust the angle the car is heading"""
        #    self._phi = np.math.radians(angle_degrees)
        #        self._theta += (1 / self._lenght) * math.tan(self._phi) * self._u1
        self._theta += np.math.radians(angle_degrees)

    def reverse(self):
        """Change forward/reverse, reset any speed to 0"""
        self._u1 = 0
        self._reversing = not self._reversing

    def draw(self, display):
        display.blit(self.img, self._centered_rect)

    @property
    def theta(self):
        return self._theta

    @property
    def position(self):
        return self._position

    @property
    def pose(self):
        return pygame.math.Vector3(self._position.x, self._position.y, self.theta)
