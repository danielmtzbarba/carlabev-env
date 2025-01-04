from enum import Enum
import numpy as np
import pygame
import math

robot_img_path = (
    "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/envs/robot-gr.png"
)


class Actions(Enum):
    nothing = 0
    left = 1
    right = 2
    gas = 3
    brake = 4


class Car(pygame.sprite.Sprite):
    _action_to_direction = {
        Actions.nothing.value: np.array([0, 0, 0]),
        Actions.left.value: np.array([0, 10, 0]),
        Actions.right.value: np.array([0, -10, 0]),
        Actions.gas.value: np.array([1, 0, 0]),
        Actions.brake.value: np.array([0, 0, 1]),
    }

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.speed = 0
        self.theta = 0
        self._setup()
        self.velocity = pygame.math.Vector2(0, 0)
        self.position = pygame.math.Vector2(self.rect.x, self.rect.y)

    def _setup(self, rotations=360):
        car_image = pygame.transform.scale(
            pygame.image.load("car.png").convert_alpha(), (15, 35)
        )
        self.rot_img = []
        self.min_angle = 360 / rotations
        for i in range(rotations):
            # This rotation has to match the angle in radians later
            # So offet the angle (0 degrees = "north") by 90° to be angled 0-radians (so 0 rad is "east")
            rotated_image = pygame.transform.rotozoom(
                car_image, 360 - 90 - (i * self.min_angle), 1
            )
            self.rot_img.append(rotated_image)

        self.min_angle = math.radians(self.min_angle)  # don't need degrees anymore
        # define the image used
        self.img = self.rot_img[0]
        self.rect = self.img.get_rect()
        x = 512 + 512
        y = 7679 + 512
        self.rect.center = (x, y)
        # movement
        self.reversing = False
        self.velocity = pygame.math.Vector2(0, 0)
        self.position = pygame.math.Vector2(x, y)

        self.box = pygame.Rect(self.rect.x, self.rect.y, self.rect.w * 2, self.rect.h)
        self.box.center = self.rect.center

    def update(self, action):
        """Sprite update function, calcualtes any new position"""
        action = self._action_to_direction[action]
        self.accelerate(action[0])
        self.turn(action[1])

        if action[1] > 0:
            self.brake()

        self.velocity.from_polar((self.speed, math.degrees(self.theta)))
        self.position += self.velocity
        self.rect.center = (round(self.position[0]), round(self.position[1]))
        # print(self.rect.x, self.rect.y, math.degrees(self.theta))

    def accelerate(self, amount):
        """Increase the speed either forward or reverse"""
        if not self.reversing:
            self.speed += amount
        else:
            self.speed -= amount

    def brake(self):
        """Slow the car by half"""
        self.speed /= 1.2
        if abs(self.speed) < 0.1:
            self.speed = 0

    def turn(self, angle_degrees):
        """Adjust the angle the car is heading, if this means using a
        different car-image, select that here too"""
        ### TODO: car shouldn't be able to turn while not moving
        self.theta += np.math.radians(angle_degrees)

    def reverse(self):
        """Change forward/reverse, reset any speed to 0"""
        self.speed = 0
        self.reversing = not self.reversing

    def draw(self, display, pos=(512, 512), originPos=(0, 0)):
        # get a rotated image
        rotated_image = pygame.transform.rotate(self.img, 0)
        rotated_image_rect = rotated_image.get_rect(center=(512, 512))
        # rotate and blit the image
        display.blit(rotated_image, rotated_image_rect)
