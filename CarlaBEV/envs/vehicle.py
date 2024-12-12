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
        Actions.left.value: np.array([0, 0.15, 0]),
        Actions.right.value: np.array([0, -0.15, 0]),
        Actions.gas.value: np.array([1, 0, 0]),
        Actions.brake.value: np.array([0, 0, 1]),
    }

    def __init__(self, spawn_position, length):
        pygame.sprite.Sprite.__init__(self)
        self.img = pygame.image.load(robot_img_path)
        self.rect = self.img.get_rect()
        self._win_center = (512, 512)
        # Robot settings
        self.x = spawn_position[0]  # X position
        self.y = spawn_position[1]  # Y position
        self.theta = math.degrees(spawn_position[2])  # Initial heading angle (rad)
        self._phi = 0  # Initial steering angle (rad)
        self._v = 0  # m/s
        self.L = self.meters_to_pixels(meters=length)  # m (meters)
        self.max_speed = self.meters_to_pixels(meters=5)  # m/s
        self.min_speed = self.meters_to_pixels(meters=0)  # m/s

        # Graphics
        self.rotated = self.img
        self.rect = self.rotated.get_rect(
            center=(self._win_center[0], self._win_center[1])
        )

        # Time variant
        self.dt = 0.0  # Delta time
        self.last_time = pygame.time.get_ticks()  # Last time recorded
        self.x_position = []
        self.y_position = []
        self.theta_orientation = []

        # Actions
        self._gas = 0.0
        self._steer = 0.0
        self._brake = 0.0
        self._movex, self._movey = 0, 0

        #
        self.LEFT_KEY, self.RIGHT_KEY, self.FACING_LEFT = False, False, False
        self.UP_KEY, self.DOWN_KEY = False, False
        self.vx, self.vy = 0, 0
        self.box = pygame.Rect(self.rect.x, self.rect.y, self.rect.w * 2, self.rect.h)
        self.box.center = self.rect.center
        self.passed = False

    def update(self):
        self.vx, self.vy = 0, 0
        self._movex, self._movex = 0, 0
        if self.UP_KEY:
            self.vy = 5
            action = 3
        elif self.DOWN_KEY:
            action = 4
            self.vy = -5
        elif self.LEFT_KEY:
            action = 1
            self.vx = 5
        elif self.RIGHT_KEY:
            self.vx = -5
            action = 2
        else:
            action = 0

        self.rect.x += self.vx
        self.rect.y += self.vy
        self.step(self._action_to_direction[action])
        # print(self._gas, self._brake, self._phi)
        print(self._movex, self._movey)

    def step(self, action):
        self.gas(action[0])
        self.steer(action[1])
        self.brake(action[2])
        self.kinematics()
        return np.array(
            [self.meters_to_pixels(self._movex), self.meters_to_pixels(self._movey)]
        )

    def kinematics(self):
        # Stablish the control input
        v = self.meters_to_pixels(self._gas + self._brake)
        self._v = np.clip(v, 0, self.max_speed)
        #
        self.u1 = self._v * (
            np.cos(np.deg2rad(self.theta) + np.sin(np.deg2rad(self.theta)))
        )
        self.u2 = (1 / self.L) * np.tan(np.deg2rad(self._phi)) * self._v

        self.move()

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        diff = gas - self._gas
        if diff > 0.1:
            diff = 0.1  # gradually increase, but stop immediately
        self._gas = diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        self._brake = np.clip(b, 0, 1)

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self._steer = np.clip(s, -1, 1)
        self._phi = self._steer * 45

    def meters_to_pixels(self, meters):
        """Converts from meters to pixels.

        Parameters
        ----------
        meters: float
                The meters to be converted into pixels.

        Returns
        -------
        float
                The meters in pixels.
        """
        return int(meters)

    def draw(self, map):
        """Draws the robot on map.

        Parameters
        ----------
        map : pygame.Surface
                The where the robot will be drawn.
        """
        map.blit(source=self.rotated, dest=self.rect)

    def move(self):
        # Car-like kinematic robot model
        self._movex = self.u1 * math.cos(math.radians(self.theta)) * self.dt
        self._movey = self.u1 * math.sin(math.radians(self.theta)) * self.dt
        self.x += self._movex
        self.y -= self._movey
        self.theta += self.u2 * self.dt

        # Create the translation and rotation animation
        self.rotated = pygame.transform.rotozoom(
            surface=self.img, angle=self.theta, scale=1
        )
        # self.rect = self.rotated.get_rect(center=(self.rect.x, self.rect.y))

        self.x_position.append(self.x)
        self.y_position.append(self.y)
        self.theta_orientation.append(self.theta)
        self.dt += 0.05

    @property
    def agent_location(self):
        return np.array([self.x, self.y])


def draw(self, display):
    display.blit(self.img, self.rect)


class Vehicle:
    """
    A class to represent a car-like robot dimensions, start position,
    heading angle, and velocity.

    Attributes
    ----------
    start : tuple
            Initial configuration of the robot in X, Y, and theta, respectively.
    robot_img : str
            The robot image path.
    length : float
            The length between the rear wheels and front wheels of the
            car-like robot.
    """

    def __init__(self, start, length):
        self._win_center = (512, 512)
        # Robot settings
        self.x = start[0]  # X position
        self.y = start[1]  # Y position
        self.theta = math.degrees(start[2])  # Initial heading angle (rad)
        self._phi = 0  # Initial steering angle (rad)
        self._v = 0  # m/s
        self.L = self.meters_to_pixels(meters=length)  # m (meters)
        self.max_speed = self.meters_to_pixels(meters=5)  # m/s
        self.min_speed = self.meters_to_pixels(meters=0)  # m/s

        # Graphics
        self.img = pygame.image.load(robot_img_path)
        self.rotated = self.img
        self.rect = self.rotated.get_rect(
            center=(self._win_center[0], self._win_center[1])
        )

        # Time variant
        self.dt = 0.0  # Delta time
        self.last_time = pygame.time.get_ticks()  # Last time recorded
        self.x_position = []
        self.y_position = []
        self.theta_orientation = []

        # Actions
        self._gas = 0.0
        self._steer = 0.0
        self._brake = 0.0
        self._movex, self._movey = 0, 0

        #

    def step(self, action):
        self.gas(action[0])
        self.steer(action[1])
        self.brake(action[2])
        self.kinematics()
        return np.array(
            [self.meters_to_pixels(self._movex), self.meters_to_pixels(self._movey)]
        )

    def kinematics(self):
        # Stablish the control input
        v = self.meters_to_pixels(self._gas + self._brake)
        self._v = np.clip(v, 0, self.max_speed)
        #
        self.u1 = self._v * (
            np.cos(np.deg2rad(self.theta) + np.sin(np.deg2rad(self.theta)))
        )
        self.u2 = (1 / self.L) * np.tan(np.deg2rad(self._phi)) * self._v

        print(self._movex, self._movey, self.agent_location)

        self.move()

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        diff = gas - self._gas
        if diff > 0.1:
            diff = 0.1  # gradually increase, but stop immediately
        self._gas = diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        self._brake = np.clip(b, 0, 1)

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self._steer = np.clip(s, -1, 1)
        self._phi = self._steer * 45

    def meters_to_pixels(self, meters):
        """Converts from meters to pixels.

        Parameters
        ----------
        meters: float
                The meters to be converted into pixels.

        Returns
        -------
        float
                The meters in pixels.
        """
        return int(meters * 5)

    def draw(self, map):
        """Draws the robot on map.

        Parameters
        ----------
        map : pygame.Surface
                The where the robot will be drawn.
        """
        map.blit(source=self.rotated, dest=self.rect)

    def move(self):
        # Car-like kinematic robot model
        self._movex = self.u1 * math.cos(math.radians(self.theta)) * self.dt
        self._movey = self.u1 * math.sin(math.radians(self.theta)) * self.dt
        self.x += self._movex
        self.y -= self._movey
        self.theta += self.u2 * self.dt

        # Create the translation and rotation animation
        self.rotated = pygame.transform.rotozoom(
            surface=self.img, angle=self.theta, scale=1
        )
        self.rect = self.rotated.get_rect(
            center=(self._win_center[0], self._win_center[1])
        )

        self.x_position.append(self.x)
        self.y_position.append(self.y)
        self.theta_orientation.append(self.theta)

    @property
    def agent_location(self):
        return np.array([self.x, self.y])
