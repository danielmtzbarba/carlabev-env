import numpy as np

from CarlaBEV.src.control.utils import angle_mod


class State:
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        #
        self.x_1 = x
        self.y_1 = y
        self.yaw_1 = yaw
        self.v_1 = v
        #

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        #
        self.x_1 = self.x
        self.y_1 = self.y
        self.yaw_1 = self.yaw
        self.v_1 = self.v
        #
        self.x += self.v * np.cos(self.yaw) * self.dt
        self.y += self.v * np.sin(self.yaw) * self.dt
        self.yaw += self.v / self.L * np.tan(delta) * self.dt
        self.v += acceleration * self.dt
        #
        self.yaw = angle_mod(self.yaw)
        self.v = np.clip(self.v, -1 * self._target_speed, self._target_speed)

    @property
    def state(self):
        """
        Get the state of the vehicle.

        :return: (list) [x, y, yaw, v]
        """
        return [self.x, self.y, self.yaw, self.v]

    @property
    def last_state(self):
        """
        Get the last state of the vehicle.

        :return: (list) [x, y, yaw, v]
        """
        return [self.x_1, self.y_1, self.yaw_1, self.v_1]
