"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""

import numpy as np
from numpy.random import rand, randint

from CarlaBEV.src.planning import cubic_spline_planner
from CarlaBEV.src.control.utils import angle_mod
from CarlaBEV.src.control.state import State


class Controller(State):
    k = 0.5  # control gain
    Kp = 1.0  # speed proportional gain
    dt = 0.1  # [s] time difference

    def __init__(self, target_speed, L=2.9) -> None:
        super().__init__()
        self.time = 0.0
        self._target_speed = target_speed
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        self.L = 2.9  # [m] Wheel base of vehicle

    def set_route(self, ax, ay, ds=1.0):
        """Stanley steering control on a cubic spline."""
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=ds)
        self.x, self.y = cx[0] + randint(-10, 10), cy[0] + randint(-10, 10)
        self.cx, self.cy = cx, cy
        self.v = 0.0
        self.cyaw = cyaw
        self.target_idx, _ = self.calc_target_index()

    def control_step(self):
        ai = self.pid_control()
        di, self.target_idx = self.stanley_control()
        self.update(ai, di)
        self.time += self.dt

    def stanley_control(self):
        """
        Stanley steering control.

        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = self.calc_target_index()

        if self.target_idx >= current_target_idx:
            current_target_idx = self.target_idx

        # theta_e corrects the heading error
        theta_e = angle_mod(self.cyaw[current_target_idx] - self.yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, self.v)
        # Steering control
        delta = theta_e + theta_d

        return delta, current_target_idx

    def pid_control(self):
        """
        Proportional control for the speed.

        :param target: (float)
        :return: (float)
        """
        return self.Kp * (self._target_speed - self.v)

    def calc_target_index(self):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fx = self.x + self.L * np.cos(self.yaw)
        fy = self.y + self.L * np.sin(self.yaw)

        # Search nearest point index
        dx = [fx - icx for icx in self.cx]
        dy = [fy - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(self.yaw + np.pi / 2), -np.sin(self.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle
