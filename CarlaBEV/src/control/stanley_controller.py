"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""

import numpy as np

from CarlaBEV.envs.geometry import (
    SURFACE_METERS_PER_PIXEL,
    distance_surface_to_meters,
    speed_surface_to_mps,
)
from CarlaBEV.src.control.utils import angle_mod, smooth_and_compute
from CarlaBEV.src.control.state import State


class Controller(State):
    k = 2.0  # control gain
    Kp = 1.0  # speed proportional gain
    dt = 0.1  # [s] time difference
    rolling_resistance_mps2 = 0.05
    aero_drag_coeff = 0.002

    def __init__(
        self,
        target_speed,
        L=2.9,
        speed_kp=None,
        max_accel_mps2=None,
        max_decel_mps2=None,
    ) -> None:
        super().__init__()
        self.time = 0.0
        self._target_speed = float(target_speed)
        self._target_speed_mps = speed_surface_to_mps(target_speed)
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        self.L = float(L)  # [m] Wheel base of vehicle
        self.speed_kp = self.Kp if speed_kp is None else float(speed_kp)
        self.max_accel_mps2 = None if max_accel_mps2 is None else float(max_accel_mps2)
        self.max_decel_mps2 = None if max_decel_mps2 is None else float(max_decel_mps2)
    
    def set_target_speed(self, target_speed):
        self._target_speed = float(target_speed)
        self._target_speed_mps = speed_surface_to_mps(target_speed)

    def set_route(self, ax, ay, v0=0.0, jitter_start=True, np_rng=None):
        """Stanley steering control on a cubic spline."""
        ax = np.asarray(ax, dtype=float)
        ay = np.asarray(ay, dtype=float)
        ax_m = ax * SURFACE_METERS_PER_PIXEL
        ay_m = ay * SURFACE_METERS_PER_PIXEL
        cx_m, cy_m, cyaw, ck, s = smooth_and_compute(ax_m, ay_m, window=11, poly=3)

        if jitter_start:
            if np_rng is None:
                np_rng = np.random.default_rng()
            jitter_x_m = distance_surface_to_meters(int(np_rng.integers(-1, 2)))
            jitter_y_m = distance_surface_to_meters(int(np_rng.integers(-1, 2)))
            self.x_m = cx_m[0] + jitter_x_m
            self.y_m = cy_m[0] + jitter_y_m
        else:
            self.x_m, self.y_m = cx_m[0], cy_m[0]
        self.cx_m, self.cy_m = cx_m, cy_m
        self.cx = cx_m / SURFACE_METERS_PER_PIXEL
        self.cy = cy_m / SURFACE_METERS_PER_PIXEL
        self.v = v0
        self.cyaw = cyaw
        self.target_idx, _ = self.calc_target_index()
        self.yaw = self.cyaw[self.target_idx]

    def control_step(self):
        # Check if we're at or past the last waypoint
        if self.target_idx >= len(self.cx) - 1:
            self._target_speed = 0.0  # stop target
            self._target_speed_mps = 0.0
            if self.v_mps <= 0.01:        # fully stopped
                return True           # signal finished
        else:
            ai = self.pid_control()
            di, self.target_idx = self.stanley_control()
            self.update(ai - self.longitudinal_resistance_accel(), di)
            self.time += self.dt
            return False  # still running

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
        theta_d = np.arctan2(self.k * error_front_axle, max(self.v_mps, 1e-3))

        # Steering control
        # TODO: smooth steering control
        delta = theta_e + theta_d
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        return delta, current_target_idx

    def pid_control(self):
        """
        Proportional control for the speed.

        :param target: (float)
        :return: (float)
        """
        accel_cmd = self.speed_kp * (self._target_speed_mps - self.v_mps)
        accel_upper = np.inf if self.max_accel_mps2 is None else self.max_accel_mps2
        decel_lower = -np.inf if self.max_decel_mps2 is None else -self.max_decel_mps2
        return float(np.clip(accel_cmd, decel_lower, accel_upper))

    def longitudinal_resistance_accel(self):
        speed_mps = abs(self.v_mps)
        if speed_mps <= 1e-6:
            return 0.0
        return self.rolling_resistance_mps2 + self.aero_drag_coeff * (speed_mps ** 2)

    def calc_target_index(self):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fx = self.x_m + self.L * np.cos(self.yaw)
        fy = self.y_m + self.L * np.sin(self.yaw)

        # Search nearest point index
        dx = [fx - icx for icx in self.cx_m]
        dy = [fy - icy for icy in self.cy_m]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(self.yaw + np.pi / 2), -np.sin(self.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle

    def next_wps(self, n):
        if self.target_idx + n <= len(self.cx):
            wps = (
                self.cx[self.target_idx : self.target_idx + n],
                self.cy[self.target_idx : self.target_idx + n],
                self.cyaw[self.target_idx : self.target_idx + n],
            )
        else:
            wps = (
                self.cx[self.target_idx : -1],
                self.cy[self.target_idx : -1],
                self.cyaw[self.target_idx : -1],
            )
        return wps 

    def next_wps_m(self, n):
        if self.target_idx + n <= len(self.cx_m):
            wps = (
                self.cx_m[self.target_idx : self.target_idx + n],
                self.cy_m[self.target_idx : self.target_idx + n],
                self.cyaw[self.target_idx : self.target_idx + n],
            )
        else:
            wps = (
                self.cx_m[self.target_idx : -1],
                self.cy_m[self.target_idx : -1],
                self.cyaw[self.target_idx : -1],
            )
        return wps

    @property
    def set_point(self):
        return np.array(
            [
                self.cx[self.target_idx],
                self.cy[self.target_idx],
                self.cyaw[self.target_idx],
            ]
        )

    @property
    def set_point_m(self):
        return np.array(
            [
                self.cx_m[self.target_idx],
                self.cy_m[self.target_idx],
                self.cyaw[self.target_idx],
            ]
        )

    @property
    def dist2wp(self):
        return np.linalg.norm(self.position - self.set_point[:-1], ord=2)

    @property
    def dist2wp_m(self):
        return np.linalg.norm(
            np.array([self.x_m, self.y_m], dtype=float) - self.set_point_m[:-1],
            ord=2,
        )


    @property
    def course(self):
        return (
            self.cx[self.target_idx :],
            self.cy[self.target_idx :],
            self.cyaw[self.target_idx :],
        )
    
    @property
    def controller_info(self):
        info = {
            "state": self.state,
            "state_m": self.state_m,
            "last_state": self.last_state,
            "last_state_m": self.last_state_m,
            "dist2wp": self.dist2wp,
            "dist2wp_m": self.dist2wp_m,
            "set_point": self.set_point,
            "set_point_m": self.set_point_m,
            "next_wps": self.next_wps(5),
            "next_wps_m": self.next_wps_m(5),
        }
        if hasattr(self, "last_control"):
            info.update(self.last_control)
        if hasattr(self, "last_comfort"):
            info.update(self.last_comfort)
        return info
