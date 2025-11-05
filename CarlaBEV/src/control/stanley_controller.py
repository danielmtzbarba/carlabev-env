"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""

import numpy as np
from numpy.random import randint

from CarlaBEV.src.planning import cubic_spline_planner
from CarlaBEV.src.control.utils import angle_mod
from CarlaBEV.src.control.state import State

from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

def refit_bspline(ax, ay, ds=0.5, smooth=5.0):
    tck, u = splprep([ax, ay], s=smooth)
    u_new = np.linspace(0, 1, int(len(ax) / ds))

    x_new, y_new = splev(u_new, tck)
    dx, dy = splev(u_new, tck, der=1)

    cyaw = np.arctan2(dy, dx)  # <-- tangent direction
    return np.array(x_new), np.array(y_new), np.unwrap(cyaw)

def moving_average(arr, window=5):
    return np.convolve(arr, np.ones(window)/window, mode="valid")

def smooth_path_mavg(ax, ay, window=5):
    if len(ax) <= window:
        return ax, ay
    ax_s = moving_average(ax, window)
    ay_s = moving_average(ay, window)
    cyaw = np.arctan2(ay_s, ax_s)  # <-- tangent direction
    return ax_s, ay_s, cyaw


def smooth_and_compute(ax, ay, window=9, poly=3):
    """
    Input:
      ax, ay : 1D arrays (raw path coordinates)
      window : odd window length for Savitzky-Golay (will be clamped)
      poly   : polynomial order for Savitzky-Golay

    Returns:
      cx, cy : smoothed coordinates (same length as inputs after dedup)
      cyaw   : tangent angles (radians), array same length
      ck     : curvature array same length
      s      : cumulative arc-length (meters or units of ax/ay)
    """
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    if ax.size != ay.size:
        raise ValueError("ax and ay must have same length")

    # remove consecutive duplicate points
    d = np.hypot(np.diff(ax), np.diff(ay))
    mask = np.concatenate(([True], d > 1e-9))
    ax = ax[mask]; ay = ay[mask]

    if len(ax) < 2:
        raise ValueError("Need at least 2 unique points")

    # ensure window is valid (odd and <= len)
    if window % 2 == 0:
        window += 1
    if window > len(ax):
        window = len(ax) if len(ax) % 2 == 1 else len(ax)-1
    if window < 3:
        window = 3

    poly = min(poly, window-1)

    # Smooth coordinates (if path too short fallback to raw)
    if len(ax) >= window:
        cx = savgol_filter(ax, window_length=window, polyorder=poly)
        cy = savgol_filter(ay, window_length=window, polyorder=poly)
    else:
        cx, cy = ax.copy(), ay.copy()

    # compute cumulative arc-length s
    dx_local = np.diff(cx)
    dy_local = np.diff(cy)
    seg_len = np.hypot(dx_local, dy_local)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = s[-1]
    if total_len <= 1e-9:
        # degenerate: all points essentially same after smoothing
        dx_ds = np.zeros_like(cx)
        dy_ds = np.zeros_like(cy)
        cyaw = np.zeros_like(cx)
        ck = np.zeros_like(cx)
        return cx, cy, cyaw, ck, s

    # derivatives w.r.t arc-length s using numpy.gradient
    # np.gradient(y, x) approximates dy/dx
    # gradient preserves original array length
    dx_ds = np.gradient(cx, s)
    dy_ds = np.gradient(cy, s)

    # yaw is tangent direction
    cyaw = np.unwrap(np.arctan2(dy_ds, dx_ds))

    # second derivatives
    d2x_ds2 = np.gradient(dx_ds, s)
    d2y_ds2 = np.gradient(dy_ds, s)

    denom = (dx_ds**2 + dy_ds**2)
    # avoid division by zero (tiny denom -> set curvature to 0)
    small = denom < 1e-9
    denom_safe = np.where(small, 1.0, denom)  # avoid nan
    ck = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (denom_safe**1.5)
    ck[small] = 0.0

    return cx, cy, cyaw, ck, s

class Controller(State):
    k = 2.0  # control gain
    Kp = 1.0  # speed proportional gain
    dt = 0.1  # [s] time difference

    def __init__(self, target_speed, L=2.9) -> None:
        super().__init__()
        self.time = 0.0
        self._target_speed = target_speed
        self.max_steer = np.radians(35.0)  # [rad] max steering angle
        self.L = 2.9  # [m] Wheel base of vehicle

    def set_route(self, ax, ay, ds=2.0):
        """Stanley steering control on a cubic spline."""
#        cx, cy, cyaw = smooth_path_mavg(ax, ay)
        cx, cy, cyaw, ck, s = smooth_and_compute(ax, ay, window=11, poly=3)

        self.x, self.y = cx[0] + randint(-1, 1), cy[0] + randint(-1, 1)
        self.cx, self.cy = cx, cy
        self.v = 0.0
        self.cyaw = cyaw
        self.target_idx, _ = self.calc_target_index()
        self.yaw = self.cyaw[self.target_idx]

    def control_step(self):
        # Check if we're at or past the last waypoint
        if self.target_idx >= len(self.cx) - 1:
            self._target_speed = 0.0  # stop target
            if self.v <= 0.01:        # fully stopped
                return True           # signal finished
        else:
            ai = self.pid_control()
            di, self.target_idx = self.stanley_control()
            self.update(ai, di)
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
        theta_d = np.arctan2(self.k * error_front_axle, max(self.v, 1e-3))

        # Steering control
        delta = theta_e + theta_d
        delta = np.clip(delta, -self.max_steer, self.max_steer)

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

    def next_wps(self, n):
        try:
            wps = (
                self.cx[self.target_idx : self.target_idx + n],
                self.cy[self.target_idx : self.target_idx + n],
                self.cyaw[self.target_idx : self.target_idx + n],
            )
        except:
            wps = (
                self.cx[self.target_idx : -1],
                self.cy[self.target_idx : -1],
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
    def dist2wp(self):
        return np.linalg.norm(self.position - self.set_point[:-1], ord=2)


    @property
    def course(self):
        return (
            self.cx[self.target_idx :],
            self.cy[self.target_idx :],
            self.cyaw[self.target_idx :],
        )
    
    @property
    def controller_info(self):
        return {
            "state": self.state,
            "last_state": self.last_state,
            "dist2wp": self.dist2wp,
            "set_point": self.set_point,
            "next_wps": self.next_wps(5)
        }
