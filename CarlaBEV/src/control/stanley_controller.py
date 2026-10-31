"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""

import numpy as np

from CarlaBEV.envs.geometry import MetricFrame
from CarlaBEV.src.control.utils import angle_mod, smooth_and_compute
from CarlaBEV.src.control.state import State
from CarlaBEV.src.sim_time import DEFAULT_PHYSICS_DT_S


class Controller(State):
    k = 2.0  # control gain
    Kp = 1.0  # speed proportional gain
    rolling_resistance_mps2 = 0.05
    aero_drag_coeff = 0.002
    route_offroad_point_tolerance = 1
    route_offroad_fraction_tolerance = 0.01
    # Deceleration used to plan a physical stop at the end of the route when
    # the owner did not supply an explicit max_decel_mps2.
    default_stop_decel_mps2 = 3.5
    # The stopping profile is planned at this fraction of the maximum
    # deceleration (comfort braking); the remaining headroom absorbs the lag of
    # the proportional speed loop via feed-forward braking.
    stop_plan_decel_fraction = 0.6
    # Minimum deceleration applied while stopping at the route end so a
    # proportional speed loop does not creep asymptotically towards zero.
    stop_min_decel_mps2 = 1.0
    stop_speed_eps_mps = 0.01
    # Route-start jitter applied by set_route(jitter_start=True): +/- one step of
    # this size in x and y (metric, so it does not depend on the raster scale;
    # equals the legacy +/-1 px at 128 px).
    route_start_jitter_m = 0.3125

    def __init__(
        self,
        target_speed,
        L=2.9,
        speed_kp=None,
        max_accel_mps2=None,
        max_decel_mps2=None,
        dt=DEFAULT_PHYSICS_DT_S,
        metric: MetricFrame | None = None,
    ) -> None:
        super().__init__(metric=metric)
        # Physics step [s]. Injected from the simulation clock; the same value
        # drives State.update(), comfort derivatives and behaviour timers.
        self.dt = float(dt)
        if not self.dt > 0.0:
            raise ValueError(f"Controller dt must be positive, got {dt!r}")
        self._target_speed = float(target_speed)
        self._target_speed_mps = self.metric.speed_surface_to_mps(target_speed)
        self._speed_limit_mps = self._target_speed_mps
        # Feed-forward acceleration [m/s^2] for a target-speed *ramp* published by
        # a behaviour (dv_target/dt). A P-only loop tracking a ramp has a
        # steady-state lag of rate/Kp; the feed-forward removes it (finding S-01).
        self._target_accel_ff_mps2 = 0.0
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        self.L = float(L)  # [m] Wheel base of vehicle
        self.speed_kp = self.Kp if speed_kp is None else float(speed_kp)
        self.max_accel_mps2 = None if max_accel_mps2 is None else float(max_accel_mps2)
        self.max_decel_mps2 = None if max_decel_mps2 is None else float(max_decel_mps2)
    
    def set_target_speed(self, target_speed, *, update_speed_limit=True, target_accel_ff_mps2=0.0):
        self._target_speed = float(target_speed)
        self._target_speed_mps = self.metric.speed_surface_to_mps(target_speed)
        self._target_accel_ff_mps2 = float(target_accel_ff_mps2)
        if update_speed_limit:
            self._speed_limit_mps = self._target_speed_mps

    def set_route(self, ax, ay, v0=0.0, jitter_start=True, np_rng=None, drivable_mask=None):
        """Stanley steering control on a cubic spline."""
        ax = np.asarray(ax, dtype=float)
        ay = np.asarray(ay, dtype=float)
        mpp = self.metric.meters_per_pixel
        ax_m = ax * mpp
        ay_m = ay * mpp
        raw_route = self._build_raw_controller_path(ax_m, ay_m)
        smoothed_route = smooth_and_compute(ax_m, ay_m, window=11, poly=3)
        offroad_points, offroad_fraction = self._route_offroad_stats(smoothed_route, drivable_mask)
        if self._should_fallback_to_raw(offroad_points, offroad_fraction):
            cx_m, cy_m, cyaw, ck, s = raw_route
            self.route_smoothing_mode = "raw_fallback"
        else:
            cx_m, cy_m, cyaw, ck, s = smoothed_route
            self.route_smoothing_mode = "smoothed"
        self.cs_m = np.asarray(s, dtype=float)
        self.route_offroad_points = int(offroad_points)
        self.route_offroad_fraction = float(offroad_fraction)

        if jitter_start:
            if np_rng is None:
                np_rng = np.random.default_rng()
            jitter_x_m = int(np_rng.integers(-1, 2)) * self.route_start_jitter_m
            jitter_y_m = int(np_rng.integers(-1, 2)) * self.route_start_jitter_m
            self.x_m = cx_m[0] + jitter_x_m
            self.y_m = cy_m[0] + jitter_y_m
        else:
            self.x_m, self.y_m = cx_m[0], cy_m[0]
        self.cx_m, self.cy_m = cx_m, cy_m
        self.cx = cx_m / mpp
        self.cy = cy_m / mpp
        self.v = v0
        self.cyaw = cyaw
        self.target_idx, _ = self.calc_target_index()
        self.yaw = self.cyaw[self.target_idx]

    @staticmethod
    def _build_raw_controller_path(ax_m, ay_m):
        cx_m = np.asarray(ax_m, dtype=float)
        cy_m = np.asarray(ay_m, dtype=float)
        dx = np.diff(cx_m)
        dy = np.diff(cy_m)
        seg_len = np.hypot(dx, dy)
        s = np.concatenate(([0.0], np.cumsum(seg_len)))
        if len(cx_m) == 1:
            cyaw = np.zeros_like(cx_m)
            ck = np.zeros_like(cx_m)
            return cx_m, cy_m, cyaw, ck, s

        heading = np.arctan2(dy, dx)
        cyaw = np.empty_like(cx_m)
        cyaw[:-1] = heading
        cyaw[-1] = heading[-1]
        cyaw = np.unwrap(cyaw)
        ck = np.zeros_like(cx_m)
        return cx_m, cy_m, cyaw, ck, s

    def _route_offroad_stats(self, route, drivable_mask):
        if drivable_mask is None:
            return 0, 0.0
        cx_m, cy_m, *_ = route
        offroad_points = 0
        total_points = len(cx_m)
        height, width = drivable_mask.shape
        mpp = self.metric.meters_per_pixel
        for x_m, y_m in zip(cx_m, cy_m):
            x = x_m / mpp
            y = y_m / mpp
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if not (0 <= xi < width and 0 <= yi < height) or not bool(drivable_mask[yi, xi]):
                offroad_points += 1
        if total_points <= 0:
            return 0, 0.0
        return offroad_points, offroad_points / total_points

    def _should_fallback_to_raw(self, offroad_points, offroad_fraction):
        return (
            offroad_points > self.route_offroad_point_tolerance
            or offroad_fraction > self.route_offroad_fraction_tolerance
        )

    def control_step(self):
        """Advance one physics step along the route.

        Returns ``True`` once the actor has physically come to rest at the end
        of its route. Motion is always integrated through :meth:`update` so the
        reported velocity matches the realised displacement: approaching the
        route end the speed request is capped by the stopping profile
        ``v <= sqrt(2 a_dec s_remaining)`` and, past the last waypoint, the
        actor decelerates to a stop instead of freezing mid-motion.
        """
        at_route_end = self.target_idx >= len(self.cx) - 1
        if at_route_end:
            # Stop target. The speed *limit* is left untouched: it is a cap on
            # the speed request, while stopping is realised by the controller
            # decelerating towards the zero target within max_decel.
            self._target_speed = 0.0
            self._target_speed_mps = 0.0
            self._target_accel_ff_mps2 = 0.0
            if abs(self.v_mps) <= self.stop_speed_eps_mps:  # fully stopped
                if self.v_mps != 0.0:
                    self._v_1_mps = self._v_mps
                    self._v_mps = 0.0
                return True  # signal finished
            ai = self.pid_control()
            if self.v_mps > 0.0:
                ai = min(ai, -min(self.stop_min_decel_mps2, self.v_mps / self.dt))
            di = 0.0
        else:
            ai = self.stop_aware_speed_control()
            di, self.target_idx = self.stanley_control()
        self.update(ai - self.longitudinal_resistance_accel(), di)
        return False  # still running

    def stop_aware_speed_control(self) -> float:
        """Speed control that guarantees a stop at the route end.

        Below the comfort stopping profile the proportional loop tracks
        ``min(target, cap)``. Above it, the required deceleration
        ``v^2 / (2 s_remaining)`` is commanded directly (feed-forward), bounded
        by the maximum deceleration, so the controller lag cannot cause an
        overshoot of the route end.
        """
        remaining_m = self.remaining_route_m()
        cap_mps = self.stop_profile_speed_cap_mps()
        v = float(self.v_mps)
        if np.isfinite(remaining_m) and v > cap_mps and v > 0.0:
            if remaining_m <= 1e-6:
                return -self._stop_decel_mps2()
            required = v * v / (2.0 * remaining_m)
            return -float(min(self._stop_decel_mps2(), required))
        return self.pid_control(cap_mps)

    def _stop_decel_mps2(self) -> float:
        if self.max_decel_mps2 is not None and self.max_decel_mps2 > 0.0:
            return float(self.max_decel_mps2)
        return float(self.default_stop_decel_mps2)

    def remaining_route_m(self) -> float:
        """Arc length left between the current target index and the route end."""
        cs_m = getattr(self, "cs_m", None)
        if cs_m is None or len(cs_m) == 0:
            return float("inf")
        idx = int(np.clip(self.target_idx, 0, len(cs_m) - 1))
        return float(max(0.0, cs_m[-1] - cs_m[idx]))

    def stop_profile_speed_cap_mps(self) -> float:
        """Maximum speed that still allows a comfort stop at the route end."""
        remaining_m = self.remaining_route_m()
        if not np.isfinite(remaining_m):
            return float("inf")
        plan_decel = self.stop_plan_decel_fraction * self._stop_decel_mps2()
        return float(np.sqrt(2.0 * plan_decel * remaining_m))

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

    def pid_control(self, speed_cap_mps=None):
        """
        Proportional control for the speed.

        :param speed_cap_mps: optional upper bound applied to the speed request
        :return: (float) commanded acceleration [m/s^2]
        """
        target_mps = self._target_speed_mps
        feed_forward = self._target_accel_ff_mps2
        if speed_cap_mps is not None and float(speed_cap_mps) < target_mps:
            target_mps = float(speed_cap_mps)
            feed_forward = min(feed_forward, 0.0)  # never accelerate into a stopping cap
        accel_cmd = self.speed_kp * (target_mps - self.v_mps) + feed_forward
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
            "target_speed_mps": self._target_speed_mps,
            "speed_limit_mps": self._speed_limit_mps,
            "route_smoothing_mode": getattr(self, "route_smoothing_mode", "smoothed"),
            "route_offroad_points": getattr(self, "route_offroad_points", 0),
            "route_offroad_fraction": getattr(self, "route_offroad_fraction", 0.0),
            "next_wps": self.next_wps(5),
            "next_wps_m": self.next_wps_m(5),
        }
        if hasattr(self, "last_control"):
            info.update(self.last_control)
        if hasattr(self, "last_traffic_policy"):
            info.update(self.last_traffic_policy)
        if hasattr(self, "last_comfort"):
            info.update(self.last_comfort)
        return info
