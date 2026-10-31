import numpy as np

from CarlaBEV.envs.geometry import DEFAULT_METRIC_FRAME, MetricFrame
from CarlaBEV.src.control.utils import angle_mod


class State:
    """
    Class representing the state of a vehicle.

    Canonical state is metric (``x_m``, ``y_m``, ``yaw``, ``v_mps``). Surface-frame
    views (``x``, ``y``, ``v``) are converted through the injected
    :class:`MetricFrame` so the state never depends on a module-level scale.

    :param x: (float) x-coordinate [surface px]
    :param y: (float) y-coordinate [surface px]
    :param yaw: (float) yaw angle [rad]
    :param v: (float) speed [surface px/s]
    :param metric: (MetricFrame) surface <-> metric conversion
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, metric: MetricFrame | None = None):
        """Instantiate the object."""
        self.metric = metric if metric is not None else DEFAULT_METRIC_FRAME
        self._x_m = self.metric.distance_surface_to_meters(x)
        self._y_m = self.metric.distance_surface_to_meters(y)
        self.yaw = yaw
        self._v_mps = self.metric.distance_surface_to_meters(v)
        #
        self._x_1_m = self._x_m
        self._y_1_m = self._y_m
        self.yaw_1 = yaw
        self._v_1_mps = self._v_mps
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
        self._x_1_m = self._x_m
        self._y_1_m = self._y_m
        self.yaw_1 = self.yaw
        self._v_1_mps = self._v_mps
        #
        self._x_m += self._v_mps * np.cos(self.yaw) * self.dt
        self._y_m += self._v_mps * np.sin(self.yaw) * self.dt
        # Kinematic bicycle heading update.
        self.yaw += self._v_mps / self.L * np.tan(delta) * self.dt
        self._v_mps += acceleration * self.dt
        self.yaw = angle_mod(self.yaw)
        speed_limit_mps = getattr(self, "_speed_limit_mps", self._target_speed_mps)
        self._v_mps = np.clip(self._v_mps, -1 * speed_limit_mps, speed_limit_mps)

    @property
    def x(self):
        return self.metric.distance_meters_to_surface(self._x_m)

    @x.setter
    def x(self, value):
        self._x_m = self.metric.distance_surface_to_meters(value)

    @property
    def y(self):
        return self.metric.distance_meters_to_surface(self._y_m)

    @y.setter
    def y(self, value):
        self._y_m = self.metric.distance_surface_to_meters(value)

    @property
    def v(self):
        return self.metric.distance_meters_to_surface(self._v_mps)

    @v.setter
    def v(self, value):
        self._v_mps = self.metric.distance_surface_to_meters(value)

    @property
    def x_1(self):
        return self.metric.distance_meters_to_surface(self._x_1_m)

    @x_1.setter
    def x_1(self, value):
        self._x_1_m = self.metric.distance_surface_to_meters(value)

    @property
    def y_1(self):
        return self.metric.distance_meters_to_surface(self._y_1_m)

    @y_1.setter
    def y_1(self, value):
        self._y_1_m = self.metric.distance_surface_to_meters(value)

    @property
    def v_1(self):
        return self.metric.distance_meters_to_surface(self._v_1_mps)

    @v_1.setter
    def v_1(self, value):
        self._v_1_mps = self.metric.distance_surface_to_meters(value)

    @property
    def x_m(self):
        return float(self._x_m)

    @x_m.setter
    def x_m(self, value):
        self._x_m = float(value)

    @property
    def y_m(self):
        return float(self._y_m)

    @y_m.setter
    def y_m(self, value):
        self._y_m = float(value)

    @property
    def v_mps(self):
        return float(self._v_mps)

    @v_mps.setter
    def v_mps(self, value):
        self._v_mps = float(value)

    @property
    def x_1_m(self):
        return float(self._x_1_m)

    @property
    def y_1_m(self):
        return float(self._y_1_m)

    @property
    def v_1_mps(self):
        return float(self._v_1_mps)

    @property
    def state(self):
        """
        Get the state of the vehicle.

        :return: (list) [x, y, yaw, v]
        """
        return [self.x, self.y, self.yaw, self.v]

    @property
    def state_m(self):
        return [self.x_m, self.y_m, self.yaw, self.v_mps]

    @property
    def last_state(self):
        """
        Get the last state of the vehicle.

        :return: (list) [x, y, yaw, v]
        """
        return [self.x_1, self.y_1, self.yaw_1, self.v_1]

    @property
    def last_state_m(self):
        return [self.x_1_m, self.y_1_m, self.yaw_1, self.v_1_mps]
