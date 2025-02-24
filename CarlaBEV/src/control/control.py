from copy import deepcopy
import math
import cvxpy
import numpy as np
from time import time

from CarlaBEV.src.control.utils import (
    calc_nearest_index,
    get_nparray_from_matrix,
    calc_speed_profile,
    smooth_yaw,
)
from CarlaBEV.src.planning.cubic_spline_planner import calc_spline_course


class State:
    _dt = 0.2
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None
        #
        self._wb = 2.5  # [m]
        self._max_steer = np.deg2rad(45.0)  # maximum steering angle [rad]

        self._max_speed = 55.0 / 3.6  # maximum speed [m/s]
        self._min_speed = -20.0 / 3.6  # minimum speed [m/s]

    def update_state(self, a, delta):
        # input check
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        self.x += self.v * math.cos(self.yaw) * self._dt
        self.y += self.v * math.sin(self.yaw) * self._dt
        self.yaw += self.v / self._wb * math.tan(delta) * self._dt
        self.v += a * self._dt

        self.v = np.clip(self.v, self.max_speed, self.min_speed)

        return self

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def min_speed(self):
        return self._min_speed

    @property
    def max_steer(self):
        return self._max_steer


class Controller(object):
    _dt = 1.0
    _dl = 1.0  # course tick
    _goal_dis = 1.5  # goal distance
    _stop_speed = 0.5 / 3.6  # stop speed

    _nx = 4  # x = x, y, v, yaw
    _nu = 2  # a = [accel, steer]
    _horizon = 5  # horizon length
    _max_iter = 3
    _du_th = 0.1

    _R = np.diag([0.01, 0.01])  # input cost matrix
    _Rd = np.diag([0.01, 1.0])  # input difference cost matrix
    _Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
    _Qf = _Q  # state final matrix

    def __init__(self) -> None:
        self._time = 0.0
        self._target_speed = 10.0 / 3.6  # [m/s] target speed
        #
        self._in_speed = -20.0 / 3.6  # minimum speed [m/s]
        self._max_acc = 1.0  # maximum accel [m/ss]
        self._wb = 2.5  # [m]
        #
        self._n_ind_search = 10
        self._max_dsteer = np.deg2rad(30.0)  # maximum steering speed [rad/s]
        #
        self._x, self._y = [], []
        self._yaw, self._v = [], []
        self._t = [0.0]
        self._d = [0.0]
        self._a = [0.0]

    def set_route(self, ax, ay, resolution=1.0):
        self._spx, self._spy, self._spyaw, self._spk, _ = calc_spline_course(
            ax, ay, ds=resolution
        )
        self._spyaw = smooth_yaw(self._spyaw)

        self._speed_profile = calc_speed_profile(
            self._spx, self._spy, self._spyaw, self._target_speed
        )

        self._goal = [self._spx[-1], self._spy[-1]]
        initial_state = State(x=self._spx[0], y=self._spy[0], yaw=self._spyaw[0], v=0.0)

        self._state = initial_state
        self._x.append(self._state.x)
        self._y.append(self._state.y)
        self._yaw.append(self._state.yaw)
        self._v.append(self._state.v)

        # initial yaw compensation
        if self._state.yaw - self._spyaw[0] >= math.pi:
            self._state.yaw -= math.pi * 2.0
        elif self._state.yaw - self._spyaw[0] <= -math.pi:
            self._state.yaw += math.pi * 2.0

        self._target_ind, _ = calc_nearest_index(
            self._state, self._spx, self._spy, self._spyaw, 0, self._n_ind_search
        )

        self._od, self._oa = None, None

    def control_step(self):
        xref, self._target_ind, dref = self.calc_ref_trajectory()

        ox, oy, oyaw, ov = self.iterative_linear_mpc_control(xref, dref)

        di, ai = 0.0, 0.0
        if self._od is not None:
            di, ai = self._od[0], self._oa[0]
            state = self._state.update_state(ai, di)

        self._time += self._dt

        if self.check_goal():
            print("Goal")

        self._x.append(state.x)
        self._y.append(state.y)
        self._yaw.append(state.yaw)
        self._v.append(state.v)
        self._t.append(time)
        self._d.append(di)
        self._a.append(ai)

        return state

    def iterative_linear_mpc_control(self, xref, dref):
        """
        MPC control with updating operational point iteratively
        """
        ox, oy, oyaw, ov = None, None, None, None

        if self._oa is None or self._od is None:
            self._oa = [0.0] * self._horizon
            self._od = [0.0] * self._horizon

        for i in range(self._max_iter):
            xbar = self.predict_motion(xref)
            poa, pod = self._oa[:], self._od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self._du_th:
                break
            self._oa, self._od = oa, od

        return ox, oy, oyaw, ov

    def predict_motion(self, xref):
        xbar = xref * 0.0
        x0 = deepcopy(self.x0)

        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for ai, di, i in zip(self._oa, self._od, range(1, self._horizon + 1)):
            state = state.update_state(ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar

    def linear_mpc_control(self, xref, xbar, dref):
        """
        linear mpc control

        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """

        x = cvxpy.Variable((self._nx, self._horizon + 1))
        u = cvxpy.Variable((self._nu, self._horizon))

        cost = 0.0
        constraints = []

        for t in range(self._horizon):
            cost += cvxpy.quad_form(u[:, t], self._R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self._Q)

            A, B, C = self.get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self._horizon - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self._Rd)
                constraints += [
                    cvxpy.abs(u[1, t + 1] - u[1, t]) <= self._max_dsteer * self._dt
                ]

        cost += cvxpy.quad_form(xref[:, self._horizon] - x[:, self._horizon], self._Qf)

        constraints += [x[:, 0] == self.x0]
        constraints += [x[2, :] <= self._state.max_speed]
        constraints += [x[2, :] >= self._state.min_speed]
        constraints += [cvxpy.abs(u[0, :]) <= self._max_acc]
        constraints += [cvxpy.abs(u[1, :]) <= self._state.max_steer]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def get_linear_model_matrix(self, v, phi, delta):
        A = np.zeros((self._nx, self._nx))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self._dt * math.cos(phi)
        A[0, 3] = -self._dt * v * math.sin(phi)
        A[1, 2] = self._dt * math.sin(phi)
        A[1, 3] = self._dt * v * math.cos(phi)
        A[3, 2] = self._dt * math.tan(delta) / self._wb

        B = np.zeros((self._nx, self._nu))
        B[2, 0] = self._dt
        B[3, 1] = self._dt * v / (self._wb * math.cos(delta) ** 2)

        C = np.zeros(self._nx)
        C[0] = self._dt * v * math.sin(phi) * phi
        C[1] = -self._dt * v * math.cos(phi) * phi
        C[3] = -self._dt * v * delta / (self._wb * math.cos(delta) ** 2)

        return A, B, C

    def calc_ref_trajectory(self):
        xref = np.zeros((self._nx, self._horizon + 1))
        dref = np.zeros((1, self._horizon + 1))
        ncourse = len(self._spx)

        ind, _ = calc_nearest_index(
            self._state,
            self._spx,
            self._spy,
            self._spyaw,
            self._target_ind,
            self._n_ind_search,
        )

        if self._target_ind >= ind:
            ind = self._target_ind

        xref[0, 0] = self._spx[ind]
        xref[1, 0] = self._spy[ind]
        xref[2, 0] = self._speed_profile[ind]
        xref[3, 0] = self._spyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(self._horizon + 1):
            travel += abs(self._state.v) * self._dt
            dind = int(round(travel / self._dl))

            if (ind + dind) < ncourse:
                xref[0, i] = self._spx[ind + dind]
                xref[1, i] = self._spy[ind + dind]
                xref[2, i] = self._speed_profile[ind + dind]
                xref[3, i] = self._spyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = self._spx[ncourse - 1]
                xref[2, i] = self._spy[ncourse - 1]
                xref[3, i] = self._spyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref

    def check_goal(self):
        # check goal
        dx = self._state.x - self._goal[0]
        dy = self._state.y - self._goal[1]
        d = math.hypot(dx, dy)

        isgoal = d <= self._goal_dis

        if abs(self._target_ind - len(self._spx)) >= 5:
            isgoal = False

        isstop = abs(self._state.v) <= self._stop_speed

        if isgoal and isstop:
            return True

        return False

    @property
    def state(self):
        return self._state

    @property
    def x0(self):
        # current state
        return [
            self._state.x,
            self._state.y,
            self._state.v,
            self._state.yaw,
        ]
