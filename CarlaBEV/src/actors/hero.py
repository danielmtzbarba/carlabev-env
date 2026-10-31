"""Policy-driven ego vehicle.

``BaseAgent`` is a :class:`Controller` (bicycle kinematics + route tracking)
with command-space longitudinal/steering authority from the unit contract and a
metric footprint. It carries no rendering state: the renderer draws its body
in the world layer and the FOV renderer draws the ego overlay.
"""

import math
import numpy as np

from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.semantics import SemanticClass, semantic_color_tuple
from CarlaBEV.src.deeprl.comfort import compute_comfort_kinematics
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT, UnitContract
from CarlaBEV.src.geometry.body import Body
from CarlaBEV.src.geometry.rects import body_surface_rect


class BaseAgent(Controller):
    """Policy-driven ego. Every physical parameter comes from ``UnitContract.ego``."""

    semantic_class = SemanticClass.EGO

    def __init__(
        self,
        route,
        window_size,
        target_speed=0.0,
        initial_speed=0.0,
        color=None,
        np_rng=None,
        drivable_mask=None,
        units: UnitContract | None = None,
        dt=None,
    ):
        units = units if units is not None else DEFAULT_UNIT_CONTRACT
        ego = units.ego
        # ``dt`` defaults to the contract's clock; an explicit value overrides it.
        dt = units.time.physics_dt_s if dt is None else float(dt)
        Controller.__init__(self, target_speed=target_speed, L=ego.wheelbase_m, dt=dt, metric=units.metric)
        self.units = units
        self.window_size = int(window_size)
        self.color = color if color is not None else semantic_color_tuple(SemanticClass.EGO)
        # Footprint is defined in meters by the unit contract and derived to px.
        self.footprint = units.footprints.ego
        self.length_m = self.footprint.length_m
        self.width_m = self.footprint.width_m
        self.w, self.l = units.ego_footprint_px

        xs, ys = route[0], route[1]
        self.acc = 0.0
        self.meters_per_pixel = units.metric.meters_per_pixel
        # --- Longitudinal authority (SI, raster independent) ---
        self.max_accel_mps2 = ego.accel_mps2
        self.max_brake_mps2 = ego.brake_decel_mps2
        self.brake_speed_ref_mps = ego.brake_speed_ref_mps
        self.brake_low_speed_floor = ego.brake_low_speed_floor
        self.accel_response_tau_s = ego.accel_response_tau_s
        self.accel_response_alpha = ego.accel_response_alpha(dt)
        self.rolling_resistance_mps2 = ego.rolling_resistance_mps2
        self.aero_drag_coeff = ego.aero_drag_coeff
        self.stop_speed_eps_mps = ego.stop_speed_eps_mps
        # --- Steering authority ---
        self.min_action_steer_deg = ego.steer_min_deg
        self.max_action_steer_deg = ego.steer_max_deg
        self.steer_speed_scale_mps = ego.steer_speed_scale_per_mps
        self.max_steer = math.radians(ego.max_steer_deg)
        self._prev_comfort_accel_long = None
        self._prev_comfort_accel_lat = None
        self._prev_comfort_yaw_rate = None
        self.last_control = {
            "cmd_gas": 0.0,
            "cmd_steer": 0.0,
            "cmd_brake": 0.0,
            "applied_delta": 0.0,
            "speed_surface": 0.0,
            "speed_mps": 0.0,
        }
        self.last_comfort = {
            "speed_mps": 0.0,
            "accel_long": 0.0,
            "accel_lat": 0.0,
            "jerk_long": 0.0,
            "jerk_lat": 0.0,
            "yaw_rate": 0.0,
            "yaw_acc": 0.0,
        }
        self.drivable_mask = drivable_mask

        self.set_route(xs, ys, initial_speed, np_rng=np_rng, drivable_mask=self.drivable_mask)
        
        # Initial control step to set target_idx
        _, self.target_idx = self.stanley_control()

    def physics_step(self, gas, steer, brake):
        """Unified physics update step."""
        # Update path tracking (just for info/debug, not used for control if we override steer)
        _, self.target_idx = self.stanley_control()

        # === Compute acceleration, steering, braking ===
        acc_val = self.accelerate(gas)
        delta = self.steering(steer)
        brake_val = self.brake(brake)

        # Total longitudinal acceleration with the same explicit resistance
        # model used by controller-driven traffic actors.
        target_acc = acc_val - brake_val - self.longitudinal_resistance_accel()

        # === Smooth acceleration (low-pass filter) ===
        alpha = self.accel_response_alpha
        self.acc = (1 - alpha) * self.acc + alpha * target_acc

        # === Update physics (bicycle model) ===
        # Controller (state.py) uses bicycle model:
        # yaw += v / L * tan(delta) * dt
        self.update(self.acc, delta)
        if abs(self.v_mps) < self.stop_speed_eps_mps:
            self.v_mps = 0.0

        self.last_control = {
            "cmd_gas": float(gas),
            "cmd_steer": float(steer),
            "cmd_brake": float(brake),
            "applied_delta": float(delta),
            "speed_surface": float(self.v),
            "speed_mps": float(self.speed_mps),
        }
        self.last_comfort = compute_comfort_kinematics(
            speed_px_s=self.v,
            prev_speed_px_s=self.v_1,
            yaw_rad=self.yaw,
            prev_yaw_rad=self.yaw_1,
            dt=self.dt,
            meters_per_pixel=self.meters_per_pixel,
            prev_accel_long=self._prev_comfort_accel_long,
            prev_accel_lat=self._prev_comfort_accel_lat,
            prev_yaw_rate_deg=self._prev_comfort_yaw_rate,
        )
        self._prev_comfort_accel_long = self.last_comfort["accel_long"]
        self._prev_comfort_accel_lat = self.last_comfort["accel_lat"]
        self._prev_comfort_yaw_rate = self.last_comfort["yaw_rate"]

    def accelerate(self, amount):
        """Return positive forward acceleration"""
        return max(0.0, amount) * self.max_accel_mps2

    @property
    def speed_surface(self) -> float:
        return float(self.v)

    @property
    def speed_mps(self) -> float:
        return float(self.v_mps)
    
    def steering(self, steer_action):
        """Map normalized steering actions to a meaningful wheel angle."""
        speed_mps = abs(self.speed_mps)
        if speed_mps < 0.1:
            return 0.0

        # Give the policy enough steering authority to make lane changes and
        # avoidance maneuvers, while tapering at higher speed to avoid twitchy turns.
        steer_deg = self.max_action_steer_deg / (
            1.0 + self.steer_speed_scale_mps * speed_mps
        )
        steer_deg = np.clip(
            steer_deg,
            self.min_action_steer_deg,
            self.max_action_steer_deg,
        )
        delta = math.radians(steer_action * steer_deg)
        return delta

    def brake(self, amount):
        """Brake deceleration [m/s^2]; authority tapers below brake_speed_ref_mps."""
        speed_factor = np.clip(
            abs(self.speed_mps) / self.brake_speed_ref_mps, self.brake_low_speed_floor, 1.0
        )
        return max(0.0, amount) * self.max_brake_mps2 * speed_factor


class DiscreteAgent(BaseAgent):
    def step(self, action):
        """
        action: [gas, steer, brake] (from spaces.py mapping)
        """
        gas = action[0]
        steer = action[1]
        brake = action[2]
        
        self.physics_step(gas, steer, brake)


class ContinuousAgent(BaseAgent):
    def step(self, action):
        """
        action: [gas, steer, brake]
        """

        gas = np.clip(action[0], 0.0, 1.0)
        steer = np.clip(action[1], -1.0, 1.0)
        brake = np.clip(action[2], 0.0, 1.0)
        
        self.physics_step(gas, steer, brake)


# ---------------------------------------------------------------------------
# Geometry / rendering interface shared by all ego classes
# ---------------------------------------------------------------------------
def _kind(self) -> str:
    return "agent"


def _body(self) -> Body:
    return Body(self.footprint, self.x_m, self.y_m, self.yaw, self.v_mps)


def _surface_rect(self):
    """Axis-aligned surface-px rect enclosing the ego body (route-target triggers)."""
    return body_surface_rect(self.body(), self.units.metric)


def _position(self):
    """Ego reference point in surface px."""
    return np.array([self.x, self.y], dtype=float)


def _pose(self):
    return np.array([self.x, self.y, self.yaw], dtype=float)


BaseAgent.kind = property(_kind)
BaseAgent.body = _body
BaseAgent.surface_rect = _surface_rect
BaseAgent.position = property(_position)
BaseAgent.pose = property(_pose)
