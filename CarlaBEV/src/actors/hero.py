import numpy as np
import pygame
import math

from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.gui.settings import Settings as cfg


class Hero(pygame.sprite.Sprite):
    def __init__(self, window_size, color=(0, 7, 175), car_size=32):
        pygame.sprite.Sprite.__init__(self)
        self.window_center = int(window_size / 2)
        self.scale = int(1024 / window_size)
        self.color = color
        #
        self.w = int(car_size / self.scale)
        self.l = self.w
        # Visual Offset Bug
        self._offx = -32
        self._offy = -32

    def _setup(self):
        center = self.window_center
        self.fov_rect = pygame.Rect((center, center, self.w, self.l))
        self.rect = pygame.Rect((self.x0, self.y0, self.w, self.l))
        # movement
        self.x = self.rect.x 
        self.y = self.rect.y

    def draw(self, display_fov, display):
        pygame.draw.rect(display_fov, self.color, self.fov_rect)
        #pygame.draw.rect(display, self.color, self.rect)

    @property
    def position(self):
        return pygame.math.Vector2(self.x, self.y)

    @property
    def pose(self):
        return pygame.math.Vector3(self.x, self.y, self.yaw)


class BaseAgent(Controller, Hero):
    dt = 0.1

    def __init__(
        self,
        route,
        window_size,
        target_speed=0.0,
        initial_speed=0.0,
        color=(0, 7, 175),
        car_size=32,
    ):
        Controller.__init__(self, target_speed=target_speed)
        Hero.__init__(self, window_size, color, car_size)

        xs, ys = route[0], route[1]
        self.x0 = int(xs[0])
        self.y0 = int(ys[0])
        self.acc = 0.0
        self._setup()

        self.set_route(xs, ys, initial_speed)
        
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

        # Total longitudinal acceleration
        # 0.05 * self.v is rolling resistance/friction
        target_acc = acc_val - brake_val - 0.05 * self.v

        # === Smooth acceleration (low-pass filter) ===
        alpha = 0.2
        self.acc = (1 - alpha) * self.acc + alpha * target_acc

        # === Update physics (bicycle model) ===
        # Controller (state.py) uses bicycle model:
        # yaw += v / L * tan(delta) * dt
        self.update(self.acc, delta)

        # === Friction & velocity stabilization ===
        self.v *= 0.9999
        if abs(self.v) < 0.05:
            self.v = 0.0

        # === Render update ===
        self.rect.center = (round(self.x) + self._offx, round(self.y) + self._offy)
        # natural drag proportional to speed
        self.v *= 0.985

    def accelerate(self, amount):
        """Return positive forward acceleration"""
        return max(0.0, amount) * 1.0 * self.scale
    
    def steering(self, steer_action):
        """Turn realistically, depending on speed."""
        if abs(self.v) < 0.1:
            return  0.0# no turn if not moving

        min_turn_scale = 0.3
        max_turn_scale = 1.2
        turn_scale = np.clip(self.v / 2.0, min_turn_scale, max_turn_scale)
        delta = math.radians(steer_action * turn_scale)
        return delta

    def brake(self, amount):
        speed_factor = np.clip(abs(self.v) / 5.0, 0.3, 1.0)
        return max(0.0, amount) * 0.6 * self.scale * speed_factor


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
