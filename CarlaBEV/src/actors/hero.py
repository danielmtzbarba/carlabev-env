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


class DiscreteAgent(Controller, Hero):
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
        _, self.target_idx = self.stanley_control()

    def step(self, action):
        """Sprite update function, calculates any new position"""

        _, self.target_idx = self.stanley_control()

        # === Compute acceleration and braking ===
        acc_val = self.accelerate(action[0])
        brake_val = self.brake(action[2])

        # Total longitudinal acceleration
        target_acc = acc_val - brake_val

        # === Smooth acceleration (low-pass filter) ===
        alpha = 0.2
        self.acc = (1 - alpha) * self.acc + alpha * target_acc

        # === Apply steering ===
        self.turn(action[1])

        # === Update physics ===
        self.update(self.acc, 0)

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
        return amount * 1.0 * self.scale  # tweak 1.0 to 0.8–1.2 for feel

    def brake(self, amount):
        speed_factor = np.clip(abs(self.v) / 5.0, 0.3, 1.0)
        return amount * 0.6 * self.scale * speed_factor

    def turn(self, angle_degrees):
        """Turn realistically, depending on speed."""
        if abs(self.v) < 0.1:
            return  # no turn if not moving

        min_turn_scale = 0.3
        max_turn_scale = 1.2
        turn_scale = np.clip(self.v / 2.0, min_turn_scale, max_turn_scale)
        self.yaw += math.radians(angle_degrees * turn_scale)


class ContinuousAgent(Controller, Hero):
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
        _, self.target_idx = self.stanley_control()

    def step(self, action):
        """Sprite update function, calculates any new position"""
        _, self.target_idx = self.stanley_control()

        # Action: [steer, gas, brake]
        steer_action = action[0]
        gas_action = action[1]
        brake_action = action[2]

        # === Compute acceleration and braking ===
        acc_val = self.accelerate(gas_action)
        brake_val = self.brake(brake_action)

        # Total longitudinal acceleration
        target_acc = acc_val - brake_val - 0.05 * self.v  # slight friction

        # === Smooth acceleration (low-pass filter) ===
        alpha = 0.2
        self.acc = (1 - alpha) * self.acc + alpha * target_acc

        # === Apply steering ===
        # Map [-1, 1] to [-max_steer, max_steer] effectively handled by Controller logic 
        # But here we treat action[0] as direct steering input or delta?
        # The original code used action[0] as steer. Let's assume action[0] is normalized steer.
        # Controller.stanley_control usually returns a delta, but here we challenge that.
        # Actually, in continuous mode, the agent policy outputs the steering *command*.
        # Stanley is for *following* a path automatically. 
        # If we are training an agent, the agent *provides* the steering.
        
        # However, to be consistent with "DiscreteAgent" which seems to follow a route but allows
        # discrete interventions? No, DiscreteAgent uses stanley_control() to get reference, 
        # but then ignores it? Wait, DiscreteAgent implementation:
        # _, self.target_idx = self.stanley_control()
        # ...
        # self.turn(action[1]) -> this modifies self.yaw directly.
        
        # So for ContinuousAgent, we should also allow direct yaw modification.
        # action[0] is steering.
        
        self.turn(steer_action)

        # === Update physics ===
        # We pass 0 for delta to update() because we manually updated yaw in turn()
        self.update(self.acc, 0)

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
        # amount is assumed to be in [0, 1] if coming from sigmoid/rectified, 
        # or [-1, 1] if raw. usually gas is [0, 1].
        # Let's clip to be safe or assume input is reasonable.
        return max(0.0, amount) * 1.0 * self.scale

    def brake(self, amount):
        speed_factor = np.clip(abs(self.v) / 5.0, 0.3, 1.0)
        return max(0.0, amount) * 0.6 * self.scale * speed_factor
    
    def turn(self, steer_input):
        """
        steer_input in [-1, 1].
        Maps to [-max_turn_angle, max_turn_angle]
        """
        if abs(self.v) < 0.1:
            return

        max_turn_degrees = 15.0 # Max degrees per step? or max steering angle?
        # If this is delta yaw per step:
        
        min_turn_scale = 0.3
        max_turn_scale = 1.2
        turn_scale = np.clip(self.v / 2.0, min_turn_scale, max_turn_scale)
        
        angle_degrees = steer_input * max_turn_degrees
        self.yaw += math.radians(angle_degrees * turn_scale)
