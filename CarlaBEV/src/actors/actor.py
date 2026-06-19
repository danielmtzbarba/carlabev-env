import pygame
import math
import numpy as np

from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.envs.geometry import speed_mps_to_surface, speed_surface_to_mps


class Node(object):
    def __init__(self, id, position, lane=None):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        self.draw_x = self._x + cfg.offx
        self.draw_y = self._y + cfg.offy
        self.btn = pygame.Rect(self.draw_x, self.draw_y, 3, 3)
        self.color = None

    def reset(self):
        self.color = None

    def render(self, screen, color=None):
        if color is not None:
            self.color = color

        if self.color is not None:
            pygame.draw.rect(screen, self.color, self.btn)

    def clicked(self, event):
        if self.btn.collidepoint(event.pos):
            self.color = cfg.red
            return True

    @property
    def scaled_pos(self):
        return [self._x, self._y]

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]

class Actor(pygame.sprite.Sprite):
    traffic_speed_kp = 0.5
    traffic_max_accel_mps2 = 2.5
    traffic_max_decel_mps2 = 3.5

    def __init__(
        self,
        start_node=None,
        end_node=None,
        id=0,
        actor_size=1,
        routeX=None,
        routeY=None,
        behavior=None,
        np_rng=None,
    ):
        pygame.sprite.Sprite.__init__(self)
        self.id = id
        self.size = actor_size
        self.behavior = behavior

        self.start_node = start_node
        self.end_node = end_node
        #
        if routeX is not None and routeY is not None:
            self.rx = list(routeX)
            self.ry = list(routeY)
        else:
            self.rx, self.ry = [], []
        self._initial_rx = list(self.rx)
        self._initial_ry = list(self.ry)
        #
        self.path = []
        self.selected = False
        self.behavior_state = "idle"
        self.target_speed = 0.0
        self.target_speed_mps = 0.0
        self.cruise_speed = 0.0
        self.cruise_speed_mps = 0.0
        self.speed_class = "medium"
        self.speed_multiplier = 1.0
        self.speed_jitter = 1.0
        self.initial_speed = None
        self.initial_speed_mps = None
        self.startup_ramp_duration_s = 0.0
        self._startup_elapsed_s = 0.0
        self.np_rng = np_rng

    def set_route_wp(self, node_id, x, y):
        self.rx.append(x)
        self.ry.append(y)
        pos = np.array([x, y])
        self.path.append(Node(node_id, pos))

    def reset(self):
        if not self._initial_rx or not self._initial_ry:
            self._initial_rx = list(self.rx)
            self._initial_ry = list(self.ry)
        self.rx = list(self._initial_rx)
        self.ry = list(self._initial_ry)
        if self.initial_speed_mps is None:
            initial_speed_mps = self.cruise_speed_mps
        else:
            initial_speed_mps = min(max(0.0, float(self.initial_speed_mps)), self.cruise_speed_mps)
        self.initial_speed_mps = initial_speed_mps
        self.initial_speed = speed_mps_to_surface(initial_speed_mps)
        self.target_speed = self.cruise_speed
        self.target_speed_mps = self.cruise_speed_mps
        self._controller = Controller(
            self.target_speed,
            speed_kp=self.traffic_speed_kp,
            max_accel_mps2=self.traffic_max_accel_mps2,
            max_decel_mps2=self.traffic_max_decel_mps2,
        )
        self._controller.set_route(
            self.rx,
            self.ry,
            v0=self.initial_speed,
            np_rng=self.np_rng,
        )
        self._startup_elapsed_s = 0.0
        self.behavior_state = "idle"

        if self.behavior:
            self.behavior.reset(self)

        self.rect = pygame.Rect(0, 0, self.size, self.size)

    def step(self, t=0.0, dt=0.05):
        # --- APPLY BEHAVIOR FIRST ---
        if self.behavior:
            self.behavior.apply(self, t, dt)

        # --- PROPAGATE TARGET SPEED TO CONTROLLER ---
        controller_target_mps = self._startup_capped_target_speed_mps()
        self._controller.set_target_speed(speed_mps_to_surface(controller_target_mps))

        finished = self._controller.control_step()
        self._startup_elapsed_s += float(dt)
        return finished

    def set_target_speed_mps(self, speed_mps):
        speed_mps = max(0.0, float(speed_mps))
        self.target_speed_mps = speed_mps
        self.target_speed = speed_mps_to_surface(speed_mps)

    def set_target_speed_surface(self, speed_surface):
        speed_surface = max(0.0, float(speed_surface))
        self.target_speed = speed_surface
        self.target_speed_mps = speed_surface_to_mps(speed_surface)

    def set_cruise_speed_mps(self, speed_mps):
        self.cruise_speed_mps = max(0.0, float(speed_mps))
        self.cruise_speed = speed_mps_to_surface(self.cruise_speed_mps)
        self.set_target_speed_mps(self.cruise_speed_mps)

    def set_initial_speed_mps(self, speed_mps):
        speed_mps = max(0.0, float(speed_mps))
        self.initial_speed_mps = min(speed_mps, self.cruise_speed_mps)
        self.initial_speed = speed_mps_to_surface(self.initial_speed_mps)

    def set_startup_ramp_duration(self, duration_s):
        self.startup_ramp_duration_s = max(0.0, float(duration_s))

    def set_behavior_state(self, state_name):
        self.behavior_state = str(state_name)

    def _startup_capped_target_speed_mps(self):
        target_speed_mps = max(0.0, float(self.target_speed_mps))
        if self.startup_ramp_duration_s <= 0.0 or self.initial_speed_mps is None:
            return target_speed_mps

        ramp_progress = min(self._startup_elapsed_s / self.startup_ramp_duration_s, 1.0)
        startup_cap_mps = self.initial_speed_mps + (
            self.cruise_speed_mps - self.initial_speed_mps
        ) * ramp_progress
        return min(target_speed_mps, startup_cap_mps)

    def set_route_surface(self, route_x, route_y, initial_speed_surface=None, jitter_start=False):
        self.rx = list(route_x)
        self.ry = list(route_y)
        if initial_speed_surface is None:
            initial_speed_surface = self.state[3] if hasattr(self, "_controller") else self.target_speed
        self._controller.set_route(
            self.rx,
            self.ry,
            v0=float(initial_speed_surface),
            jitter_start=jitter_start,
        )

    def sync_rect(self, frame):
        self.rect = frame.rect_from_world_center(
            (self._controller.x, self._controller.y), (self.size, self.size)
        )

    def draw(self, screen, frame):
        self.sync_rect(frame)
        if self.selected:
            self.start_node.render(screen, cfg.green)
            self.end_node.render(screen, cfg.red)
            for node in self.path:
                node.render(screen, cfg.blue)

        self.rect = pygame.draw.rect(screen, self._color, self.rect)

    def isCollided(self, hero, offset=None):
        result = hero.rect.colliderect(self.rect)

        # --- Distance estimation (center-to-center) ---
        hero_center = hero.rect.center
        obj_center = self.rect.center
        dx = hero_center[0] - obj_center[0]
        dy = hero_center[1] - obj_center[1]
        distance = math.hypot(dx, dy)  # Euclidean distance in pixels

        return self.id, result, distance

    @property
    def data(self):
        return [
            None,
            self.id,
            self.start_node.scaled_pos,
            self.end_node.scaled_pos,
            self.rx,
            self.ry,
        ]

    @property
    def state(self):
        return self._controller.state
