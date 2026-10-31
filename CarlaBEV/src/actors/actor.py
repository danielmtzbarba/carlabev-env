"""Background actors: route-following bodies with behaviours (no rendering here).

Actors own their kinematics (controller), behaviour and metric footprint. They
expose ``body()``/``semantic_class`` for the renderer and the collision system
and never touch pygame; drawing is the renderer's job.
"""

import numpy as np

from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.actors.traffic_policy import compute_safe_target_speed_mps, compute_traffic_command
from CarlaBEV.src.sim_time import TimeConfig
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT
from CarlaBEV.src.geometry.body import Body
from CarlaBEV.src.geometry.rects import body_surface_rect
from CarlaBEV.semantics import SemanticClass

# Scene-designer draw offsets (GUI concern): imported lazily to keep pygame out of the domain.
_GUI_OFFSET = None


def _gui_offset():
    global _GUI_OFFSET
    if _GUI_OFFSET is None:
        try:
            from CarlaBEV.src.gui.settings import Settings as cfg

            _GUI_OFFSET = (cfg.offx, cfg.offy)
        except Exception:  # pragma: no cover - GUI settings unavailable
            _GUI_OFFSET = (0, 0)
    return _GUI_OFFSET


class Node(object):
    """Graph node reference (id, surface position, lane). GUI helpers import pygame lazily."""

    def __init__(self, id, position, lane=None):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        offx, offy = _gui_offset()
        self.draw_x = self._x + offx
        self.draw_y = self._y + offy
        self.color = None
        self._btn = None

    @property
    def btn(self):
        if self._btn is None:
            import pygame

            self._btn = pygame.Rect(self.draw_x, self.draw_y, 3, 3)
        return self._btn

    def reset(self):
        self.color = None

    def render(self, screen, color=None):
        import pygame

        if color is not None:
            self.color = color
        if self.color is not None:
            pygame.draw.rect(screen, self.color, self.btn)

    def clicked(self, event):
        from CarlaBEV.src.gui.settings import Settings as cfg

        if self.btn.collidepoint(event.pos):
            self.color = cfg.red
            return True

    @property
    def scaled_pos(self):
        return [self._x, self._y]

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]


class Actor:
    traffic_speed_kp = 0.5
    traffic_max_accel_mps2 = 2.5
    traffic_max_decel_mps2 = 3.5
    # Kinematic bicycle wheelbase [m] used by the route-tracking controller.
    # Also the Stanley front-axle look-ahead, so it must be small relative to
    # the shortest route the actor is expected to follow.
    wheelbase_m = 2.9
    reacts_to_traffic = False
    # Rule compliance flags (scenario adversaries clear them, e.g. a red-light runner).
    obeys_signals = True
    yields_at_junctions = True

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
        # Simulation clock description. Bound by the ActorManager at load time
        # (same pattern as drivable_mask); defaults to the documented clock so
        # stand-alone actors remain usable.
        self.time_config = TimeConfig()
        self.units = DEFAULT_UNIT_CONTRACT
        # Metric footprint (Footprint); subclasses bind it from the unit contract.
        self.footprint = self.units.footprints.vehicle
        self.length_m = self.footprint.length_m
        self.width_m = self.footprint.width_m
        self.np_rng = np_rng
        # Lifecycle: procedurally generated, behaviour-free traffic may be recycled
        # onto a new route once it finishes (see managers/traffic_lifecycle.py).
        self.recyclable = False
        self.recycle_count = 0
        self.lane = getattr(start_node, "lane", None)
        self.drivable_mask = None
        # Junction zones for right-of-way (bound by the ActorManager; None = no rule).
        self.junctions = None
        self._yield_wait_s = 0.0
        self.last_traffic_policy = {
            "traffic_braking_active": False,
            "lead_gap_m": None,
            "lead_speed_mps": None,
            "lead_ttc_s": None,
            "nominal_target_speed_mps": 0.0,
            "safe_target_speed_mps": 0.0,
        }

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
        self.initial_speed = self.units.metric.speed_mps_to_surface(initial_speed_mps)
        self.target_speed = self.cruise_speed
        self.target_speed_mps = self.cruise_speed_mps
        self._startup_elapsed_s = 0.0
        self.behavior_state = "idle"
        self._prev_nominal_target_speed_mps = None
        self._yield_wait_s = 0.0

        # Behaviours reset first so they can define the spawn state (e.g. a
        # waiting pedestrian starts at rest) before the controller is built.
        if self.behavior:
            self.behavior.reset(self)
            self.initial_speed = self.units.metric.speed_mps_to_surface(self.initial_speed_mps)

        # The controller speed *limit* is the cruise cap; the momentary target
        # (possibly 0 for a waiting behaviour) is applied separately.
        self._controller = Controller(
            self.cruise_speed,
            L=self.wheelbase_m,
            speed_kp=self.traffic_speed_kp,
            max_accel_mps2=self.traffic_max_accel_mps2,
            max_decel_mps2=self.traffic_max_decel_mps2,
            dt=self.physics_dt_s,
            metric=self.units.metric,
        )
        self._controller.set_target_speed(self.target_speed, update_speed_limit=False)
        self._controller.set_route(
            self.rx,
            self.ry,
            v0=self.initial_speed,
            np_rng=self.np_rng,
            drivable_mask=self.drivable_mask,
        )

        self.rect = None  # legacy; surface rects are derived on demand (surface_rect)

    def set_time_config(self, time_config):
        """Bind the simulation clock description used to build the controller."""
        self.time_config = time_config if time_config is not None else TimeConfig()
        controller = getattr(self, "_controller", None)
        if controller is not None:
            controller.dt = self.physics_dt_s

    @property
    def physics_dt_s(self) -> float:
        return float(self.time_config.physics_dt_s)

    def set_units(self, units):
        """Bind the unit contract (metric frame, footprints, clock)."""
        self.units = units if units is not None else DEFAULT_UNIT_CONTRACT
        self.set_time_config(self.units.time)
        self._apply_footprint()

    def _apply_footprint(self):
        """Bind the metric footprint and derive the legacy pixel size. Subclasses override."""
        self.footprint = self.units.footprints.vehicle
        self.length_m = self.footprint.length_m
        self.width_m = self.footprint.width_m
        self.size = self.units.px_int(self.length_m)

    @property
    def kind(self) -> str:
        return "actor"

    @property
    def color(self):
        from CarlaBEV.semantics import semantic_color_tuple

        return semantic_color_tuple(self.semantic_class)

    def body(self) -> Body:
        """Oriented metric body at the current controller pose."""
        ctrl = self._controller
        return Body(self.footprint, ctrl.x_m, ctrl.y_m, ctrl.yaw, ctrl.v_mps)

    def step(self, t, dt, other_actors=None, signals=None):
        """Advance one physics step.

        ``t`` is the end-of-step simulated time observed during this step and
        ``dt`` the physics step, both provided by the scene's SimClock. The
        integrator uses the same ``dt``; any mismatch is a wiring error.
        ``other_actors`` are the perceived traffic actors (incl. the ego) and
        ``signals`` the scene's traffic lights.
        """
        dt = float(dt)
        if abs(dt - self._controller.dt) > 1e-12:
            raise ValueError(
                f"Actor.step received dt={dt} but its controller integrates with dt={self._controller.dt}; "
                "bind the actor to the scene TimeConfig via ActorManager."
            )
        # Startup ramp and behaviours both observe end-of-step time.
        self._startup_elapsed_s = self.time_config.snap(self._startup_elapsed_s + dt)

        # --- APPLY BEHAVIOR FIRST (with perception: other actors, the ego, signals) ---
        if self.behavior:
            ego = next((o for o in (other_actors or ()) if getattr(o, "kind", None) == "agent"), None)
            self.behavior.apply(self, t, dt, other_actors=other_actors, ego=ego, signals=signals or ())

        # --- PROPAGATE TARGET SPEED TO CONTROLLER ---
        nominal_target_speed_mps = max(0.0, float(self.target_speed_mps))
        # Behaviour-driven target *ramp* rate, fed forward to the speed loop so a
        # requested deceleration is realised rather than lagged (S-01). Step
        # changes (cruise -> 0) are clipped by the controller's accel limits.
        if self._prev_nominal_target_speed_mps is None:
            target_accel_ff_mps2 = 0.0
        else:
            target_accel_ff_mps2 = (nominal_target_speed_mps - self._prev_nominal_target_speed_mps) / dt
            target_accel_ff_mps2 = float(
                np.clip(target_accel_ff_mps2, -self.traffic_max_decel_mps2, self.traffic_max_accel_mps2)
            )
        self._prev_nominal_target_speed_mps = nominal_target_speed_mps
        rules = self.units.rules
        if self.reacts_to_traffic and rules.car_following == "idm":
            return self._step_rule_based(nominal_target_speed_mps, target_accel_ff_mps2, other_actors, signals)

        safe_target_speed_mps = nominal_target_speed_mps
        if self.reacts_to_traffic:
            safe_target_speed_mps, self.last_traffic_policy = compute_safe_target_speed_mps(
                self,
                other_actors,
                nominal_target_speed_mps,
            )
        else:
            self.last_traffic_policy = {
                "traffic_braking_active": False,
                "lead_gap_m": None,
                "lead_speed_mps": None,
                "lead_ttc_s": None,
                "nominal_target_speed_mps": nominal_target_speed_mps,
                "safe_target_speed_mps": nominal_target_speed_mps,
            }

        controller_target_mps = self._startup_capped_target_speed_mps(safe_target_speed_mps)
        self.last_traffic_policy["safe_target_speed_mps"] = float(controller_target_mps)
        # The startup ramp / traffic policy may cap the request; a cap below the
        # nominal target must not receive positive feed-forward.
        if controller_target_mps < nominal_target_speed_mps - 1e-9:
            target_accel_ff_mps2 = min(target_accel_ff_mps2, 0.0)
        self._controller.set_target_speed(
            self.units.metric.speed_mps_to_surface(controller_target_mps),
            update_speed_limit=False,
            target_accel_ff_mps2=target_accel_ff_mps2,
        )

        return self._controller.control_step()

    def _step_rule_based(self, nominal_target_speed_mps, target_accel_ff_mps2, other_actors, signals):
        """Rule-based longitudinal step: IDM following, stop lines, junction yielding.

        Without a traffic constraint the speed loop is identical to the legacy
        path (``kp (target - v) + ff``). With a constraint, the IDM acceleration
        is commanded exactly: the controller target is set to the current speed
        and the acceleration is passed as feed-forward, so the realised
        acceleration equals the rule's command (clipped by the actor's accel
        limits and the route-end stopping profile).
        """
        ctrl = self._controller
        v = float(ctrl.v_mps)
        controller_target_mps = self._startup_capped_target_speed_mps(nominal_target_speed_mps)
        if controller_target_mps < nominal_target_speed_mps - 1e-9:
            target_accel_ff_mps2 = min(target_accel_ff_mps2, 0.0)
        behaviour_accel = self.traffic_speed_kp * (controller_target_mps - v) + target_accel_ff_mps2
        accel_cmd, debug = compute_traffic_command(
            self,
            other_actors,
            controller_target_mps,
            behaviour_accel_mps2=behaviour_accel,
            signals=signals or (),
            junctions=self.junctions,
            rules=self.units.rules,
            waited_s=self._yield_wait_s,
        )
        constraint = debug["constraint"]
        # Yield-wait bookkeeping: time spent (nearly) stopped at a yield point.
        if constraint == "yield" and v <= 0.3:
            self._yield_wait_s = self.time_config.snap(self._yield_wait_s + self.physics_dt_s)
        elif debug["yield_to"] is None:
            self._yield_wait_s = 0.0
        self.last_traffic_policy = debug
        if constraint is None:
            self.last_traffic_policy["safe_target_speed_mps"] = float(controller_target_mps)
            ctrl.set_target_speed(
                self.units.metric.speed_mps_to_surface(controller_target_mps),
                update_speed_limit=False,
                target_accel_ff_mps2=target_accel_ff_mps2,
            )
        else:
            accel_cmd = float(np.clip(accel_cmd, -self.traffic_max_decel_mps2, self.traffic_max_accel_mps2))
            # A virtual/real obstacle closer than the standstill gap yields a negative
            # command at rest; never integrate below zero speed (no reversing).
            stop_floor = -max(v, 0.0) / self.physics_dt_s + ctrl.longitudinal_resistance_accel()
            accel_cmd = max(accel_cmd, min(stop_floor, 0.0))
            if debug["settle"]:
                accel_cmd = min(stop_floor, 0.0)  # come to a full stop instead of creeping
            # Command the rule's acceleration exactly; the proportional term is zero
            # because the target equals the current speed. Never accelerate past
            # the (possibly startup-capped) nominal target.
            target_mps = min(v, controller_target_mps) if accel_cmd > 0.0 else v
            self.last_traffic_policy["safe_target_speed_mps"] = float(max(0.0, v + accel_cmd * self.physics_dt_s))
            ctrl.set_target_speed(
                self.units.metric.speed_mps_to_surface(target_mps),
                update_speed_limit=False,
                target_accel_ff_mps2=accel_cmd,
            )
        return ctrl.control_step()

    def set_target_speed_mps(self, speed_mps):
        speed_mps = max(0.0, float(speed_mps))
        self.target_speed_mps = speed_mps
        self.target_speed = self.units.metric.speed_mps_to_surface(speed_mps)

    def set_target_speed_surface(self, speed_surface):
        speed_surface = max(0.0, float(speed_surface))
        self.target_speed = speed_surface
        self.target_speed_mps = self.units.metric.speed_surface_to_mps(speed_surface)

    def set_cruise_speed_mps(self, speed_mps):
        self.cruise_speed_mps = max(0.0, float(speed_mps))
        self.cruise_speed = self.units.metric.speed_mps_to_surface(self.cruise_speed_mps)
        self.set_target_speed_mps(self.cruise_speed_mps)

    def set_initial_speed_mps(self, speed_mps):
        speed_mps = max(0.0, float(speed_mps))
        self.initial_speed_mps = min(speed_mps, self.cruise_speed_mps)
        self.initial_speed = self.units.metric.speed_mps_to_surface(self.initial_speed_mps)

    def set_startup_ramp_duration(self, duration_s):
        self.startup_ramp_duration_s = max(0.0, float(duration_s))

    def set_behavior_state(self, state_name):
        self.behavior_state = str(state_name)

    def _startup_capped_target_speed_mps(self, target_speed_mps=None):
        if target_speed_mps is None:
            target_speed_mps = self.target_speed_mps
        target_speed_mps = max(0.0, float(target_speed_mps))
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
            drivable_mask=self.drivable_mask,
        )

    @property
    def semantic_class(self) -> SemanticClass:
        return SemanticClass.VEHICLE

    def surface_rect(self):
        """Axis-aligned surface-px rect enclosing the body (legacy trigger semantics)."""
        return body_surface_rect(self.body(), self.units.metric)

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
