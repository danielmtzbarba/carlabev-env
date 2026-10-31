"""Declarative behaviour framework: finite-state machines with triggers and actions.

A :class:`Behavior` is a set of :class:`State`s. Each step (end-of-step time
``t`` from the scene clock) the current state's per-step actions run, then its
transitions are evaluated in order; the first trigger that fires switches
state, records the entry time and runs the new state's ``on_enter`` actions
(at most one transition per step). Timers are derived from ``t`` (never
accumulated), so ``AfterS(1.0)`` fires exactly at 1.0 s.

Triggers see a :class:`BehaviorContext` (actor, clock, perceived actors, ego,
signals) so behaviours can react to the world (gap acceptance, TTC, signals),
not only to time.

Behaviours are authorable from a JSON-like spec::

    {"initial": "cruise",
     "states": {
        "cruise":  {"transitions": [{"trigger": {"type": "at_time_s", "seconds": 2.5}, "to": "braking"}]},
        "braking": {"each_step": [{"type": "ramp_speed", "rate_mps2": -4.0}]}}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from CarlaBEV.src.sim_time import TIME_EPS_S, time_reached, timer_elapsed


# --------------------------------------------------------------------------- context
@dataclass
class BehaviorContext:
    actor: object
    t: float
    dt: float
    other_actors: list = field(default_factory=list)
    ego: object | None = None
    signals: list = field(default_factory=list)

    def state_elapsed(self, entered_t: float) -> float:
        return round((self.t - entered_t) / self.dt) * self.dt


# --------------------------------------------------------------------------- triggers
class Trigger:
    def fires(
        self, ctx: BehaviorContext, entered_t: float
    ) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def to_spec(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class Always(Trigger):
    def fires(self, ctx, entered_t):
        return True

    def to_spec(self):
        return {"type": "always"}


@dataclass(frozen=True)
class AtTimeS(Trigger):
    """Absolute simulated time since the behaviour started (end-of-step convention)."""

    seconds: float

    def fires(self, ctx, entered_t):
        return time_reached(ctx.t - ctx.actor.behavior._t_start, self.seconds)

    def to_spec(self):
        return {"type": "at_time_s", "seconds": self.seconds}


@dataclass(frozen=True)
class AfterS(Trigger):
    """Seconds elapsed in the current state."""

    seconds: float

    def fires(self, ctx, entered_t):
        return timer_elapsed(ctx.t, entered_t, self.seconds, ctx.dt)

    def to_spec(self):
        return {"type": "after_s", "seconds": self.seconds}


@dataclass(frozen=True)
class RouteIndexAtLeast(Trigger):
    """Controller target index reached a fraction of the route (geometry-triggered)."""

    fraction: float

    def fires(self, ctx, entered_t):
        actor = ctx.actor
        n = len(actor.rx)
        idx = max(1, min(n - 1, int(self.fraction * (n - 1))))
        return actor._controller.target_idx >= idx

    def to_spec(self):
        return {"type": "route_index_at_least", "fraction": self.fraction}


@dataclass(frozen=True)
class RouteComplete(Trigger):
    def fires(self, ctx, entered_t):
        actor = ctx.actor
        return actor._controller.target_idx >= len(actor.rx) - 1

    def to_spec(self):
        return {"type": "route_complete"}


@dataclass(frozen=True)
class NearPoint(Trigger):
    """Actor within ``radius_surface`` px of a point stored on the behaviour (``attr``)."""

    attr: str
    radius_surface: float = 1.0

    def fires(self, ctx, entered_t):
        goal = getattr(ctx.actor.behavior, self.attr, None)
        if goal is None:
            return False
        return (
            float(
                np.linalg.norm(
                    np.asarray(ctx.actor.state[:2], dtype=float)
                    - np.asarray(goal, dtype=float)
                )
            )
            <= self.radius_surface
        )

    def to_spec(self):
        return {
            "type": "near_point",
            "attr": self.attr,
            "radius_surface": self.radius_surface,
        }


@dataclass(frozen=True)
class EgoWithinM(Trigger):
    meters: float

    def fires(self, ctx, entered_t):
        if ctx.ego is None:
            return False
        c = ctx.actor._controller
        return math.hypot(ctx.ego.x_m - c.x_m, ctx.ego.y_m - c.y_m) <= self.meters

    def to_spec(self):
        return {"type": "ego_within_m", "meters": self.meters}


@dataclass(frozen=True)
class GapAccepted(Trigger):
    """No approaching vehicle (incl. the ego) would reach the actor's position within ``min_ttc_s``.

    Used by pedestrians before stepping off the curb: for every perceived vehicle
    heading towards the actor, the time to reach the actor's lateral line must
    exceed ``min_ttc_s`` (or the vehicle must be moving away / already past).
    """

    min_ttc_s: float
    lateral_margin_m: float = 3.0

    def fires(self, ctx, entered_t):
        from CarlaBEV.src.actors.traffic_policy import kinematics_of

        c = ctx.actor._controller
        candidates = list(ctx.other_actors) + ([ctx.ego] if ctx.ego is not None else [])
        for other in candidates:
            k = kinematics_of(other)
            if k is None or getattr(other, "kind", "vehicle") == "pedestrian":
                continue
            dx, dy = c.x_m - k.x_m, c.y_m - k.y_m
            forward = dx * math.cos(k.yaw) + dy * math.sin(
                k.yaw
            )  # actor position along the vehicle heading
            lateral = -dx * math.sin(k.yaw) + dy * math.cos(k.yaw)
            if forward <= 0.0 or abs(lateral) > self.lateral_margin_m:
                continue  # already past, or not on this vehicle's path
            speed = max(float(k.v_mps), 1e-3)
            if forward / speed < self.min_ttc_s:
                return False
        return True

    def to_spec(self):
        return {
            "type": "gap_accepted",
            "min_ttc_s": self.min_ttc_s,
            "lateral_margin_m": self.lateral_margin_m,
        }


@dataclass(frozen=True)
class AllOf(Trigger):
    triggers: tuple

    def fires(self, ctx, entered_t):
        return all(trigger.fires(ctx, entered_t) for trigger in self.triggers)

    def to_spec(self):
        return {"type": "all_of", "triggers": [t.to_spec() for t in self.triggers]}


@dataclass(frozen=True)
class AnyOf(Trigger):
    triggers: tuple

    def fires(self, ctx, entered_t):
        return any(trigger.fires(ctx, entered_t) for trigger in self.triggers)

    def to_spec(self):
        return {"type": "any_of", "triggers": [t.to_spec() for t in self.triggers]}


# --------------------------------------------------------------------------- actions
class Action:
    def apply(self, actor, ctx: BehaviorContext):  # pragma: no cover - interface
        raise NotImplementedError

    def to_spec(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class SetTargetSpeed(Action):
    """Set the target speed; ``None`` means the actor's cruise speed."""

    speed_mps: float | None

    def apply(self, actor, ctx):
        actor.set_target_speed_mps(
            actor.cruise_speed_mps if self.speed_mps is None else self.speed_mps
        )

    def to_spec(self):
        return {"type": "set_target_speed", "speed_mps": self.speed_mps}


@dataclass(frozen=True)
class RampSpeed(Action):
    """Change the target speed by ``rate_mps2 * dt`` per step (negative = brake)."""

    rate_mps2: float
    floor_mps: float = 0.0

    def apply(self, actor, ctx):
        actor.set_target_speed_mps(
            max(self.floor_mps, actor.target_speed_mps + self.rate_mps2 * ctx.dt)
        )

    def to_spec(self):
        return {
            "type": "ramp_speed",
            "rate_mps2": self.rate_mps2,
            "floor_mps": self.floor_mps,
        }


@dataclass(frozen=True)
class Stop(Action):
    def apply(self, actor, ctx):
        actor.set_target_speed_mps(0.0)

    def to_spec(self):
        return {"type": "stop"}


@dataclass(frozen=True)
class Retreat(Action):
    """Reverse the route back towards its start (pedestrian stop-and-return)."""

    goal_attr: str = "_retreat_goal"

    def apply(self, actor, ctx):
        current_idx = max(0, min(actor._controller.target_idx, len(actor.rx) - 1))
        retreat_rx = [actor.state[0]] + list(actor._initial_rx[: current_idx + 1][::-1])
        retreat_ry = [actor.state[1]] + list(actor._initial_ry[: current_idx + 1][::-1])
        setattr(
            actor.behavior,
            self.goal_attr,
            np.array([actor._initial_rx[0], actor._initial_ry[0]], dtype=float),
        )
        actor.set_route_surface(
            retreat_rx,
            retreat_ry,
            initial_speed_surface=actor.state[3],
            jitter_start=False,
        )
        actor.set_target_speed_mps(actor.cruise_speed_mps)

    def to_spec(self):
        return {"type": "retreat"}


@dataclass(frozen=True)
class SetInitialSpeed(Action):
    """Define the spawn speed (applied at reset, before the controller is built)."""

    speed_mps: float

    def apply(self, actor, ctx):
        actor.set_initial_speed_mps(self.speed_mps)

    def to_spec(self):
        return {"type": "set_initial_speed", "speed_mps": self.speed_mps}


@dataclass(frozen=True)
class SetFlag(Action):
    """Set an attribute on the actor (e.g. ``obeys_signals=False``)."""

    name: str
    value: object

    def apply(self, actor, ctx):
        setattr(actor, self.name, self.value)

    def to_spec(self):
        return {"type": "set_flag", "name": self.name, "value": self.value}


# --------------------------------------------------------------------------- machine
@dataclass
class State:
    name: str
    on_enter: tuple = ()
    each_step: tuple = ()
    transitions: tuple = ()  # of (Trigger, next_state_name)
    speed_mps: float | None = (
        "unset"  # convenience: per-step SetTargetSpeed (None = cruise)
    )

    def step_actions(self):
        actions = list(self.each_step)
        if self.speed_mps != "unset":
            actions.insert(0, SetTargetSpeed(self.speed_mps))
        return actions


class Behavior:
    """Finite-state behaviour driven by the scene clock and perception."""

    def __init__(self, states: dict[str, State], initial: str, *, name: str = "fsm"):
        if initial not in states:
            raise ValueError(
                f"initial state {initial!r} not in states {sorted(states)}"
            )
        self.states = states
        self.initial = initial
        self.name = name
        self.state_name = initial
        self._t_start: float | None = None
        self._t_state_entered: float | None = None
        self._last_ctx: BehaviorContext | None = None

    # --- lifecycle -------------------------------------------------------------
    def reset(self, actor):
        self.state_name = self.initial
        self._t_start = None
        self._t_state_entered = None
        self._last_ctx = None
        actor.set_behavior_state(self.state_name)
        ctx = BehaviorContext(actor, 0.0, 1.0)
        state = self.states[self.initial]
        for action in state.on_enter:
            action.apply(actor, ctx)
        if state.speed_mps != "unset":
            SetTargetSpeed(state.speed_mps).apply(actor, ctx)

    @property
    def state(self) -> State:
        return self.states[self.state_name]

    @property
    def elapsed_s(self) -> float:
        if self._last_ctx is None or self._t_start is None:
            return 0.0
        return self._last_ctx.state_elapsed(self._t_start)

    @property
    def state_elapsed_s(self) -> float:
        if self._last_ctx is None or self._t_state_entered is None:
            return 0.0
        return self._last_ctx.state_elapsed(self._t_state_entered)

    def _enter(self, actor, name: str, ctx: BehaviorContext):
        self.state_name = name
        self._t_state_entered = ctx.t
        actor.set_behavior_state(name)
        state = self.states[name]
        for action in state.on_enter:
            action.apply(actor, ctx)
        if state.speed_mps != "unset":
            SetTargetSpeed(state.speed_mps).apply(actor, ctx)

    def apply(self, actor, t, dt, other_actors=None, ego=None, signals=None):
        ctx = BehaviorContext(
            actor,
            float(t),
            float(dt),
            list(other_actors or ()),
            ego,
            list(signals or ()),
        )
        self._last_ctx = ctx
        if self._t_start is None:
            # First step after reset: the behaviour started one dt before the
            # observed end-of-step time.
            self._t_start = ctx.t - ctx.dt
            self._t_state_entered = self._t_start
        state = self.state
        for action in state.step_actions():
            action.apply(actor, ctx)
        for trigger, target in state.transitions:
            if trigger.fires(ctx, self._t_state_entered):
                self._enter(actor, target, ctx)
                break

    # --- spec -------------------------------------------------------------------
    def to_spec(self) -> dict:
        return {
            "type": "fsm",
            "name": self.name,
            "initial": self.initial,
            "states": {
                name: {
                    **({"speed_mps": st.speed_mps} if st.speed_mps != "unset" else {}),
                    "on_enter": [a.to_spec() for a in st.on_enter],
                    "each_step": [a.to_spec() for a in st.each_step],
                    "transitions": [
                        {"trigger": tr.to_spec(), "to": to} for tr, to in st.transitions
                    ],
                }
                for name, st in self.states.items()
            },
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "Behavior":
        states = {}
        for name, st in spec["states"].items():
            states[name] = State(
                name=name,
                on_enter=tuple(action_from_spec(a) for a in st.get("on_enter", [])),
                each_step=tuple(action_from_spec(a) for a in st.get("each_step", [])),
                transitions=tuple(
                    (trigger_from_spec(tr["trigger"]), tr["to"])
                    for tr in st.get("transitions", [])
                ),
                speed_mps=st.get("speed_mps", "unset"),
            )
        return cls(states, spec["initial"], name=spec.get("name", "fsm"))


TRIGGERS = {
    "always": lambda s: Always(),
    "at_time_s": lambda s: AtTimeS(float(s["seconds"])),
    "after_s": lambda s: AfterS(float(s["seconds"])),
    "route_index_at_least": lambda s: RouteIndexAtLeast(float(s["fraction"])),
    "route_complete": lambda s: RouteComplete(),
    "near_point": lambda s: NearPoint(
        str(s["attr"]), float(s.get("radius_surface", 1.0))
    ),
    "ego_within_m": lambda s: EgoWithinM(float(s["meters"])),
    "gap_accepted": lambda s: GapAccepted(
        float(s["min_ttc_s"]), float(s.get("lateral_margin_m", 3.0))
    ),
    "all_of": lambda s: AllOf(tuple(trigger_from_spec(t) for t in s["triggers"])),
    "any_of": lambda s: AnyOf(tuple(trigger_from_spec(t) for t in s["triggers"])),
}

ACTIONS = {
    "set_target_speed": lambda s: SetTargetSpeed(s.get("speed_mps")),
    "ramp_speed": lambda s: RampSpeed(
        float(s["rate_mps2"]), float(s.get("floor_mps", 0.0))
    ),
    "stop": lambda s: Stop(),
    "retreat": lambda s: Retreat(),
    "set_flag": lambda s: SetFlag(str(s["name"]), s["value"]),
    "set_initial_speed": lambda s: SetInitialSpeed(float(s["speed_mps"])),
}


def trigger_from_spec(spec: dict) -> Trigger:
    try:
        return TRIGGERS[spec["type"]](spec)
    except KeyError as exc:
        raise ValueError(f"unknown trigger spec {spec!r}") from exc


def action_from_spec(spec: dict) -> Action:
    try:
        return ACTIONS[spec["type"]](spec)
    except KeyError as exc:
        raise ValueError(f"unknown action spec {spec!r}") from exc


__all__ = [
    "Behavior",
    "BehaviorContext",
    "State",
    "Trigger",
    "Action",
    "Always",
    "AtTimeS",
    "AfterS",
    "RouteIndexAtLeast",
    "RouteComplete",
    "NearPoint",
    "EgoWithinM",
    "GapAccepted",
    "AllOf",
    "AnyOf",
    "SetTargetSpeed",
    "RampSpeed",
    "Stop",
    "Retreat",
    "SetFlag",
    "trigger_from_spec",
    "action_from_spec",
    "TIME_EPS_S",
]
