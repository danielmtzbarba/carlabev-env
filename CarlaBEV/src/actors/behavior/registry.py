from __future__ import annotations

from dataclasses import dataclass

from CarlaBEV.src.actors.behavior.jaywalk import (
    CrossBehavior,
    StopMidBehavior,
    StopReturnBehavior,
)
from CarlaBEV.src.actors.behavior.framework import Behavior
from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior


@dataclass(frozen=True)
class BehaviorField:
    key: str
    label: str
    default: float
    cast: type = float

    def parse(self, value):
        if value in (None, ""):
            return self.cast(self.default)
        return self.cast(value)


@dataclass(frozen=True)
class BehaviorSpec:
    behavior_id: str
    label: str
    fields: tuple[BehaviorField, ...] = ()


BEHAVIOR_LIBRARY = {
    "agent": {
        "none": BehaviorSpec("none", "None"),
    },
    "vehicle": {
        "constant_speed": BehaviorSpec("constant_speed", "Constant Speed"),
        "timed_brake": BehaviorSpec(
            "timed_brake",
            "Timed Brake",
            fields=(
                BehaviorField("start_brake_t", "Brake Start (s)", 3.5),
                BehaviorField("decel_mps2", "Decel (m/s^2)", 1.0),
            ),
        ),
    },
    "pedestrian": {
        "cross": BehaviorSpec(
            "cross",
            "Cross",
            fields=(
                BehaviorField("start_delay", "Start Delay (s)", 0.0),
                BehaviorField("gap_accept_s", "Gap Acceptance TTC (s, 0 = off)", 0.0),
            ),
        ),
        "stop_mid": BehaviorSpec(
            "stop_mid",
            "Stop Mid",
            fields=(
                BehaviorField("start_delay", "Start Delay (s)", 0.0),
                BehaviorField("gap_accept_s", "Gap Acceptance TTC (s, 0 = off)", 0.0),
            ),
        ),
        "yield_return": BehaviorSpec(
            "yield_return",
            "Yield Return",
            fields=(
                BehaviorField("start_delay", "Start Delay (s)", 0.0),
                BehaviorField("yield_duration", "Yield Duration (s)", 1.0),
                BehaviorField("gap_accept_s", "Gap Acceptance TTC (s, 0 = off)", 0.0),
            ),
        ),
    },
}

# Declarative finite-state behaviours authored inline: {"type": "fsm", "initial": ..., "states": {...}}
FSM_SPEC_TYPE = "fsm"

LEGACY_BEHAVIOR_NAMES = {
    "Normal": "constant_speed",
    "CrossBehavior": "cross",
    "StopMidBehavior": "stop_mid",
    "StopReturnBehavior": "yield_return",
    "LeadBrakeBehavior": "timed_brake",
}


def behavior_options_for_actor(actor_type: str) -> list[str]:
    return list(
        BEHAVIOR_LIBRARY.get(actor_type, {"none": BehaviorSpec("none", "None")}).keys()
    )


def behavior_label_map_for_actor(actor_type: str) -> dict[str, str]:
    return {
        behavior_id: spec.label
        for behavior_id, spec in BEHAVIOR_LIBRARY.get(
            actor_type, {"none": BehaviorSpec("none", "None")}
        ).items()
    }


def get_behavior_spec(actor_type: str, behavior_id: str) -> BehaviorSpec:
    actor_specs = BEHAVIOR_LIBRARY.get(actor_type, {})
    if behavior_id not in actor_specs:
        if actor_specs:
            return next(iter(actor_specs.values()))
        return BehaviorSpec("none", "None")
    return actor_specs[behavior_id]


def normalize_behavior_spec(actor_type: str, behavior):
    actor_specs = BEHAVIOR_LIBRARY.get(actor_type, {})
    if not actor_specs:
        return {"type": "none", "params": {}}

    if behavior in (None, "", "Normal"):
        default_id = "none" if "none" in actor_specs else next(iter(actor_specs.keys()))
        return {"type": default_id, "params": {}}

    if isinstance(behavior, str):
        behavior_id = LEGACY_BEHAVIOR_NAMES.get(behavior, behavior)
        spec = get_behavior_spec(actor_type, behavior_id)
        return {"type": spec.behavior_id, "params": {}}

    if behavior.get("type") == FSM_SPEC_TYPE:
        # Inline state machine: validated by constructing it.
        Behavior.from_spec(behavior)
        return {"type": FSM_SPEC_TYPE, "params": {}, "spec": dict(behavior)}

    behavior_id = LEGACY_BEHAVIOR_NAMES.get(
        behavior.get("type", ""), behavior.get("type", "")
    )
    spec = get_behavior_spec(actor_type, behavior_id)
    raw_params = behavior.get("params", {}) or behavior.get("behavior_kwargs", {}) or {}
    params = {
        field.key: field.parse(raw_params.get(field.key)) for field in spec.fields
    }
    return {"type": spec.behavior_id, "params": params}


def build_behavior(actor_type: str, behavior):
    normalized = normalize_behavior_spec(actor_type, behavior)
    behavior_id = normalized["type"]
    params = normalized["params"]

    if behavior_id in {"none", "constant_speed"}:
        return None, normalized
    if behavior_id == FSM_SPEC_TYPE:
        return Behavior.from_spec(normalized["spec"]), normalized
    gap_accept_s = params.get("gap_accept_s", 0.0) or None  # 0 = disabled
    if behavior_id == "cross":
        return CrossBehavior(start_delay=params.get("start_delay", 0.0), gap_accept_s=gap_accept_s), normalized
    if behavior_id == "stop_mid":
        return StopMidBehavior(start_delay=params.get("start_delay", 0.0), gap_accept_s=gap_accept_s), normalized
    if behavior_id == "yield_return":
        return StopReturnBehavior(
            start_delay=params.get("start_delay", 0.0),
            yield_duration=params.get("yield_duration", 1.0),
            gap_accept_s=gap_accept_s,
        ), normalized
    if behavior_id == "timed_brake":
        return LeadBrakeBehavior(
            start_brake_t=params.get("start_brake_t", 3.5),
            dec_rate=params.get("decel_mps2", 1.0),
        ), normalized
    return None, normalized


def behavior_to_spec(behavior) -> dict | None:
    """Serializable spec of a behaviour instance (inverse of :func:`build_behavior`).

    Registry behaviours round-trip through their constructor parameters (the
    class carries scripted logic a plain FSM spec cannot express); any other
    :class:`Behavior` round-trips through its declarative FSM spec.
    """
    if behavior is None:
        return None
    if isinstance(behavior, LeadBrakeBehavior):
        return {
            "type": "timed_brake",
            "params": {"start_brake_t": behavior.start_brake_t, "decel_mps2": behavior.dec_rate},
        }
    if isinstance(behavior, StopReturnBehavior):
        return {
            "type": "yield_return",
            "params": {
                "start_delay": behavior.start_delay,
                "yield_duration": float(behavior.stop_duration if behavior.stop_duration is not None else 1.0),
                "gap_accept_s": float(behavior.gap_accept_s or 0.0),
            },
        }
    if isinstance(behavior, StopMidBehavior):
        return {
            "type": "stop_mid",
            "params": {"start_delay": behavior.start_delay, "gap_accept_s": float(behavior.gap_accept_s or 0.0)},
        }
    if isinstance(behavior, CrossBehavior):
        return {
            "type": "cross",
            "params": {"start_delay": behavior.start_delay, "gap_accept_s": float(behavior.gap_accept_s or 0.0)},
        }
    if isinstance(behavior, Behavior):
        return {"type": FSM_SPEC_TYPE, "params": {}, "spec": behavior.to_spec()}
    raise TypeError(f"Cannot serialize behaviour of type {type(behavior).__name__}")


def behavior_from_spec(actor_type: str, spec):
    """Rebuild a behaviour from :func:`behavior_to_spec` output (``None`` -> no behaviour)."""
    if spec is None:
        return None
    behavior, _ = build_behavior(actor_type, spec)
    return behavior
