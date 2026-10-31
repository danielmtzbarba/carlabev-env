"""Intelligent Driver Model (IDM) car-following on metric shape gaps.

Treiber, Hennecke & Helbing (2000). The acceleration command is

    a = a_max * [1 - (v / v0)^delta - (s* / s)^2]
    s* = s0 + max(0, v T + v dv / (2 sqrt(a_max b)))

with ``v`` the own speed, ``v0`` the desired speed, ``s`` the bumper-to-bumper
gap to the lead, ``dv = v - v_lead`` the closing speed, ``T`` the desired time
headway, ``s0`` the standstill gap and ``b`` the comfortable deceleration.

The same equation serves stop lines and yield points: a virtual lead at rest
(``v_lead = 0``) placed at the stopping position makes the vehicle settle
``s0`` metres before it. All quantities are SI; nothing here knows about
pixels or the controller — the actor feeds the result to its speed loop as a
feed-forward acceleration (see :mod:`CarlaBEV.src.actors.traffic_policy`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class IDMParams:
    max_accel_mps2: float = 2.0
    """``a_max``: free-road acceleration authority."""
    comfort_decel_mps2: float = 2.5
    """``b``: comfortable braking deceleration (the model can exceed it in emergencies)."""
    min_gap_m: float = 2.0
    """``s0``: bumper-to-bumper gap kept at standstill."""
    time_headway_s: float = 1.5
    """``T``: desired time gap to the lead."""
    delta: float = 4.0
    """Free-road acceleration exponent."""

    def __post_init__(self):
        for name in (
            "max_accel_mps2",
            "comfort_decel_mps2",
            "min_gap_m",
            "time_headway_s",
            "delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"IDMParams.{name} must be positive, got {value!r}")

    def desired_gap_m(self, v_mps: float, closing_speed_mps: float) -> float:
        """``s*``: dynamic desired gap at speed ``v`` closing at ``dv``."""
        v = max(0.0, float(v_mps))
        dv = float(closing_speed_mps)
        dyn = v * self.time_headway_s + v * dv / (
            2.0 * math.sqrt(self.max_accel_mps2 * self.comfort_decel_mps2)
        )
        return self.min_gap_m + max(0.0, dyn)

    def free_accel_mps2(self, v_mps: float, desired_speed_mps: float) -> float:
        """Free-road term ``a_max (1 - (v/v0)^delta)``; a stop request (``v0 = 0``) brakes at ``-b``."""
        v = max(0.0, float(v_mps))
        v0 = max(0.0, float(desired_speed_mps))
        if v0 <= 1e-9:
            return -self.comfort_decel_mps2 if v > 0.0 else 0.0
        return self.max_accel_mps2 * (1.0 - (v / v0) ** self.delta)

    def interaction_accel_mps2(
        self, v_mps: float, gap_m: float, lead_speed_mps: float
    ) -> float:
        """Interaction term ``-a_max (s*/s)^2`` for a lead ``gap_m`` ahead at ``lead_speed_mps``."""
        v = max(0.0, float(v_mps))
        gap = max(1e-3, float(gap_m))
        s_star = self.desired_gap_m(v, v - max(0.0, float(lead_speed_mps)))
        return -self.max_accel_mps2 * (s_star / gap) ** 2

    def acceleration_mps2(
        self,
        v_mps: float,
        desired_speed_mps: float,
        gap_m: float | None = None,
        lead_speed_mps: float = 0.0,
    ) -> float:
        """Full IDM acceleration; ``gap_m=None`` means free road."""
        a = self.free_accel_mps2(v_mps, desired_speed_mps)
        if gap_m is not None:
            a += self.interaction_accel_mps2(v_mps, gap_m, lead_speed_mps)
        return float(a)

    def equilibrium_gap_m(self, v_mps: float) -> float:
        """Steady-state following gap at speed ``v`` (``dv = 0``): ``s0 + v T`` scaled by the free term."""
        # At equilibrium a = 0 with dv = 0: (s*/s)^2 = 1 - (v/v0)^delta. For v << v0 this is s ~ s*.
        return self.desired_gap_m(v_mps, 0.0)


# Per speed class (scene generator "slow" / "medium" / "fast" drivers). Faster
# drivers keep shorter headways and accelerate harder; slower drivers are more
# conservative. Values are within the ranges used in the traffic literature.
IDM_PARAMS_BY_SPEED_CLASS: dict[str, IDMParams] = {
    "slow": IDMParams(
        max_accel_mps2=1.5, comfort_decel_mps2=2.0, min_gap_m=2.5, time_headway_s=1.8
    ),
    "medium": IDMParams(
        max_accel_mps2=2.0, comfort_decel_mps2=2.5, min_gap_m=2.0, time_headway_s=1.5
    ),
    "fast": IDMParams(
        max_accel_mps2=2.5, comfort_decel_mps2=3.0, min_gap_m=1.5, time_headway_s=1.2
    ),
}


def idm_params_for(speed_class: str | None) -> IDMParams:
    return IDM_PARAMS_BY_SPEED_CLASS.get(
        str(speed_class or "medium"), IDM_PARAMS_BY_SPEED_CLASS["medium"]
    )
