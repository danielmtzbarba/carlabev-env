"""Timed braking lead vehicle, expressed on the behaviour framework."""

from __future__ import annotations

from CarlaBEV.src.actors.behavior.framework import AtTimeS, Behavior, RampSpeed, State


class LeadBrakeBehavior(Behavior):
    """Cruise, then from ``start_brake_t`` [s] ramp the target speed down at ``dec_rate`` [m/s^2].

    The trigger fires at the first step whose end-of-step time reaches
    ``start_brake_t``; the ramp is applied in that same step (entry) and every
    step after, and the rate is fed forward to the speed loop by the actor.
    """

    def __init__(self, start_brake_t=3.5, dec_rate=1.0):
        self.start_brake_t = float(start_brake_t)
        self.dec_rate = float(dec_rate)
        ramp = RampSpeed(-self.dec_rate)
        super().__init__(
            {
                "cruise": State(
                    "cruise", transitions=((AtTimeS(self.start_brake_t), "braking"),)
                ),
                "braking": State("braking", on_enter=(ramp,), each_step=(ramp,)),
            },
            initial="cruise",
            name="timed_brake",
        )

    @property
    def braking(self) -> bool:
        return self.state_name == "braking"
