"""Jaywalking pedestrian behaviours, expressed on the behaviour framework.

States: ``waiting`` (at rest on the curb) -> ``entering``/``crossing`` ->
``yielding``/``stalled`` -> ``crossing`` | ``retreating`` -> ``cleared`` |
``retreated``. Timers run on the scene clock (``AtTimeS``/``AfterS``); the
conflict-zone and completion triggers are geometric (route index).

``gap_accept_s`` (optional) adds gap acceptance: the pedestrian only steps off
the curb when no vehicle (including the ego) heading towards it would reach
it within that many seconds.
"""

from __future__ import annotations

from CarlaBEV.src.actors.behavior.framework import (
    AfterS,
    AllOf,
    AnyOf,
    AtTimeS,
    Behavior,
    GapAccepted,
    NearPoint,
    Retreat,
    RouteComplete,
    RouteIndexAtLeast,
    SetInitialSpeed,
    State,
)


def _start_trigger(start_delay: float, gap_accept_s: float | None):
    timer = AtTimeS(float(start_delay))
    if gap_accept_s is None:
        return timer
    return AllOf((timer, GapAccepted(float(gap_accept_s))))


class BaseJaywalkBehavior(Behavior):
    """Wait, enter, then yield (and optionally retreat) or stall at the conflict zone."""

    def __init__(
        self,
        start_delay=0.0,
        trigger_fraction=0.5,
        stop_duration=None,
        retreat=False,
        gap_accept_s=None,
    ):
        self.start_delay = float(start_delay)
        self.trigger_fraction = float(trigger_fraction)
        self.stop_duration = stop_duration
        self.retreat = retreat
        self.gap_accept_s = gap_accept_s
        self._retreat_goal = None
        after_conflict = (
            "yielding" if (retreat or stop_duration is not None) else "stalled"
        )
        states = {
            "waiting": State(
                "waiting",
                speed_mps=0.0,
                on_enter=(SetInitialSpeed(0.0),),
                transitions=(
                    (_start_trigger(self.start_delay, gap_accept_s), "entering"),
                ),
            ),
            "entering": State(
                "entering",
                speed_mps=None,
                transitions=(
                    (RouteIndexAtLeast(self.trigger_fraction), after_conflict),
                    (RouteComplete(), "cleared"),
                ),
            ),
            "yielding": State(
                "yielding",
                speed_mps=0.0,
                transitions=()
                if stop_duration is None
                else (
                    (
                        AfterS(float(stop_duration)),
                        "retreating" if retreat else "crossing",
                    ),
                ),
            ),
            "crossing": State(
                "crossing", speed_mps=None, transitions=((RouteComplete(), "cleared"),)
            ),
            "stalled": State("stalled", speed_mps=0.0),
            "retreating": State(
                "retreating",
                speed_mps=None,
                on_enter=(Retreat(),),
                transitions=(
                    (
                        AnyOf((NearPoint("_retreat_goal", 1.0), RouteComplete())),
                        "retreated",
                    ),
                ),
            ),
            "cleared": State("cleared", speed_mps=0.0),
            "retreated": State("retreated", speed_mps=0.0),
        }
        super().__init__(states, initial="waiting", name="jaywalk")

    def reset(self, actor):
        self._retreat_goal = None
        super().reset(actor)

    # Legacy introspection used by tools/tests.
    @property
    def _elapsed(self) -> float:
        return self.elapsed_s

    @property
    def _state_elapsed(self) -> float:
        return self.state_elapsed_s


class CrossBehavior(Behavior):
    """Wait, then cross the road and stop at the far side."""

    def __init__(self, start_delay=0.0, gap_accept_s=None):
        self.start_delay = float(start_delay)
        self.gap_accept_s = gap_accept_s
        super().__init__(
            {
                "waiting": State(
                    "waiting",
                    speed_mps=0.0,
                    on_enter=(SetInitialSpeed(0.0),),
                    transitions=(
                        (_start_trigger(self.start_delay, gap_accept_s), "crossing"),
                    ),
                ),
                "crossing": State(
                    "crossing",
                    speed_mps=None,
                    transitions=((RouteComplete(), "cleared"),),
                ),
                "cleared": State("cleared", speed_mps=0.0),
            },
            initial="waiting",
            name="cross",
        )


class StopMidBehavior(BaseJaywalkBehavior):
    """Wait, enter the road, then stop in the ego lane."""

    def __init__(self, start_delay=0.0, gap_accept_s=None):
        super().__init__(
            start_delay=start_delay,
            trigger_fraction=0.5,
            stop_duration=None,
            retreat=False,
            gap_accept_s=gap_accept_s,
        )


class StopReturnBehavior(BaseJaywalkBehavior):
    """Wait, enter the road, yield briefly, then retreat to the curb."""

    def __init__(self, start_delay=0.0, yield_duration=1.0, gap_accept_s=None):
        super().__init__(
            start_delay=start_delay,
            trigger_fraction=1.0 / 3.0,
            stop_duration=yield_duration,
            retreat=True,
            gap_accept_s=gap_accept_s,
        )
