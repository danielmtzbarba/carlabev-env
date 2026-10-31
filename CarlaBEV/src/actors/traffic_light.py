"""Stop-line traffic light: a semantic strip with a signal state.

The state is either static (as authored) or derived from the scene clock
through a :class:`SignalCycle` (``state = f(t)``; never accumulated, so it is
independent of rendering FPS and identical under any step history).

A light *controls* the traffic travelling in ``controls_heading`` (name of the
travel direction: ``"north"`` means vehicles heading up the map, i.e. surface
yaw -pi/2). Lights without a controlled heading are decorative: they are drawn
but no vehicle obeys them (legacy authored scenes).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from CarlaBEV.src.actors.actor import Actor
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT
from CarlaBEV.semantics import SemanticClass, semantic_color_tuple


class TrafficLightState:
    RED = 0
    YELLOW = 1
    GREEN = 2


STATE_NAMES = {
    TrafficLightState.RED: "red",
    TrafficLightState.YELLOW: "yellow",
    TrafficLightState.GREEN: "green",
}
STATE_BY_NAME = {name: value for value, name in STATE_NAMES.items()}

# Travel direction name -> surface yaw [rad] (y grows downwards on the surface).
HEADING_YAW = {
    "east": 0.0,
    "south": math.pi / 2,
    "west": math.pi,
    "north": -math.pi / 2,
}


@dataclass(frozen=True)
class SignalCycle:
    """Fixed-time phase plan ``green -> yellow -> red`` repeated with period ``green+yellow+red``.

    ``offset_s`` shifts the plan so that two lights can run complementary
    phases (e.g. the crossing approach uses ``offset_s = green_s + yellow_s``).
    """

    green_s: float = 10.0
    yellow_s: float = 3.0
    red_s: float = 13.0
    offset_s: float = 0.0

    def __post_init__(self):
        for name in ("green_s", "yellow_s", "red_s"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"SignalCycle.{name} must be >= 0")
        if self.period_s <= 0.0:
            raise ValueError("SignalCycle period must be positive")

    @property
    def period_s(self) -> float:
        return float(self.green_s + self.yellow_s + self.red_s)

    def state_at(self, t: float) -> int:
        """Signal state at simulated time ``t`` [s] (end-of-step convention)."""
        phase = math.fmod(float(t) + float(self.offset_s), self.period_s)
        if phase < 0.0:
            phase += self.period_s
        # Snap to the physics grid-independent phase boundary (t is already snapped
        # by the clock; the small epsilon guards float residue at boundaries).
        eps = 1e-9
        if phase + eps < self.green_s:
            return TrafficLightState.GREEN
        if phase + eps < self.green_s + self.yellow_s:
            return TrafficLightState.YELLOW
        return TrafficLightState.RED

    def time_to_red_s(self, t: float) -> float:
        """Seconds until the next red phase starts (0 when already red)."""
        phase = math.fmod(float(t) + float(self.offset_s), self.period_s)
        if phase < 0.0:
            phase += self.period_s
        red_start = self.green_s + self.yellow_s
        return 0.0 if phase >= red_start - 1e-9 else red_start - phase

    def to_spec(self) -> dict:
        return {
            "green_s": self.green_s,
            "yellow_s": self.yellow_s,
            "red_s": self.red_s,
            "offset_s": self.offset_s,
        }

    @classmethod
    def from_spec(cls, spec) -> "SignalCycle | None":
        if spec is None:
            return None
        if isinstance(spec, SignalCycle):
            return spec
        return cls(
            green_s=float(spec.get("green_s", 10.0)),
            yellow_s=float(spec.get("yellow_s", 3.0)),
            red_s=float(spec.get("red_s", 13.0)),
            offset_s=float(spec.get("offset_s", 0.0)),
        )


class TrafficLight(Actor):
    def __init__(
        self,
        pos_x,
        pos_y,
        map_size=128,
        orientation="horizontal",  # 'horizontal' or 'vertical' strip
        signal_state=TrafficLightState.RED,
        width=None,
        length=None,
        units=None,
        controls_heading=None,
        cycle=None,
    ):
        """
        pos_x, pos_y: Center position of the semantic stop-line strip (surface px).
        controls_heading: travel direction the light applies to ("north"/"south"/
            "east"/"west") or None for a decorative light.
        cycle: optional SignalCycle (or its dict spec); the state then follows the clock.
        """
        super().__init__(id="traffic_light", actor_size=1)
        self.units = units if units is not None else DEFAULT_UNIT_CONTRACT
        metric = self.units.metric

        self.x = pos_x
        self.y = pos_y
        self._map_size = map_size
        self.orientation = orientation
        self.controls_heading = (
            None if controls_heading is None else str(controls_heading)
        )
        if (
            self.controls_heading is not None
            and self.controls_heading not in HEADING_YAW
        ):
            raise ValueError(
                f"controls_heading must be one of {sorted(HEADING_YAW)}, got {controls_heading!r}"
            )
        self.cycle = SignalCycle.from_spec(cycle)
        self.initial_signal_state = signal_state
        self.signal_state = (
            self.cycle.state_at(0.0) if self.cycle is not None else signal_state
        )
        self.width = (
            float(width)
            if width is not None
            else max(1.0, metric.distance_meters_to_surface(0.45)) + 1.0
        )
        self.length = (
            float(length)
            if length is not None
            else max(4.0, metric.distance_meters_to_surface(8.5))
        )

        self._set_color()
        self._update_rect()

    @property
    def semantic_class(self) -> SemanticClass:
        if self.signal_state == TrafficLightState.RED:
            return SemanticClass.TRAFFIC_LIGHT_RED
        if self.signal_state == TrafficLightState.YELLOW:
            return SemanticClass.TRAFFIC_LIGHT_YELLOW
        if self.signal_state == TrafficLightState.GREEN:
            return SemanticClass.TRAFFIC_LIGHT_GREEN
        return SemanticClass.NON_DRIVABLE

    def _set_color(self):
        self._color = semantic_color_tuple(self.semantic_class)

    @property
    def kind(self) -> str:
        return "traffic_light"

    def _strip_size(self):
        if self.orientation == "horizontal":
            return self.length, self.width
        return self.width, self.length

    def rect_px(self) -> tuple[float, float, float, float]:
        """(left, top, width, height) of the stop-line strip in surface px."""
        w, h = self._strip_size()
        return self.x - w / 2, self.y - h / 2, w, h

    def _update_rect(self):
        self.rect = self.rect_px()

    def set_signal_state(self, new_state):
        self.signal_state = new_state
        self._set_color()

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.signal_state, "unknown")

    @property
    def controlled_yaw(self) -> float | None:
        """Surface yaw of the controlled travel direction, or None (decorative)."""
        if self.controls_heading is None:
            return None
        return HEADING_YAW[self.controls_heading]

    def stop_line_m(self) -> tuple[float, float, float]:
        """(x_m, y_m, half_span_m) of the stop line in the metric frame."""
        metric = self.units.metric
        x_m, y_m = metric.surface_to_meters((self.x, self.y))
        w, h = self._strip_size()
        half_span_px = 0.5 * (w if self.orientation == "horizontal" else h)
        return (
            float(x_m),
            float(y_m),
            float(metric.distance_surface_to_meters(half_span_px)),
        )

    def reset(self):
        if self.cycle is not None:
            self.set_signal_state(self.cycle.state_at(0.0))
        else:
            self.set_signal_state(self.initial_signal_state)
        self._update_rect()

    def step(self, t, dt, other_actors=None):
        if self.cycle is not None:
            new_state = self.cycle.state_at(t)
            if new_state != self.signal_state:
                self.set_signal_state(new_state)
        return

    def trigger(self, hero_rect):
        return self.id, False

    def isCollided(self, hero, offset=None):
        return self.id, False, 9999
