"""Explicit simulation clock.

This module is the single source of truth for *simulated time* in CarlaBEV.

Contract (see docs/simulation_clock_and_units.md):

* One ``env.step()`` (a *control step*) advances simulated time by exactly
  ``control_dt_s = action_repeat * physics_dt_s``: the action is held over
  ``action_repeat`` physics sub-steps of ``physics_dt_s`` each (default 1, so a
  control step is one physics step).
* Every consumer (ego integrator, NPC controllers, scenario behaviours, startup
  ramps, reward/comfort derivatives, video metadata) reads ``dt`` from the same
  :class:`TimeConfig`; no module carries its own time literal.
* Time is computed multiplicatively from the step index, never accumulated, so
  ``10 steps of 0.1 s`` is exactly ``1.0 s``.
* End-of-step convention: while step ``k`` (0-based) is being computed, every
  consumer observes ``t = (k + 1) * dt``. A timer of ``T`` seconds fires at the
  first step whose observed time satisfies ``t >= T``.
* Rendering frame rate is derived from the clock (one observation per control
  step: ``render_fps = 1 / control_dt``) and never feeds back into it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


DEFAULT_PHYSICS_DT_S = 0.1
"""Default physics step in seconds (10 Hz)."""

TIME_EPS_S = 1e-9
"""Tolerance used when comparing simulated time against timer thresholds."""


@dataclass(frozen=True)
class TimeConfig:
    """Immutable description of the simulation clock."""

    physics_dt_s: float = DEFAULT_PHYSICS_DT_S
    action_repeat: int = 1
    """Physics sub-steps per control step (agent decision)."""

    def __post_init__(self):
        dt = float(self.physics_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                f"physics_dt_s must be a positive finite number, got {self.physics_dt_s!r}"
            )
        if dt > 1.0:
            raise ValueError(
                f"physics_dt_s must be <= 1.0 s for a meaningful control loop, got {dt}"
            )
        object.__setattr__(self, "physics_dt_s", dt)
        k = int(self.action_repeat)
        if k < 1 or k != self.action_repeat:
            raise ValueError(
                f"action_repeat must be a positive integer, got {self.action_repeat!r}"
            )
        object.__setattr__(self, "action_repeat", k)

    @property
    def physics_rate_hz(self) -> float:
        return 1.0 / self.physics_dt_s

    @property
    def control_dt_s(self) -> float:
        """Simulated seconds advanced by one ``env.step()``."""
        return self.action_repeat * self.physics_dt_s

    @property
    def control_rate_hz(self) -> float:
        return 1.0 / self.control_dt_s

    @property
    def render_fps(self) -> int:
        """Frame rate at which one rendered frame per control step plays in real time."""
        return max(1, int(round(self.control_rate_hz)))

    @classmethod
    def from_env_config(cls, cfg) -> "TimeConfig":
        """Build from any object exposing ``physics_dt`` / ``action_repeat`` (EnvConfig or mapping)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, dict):
            dt = cfg.get("physics_dt", DEFAULT_PHYSICS_DT_S)
            k = cfg.get("action_repeat", 1)
        else:
            dt = getattr(cfg, "physics_dt", DEFAULT_PHYSICS_DT_S)
            k = getattr(cfg, "action_repeat", 1)
        return cls(
            physics_dt_s=DEFAULT_PHYSICS_DT_S if dt is None else float(dt),
            action_repeat=1 if k is None else int(k),
        )

    def steps_for(self, seconds: float) -> int:
        """Number of physics steps needed so that observed time reaches ``seconds``."""
        return int(math.ceil(float(seconds) / self.physics_dt_s - TIME_EPS_S))

    def snap(self, seconds: float) -> float:
        """Snap a duration to the nearest multiple of ``dt`` (removes float drift)."""
        return round(float(seconds) / self.physics_dt_s) * self.physics_dt_s


def timer_elapsed(
    t_now_s: float, t_start_s: float, threshold_s: float, dt_s: float
) -> bool:
    """True if ``threshold_s`` seconds have elapsed since ``t_start_s`` at time ``t_now_s``.

    The elapsed duration is snapped to the ``dt`` grid so that differences of
    grid-aligned times do not suffer from floating-point drift.
    """
    elapsed = round((float(t_now_s) - float(t_start_s)) / float(dt_s)) * float(dt_s)
    return elapsed + TIME_EPS_S >= float(threshold_s)


def time_reached(t_now_s: float, threshold_s: float) -> bool:
    """True if simulated time ``t_now_s`` has reached the absolute threshold."""
    return float(t_now_s) + TIME_EPS_S >= float(threshold_s)


@dataclass(frozen=True)
class EpisodeHorizon:
    """Episode termination horizons defined in simulated seconds.

    The task is defined in physical time; the learner sees steps. Horizons are
    resolved to step counts once per configuration through :class:`TimeConfig`
    so that changing ``physics_dt`` does not redefine the task.
    """

    max_episode_duration_s: float = 500.0
    """Episode is truncated (Gymnasium ``truncated=True``) once this much simulated time elapsed."""
    offroad_terminate_after_s: float = 4.0
    """Continuous off-road time after which the shaping reward terminates the episode."""

    def __post_init__(self):
        for name in ("max_episode_duration_s", "offroad_terminate_after_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be a positive finite number of seconds, got {value!r}"
                )

    def max_episode_steps(self, time_config: TimeConfig) -> int:
        return max(1, time_config.steps_for(self.max_episode_duration_s))

    def offroad_terminate_after_steps(self, time_config: TimeConfig) -> int:
        return max(1, time_config.steps_for(self.offroad_terminate_after_s))

    def describe(self, time_config: TimeConfig) -> dict:
        return {
            "physics_dt_s": time_config.physics_dt_s,
            "action_repeat": time_config.action_repeat,
            "control_dt_s": time_config.control_dt_s,
            "max_episode_duration_s": self.max_episode_duration_s,
            "max_episode_steps": self.max_episode_steps(time_config),
            "offroad_terminate_after_s": self.offroad_terminate_after_s,
            "offroad_terminate_after_steps": self.offroad_terminate_after_steps(
                time_config
            ),
        }


class SimClock:
    """Step-indexed simulation clock.

    ``t_s`` is always ``step_index * dt_s``. :meth:`tick` is called once at the
    start of every ``env.step()``; the returned value is the end-of-step time
    that all consumers observe during that step.
    """

    def __init__(self, time_config: TimeConfig | None = None):
        self.time_config = time_config if time_config is not None else TimeConfig()
        self.step_index = 0
        self.control_index = 0

    @property
    def dt_s(self) -> float:
        return self.time_config.physics_dt_s

    @property
    def control_dt_s(self) -> float:
        return self.time_config.control_dt_s

    @property
    def t_s(self) -> float:
        return self.step_index * self.dt_s

    def reset(self) -> None:
        self.step_index = 0
        self.control_index = 0

    def tick(self) -> float:
        """Advance one physics step; returns the end-of-step time."""
        self.step_index += 1
        return self.t_s

    def begin_control_step(self) -> int:
        """Count one agent decision (``env.step()``); returns the new control index."""
        self.control_index += 1
        return self.control_index

    def elapsed_since(self, t_start_s: float) -> float:
        return self.time_config.snap(self.t_s - float(t_start_s))

    def __repr__(self) -> str:
        return (
            f"SimClock(step_index={self.step_index}, control_index={self.control_index}, "
            f"t_s={self.t_s:.3f}, dt_s={self.dt_s})"
        )
