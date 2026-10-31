"""Phase-level wall-clock instrumentation for the simulation pipeline.

The simulator marks its pipeline phases with :func:`phase` (context manager) or
:func:`profiled` (decorator). When the registry is disabled — the default — a
phase costs one attribute check; when enabled, each phase records its duration
with ``time.perf_counter_ns`` into per-name statistics.

Phase names are hierarchical, dot-separated, and stable: they are the contract
the benchmark and future optimisation work compare against.

    env.step                      one CarlaBEV.step() call (everything below)
    ├─ env.simulate               action decode + map.step
    │  ├─ scene.hero_step         ego physics + camera scroll
    │  ├─ scene.npc_step          all background actors (behaviours, traffic policy, controllers)
    │  ├─ scene.traffic_lifecycle recycling of finished background vehicles (when any finished)
    │  ├─ scene.draw_actors       dirty-rect restore + bodies/targets/lights onto the world id layer
    │  └─ map.draw_fov            crop / rotate / compose the observation + semantic tile lookup
    ├─ env.outcome                collision check + reward
    │  ├─ scene.collision_check
    │  └─ reward.step
    ├─ env.stats                  per-step statistics aggregation
    └─ env.render                 observation extraction (+ renderer.human when render_mode="human")
       └─ renderer.human          window blit + wall-clock pacing
    env.reset                     one CarlaBEV.reset() call
    ├─ reset.scene_generation     SceneGenerator.build_scene (cache miss) or cache load
    └─ reset.map_reset            actor spawn, validation, first FOV draw

Usage::

    from CarlaBEV.src.diagnostics import profiling
    profiling.enable()
    ... run episodes ...
    table = profiling.snapshot()        # {name: PhaseStats}
    profiling.disable()
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import math
import statistics
import time


@dataclass
class PhaseStats:
    """Duration statistics for one phase (milliseconds)."""

    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = math.inf
    max_ms: float = 0.0
    samples_ms: list[float] = field(default_factory=list)

    def add(self, duration_ms: float, *, keep_samples: bool) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        if keep_samples:
            self.samples_ms.append(duration_ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def percentile_ms(self, q: float) -> float:
        if not self.samples_ms:
            return float("nan")
        ordered = sorted(self.samples_ms)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]

    @property
    def std_ms(self) -> float:
        return statistics.pstdev(self.samples_ms) if len(self.samples_ms) > 1 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms if self.count else 0.0,
            "p50_ms": self.percentile_ms(0.50),
            "p95_ms": self.percentile_ms(0.95),
            "p99_ms": self.percentile_ms(0.99),
            "max_ms": self.max_ms,
        }


class ProfileRegistry:
    """Collects phase durations. Disabled by default (near-zero overhead)."""

    def __init__(self, *, keep_samples: bool = True, max_samples: int = 20000):
        self.enabled = False
        self.keep_samples = keep_samples
        self.max_samples = max_samples
        self._stats: dict[str, PhaseStats] = {}

    # --- control ----------------------------------------------------------
    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        self._stats = {}

    # --- recording --------------------------------------------------------
    def record(self, name: str, duration_ns: int) -> None:
        stats = self._stats.get(name)
        if stats is None:
            stats = self._stats[name] = PhaseStats(name)
        keep = self.keep_samples and len(stats.samples_ms) < self.max_samples
        stats.add(duration_ns / 1e6, keep_samples=keep)

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(name, time.perf_counter_ns() - start)

    # --- reporting --------------------------------------------------------
    def snapshot(self) -> dict[str, PhaseStats]:
        return {name: stats for name, stats in sorted(self._stats.items())}

    def to_dict(self) -> dict[str, dict]:
        return {name: stats.to_dict() for name, stats in self.snapshot().items()}

    def table(self) -> str:
        rows = [
            f"{'phase':<28}{'count':>8}{'mean ms':>10}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}{'total ms':>12}"
        ]
        for name, st in self.snapshot().items():
            rows.append(
                f"{name:<28}{st.count:>8d}{st.mean_ms:>10.3f}{st.percentile_ms(0.5):>10.3f}"
                f"{st.percentile_ms(0.95):>10.3f}{st.max_ms:>10.3f}{st.total_ms:>12.1f}"
            )
        return "\n".join(rows)


# Module-level default registry used by the simulator's instrumentation points.
registry = ProfileRegistry()


def enable() -> None:
    registry.enable()


def disable() -> None:
    registry.disable()


def reset() -> None:
    registry.reset()


def is_enabled() -> bool:
    return registry.enabled


def snapshot() -> dict[str, PhaseStats]:
    return registry.snapshot()


def phase(name: str):
    """Context manager marking a pipeline phase on the default registry."""
    return registry.phase(name)


def profiled(name: str):
    """Decorator marking a function as a pipeline phase on the default registry."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not registry.enabled:
                return fn(*args, **kwargs)
            start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                registry.record(name, time.perf_counter_ns() - start)

        return wrapper

    return decorator


@contextmanager
def profiling_session(*, reset_first: bool = True):
    """Enable profiling for the duration of a block and return the registry."""
    if reset_first:
        registry.reset()
    was_enabled = registry.enabled
    registry.enable()
    try:
        yield registry
    finally:
        registry.enabled = was_enabled
