"""BENCHMARK / instrumentation: pipeline phases are recorded when enabled and free when disabled.

Expected: every documented phase name appears with one sample per env.step();
phase totals are consistent (children <= parent); the registry is disabled by
default and adds no measurable per-step cost.
"""

from __future__ import annotations

import time

from CarlaBEV.src.diagnostics import profiling

from physics_time_helpers import LEAD_BRAKE_OPTIONS, brake_only_action, make_env

STEP_PHASES = (
    "env.step",
    "env.simulate",
    "scene.hero_step",
    "scene.npc_step",
    "scene.draw_actors",
    "map.draw_fov",
    "env.outcome",
    "scene.collision_check",
    "reward.step",
    "env.stats",
    "env.render",
)


def test_profiling_is_disabled_by_default_and_records_nothing():
    assert not profiling.is_enabled()
    profiling.reset()
    env = make_env()
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        for k in range(3):
            env.step(brake_only_action(k))
    finally:
        env.close()
    assert profiling.snapshot() == {}


def test_every_pipeline_phase_is_recorded_once_per_step():
    env = make_env()
    try:
        with profiling.profiling_session() as reg:
            env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
            n = 12
            for k in range(n):
                env.step(brake_only_action(k))
        stats = reg.snapshot()
    finally:
        env.close()
    assert not profiling.is_enabled()
    for name in STEP_PHASES:
        assert name in stats, f"phase {name} not recorded"
        assert stats[name].count == n, (name, stats[name].count)
    assert stats["env.reset"].count == 1
    assert (
        stats["reset.map_reset"].count == 1
        and stats["reset.scene_generation"].count == 1
    )
    # Children never exceed their parent.
    step, simulate, outcome = (
        stats["env.step"].total_ms,
        stats["env.simulate"].total_ms,
        stats["env.outcome"].total_ms,
    )
    assert (
        simulate + outcome + stats["env.render"].total_ms + stats["env.stats"].total_ms
        <= step * 1.05
    )
    # Physics children live under env.simulate; drawing under env.render (Phase 5 split).
    physics_children = sum(stats[n].total_ms for n in ("scene.hero_step", "scene.npc_step"))
    assert physics_children <= simulate * 1.05
    draw_children = sum(stats[n].total_ms for n in ("scene.draw_actors", "map.draw_fov"))
    assert draw_children <= stats["env.render"].total_ms * 1.05
    table = reg.table()
    assert "env.step" in table and "scene.draw_actors" in table


def test_human_render_phase_recorded_only_in_human_mode():
    env = make_env(render_mode="human", fps=1000)
    try:
        with profiling.profiling_session() as reg:
            env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
            env.step(brake_only_action(0))
        assert (
            reg.snapshot()["renderer.human"].count == 2
        )  # reset observation + one step
    finally:
        env.close()
    env = make_env(render_mode="rgb_array")
    try:
        with profiling.profiling_session() as reg:
            env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
            env.step(brake_only_action(0))
        assert "renderer.human" not in reg.snapshot()
    finally:
        env.close()


def test_disabled_profiler_overhead_is_negligible():
    """The phase() fast path is one attribute check; 10k calls must stay far below a step's cost."""
    profiling.disable()
    t0 = time.perf_counter()
    for _ in range(10_000):
        with profiling.phase("noop"):
            pass
    per_call_us = (time.perf_counter() - t0) / 10_000 * 1e6
    assert per_call_us < 5.0, f"{per_call_us:.2f} us per disabled phase"
    assert profiling.snapshot() == {} or "noop" not in profiling.snapshot()


def test_phase_stats_percentiles():
    reg = profiling.ProfileRegistry()
    reg.enable()
    for ns in (1_000_000, 2_000_000, 3_000_000, 4_000_000):
        reg.record("x", ns)
    st = reg.snapshot()["x"]
    assert st.count == 4 and st.mean_ms == 2.5 and st.max_ms == 4.0 and st.min_ms == 1.0
    assert st.percentile_ms(0.5) in (2.0, 3.0) and st.percentile_ms(1.0) == 4.0
    d = st.to_dict()
    assert set(d) >= {"count", "mean_ms", "p50_ms", "p95_ms", "max_ms", "total_ms"}
