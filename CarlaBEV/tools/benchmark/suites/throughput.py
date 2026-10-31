"""THROUGHPUT suite: steps/s, ms/step, phase breakdown (instrumentation), reset latency."""

from __future__ import annotations

import os
import time

from CarlaBEV.src.diagnostics import profiling
from CarlaBEV.tools.benchmark.schema import Metric, Section
from CarlaBEV.tools.benchmark.suites.common import HALF_GAS, make_env
from CarlaBEV.src.diagnostics.rollouts import RANDOM_NAV_OPTIONS


def _measure(size, render_mode, steps, options, *, physics_dt, action_repeat=1):
    """Time env.step() only; resets (warm-up and in-loop episode restarts) are measured separately."""
    env = make_env(
        size=size,
        physics_dt=physics_dt,
        action_repeat=action_repeat,
        render_mode=render_mode,
        fps=10_000 if render_mode == "human" else None,
    )
    try:
        t0 = time.perf_counter()
        env.reset(seed=3, options=dict(options))
        reset_s = time.perf_counter() - t0
        for _ in range(5):  # warm-up
            env.step(HALF_GAS)
        step_wall = 0.0
        resets_in_loop = 0
        done = 0
        with profiling.profiling_session() as reg:
            for _ in range(steps):
                t0 = time.perf_counter()
                _, _, term, trunc, _ = env.step(HALF_GAS)
                step_wall += time.perf_counter() - t0
                done += 1
                if term or trunc:
                    env.reset(seed=3, options=dict(options))
                    resets_in_loop += 1
        phases = reg.to_dict()
        return {
            "reset_s": reset_s,
            "steps": done,
            "resets_in_loop": resets_in_loop,
            "wall_s": step_wall,
            "ms_per_step": 1000.0 * step_wall / done,
            "steps_per_s": done / step_wall,
            "phases": phases,
        }
    finally:
        env.close()


def run(profile: dict) -> Section:
    started = time.perf_counter()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sec = Section(
        "throughput", "Throughput: env.step() cost, phase breakdown, reset latency"
    )
    dt = profile["physics_dt"]
    steps = profile["throughput_steps"]
    rows, breakdown = [], {}
    for size in profile["throughput_sizes"]:
        for mode in profile["throughput_modes"]:
            for scene_name, options in (
                (
                    "lead_brake",
                    {
                        "scene": "lead_brake",
                        "level": 1,
                        "ego_speed": 8.0,
                        "lead_speed": 10.0,
                        "lead_gap": 20.0,
                        "brake_delay": 1.25,
                        "brake_strength": 4.0,
                    },
                ),
                ("random_nav", RANDOM_NAV_OPTIONS),
            ):
                if scene_name == "random_nav" and (size != 128 or mode != "rgb_array"):
                    continue
                m = _measure(size, mode, steps, options, physics_dt=dt)
                step_stats = m["phases"].get("env.step", {})
                rows.append(
                    {
                        "size": size,
                        "render_mode": mode,
                        "scene": scene_name,
                        "steps": m["steps"],
                        "ms_per_step": m["ms_per_step"],
                        "p50_ms": step_stats.get("p50_ms"),
                        "p95_ms": step_stats.get("p95_ms"),
                        "p99_ms": step_stats.get("p99_ms"),
                        "max_ms": step_stats.get("max_ms"),
                        "steps_per_s": m["steps_per_s"],
                        "real_time_x": m["steps_per_s"] * dt,
                        "reset_s": m["reset_s"],
                        "resets_in_loop": m["resets_in_loop"],
                    }
                )
                label = f"{scene_name}@{size}px/{mode}"
                breakdown[label] = [
                    {"phase": name, **{k: v for k, v in st.items() if k != "name"}}
                    for name, st in m["phases"].items()
                    if name.startswith(
                        ("env.", "scene.", "map.", "reward.", "renderer.")
                    )
                ]
    sec.tables["throughput"] = rows
    sec.tables["phase_breakdown"] = breakdown

    ref = next(
        r
        for r in rows
        if r["size"] == 128
        and r["render_mode"] == "rgb_array"
        and r["scene"] == "lead_brake"
    )
    sec.add(Metric.info("ms_per_step_128_rgb_lead_brake", ref["ms_per_step"], "ms"))
    sec.add(
        Metric.info("steps_per_s_128_rgb_lead_brake", ref["steps_per_s"], "steps/s")
    )
    sec.add(
        Metric.info(
            "p99_ms_per_step_128_rgb_lead_brake",
            ref["p99_ms"],
            "ms",
            detail="99th percentile of env.step wall time",
        )
    )
    sec.add(
        Metric.info(
            "real_time_factor_128_rgb",
            ref["real_time_x"],
            "x",
            detail="simulated seconds per wall second at this dt",
        )
    )
    sec.add(Metric.info("reset_s_128_lead_brake", ref["reset_s"], "s (wall)"))
    rn = next((r for r in rows if r["scene"] == "random_nav"), None)
    if rn:
        sec.add(Metric.info("ms_per_step_128_rgb_random_nav", rn["ms_per_step"], "ms"))
        sec.add(
            Metric.info(
                "reset_s_128_random_nav",
                rn["reset_s"],
                "s (wall)",
                detail="includes route planning and traffic generation (cache disabled)",
            )
        )
    human = next(
        (
            r
            for r in rows
            if r["size"] == 128
            and r["render_mode"] == "human"
            and r["scene"] == "lead_brake"
        ),
        None,
    )
    if human:
        sec.add(
            Metric.info(
                "render_overhead_human_vs_rgb",
                human["ms_per_step"] - ref["ms_per_step"],
                "ms",
                detail="window blit + display update (pacing disabled)",
            )
        )
    phases = {
        row["phase"]: row for row in breakdown.get("lead_brake@128px/rgb_array", [])
    }
    for phase in (
        "env.simulate",
        "scene.redraw_world",
        "scene.draw_actors",
        "map.draw_fov",
        "scene.npc_step",
        "scene.hero_step",
        "env.outcome",
        "env.render",
    ):
        if phase in phases:
            sec.add(
                Metric.info(f"phase_mean_ms.{phase}", phases[phase]["mean_ms"], "ms")
            )

    # Multi-rate presets: decisions/s and physics steps/s per (physics_dt, action_repeat)
    # at a fixed 0.1 s control step (lead_brake, 128 px, rgb_array).
    lead_brake = {
        "scene": "lead_brake",
        "level": 1,
        "ego_speed": 8.0,
        "lead_speed": 10.0,
        "lead_gap": 20.0,
        "brake_delay": 1.25,
        "brake_strength": 4.0,
    }
    multi_rows = []
    for phys_dt, k in profile.get(
        "multi_rate_presets", ((0.1, 1), (0.05, 2), (0.025, 4))
    ):
        m = _measure(
            128,
            "rgb_array",
            max(50, steps // 2),
            lead_brake,
            physics_dt=phys_dt,
            action_repeat=k,
        )
        multi_rows.append(
            {
                "physics_dt": phys_dt,
                "action_repeat": k,
                "control_dt": phys_dt * k,
                "ms_per_decision": m["ms_per_step"],
                "decisions_per_s": m["steps_per_s"],
                "physics_steps_per_s": m["steps_per_s"] * k,
                "real_time_x": m["steps_per_s"] * phys_dt * k,
                "simulate_mean_ms": m["phases"].get("env.simulate", {}).get("mean_ms"),
                "render_mean_ms": m["phases"].get("env.render", {}).get("mean_ms"),
            }
        )
    sec.tables["multi_rate"] = multi_rows
    for row in multi_rows:
        tag = f"dt{row['physics_dt']}_k{row['action_repeat']}"
        sec.add(
            Metric.info(f"decisions_per_s_{tag}", row["decisions_per_s"], "decisions/s")
        )
        sec.add(
            Metric.info(
                f"physics_steps_per_s_{tag}", row["physics_steps_per_s"], "steps/s"
            )
        )

    sec.wall_time_s = time.perf_counter() - started
    return sec
