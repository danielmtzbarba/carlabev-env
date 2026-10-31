"""REPLAY suite: golden-snapshot fidelity, same-machine determinism, cross-size agreement."""

from __future__ import annotations

import json
import time

from CarlaBEV.src.diagnostics.rollouts import (
    GOLDEN_DIR,
    RANDOM_NAV_OPTIONS,
    JAYWALK_OPTIONS,
    rollout,
    scripted_action,
    trajectory_differences,
    trajectory_digest,
)
from CarlaBEV.tools.benchmark.schema import Metric, Section
from CarlaBEV.tools.benchmark.suites.common import (
    HALF_GAS,
    OFFGRID_LEAD_BRAKE,
    hypot,
    make_env,
)

LEGACY = {"geometry_profile": "legacy"}  # snapshots of the historical footprints (see tests/.../regression)
GOLDEN_CASES = {
    "lead_brake_l1_ongrid": {
        "options": {
            **OFFGRID_LEAD_BRAKE,
            "ego_speed": 10.0,
            "lead_gap": 12.0,
            "brake_delay": 1.0,
        },
        "num_steps": 80,
        "env": LEGACY,
    },
    "jaywalk_l3_ongrid": {"options": JAYWALK_OPTIONS, "num_steps": 80, "env": LEGACY},
    "lead_brake_l1_offgrid": {"options": OFFGRID_LEAD_BRAKE, "num_steps": 100, "env": LEGACY},
    "jaywalk_l3_offgrid": {
        "options": {**JAYWALK_OPTIONS, "cross_delay": 1.25, "yield_duration": 1.25},
        "num_steps": 100,
        "env": LEGACY,
    },
    "random_nav_easy": {"options": RANDOM_NAV_OPTIONS, "num_steps": 60, "env": LEGACY},
    "lead_brake_l1_offgrid_realistic": {"options": OFFGRID_LEAD_BRAKE, "num_steps": 100},
    "random_nav_easy_realistic": {"options": RANDOM_NAV_OPTIONS, "num_steps": 60},
}


def _metric_trace(size, options, num_steps):
    env = make_env(size=size)
    try:
        env.reset(seed=3, options=dict(options))
        npcs = (
            env.map.actor_manager.actors["vehicle"]
            + env.map.actor_manager.actors["pedestrian"]
        )
        trace = []
        for _ in range(num_steps):
            _, _, term, trunc, _ = env.step(HALF_GAS)
            trace.append(
                (
                    (env.map.hero.x_m, env.map.hero.y_m),
                    [(a._controller.x_m, a._controller.y_m) for a in npcs],
                )
            )
            if term or trunc:
                break
        return env.len_ego_route, trace
    finally:
        env.close()


def run(profile: dict) -> Section:
    started = time.perf_counter()
    sec = Section(
        "replay", "Replay: golden fidelity, determinism, cross-size agreement"
    )

    # 1. Golden snapshots
    rows = []
    worst = 0.0
    for name, case in GOLDEN_CASES.items():
        path = GOLDEN_DIR / f"{name}.json"
        if not path.exists():
            rows.append({"snapshot": name, "status": "missing"})
            continue
        expected = json.loads(path.read_text(encoding="utf-8"))
        env = make_env(**case.get("env", {}))
        try:
            actual = rollout(
                env,
                options=case["options"],
                num_steps=case["num_steps"],
                policy=scripted_action,
            )
        finally:
            env.close()
        diff = trajectory_differences(actual, expected)
        max_diff = max(diff["max_abs_diff"].values())
        worst = max(worst, max_diff)
        rows.append(
            {
                "snapshot": name,
                "length_actual": diff["length_actual"],
                "length_expected": diff["length_expected"],
                "max_abs_diff": max_diff,
                "first_diff_step": diff["first_diff_step"],
                "first_diff_key": diff["first_diff_key"],
                "state_mismatches": diff["state_mismatches"],
                "digest_equal": trajectory_digest(actual) == expected.get("digest"),
                "platform_rounding_only": max_diff <= 1e-9,
            }
        )
    sec.tables["golden_snapshots"] = rows
    sec.add(
        Metric.checked(
            "golden_max_abs_diff",
            worst,
            "",
            tolerance=1e-7,
            detail="max |Δ| over all keys/snapshots (headings modulo 2π)",
        )
    )
    sec.add(
        Metric.boolean(
            "golden_lengths_match",
            all(r.get("length_actual") == r.get("length_expected") for r in rows),
        )
    )
    sec.add(
        Metric.info(
            "golden_digests_equal",
            sum(1 for r in rows if r.get("digest_equal")),
            "snapshots",
            detail=f"of {len(rows)}; bit-exact digests are platform specific (P-01)",
        )
    )

    # 2. Same-machine determinism
    env = make_env()
    try:
        a = rollout(env, options=OFFGRID_LEAD_BRAKE, num_steps=40)
        b = rollout(env, options=OFFGRID_LEAD_BRAKE, num_steps=40)
    finally:
        env.close()
    sec.add(
        Metric.boolean(
            "same_process_repeat_bitwise_identical",
            trajectory_digest(a) == trajectory_digest(b),
        )
    )

    # 3. Cross-size agreement
    rows = []
    ref_len, ref_trace = _metric_trace(128, OFFGRID_LEAD_BRAKE, profile["replay_steps"])
    for size in profile["replay_sizes"]:
        if size == 128:
            continue
        length, trace = _metric_trace(size, OFFGRID_LEAD_BRAKE, profile["replay_steps"])
        n = min(len(trace), len(ref_trace))
        ego_err = max(hypot(trace[i][0], ref_trace[i][0]) for i in range(n))
        npc_err = max(
            hypot(p, q) for i in range(n) for p, q in zip(trace[i][1], ref_trace[i][1])
        )
        rows.append(
            {
                "size": size,
                "route_len_diff_m": abs(length - ref_len),
                "max_ego_pos_err_m": ego_err,
                "max_npc_pos_err_m": npc_err,
                "steps_compared": n,
            }
        )
    sec.tables["cross_size"] = rows
    if rows:
        sec.add(
            Metric.checked(
                "cross_size_max_ego_pos_err",
                max(r["max_ego_pos_err_m"] for r in rows),
                "m",
                tolerance=1.0,
                detail="vs the 128-px run at equal sim time",
            )
        )
        sec.add(
            Metric.checked(
                "cross_size_max_npc_pos_err",
                max(r["max_npc_pos_err_m"] for r in rows),
                "m",
                tolerance=1.0,
            )
        )
        sec.add(
            Metric.checked(
                "cross_size_route_len_diff",
                max(r["route_len_diff_m"] for r in rows),
                "m",
                tolerance=0.5,
            )
        )

    sec.wall_time_s = time.perf_counter() - started
    return sec
