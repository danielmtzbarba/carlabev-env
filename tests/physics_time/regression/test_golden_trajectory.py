"""REGRESSION / golden: bit-level snapshots of seeded scenarios at dt = 0.1 s.

Expected: behaviour-preserving refactors leave every snapshot unchanged.
``*_ongrid`` cases use 1.0 s timers and are expected to shift by exactly one
trigger step when C-03 is fixed; ``*_offgrid`` (1.25 s timers) must never change.

Regenerate deliberately after reviewing the diff:
    CARLABEV_REGEN_GOLDEN=1 uv run pytest tests/physics_time/regression

Portability: snapshots are compared within ``atol = 1e-7`` with headings taken
modulo 2*pi. Floating-point results differ at the 1e-13..1e-14 level between
platforms (BLAS/SIMD/FMA paths, e.g. Linux x86 vs macOS Apple Silicon), so the
bit-exact digest is only asserted when ``CARLABEV_STRICT_GOLDEN=1`` (same-machine
determinism is covered by tests/physics_time/rendering instead).
"""

from __future__ import annotations

import json
import os

import pytest

from physics_time_helpers import (
    GOLDEN_DIR,
    JAYWALK_OPTIONS,
    LEAD_BRAKE_OPTIONS,
    RANDOM_NAV_OPTIONS,
    assert_trajectories_close,
    make_env,
    rollout,
    scripted_action,
    trajectory_digest,
)


REGEN = os.environ.get("CARLABEV_REGEN_GOLDEN", "") not in ("", "0", "false")
STRICT_DIGEST = os.environ.get("CARLABEV_STRICT_GOLDEN", "") not in ("", "0", "false")

LEGACY = {"geometry_profile": "legacy"}  # the historical 1.25 m squares (realistic is the default since Phase 6)

CASES = {
    "lead_brake_l1_ongrid": {"options": LEAD_BRAKE_OPTIONS, "num_steps": 80, "env": LEGACY},
    "jaywalk_l3_ongrid": {"options": JAYWALK_OPTIONS, "num_steps": 80, "env": LEGACY},
    "lead_brake_l1_offgrid": {
        "options": {
            **LEAD_BRAKE_OPTIONS,
            "ego_speed": 8.0,
            "lead_gap": 20.0,
            "brake_delay": 1.25,
        },
        "num_steps": 100,
        "env": LEGACY,
    },
    "jaywalk_l3_offgrid": {
        "options": {**JAYWALK_OPTIONS, "cross_delay": 1.25, "yield_duration": 1.25},
        "num_steps": 100,
        "env": LEGACY,
    },
    "random_nav_easy": {"options": RANDOM_NAV_OPTIONS, "num_steps": 60, "env": LEGACY},
    # Realistic geometry profile (oriented 4.5 x 1.8 m cars, 0.3 m pedestrian circles).
    "lead_brake_l1_offgrid_realistic": {
        "options": {
            **LEAD_BRAKE_OPTIONS,
            "ego_speed": 8.0,
            "lead_gap": 20.0,
            "brake_delay": 1.25,
        },
        "num_steps": 100,
        "env": {"geometry_profile": "realistic"},
    },
    "jaywalk_l3_offgrid_realistic": {
        "options": {**JAYWALK_OPTIONS, "cross_delay": 1.25, "yield_duration": 1.25},
        "num_steps": 100,
        "env": {"geometry_profile": "realistic"},
    },
    "random_nav_easy_realistic": {
        "options": RANDOM_NAV_OPTIONS,
        "num_steps": 60,
        "env": {"geometry_profile": "realistic"},
    },
}


def _generate(case: dict) -> dict:
    env = make_env(
        action_mode=case.get("action_mode", "continuous"), **case.get("env", {})
    )
    try:
        traj = rollout(
            env,
            options=case["options"],
            num_steps=case["num_steps"],
            policy=scripted_action,
        )
    finally:
        env.close()
    traj["digest"] = trajectory_digest(traj)
    return traj


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_trajectory(name):
    path = GOLDEN_DIR / f"{name}.json"
    actual = _generate(CASES[name])

    if REGEN or not path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(actual, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        if REGEN:
            pytest.skip(f"regenerated golden snapshot {path.name}")
        pytest.fail(
            f"golden snapshot {path.name} did not exist and was created; re-run to validate"
        )

    expected = json.loads(path.read_text(encoding="utf-8"))
    assert len(actual["steps"]) > 10, "rollout too short to be a meaningful snapshot"
    assert_trajectories_close(actual, expected, atol=1e-7)
    if STRICT_DIGEST:
        assert actual["digest"] == expected["digest"]
