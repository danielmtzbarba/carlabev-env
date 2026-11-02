"""``carlabev validate replay`` — prove that a seeded episode replays identically.

Runs the same seeded reset and the same action sequence ``N`` times in
**independent interpreters** (each with a different ``PYTHONHASHSEED``, so any
randomness derived from ``hash(...)`` becomes visible) and asserts that the
simulator state is identical at every step ``k``:

* at reset: scene key / cache hit, ego spawn pose, ego route, checkpoint and
  goal positions, every NPC (identity, route, resolved spawn pose, speeds,
  behaviour, rule flags) and every traffic signal;
* at each step: ego pose/speed, every NPC pose/speed/target/behaviour state/
  recycle count, signal states, reward, termination cause and (optionally) the
  observation hash.

Examples::

    carlabev validate replay                                   # 3 runs, seed 0, medium random traffic, 200 steps
    carlabev validate replay --seed 11 --runs 4 --steps 300
    carlabev validate replay --cache-mode miss-then-hit        # run 0 populates a fresh library, runs 1.. replay hits
    carlabev validate replay --cache-mode hit-only --scene-library runs/lib.db   # strict: read-only + require-hit
    carlabev validate replay --reset-mode scenario --scenario-id lead_brake --policy scripted
    carlabev validate replay --policy random --action-seed 7 --report runs/replay.json
    carlabev validate replay --isolation in-process            # same interpreter (cannot detect hash seeding)

Run ``k`` is compared against run ``0``; the first divergence (step, key,
values, delta) is reported. Exit code 0 = every run identical, 1 = divergence,
2 = a run failed. Same-machine runs are expected to be bit-identical
(``--atol 0``); across machines use ``--atol 1e-7`` (see
``docs/numerical_determinism.md``). Headings are compared modulo 2π.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_MODULE = "CarlaBEV.tools.validate_replay"
CACHE_MODES = ("off", "miss-then-hit", "hit-only")
POLICIES = ("scripted", "forward", "brake", "idle", "random", "explicit")
RESET_MODES = ("random", "scenario", "scenario-preset", "authored")


# ---------------------------------------------------------------------------
# Replay specification (fully explicit -> serializable -> identical in every run)
# ---------------------------------------------------------------------------
@dataclass
class ReplaySpec:
    env: dict[str, Any]
    seed: int
    options: dict[str, Any]
    actions: list[Any]
    include_obs: bool = True
    label: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "env": dict(self.env),
            "seed": int(self.seed),
            "options": _jsonable(self.options),
            "actions": _jsonable(self.actions),
            "include_obs": bool(self.include_obs),
            "label": dict(self.label),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplaySpec":
        return cls(
            env=dict(data["env"]),
            seed=int(data["seed"]),
            options=dict(data["options"]),
            actions=list(data["actions"]),
            include_obs=bool(data.get("include_obs", True)),
            label=dict(data.get("label", {})),
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Action sequences
# ---------------------------------------------------------------------------
def build_action_sequence(
    policy: str,
    *,
    steps: int,
    action_mode: str,
    action_seed: int,
    actions_file: str | None = None,
    action_profile_id: str | None = None,
) -> list[Any]:
    """Materialize the full action list once, so every run applies the exact same inputs."""
    from CarlaBEV.config import EnvConfig
    from CarlaBEV.envs.spaces import get_action_space
    from CarlaBEV.src.diagnostics.rollouts import scripted_action

    if policy == "explicit":
        if not actions_file:
            raise ValueError("--policy explicit requires --actions FILE.json")
        with open(actions_file, encoding="utf-8") as handle:
            actions = json.load(handle)
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"{actions_file}: expected a non-empty JSON list of actions")
        return actions[:steps] if steps else actions

    continuous = action_mode == "continuous"
    if policy == "random":
        cfg_kwargs = {"action_mode": action_mode, "render_mode": "rgb_array"}
        if action_profile_id:
            cfg_kwargs["action_profile_id"] = action_profile_id
        space, _ = get_action_space(EnvConfig(**cfg_kwargs))
        rng = np.random.default_rng(int(action_seed))
        if continuous:
            low, high = np.asarray(space.low, dtype=float), np.asarray(space.high, dtype=float)
            return [
                [float(v) for v in rng.uniform(low, high).astype(np.float32)] for _ in range(steps)
            ]
        return [int(v) for v in rng.integers(0, int(space.n), size=steps)]

    if continuous:
        table = {
            "forward": lambda k: [1.0, 0.0, 0.0],
            "brake": lambda k: [0.0, 0.0, 1.0],
            "idle": lambda k: [0.0, 0.0, 0.0],
            "scripted": lambda k: [float(v) for v in scripted_action(k)],
        }
    else:
        # discrete9_v1 layout: 0 idle, 1 gas, 2 brake, 3 gas+left, 4 gas+right, ...
        # ``scripted`` cycles through every discrete action id so all channels are exercised.
        cfg_kwargs = {"action_mode": action_mode, "render_mode": "rgb_array"}
        if action_profile_id:
            cfg_kwargs["action_profile_id"] = action_profile_id
        space, _ = get_action_space(EnvConfig(**cfg_kwargs))
        n_actions = int(space.n)
        table = {
            "forward": lambda k: 1,
            "brake": lambda k: 2,
            "idle": lambda k: 0,
            "scripted": lambda k: (k // 10) % n_actions,
        }
    return [table[policy](k) for k in range(steps)]


# ---------------------------------------------------------------------------
# State fingerprints
# ---------------------------------------------------------------------------
def _digest(values) -> str:
    return hashlib.sha256(json.dumps(_jsonable(values)).encode("utf-8")).hexdigest()[:16]


def npc_identity(actor, kind: str, ordinal: int) -> str:
    """Stable NPC id: the actor's ``npc_id`` if it has one, else the scene-scoped ordinal."""
    npc_id = getattr(actor, "npc_id", None)
    return str(npc_id) if npc_id else f"{kind}:{ordinal:03d}"


def _iter_npcs(env):
    ordinal = 0
    for kind in ("vehicle", "pedestrian"):
        for actor in env.map.actor_manager.actors.get(kind, []):
            yield npc_identity(actor, kind, ordinal), kind, actor
            ordinal += 1


def reset_fingerprint(env, info: dict) -> dict[str, Any]:
    hero = env.map.hero
    scenario = info.get("scenario", {}) or {}
    cache = info.get("scene_cache", {}) or {}
    descriptor = info.get("scene_descriptor", {}) or {}
    route_x, route_y = env.map.route
    npcs = []
    for npc_id, kind, actor in _iter_npcs(env):
        ctrl = actor._controller
        npcs.append(
            {
                "npc_id": npc_id,
                "kind": kind,
                "traffic_role": str(getattr(actor, "traffic_role", "unassigned")),
                "lane": getattr(actor, "lane", None),
                "route_digest": _digest([list(actor.rx), list(actor.ry)]),
                "route_len": int(len(actor.rx)),
                "x_m": float(ctrl.x_m),
                "y_m": float(ctrl.y_m),
                "yaw": float(ctrl.yaw),
                "v_mps": float(ctrl.v_mps),
                "initial_speed_mps": float(actor.initial_speed_mps or 0.0),
                "cruise_speed_mps": float(actor.cruise_speed_mps),
                "startup_ramp_duration_s": float(actor.startup_ramp_duration_s),
                "behavior": type(actor.behavior).__name__ if actor.behavior else None,
                "obeys_signals": bool(getattr(actor, "obeys_signals", True)),
                "yields_at_junctions": bool(getattr(actor, "yields_at_junctions", True)),
                "recyclable": bool(getattr(actor, "recyclable", False)),
                "length_m": float(actor.length_m),
                "width_m": float(actor.width_m),
            }
        )
    signals = [
        {"x": float(light.x), "y": float(light.y), "state": str(light.state_name)}
        for light in env.map.actor_manager.actors.get("traffic_light", [])
    ]
    targets = [
        [float(t.position[0]), float(t.position[1])]
        for t in env.map.actor_manager.actors.get("target", [])
    ]
    return {
        "scene_key": cache.get("scene_key"),
        "seeds": {
            key: scenario.get(f"scenario_param_{key}")
            for key in ("scene_seed", "route_seed", "traffic_seed", "scenario_seed")
        },
        "descriptor": {
            key: descriptor.get(key)
            for key in (
                "spawned_num_vehicles",
                "actor_roles",
                "main_actor_role",
                "route_profile",
                "variation_seed",
                "variation_profile",
                "variation_applied",
            )
        },
        "spawn_valid": bool((info.get("spawn_validation") or {}).get("valid", True)),
        "ego": {
            "x_m": float(hero.x_m),
            "y_m": float(hero.y_m),
            "yaw": float(hero.yaw),
            "v_mps": float(hero.v_mps),
            "route_digest": _digest([[int(v) for v in route_x], [int(v) for v in route_y]]),
            "route_len": int(len(route_x)),
            "route_length_m": float(env.len_ego_route),
        },
        "targets": targets,
        "npcs": npcs,
        "signals": signals,
    }


def step_fingerprint(env, k: int, reward, terminated, truncated, obs, *, include_obs: bool) -> dict[str, Any]:
    hero = env.map.hero
    current = getattr(env, "current_info", {}) or {}
    cause = (current.get("reward") or {}).get("cause")
    lifecycle = env.map.traffic_lifecycle
    record = {
        "step": int(k),
        "sim_t": float(env.map.clock.t_s),
        "hero": {
            "x_m": float(hero.x_m),
            "y_m": float(hero.y_m),
            "yaw": float(hero.yaw),
            "v_mps": float(hero.v_mps),
        },
        "npcs": {
            npc_id: {
                "x_m": float(actor._controller.x_m),
                "y_m": float(actor._controller.y_m),
                "yaw": float(actor._controller.yaw),
                "v_mps": float(actor._controller.v_mps),
                "target_speed_mps": float(actor.target_speed_mps),
                "behavior_state": str(actor.behavior_state),
                "recycle_count": int(getattr(actor, "recycle_count", 0)),
            }
            for npc_id, _, actor in _iter_npcs(env)
        },
        "signals": [str(light.state_name) for light in env.map.actor_manager.actors.get("traffic_light", [])],
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "cause": cause,
        "recycled": 0 if lifecycle is None else int(lifecycle.stats.recycled),
    }
    if include_obs:
        record["obs_sha256"] = hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()[:16]
    return record


# ---------------------------------------------------------------------------
# Worker: one run -> one trace
# ---------------------------------------------------------------------------
def run_replay(spec: ReplaySpec) -> dict[str, Any]:
    """Execute one seeded replay in this interpreter and return its trace."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from CarlaBEV.config import EnvConfig
    from CarlaBEV.envs.carlabev import CarlaBEV

    started = time.perf_counter()
    env = CarlaBEV(EnvConfig(**spec.env))
    try:
        obs, info = env.reset(seed=int(spec.seed), options=dict(spec.options))
        cache = info.get("scene_cache", {}) or {}
        trace = {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "pid": os.getpid(),
            # Cache bookkeeping is metadata (expected to differ between a miss
            # and its hits); the compared state lives in ``reset`` and ``steps``.
            "cache": {
                "enabled": bool(cache.get("enabled", False)),
                "hit": bool(cache.get("hit", False)),
                "scene_key": cache.get("scene_key"),
            },
            "reset": reset_fingerprint(env, info),
            "steps": [],
        }
        for k, action in enumerate(spec.actions):
            obs, reward, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.float32) if isinstance(action, list) else action
            )
            trace["steps"].append(
                step_fingerprint(env, k, reward, terminated, truncated, obs, include_obs=spec.include_obs)
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    trace["wall_time_s"] = time.perf_counter() - started
    trace["digest"] = _digest({"reset": trace["reset"], "steps": trace["steps"]})
    return trace


def _worker_main(spec_path: str, out_path: str) -> int:
    with open(spec_path, encoding="utf-8") as handle:
        spec = ReplaySpec.from_dict(json.load(handle))
    trace = run_replay(spec)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(trace, handle)
    return 0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
_TWO_PI = 2.0 * math.pi


def _angle_difference(a: float, b: float) -> float:
    """|a - b| modulo 2π, exact for the ±π representative flip (bitwise-safe at atol=0)."""
    raw = abs(a - b)
    return min(raw, abs(raw - _TWO_PI))


def _compare_values(path: str, a: Any, b: Any, atol: float, out: list[dict]) -> None:
    """Recursively compare two JSON-like values; append mismatches to ``out``."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a, keys_b = list(a.keys()), list(b.keys())
        if keys_a != keys_b:
            out.append({"key": path, "a": keys_a, "b": keys_b, "delta": None, "what": "keys differ"})
            return
        for key in keys_a:
            _compare_values(f"{path}.{key}" if path else str(key), a[key], b[key], atol, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"key": path, "a": len(a), "b": len(b), "delta": None, "what": "length differs"})
            return
        for idx, (va, vb) in enumerate(zip(a, b)):
            _compare_values(f"{path}[{idx}]", va, vb, atol, out)
        return
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        if path.endswith("yaw"):
            delta = _angle_difference(float(a), float(b))
        else:
            delta = abs(float(a) - float(b))
        if delta > atol or (math.isnan(delta) and not (math.isnan(a) and math.isnan(b))):
            out.append({"key": path, "a": a, "b": b, "delta": delta, "what": "value differs"})
        return
    if a != b:
        out.append({"key": path, "a": a, "b": b, "delta": None, "what": "value differs"})


def compare_traces(reference: dict, other: dict, *, atol: float = 0.0) -> dict[str, Any]:
    """Compare ``other`` against ``reference``; report the first divergence and totals."""
    reset_mismatches: list[dict] = []
    _compare_values("reset", reference["reset"], other["reset"], atol, reset_mismatches)

    first = None
    step_mismatch_count = 0
    max_delta = 0.0
    ref_steps, oth_steps = reference["steps"], other["steps"]
    for step_ref, step_oth in zip(ref_steps, oth_steps):
        mismatches: list[dict] = []
        _compare_values("", step_ref, step_oth, atol, mismatches)
        if mismatches:
            step_mismatch_count += len(mismatches)
            for m in mismatches:
                if m["delta"] is not None and math.isfinite(m["delta"]):
                    max_delta = max(max_delta, m["delta"])
            if first is None:
                first = {"step": step_ref["step"], "sim_t": step_ref["sim_t"], **mismatches[0]}
    length_mismatch = len(ref_steps) != len(oth_steps)
    if length_mismatch and first is None:
        first = {
            "step": min(len(ref_steps), len(oth_steps)),
            "sim_t": None,
            "key": "steps",
            "a": len(ref_steps),
            "b": len(oth_steps),
            "delta": None,
            "what": "episode length differs",
        }
    for m in reset_mismatches:
        if m["delta"] is not None and math.isfinite(m["delta"]):
            max_delta = max(max_delta, m["delta"])
    identical = not reset_mismatches and first is None and not length_mismatch
    return {
        "identical": identical,
        "reset_identical": not reset_mismatches,
        "reset_mismatches": reset_mismatches[:20],
        "reset_mismatch_count": len(reset_mismatches),
        "first_divergence": first,
        "step_mismatch_count": step_mismatch_count,
        "length_mismatch": length_mismatch,
        "max_abs_delta": max_delta,
        "digest_equal": reference.get("digest") == other.get("digest"),
    }


IDENTITY_RESET_KEYS = ("scene_key", "seeds", "ego.route_digest", "ego.route_len", "targets", "signals")


def compare_variation_independence(reference: dict, variant: dict) -> dict[str, Any]:
    """Two runs with different variation seeds must share the scene identity but not the spawn state.

    Identity: scene key, seeds, ego route, targets, signals, and the NPC set
    (ids, kinds, roles, routes' digests for scripted actors). State: at least
    one spawn pose must differ.
    """
    ref_reset, var_reset = reference["reset"], variant["reset"]
    identity_mismatches: list[dict] = []

    def get(d, dotted):
        for part in dotted.split("."):
            d = d[part]
        return d

    for key in IDENTITY_RESET_KEYS:
        _compare_values(f"reset.{key}", get(ref_reset, key), get(var_reset, key), 0.0, identity_mismatches)
    ref_npcs = {n["npc_id"]: n for n in ref_reset["npcs"]}
    var_npcs = {n["npc_id"]: n for n in var_reset["npcs"]}
    if list(ref_npcs) != list(var_npcs):
        identity_mismatches.append({"key": "reset.npcs[ids]", "a": list(ref_npcs), "b": list(var_npcs), "delta": None, "what": "npc set differs"})
    else:
        for npc_id, ref_npc in ref_npcs.items():
            var_npc = var_npcs[npc_id]
            for key in ("kind", "traffic_role", "behavior", "cruise_speed_mps", "obeys_signals", "yields_at_junctions"):
                _compare_values(f"reset.npcs[{npc_id}].{key}", ref_npc[key], var_npc[key], 0.0, identity_mismatches)
    state_keys = ["ego.x_m", "ego.y_m"] + [f"npcs[{i}].x_m" for i in range(len(ref_reset["npcs"]))]
    state_diffs = []
    for key in state_keys:
        if key.startswith("npcs["):
            idx = int(key[5 : key.index("]")])
            a, b = ref_reset["npcs"][idx]["x_m"], var_reset["npcs"][idx]["x_m"]
        else:
            a, b = get(ref_reset, key), get(var_reset, key)
        if a != b:
            state_diffs.append(key)
    same_variation_seed = (
        ref_reset["descriptor"].get("variation_seed") == var_reset["descriptor"].get("variation_seed")
    )
    return {
        "identity_preserved": not identity_mismatches,
        "identity_mismatches": identity_mismatches[:20],
        "state_differs": bool(state_diffs),
        "state_diff_keys": state_diffs[:20],
        "variant_seed_recorded": var_reset["descriptor"].get("variation_seed"),
        "passed": not identity_mismatches and bool(state_diffs) and not same_variation_seed,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _hash_seed_for_run(index: int, *, vary: bool) -> str:
    # Distinct, deterministic per-run hash seeds (so the report is reproducible)
    return str(1009 * (index + 1) + 7) if vary else "0"


def _run_in_subprocess(spec: ReplaySpec, workdir: Path, index: int, hash_seed: str) -> dict[str, Any]:
    spec_path = workdir / f"spec_{index}.json"
    out_path = workdir / f"trace_{index}.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    proc = subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, "--worker", str(spec_path), str(out_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"replay worker {index} failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


def run_validation(
    spec: ReplaySpec,
    *,
    runs: int = 3,
    isolation: str = "process",
    vary_hash_seed: bool = True,
    atol: float = 0.0,
    cache_mode: str = "off",
    workdir: Path | None = None,
    progress=None,
    variation_independence_spec: ReplaySpec | None = None,
) -> dict[str, Any]:
    """Run ``runs`` replays of ``spec`` and compare each against run 0.

    ``variation_independence_spec`` (same scene, different variation seed) adds
    one extra run that must keep the scene identity while changing the state.
    """
    if runs < 2:
        raise ValueError("--runs must be >= 2 (run 0 is the reference)")
    started = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="carlabev-replay-")) if workdir is None else Path(workdir)
    traces: list[dict[str, Any] | None] = []
    errors: list[dict[str, Any]] = []
    for index in range(runs):
        hash_seed = _hash_seed_for_run(index, vary=vary_hash_seed)
        try:
            if isolation == "process":
                trace = _run_in_subprocess(spec, workdir, index, hash_seed)
            else:
                trace = run_replay(spec)
                trace["python_hash_seed"] = os.environ.get("PYTHONHASHSEED", "n/a")
        except Exception as exc:  # keep going: report every failing run
            errors.append({"run": index, "error": str(exc)})
            traces.append(None)
            if progress:
                progress(index, None, None)
            continue
        trace["run"] = index
        traces.append(trace)
        if progress:
            progress(index, trace, None)

    reference = next((t for t in traces if t is not None), None)
    variation_check = None
    if variation_independence_spec is not None and reference is not None:
        try:
            if isolation == "process":
                variant = _run_in_subprocess(variation_independence_spec, workdir, runs, _hash_seed_for_run(runs, vary=vary_hash_seed))
            else:
                variant = run_replay(variation_independence_spec)
            variation_check = compare_variation_independence(reference, variant)
            variation_check["variant_digest"] = variant["digest"]
            variation_check["variant_steps"] = len(variant["steps"])
            if progress:
                progress(runs, variant, "variant")
        except Exception as exc:
            errors.append({"run": runs, "error": f"variation run failed: {exc}"})
    comparisons = []
    for trace in traces:
        if trace is None or reference is None or trace is reference:
            comparisons.append(None)
            continue
        comparisons.append(compare_traces(reference, trace, atol=atol))

    expected_hits = {
        "off": [False] * runs,
        "miss-then-hit": [False] + [True] * (runs - 1),
        "hit-only": [True] * runs,
    }[cache_mode]
    cache_ok = all(
        trace is not None and trace["cache"]["hit"] == expected
        for trace, expected in zip(traces, expected_hits)
    )
    all_identical = (
        reference is not None
        and not errors
        and all(c is None or c["identical"] for c in comparisons)
    )
    variation_ok = variation_check is None or bool(variation_check["passed"])
    report = {
        "spec": spec.to_dict(),
        "runs": runs,
        "isolation": isolation,
        "vary_hash_seed": vary_hash_seed,
        "atol": atol,
        "cache_mode": cache_mode,
        "cache_expectation_ok": cache_ok,
        "results": [
            {
                "run": index,
                "python_hash_seed": None if trace is None else trace.get("python_hash_seed"),
                "pid": None if trace is None else trace.get("pid"),
                "cache_enabled": None if trace is None else trace["cache"]["enabled"],
                "cache_hit": None if trace is None else trace["cache"]["hit"],
                "steps": None if trace is None else len(trace["steps"]),
                "cause": None if trace is None or not trace["steps"] else trace["steps"][-1]["cause"],
                "recycled": None if trace is None or not trace["steps"] else trace["steps"][-1]["recycled"],
                "num_npcs": None if trace is None else len(trace["reset"]["npcs"]),
                "digest": None if trace is None else trace["digest"],
                "wall_time_s": None if trace is None else trace["wall_time_s"],
                "comparison": comparison,
            }
            for index, (trace, comparison) in enumerate(zip(traces, comparisons))
        ],
        "errors": errors,
        "all_identical": bool(all_identical),
        "variation_check": variation_check,
        "passed": bool(all_identical and cache_ok and variation_ok),
        "wall_time_s": time.perf_counter() - started,
        "workdir": str(workdir),
    }
    report["_traces"] = traces
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carlabev validate replay",
        description="Replay the same seeded episode N times in isolated processes and assert identical state at every step.",
    )
    group = parser.add_argument_group("replay")
    group.add_argument("--seed", type=int, default=0, help="Reset seed (= scene_seed unless overridden)")
    group.add_argument("--route-seed", type=int, default=None)
    group.add_argument("--traffic-seed", type=int, default=None)
    group.add_argument("--scenario-seed", type=int, default=None)
    group.add_argument("--runs", type=int, default=3, help="Independent runs (run 0 is the reference)")
    group.add_argument("--steps", type=int, default=200, help="Actions applied per run (episode may end earlier)")
    group.add_argument("--policy", choices=POLICIES, default="scripted")
    group.add_argument("--action-seed", type=int, default=None, help="Seed for --policy random (default: --seed)")
    group.add_argument("--actions", default=None, metavar="FILE.json", help="Explicit action list (--policy explicit)")
    group.add_argument(
        "--isolation",
        choices=("process", "in-process"),
        default="process",
        help="process = fresh interpreter per run with a distinct PYTHONHASHSEED (default)",
    )
    group.add_argument("--no-vary-hash-seed", action="store_true", help="Use PYTHONHASHSEED=0 for every run")
    group.add_argument("--atol", type=float, default=0.0, help="Tolerance (0 = bit-identical; 1e-7 across machines)")
    group.add_argument("--no-obs", action="store_true", help="Do not include the observation hash in the fingerprint")
    group.add_argument(
        "--variation-seed", type=int, default=None,
        help="Replay a specific variant of the scene (default: frozen realization)",
    )
    group.add_argument("--variation-profile", default=None, help="Variation profile (spawn_v1 | spawn_phase_v1)")
    group.add_argument(
        "--variation-independence", action="store_true",
        help="Extra run with a different variation seed: same scene identity, different spawn state",
    )
    # --- cache
    group = parser.add_argument_group("scene cache")
    group.add_argument(
        "--cache-mode", choices=CACHE_MODES, default="off",
        help="off | miss-then-hit (run 0 populates a temp library) | hit-only (existing library opened read-only with require-hit)",
    )
    group.add_argument("--scene-library", default=None, metavar="PATH", help="SQLite library (default: temp file)")
    group.add_argument("--keep-workdir", action="store_true", help="Keep specs/traces/temp library on disk")
    # --- env
    group = parser.add_argument_group("environment")
    group.add_argument("--size", type=int, default=128)
    group.add_argument("--physics-dt", type=float, default=0.1)
    group.add_argument("--action-repeat", type=int, default=1)
    group.add_argument("--action-mode", choices=("continuous", "discrete"), default="continuous")
    group.add_argument("--obs-mode", choices=("bev_rgb", "bev_semantic"), default="bev_rgb")
    group.add_argument("--geometry-profile", choices=("legacy", "realistic"), default="realistic")
    group.add_argument("--max-episode-duration-s", type=float, default=None)
    group.add_argument("--car-following", choices=("idm", "legacy"), default="idm")
    group.add_argument("--no-signals", action="store_true")
    group.add_argument("--no-right-of-way", action="store_true")
    group.add_argument("--no-traffic-lifecycle", action="store_true")
    # --- scene
    group = parser.add_argument_group("scene")
    group.add_argument("--reset-mode", choices=RESET_MODES, default="random")
    group.add_argument("--difficulty-preset-id", default="medium")
    group.add_argument("--route-min-distance", type=int, default=30)
    group.add_argument("--route-max-distance", type=int, default=130)
    group.add_argument("--num-vehicles", type=int, default=None)
    group.add_argument("--no-traffic", action="store_true")
    group.add_argument("--num-vehicles-near-ego", type=int, default=None)
    group.add_argument("--traffic-role-profile", default=None)
    group.add_argument("--guaranteed-candidate-role", default=None)
    group.add_argument("--ego-route-graph", default="canonical")
    group.add_argument("--scenario-id", default=None)
    group.add_argument("--scenario-level", type=int, default=1)
    group.add_argument("--preset-id", default=None)
    group.add_argument("--scene-file", default=None)
    # --- output
    group = parser.add_argument_group("output")
    group.add_argument("--report", default=None, metavar="FILE.json", help="Write the JSON report")
    group.add_argument("--save-traces", action="store_true", help="Embed the full per-run traces in the report")
    group.add_argument("--quiet", action="store_true")
    return parser


def _build_reset_options(args) -> dict[str, Any]:
    from CarlaBEV.config import (
        AuthoredSceneReset,
        RandomNavigationReset,
        ScenarioConfigReset,
        ScenarioPresetReset,
        build_authored_scene_options,
        build_random_navigation_options,
        build_scenario_config_options,
        build_scenario_preset_options,
    )

    if args.reset_mode == "random":
        request = RandomNavigationReset(
            difficulty_preset_id=args.difficulty_preset_id,
            route_dist_range=(args.route_min_distance, args.route_max_distance),
            num_vehicles=0 if args.no_traffic else args.num_vehicles,
            num_vehicles_near_ego=args.num_vehicles_near_ego,
            traffic_role_profile=args.traffic_role_profile,
            guaranteed_candidate_role=args.guaranteed_candidate_role,
            ego_route_graph=args.ego_route_graph,
            scene_seed=int(args.seed),
            route_seed=args.route_seed,
            traffic_seed=args.traffic_seed,
            scenario_seed=args.scenario_seed,
            variation_seed=args.variation_seed,
            variation_profile=args.variation_profile,
        )
        return build_random_navigation_options(request)
    if args.reset_mode == "scenario":
        request = ScenarioConfigReset(scenario_id=args.scenario_id or "lead_brake", level=int(args.scenario_level))
        options = build_scenario_config_options(request)
        if args.variation_seed is not None:
            options["variation_seed"] = int(args.variation_seed)
        if args.variation_profile is not None:
            options["variation_profile"] = str(args.variation_profile)
        return options
    if args.reset_mode == "scenario-preset":
        if not args.preset_id:
            raise SystemExit("--preset-id is required with --reset-mode scenario-preset")
        return build_scenario_preset_options(ScenarioPresetReset(preset_id=args.preset_id))
    if args.reset_mode == "authored":
        if not args.scene_file:
            raise SystemExit("--scene-file is required with --reset-mode authored")
        return build_authored_scene_options(AuthoredSceneReset(config_file=args.scene_file))
    raise SystemExit(f"Unsupported --reset-mode {args.reset_mode}")


def _strip_reset_mask(options: dict[str, Any]) -> dict[str, Any]:
    # The public builders add a vector-env reset mask; the single env ignores it,
    # and it is not JSON serializable.
    return {k: v for k, v in options.items() if k != "reset_mask"}


def build_spec_from_args(args, *, scene_library_path: str | None) -> ReplaySpec:
    env_kwargs: dict[str, Any] = dict(
        map_name="Town01",
        render_mode="rgb_array",
        obs_mode=args.obs_mode,
        action_mode=args.action_mode,
        size=int(args.size),
        physics_dt=float(args.physics_dt),
        action_repeat=int(args.action_repeat),
        geometry_profile=args.geometry_profile,
        traffic_enabled=not args.no_traffic,
        traffic_lifecycle_enabled=not args.no_traffic_lifecycle,
        car_following=args.car_following,
        traffic_signals_enabled=not args.no_signals,
        junction_right_of_way_enabled=not args.no_right_of_way,
        scene_library_enabled=args.cache_mode != "off",
    )
    if args.max_episode_duration_s is not None:
        env_kwargs["max_episode_duration_s"] = float(args.max_episode_duration_s)
    if args.cache_mode != "off":
        env_kwargs["scene_library_path"] = str(scene_library_path)
    if args.cache_mode == "hit-only":
        # Strict corpus replay: immutable library, a miss is an error (never a silent regeneration).
        env_kwargs["scene_library_read_only"] = True
        env_kwargs["scene_library_require_hit"] = True
    actions = build_action_sequence(
        args.policy,
        steps=int(args.steps),
        action_mode=args.action_mode,
        action_seed=int(args.seed if args.action_seed is None else args.action_seed),
        actions_file=args.actions,
    )
    return ReplaySpec(
        env=env_kwargs,
        seed=int(args.seed),
        options=_strip_reset_mask(_build_reset_options(args)),
        actions=actions,
        include_obs=not args.no_obs,
        label={"reset_mode": args.reset_mode, "policy": args.policy},
    )


def _fmt(value, digits=6):
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _describe_divergence(comparison: dict[str, Any]) -> str:
    if not comparison["reset_identical"]:
        m = comparison["reset_mismatches"][0]
        extra = f" (+{comparison['reset_mismatch_count'] - 1} more)" if comparison["reset_mismatch_count"] > 1 else ""
        return f"reset differs at {m['key']}: {_fmt(m['a'])} vs {_fmt(m['b'])}{extra}"
    first = comparison["first_divergence"]
    if first is None:
        return "differs"
    delta = f" (Δ={first['delta']:.3e})" if first.get("delta") is not None else ""
    sim_t = "" if first.get("sim_t") is None else f" (t={first['sim_t']:.2f}s)"
    return (
        f"step {first['step']}{sim_t}: {first['key']}: {_fmt(first['a'])} vs {_fmt(first['b'])}{delta}; "
        f"{comparison['step_mismatch_count']} mismatching values, max |Δ|={comparison['max_abs_delta']:.3e}"
    )


def render_report(report: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    # Non-TTY output (CI logs, pipes) defaults to 80 columns; the table needs ~100.
    console = Console() if sys.stdout.isatty() else Console(width=110)
    spec = report["spec"]
    console.print(
        f"[bold]carlabev validate replay[/bold]  seed={spec['seed']}  reset={spec['label'].get('reset_mode')}  "
        f"policy={spec['label'].get('policy')}  actions={len(spec['actions'])}  runs={report['runs']}  "
        f"isolation={report['isolation']}  cache={report['cache_mode']}  atol={report['atol']:g}  "
        f"variation_seed={spec['options'].get('variation_seed')}"
    )
    table = Table(show_lines=False, pad_edge=False)
    for column in ("run", "hash", "cache", "npcs", "steps", "end", "recyc", "digest", "time", "vs run 0"):
        table.add_column(column, no_wrap=True)
    details = []
    for result in report["results"]:
        comparison = result["comparison"]
        if result["digest"] is None:
            status = "[red]ERROR[/red]"
        elif comparison is None:
            status = "[cyan]reference[/cyan]"
        elif comparison["identical"]:
            status = "[green]identical[/green]"
        elif not comparison["reset_identical"]:
            status = "[red]RESET DIFFERS[/red]"
            details.append((result["run"], _describe_divergence(comparison)))
        else:
            first = comparison["first_divergence"]
            status = f"[red]DIVERGED @ step {first['step']}[/red]" if first else "[red]DIFFERS[/red]"
            details.append((result["run"], _describe_divergence(comparison)))
        if result["cache_hit"] is None:
            cache = "-"
        elif not result["cache_enabled"]:
            cache = "off"
        else:
            cache = "hit" if result["cache_hit"] else "miss"
        table.add_row(
            str(result["run"]),
            str(result["python_hash_seed"]),
            cache,
            str(result["num_npcs"]),
            str(result["steps"]),
            "-" if result["cause"] is None else str(result["cause"]),
            str(result["recycled"]),
            "-" if result["digest"] is None else str(result["digest"])[:8],
            "-" if result["wall_time_s"] is None else f"{result['wall_time_s']:.0f}s",
            status,
        )
    console.print(table)
    for run, text in details:
        console.print(f"  [red]run {run}[/red]: {text}", highlight=False)
    for error in report["errors"]:
        console.print(f"[red]run {error['run']} failed:[/red] {error['error']}")
    if not report["cache_expectation_ok"]:
        console.print(
            f"[yellow]cache expectation not met for --cache-mode {report['cache_mode']} "
            f"(hits: {[r['cache_hit'] for r in report['results']]})[/yellow]"
        )
    check = report.get("variation_check")
    if check is not None:
        status = "[green]ok[/green]" if check["passed"] else "[red]FAILED[/red]"
        console.print(
            f"  variation independence (variant seed {check['variant_seed_recorded']}): {status} — "
            f"identity_preserved={check['identity_preserved']} state_differs={check['state_differs']}"
            + (f" mismatches={check['identity_mismatches'][:3]}" if check["identity_mismatches"] else "")
            + (f" diff_keys={check['state_diff_keys'][:4]}" if check["state_diff_keys"] else ""),
            highlight=False,
        )
    verdict = "[bold green]PASS[/bold green]" if report["passed"] else "[bold red]FAIL[/bold red]"
    console.print(
        f"{verdict}  all_identical={report['all_identical']}  cache_expectation_ok={report['cache_expectation_ok']}  "
        f"wall={report['wall_time_s']:.1f}s"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker":
        if len(argv) != 3:
            raise SystemExit("usage: --worker SPEC.json OUT.json")
        return _worker_main(argv[1], argv[2])

    args = _build_parser().parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    workdir = Path(tempfile.mkdtemp(prefix="carlabev-replay-"))
    scene_library_path = None
    if args.cache_mode != "off":
        scene_library_path = args.scene_library or str(workdir / "scene_library.db")
        if args.cache_mode == "hit-only" and not Path(scene_library_path).exists():
            raise SystemExit(f"--cache-mode hit-only requires an existing --scene-library (got {scene_library_path})")

    spec = build_spec_from_args(args, scene_library_path=scene_library_path)
    variation_spec = None
    if args.variation_independence:
        base_seed = 0 if args.variation_seed is None else int(args.variation_seed)
        variation_spec = ReplaySpec(
            env=dict(spec.env),
            seed=spec.seed,
            options={**spec.options, "variation_seed": base_seed + 1,
                     "variation_profile": args.variation_profile or "spawn_phase_v1"},
            actions=list(spec.actions),
            include_obs=spec.include_obs,
            label={**spec.label, "variant": True},
        )

    def progress(index, trace, tag):
        if args.quiet:
            return
        label = f"run {index}" if tag is None else f"{tag} run {index}"
        if trace is None:
            print(f"  {label}: FAILED", flush=True)
        else:
            print(
                f"  {label}: pid={trace.get('pid')} hash_seed={trace.get('python_hash_seed')} "
                f"cache_hit={trace['cache']['hit']} steps={len(trace['steps'])} "
                f"digest={trace['digest']} ({trace['wall_time_s']:.1f}s)",
                flush=True,
            )

    if not args.quiet:
        print(f"[validate replay] workdir={workdir}", flush=True)
    report = run_validation(
        spec,
        runs=int(args.runs),
        isolation=args.isolation,
        vary_hash_seed=not args.no_vary_hash_seed,
        atol=float(args.atol),
        cache_mode=args.cache_mode,
        workdir=workdir,
        progress=progress,
        variation_independence_spec=variation_spec,
    )
    traces = report.pop("_traces")
    if args.report:
        payload = dict(report)
        if args.save_traces:
            payload["traces"] = traces
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"[validate replay] report written to {args.report}")
    if not args.quiet:
        render_report(report)
    if not args.keep_workdir:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    if report["errors"]:
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
