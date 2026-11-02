"""``carlabev scene verify-library`` — regenerate cached scenes and compare their spec digests.

Random-scene generation is deterministic on one machine but its decisions
depend on the platform's floating-point rounding (BLAS ``dot``/``norm``,
libm ``pow``), so a seed may realise a different — equally valid — scene on
another OS/CPU. The scene library is the portability mechanism: a cached
scene replays identically everywhere because a hit never re-runs generation.

This tool answers "does *this* machine's generator agree with the library's
origin?" by regenerating a sample of cached scenes from their stored seeds
and options and comparing the resulting :class:`SceneSpec` digests. Use it

* in CI on every supported platform (detects platform drift), and
* after any generator change (detects unintended decision changes; an
  intended change must bump ``scene_library_generator_version``).

Examples::

    carlabev scene verify-library --scene-library-path runs/lib.db
    carlabev scene verify-library --scene-library-path runs/lib.db --sample 20 --seed 0 --json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from CarlaBEV.src.scenes.scene_spec import SceneSpec
from CarlaBEV.src.scenes.scene_identity import SceneGenerationIdentity
from CarlaBEV.src.managers.scene_library import SceneLibraryInvalidEntryError, load_scene_payload


def _fetch_scene_rows(db_path: str, *, sample: int | None, seed: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        rows = conn.execute(
            "SELECT scene_key, generator_version, scene_kind, actors_json, resolved_options_json, scene_seed "
            "FROM scenes ORDER BY created_at, scene_id"
        ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            try:
                record["actors_payload"] = load_scene_payload(conn, record["scene_key"], json.loads(record.pop("actors_json")))
                record["resolved_options"] = json.loads(record.pop("resolved_options_json"))
            except (SceneLibraryInvalidEntryError, ValueError, TypeError) as exc:
                record["entry_error"] = str(exc)
            records.append(record)
    finally:
        conn.close()
    if sample is not None and sample < len(records):
        rng = random.Random(int(seed))
        records = rng.sample(records, int(sample))
    return records


def _first_difference(a: Any, b: Any, path: str = "") -> str | None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: present on one side only"
            diff = _first_difference(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for idx, (va, vb) in enumerate(zip(a, b)):
            diff = _first_difference(va, vb, f"{path}[{idx}]")
            if diff:
                return diff
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def verify_scene_library(
    db_path: str,
    *,
    sample: int | None = None,
    seed: int = 0,
    progress=None,
) -> dict[str, Any]:
    """Regenerate cached scenes on this machine and compare spec digests with the stored ones."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from CarlaBEV.config import EnvConfig
    from CarlaBEV.envs.carlabev import CarlaBEV

    started = time.perf_counter()
    records = _fetch_scene_rows(db_path, sample=sample, seed=seed)
    envs: dict[tuple, CarlaBEV] = {}
    results = []
    try:
        for index, record in enumerate(records):
            if "entry_error" in record:
                results.append({"scene_key": record["scene_key"], "status": "unreadable", "detail": record["entry_error"]})
                continue
            try:
                stored = SceneSpec.from_payload(record["actors_payload"])
                identity = SceneGenerationIdentity.model_validate(stored.identity.get("generation"))
            except ValueError as exc:
                results.append({"scene_key": record["scene_key"], "status": "unreadable", "detail": str(exc)})
                continue
            environment = identity.config.environment.model_dump()
            env_key = (json.dumps(environment, sort_keys=True), record["generator_version"])
            env = envs.get(env_key)
            if env is None:
                env = CarlaBEV(
                    EnvConfig(
                        **environment,
                        render_mode="rgb_array",
                        obs_mode="bev_rgb",
                        action_mode="discrete",
                        scene_library_enabled=False,
                        scene_library_generator_version=record["generator_version"],
                    )
                )
                envs[env_key] = env
            options = dict(record["resolved_options"])
            t0 = time.perf_counter()
            try:
                env.reset(seed=int(options.get("scene_seed", record["scene_seed"] or 0)), options=options)
                regenerated = env.scene_spec
            except Exception as exc:  # report, keep verifying the rest
                results.append({"scene_key": record["scene_key"], "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
                continue
            wall = time.perf_counter() - t0
            if regenerated.scene_key != stored.scene_key:
                results.append(
                    {
                        "scene_key": record["scene_key"],
                        "status": "key_mismatch",
                        "detail": f"regenerated key {regenerated.scene_key} (generation inputs or map assets differ)",
                        "wall_s": wall,
                    }
                )
                continue
            identical = regenerated.digest() == stored.digest()
            entry = {
                "scene_key": record["scene_key"],
                "status": "identical" if identical else "mismatch",
                "npcs": len(stored.npcs),
                "wall_s": wall,
            }
            if not identical:
                entry["detail"] = _first_difference(stored.to_payload(), regenerated.to_payload(), "spec")
            results.append(entry)
            if progress:
                progress(index, len(records), entry)
    finally:
        for env in envs.values():
            env.close()
    counts: dict[str, int] = {}
    for entry in results:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {
        "scene_library_path": db_path,
        "scenes_in_library": len(records) if sample is None else None,
        "checked": len(results),
        "counts": counts,
        "passed": counts.get("identical", 0) == len(results) and bool(results),
        "results": results,
        "wall_time_s": time.perf_counter() - started,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carlabev scene verify-library",
        description="Regenerate cached scenes on this machine and compare their spec digests with the library.",
    )
    parser.add_argument("--scene-library-path", default="CarlaBEV/assets/scenes/scene_library.db")
    parser.add_argument("--sample", type=int, default=None, help="Verify a random sample of N scenes (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--report", default=None, metavar="FILE.json")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if not os.path.exists(args.scene_library_path):
        raise SystemExit(f"scene library not found: {args.scene_library_path}")

    def progress(index, total, entry):
        if args.quiet or args.as_json:
            return
        mark = "ok " if entry["status"] == "identical" else "!! "
        print(f"  [{index + 1}/{total}] {mark}{entry['scene_key'][:16]}… {entry['status']} ({entry.get('wall_s', 0):.2f}s)", flush=True)

    report = verify_scene_library(args.scene_library_path, sample=args.sample, seed=args.seed, progress=progress)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    if args.as_json:
        print(json.dumps(report, indent=2))
    elif not args.quiet:
        from rich.console import Console

        console = Console() if sys.stdout.isatty() else Console(width=110)
        verdict = "[bold green]PASS[/bold green]" if report["passed"] else "[bold red]FAIL[/bold red]"
        console.print(
            f"{verdict}  verified {report['checked']} scenes from {args.scene_library_path}: {report['counts']}  "
            f"wall={report['wall_time_s']:.1f}s"
        )
        for entry in report["results"]:
            if entry["status"] != "identical":
                console.print(f"  [red]{entry['status']}[/red] {entry['scene_key']}: {entry.get('detail')}", highlight=False)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
