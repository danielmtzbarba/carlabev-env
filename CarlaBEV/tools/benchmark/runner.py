"""``carlabev benchmark`` entry point: run / compare / baseline."""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from CarlaBEV.tools.benchmark.report_html import render_html
from CarlaBEV.tools.benchmark.schema import Report, collect_provenance, compare_reports

SUITES = ("clock", "physics", "rendering", "replay", "throughput")
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = REPO_ROOT / "CarlaBEV" / "assets" / "benchmarks"

PROFILES = {
    "quick": {
        "physics_dt": 0.1,
        "convergence_dts": (0.1, 0.05),
        "convergence_reference": True,
        "rendering_steps": 40,
        "render_configs": ((60, "rgb_array"), (5, "human")),
        "footprint_sizes": (128,),
        "replay_steps": 30,
        "replay_sizes": (128, 256),
        "throughput_steps": 150,
        "throughput_sizes": (128,),
        "throughput_modes": ("rgb_array", "human"),
    },
    "standard": {
        "physics_dt": 0.1,
        "convergence_dts": (0.1, 0.05),
        "convergence_reference": True,
        "rendering_steps": 60,
        "render_configs": (
            (60, "rgb_array"),
            (1, "rgb_array"),
            (5, "human"),
            (60, "human"),
        ),
        "footprint_sizes": (64, 128, 256),
        "replay_steps": 40,
        "replay_sizes": (64, 128, 256),
        "throughput_steps": 400,
        "throughput_sizes": (64, 128, 256),
        "throughput_modes": ("rgb_array", "human"),
    },
    "full": {
        "physics_dt": 0.1,
        "convergence_dts": (0.1, 0.05, 0.025),
        "convergence_reference": True,
        "rendering_steps": 80,
        "render_configs": (
            (60, "rgb_array"),
            (1, "rgb_array"),
            (5, "human"),
            (60, "human"),
        ),
        "footprint_sizes": (64, 128, 256, 512),
        "replay_steps": 60,
        "replay_sizes": (64, 128, 256, 512),
        "throughput_steps": 1000,
        "throughput_sizes": (64, 128, 256, 512),
        "throughput_modes": ("rgb_array", "human"),
    },
}


def default_baseline_path() -> Path:
    system = platform.system().lower()
    machine = platform.machine().lower()
    return BASELINE_DIR / f"baseline-{system}-{machine}.json"


def run_benchmark(
    *,
    suites,
    profile_name,
    physics_dt=None,
    out_dir=None,
    baseline=None,
    strict=False,
    formats=("json", "html"),
):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    profile = dict(PROFILES[profile_name])
    if physics_dt is not None:
        profile["physics_dt"] = float(physics_dt)
    provenance = collect_provenance(
        {"profile": profile_name, **profile, "suites": list(suites)}
    )
    report = Report(provenance=provenance)
    for suite in suites:
        module = importlib.import_module(f"CarlaBEV.tools.benchmark.suites.{suite}")
        t0 = time.perf_counter()
        section = module.run(profile)
        section.wall_time_s = time.perf_counter() - t0
        report.sections.append(section)
        print(
            f"[{section.verdict:<4}] {suite:<11} {section.wall_time_s:6.1f} s  {len(section.metrics)} metrics"
        )

    baseline_report = None
    compare_rows = None
    if baseline is not None:
        baseline_path = Path(baseline)
        if baseline_path.exists():
            baseline_report = Report.load_json(baseline_path)
            compare_rows = compare_reports(baseline_report, report.to_dict())
        else:
            print(f"[WARN] baseline {baseline_path} not found; skipping comparison")

    if out_dir is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = REPO_ROOT / "runs" / "bench" / f"{stamp}-{platform.node()}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "json" in formats:
        report.save_json(out_dir / "report.json")
    if "html" in formats:
        (out_dir / "report.html").write_text(
            render_html(
                report.to_dict(), baseline=baseline_report, compare_rows=compare_rows
            ),
            encoding="utf-8",
        )
    if compare_rows:
        (out_dir / "compare.json").write_text(
            json.dumps(compare_rows, indent=1, default=str) + "\n", encoding="utf-8"
        )

    print("-" * 72)
    print(f"verdict: {report.verdict}   report: {out_dir}")
    if compare_rows:
        counts = {}
        for row in compare_rows:
            counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        print(f"vs baseline: {counts}")
    return report, out_dir, (1 if strict and report.verdict == "FAIL" else 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carlabev benchmark",
        description="Clock / physics / rendering / replay / throughput benchmark report.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run suites and write report.json / report.html")
    run.add_argument("--suite", action="append", choices=[*SUITES, "all"], default=None)
    run.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    run.add_argument("--physics-dt", type=float, default=None)
    run.add_argument(
        "--out",
        default=None,
        help="Output directory (default runs/bench/<timestamp>-<host>)",
    )
    run.add_argument(
        "--baseline",
        default=None,
        help="Baseline report.json to compare against (default: committed baseline for this platform if present)",
    )
    run.add_argument("--no-baseline", action="store_true")
    run.add_argument("--format", default="json,html", help="Comma-separated: json,html")
    run.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the verdict is FAIL (default: report only)",
    )
    cmp = sub.add_parser("compare", help="Diff two report.json files")
    cmp.add_argument("report_a")
    cmp.add_argument("report_b")
    cmp.add_argument(
        "--out",
        default=None,
        help="Optional compare.json / compare.html output directory",
    )
    base = sub.add_parser(
        "baseline",
        help="Promote a report.json to the committed baseline for this platform",
    )
    base.add_argument("report")
    base.add_argument("--path", default=None)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        suites = (
            SUITES
            if (not args.suite or "all" in args.suite)
            else tuple(dict.fromkeys(args.suite))
        )
        baseline = (
            None
            if args.no_baseline
            else (
                args.baseline
                or (
                    default_baseline_path()
                    if default_baseline_path().exists()
                    else None
                )
            )
        )
        _, _, code = run_benchmark(
            suites=suites,
            profile_name=args.profile,
            physics_dt=args.physics_dt,
            out_dir=args.out,
            baseline=baseline,
            strict=args.strict,
            formats=tuple(f.strip() for f in args.format.split(",")),
        )
        return code
    if args.command == "compare":
        a, b = Report.load_json(args.report_a), Report.load_json(args.report_b)
        rows = compare_reports(a, b)
        counts = {}
        for row in rows:
            counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        print(f"{'metric':<56}{'a':>14}{'b':>14}{'delta':>12}  class")
        for row in rows:
            if row["classification"] in {"equal"}:
                continue
            fa = (
                "—"
                if row["a"] is None
                else (
                    f"{row['a']:.6g}"
                    if isinstance(row["a"], (int, float))
                    and not isinstance(row["a"], bool)
                    else str(row["a"])
                )
            )
            fb = (
                "—"
                if row["b"] is None
                else (
                    f"{row['b']:.6g}"
                    if isinstance(row["b"], (int, float))
                    and not isinstance(row["b"], bool)
                    else str(row["b"])
                )
            )
            fd = "—" if row["delta"] is None else f"{row['delta']:+.3g}"
            print(
                f"{row['metric']:<56}{fa:>14}{fb:>14}{fd:>12}  {row['classification']}"
            )
        print("-" * 72)
        print(
            f"summary: {counts}  (a: {a['provenance'].get('platform')} | b: {b['provenance'].get('platform')})"
        )
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "compare.json").write_text(
                json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8"
            )
            (out / "compare.html").write_text(
                render_html(b, baseline=a, compare_rows=rows), encoding="utf-8"
            )
        return 0
    if args.command == "baseline":
        target = Path(args.path) if args.path else default_baseline_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = Report.load_json(args.report)
        target.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        print(f"baseline written: {target}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
