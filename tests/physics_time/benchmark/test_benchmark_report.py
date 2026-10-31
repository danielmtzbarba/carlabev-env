"""BENCHMARK / report: `carlabev benchmark run|compare|baseline` produce schema-valid reports.

Expected: the quick profile runs every suite headlessly, writes report.json and a
self-contained report.html, every metric carries name/value/unit/tolerance/status,
and `compare` classifies metric differences (equal / platform_rounding /
within_tolerance / performance / behavioural).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from CarlaBEV.tools.benchmark import runner
from CarlaBEV.tools.benchmark.schema import (
    PLATFORM_ROUNDING_ATOL,
    Metric,
    Report,
    compare_reports,
    metric_index,
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

SMOKE_PROFILE = {
    "physics_dt": 0.1,
    "convergence_dts": (0.1, 0.05),
    "convergence_reference": True,
    "rendering_steps": 15,
    "render_configs": ((60, "rgb_array"), (5, "human")),
    "footprint_sizes": (128,),
    "replay_steps": 12,
    "replay_sizes": (128, 256),
    "throughput_steps": 40,
    "throughput_sizes": (128,),
    "throughput_modes": ("rgb_array", "human"),
}


@pytest.fixture(scope="module")
def smoke_report(tmp_path_factory):
    runner.PROFILES["smoke"] = SMOKE_PROFILE
    out = tmp_path_factory.mktemp("bench")
    report, out_dir, code = runner.run_benchmark(
        suites=runner.SUITES,
        profile_name="smoke",
        out_dir=out,
        baseline=None,
        strict=False,
    )
    return report, Path(out_dir), code


def test_benchmark_runs_all_suites_and_writes_artifacts(smoke_report):
    report, out_dir, code = smoke_report
    assert code == 0
    assert [s.suite for s in report.sections] == list(runner.SUITES)
    assert (out_dir / "report.json").exists() and (out_dir / "report.html").exists()
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "CarlaBEV benchmark" in html
    assert (
        "class='tabgroup'" in html and "Plotly.newPlot" in html
    )  # tabbed details + embedded charts
    assert "p99" in html
    data = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert data["provenance"]["schema_version"] == "1.0"
    assert {"platform", "numpy", "scipy", "pygame", "python"} <= set(data["provenance"])


def test_metrics_are_well_formed_and_contract_metrics_pass(smoke_report):
    report, _, _ = smoke_report
    statuses = {}
    for section in report.sections:
        assert section.metrics, section.suite
        for m in section.metrics:
            assert m.status in {"PASS", "WARN", "FAIL", "INFO"}
            assert isinstance(m.name, str) and m.name
            statuses.setdefault(section.suite, set()).add(m.status)
    # Contract checks (non-INFO) must not FAIL on a healthy tree.
    failing = [
        (s.suite, m.name, m.value)
        for s in report.sections
        for m in s.metrics
        if m.status == "FAIL"
    ]
    assert not failing, failing
    idx = metric_index(report.to_dict())
    assert idx["clock.timer_max_error_steps"]["value"] == 0
    assert idx["rendering.max_state_diff_across_fps_and_modes"]["value"] == 0
    assert idx["replay.same_process_repeat_bitwise_identical"]["value"] is True
    assert idx["throughput.ms_per_step_128_rgb_lead_brake"]["value"] > 0


def test_throughput_section_has_phase_breakdown(smoke_report):
    report, _, _ = smoke_report
    thr = next(s for s in report.sections if s.suite == "throughput")
    breakdown = thr.tables["phase_breakdown"]
    assert (
        "lead_brake@128px/rgb_array" in breakdown
        and "lead_brake@128px/human" in breakdown
    )
    phases = {row["phase"] for row in breakdown["lead_brake@128px/rgb_array"]}
    assert {
        "env.step",
        "env.simulate",
        "scene.npc_step",
        "map.draw_fov",
        "reward.step",
    } <= phases
    assert {"renderer.human"} <= {
        row["phase"] for row in breakdown["lead_brake@128px/human"]
    }
    rows = thr.tables["throughput"]
    assert all(r["ms_per_step"] > 0 and r["steps"] > 0 for r in rows)


def test_compare_classifies_differences(smoke_report):
    report, out_dir, _ = smoke_report
    a = report.to_dict()
    b = json.loads(json.dumps(a))
    idx_b = metric_index(b)
    idx_b["clock.ego_displacement_error_per_update"]["value"] += (
        PLATFORM_ROUNDING_ATOL / 10
    )  # rounding noise
    idx_b["physics.ego_pos_err_coarse_vs_fine"]["value"] += (
        0.05  # still within tolerance
    )
    idx_b["throughput.ms_per_step_128_rgb_lead_brake"]["value"] *= 2  # performance
    idx_b["physics.route_end_max_overshoot"]["value"] = 50.0  # behavioural
    rows = {r["metric"]: r["classification"] for r in compare_reports(a, b)}
    assert rows["clock.ego_displacement_error_per_update"] == "platform_rounding"
    assert rows["physics.ego_pos_err_coarse_vs_fine"] == "within_tolerance"
    assert rows["throughput.ms_per_step_128_rgb_lead_brake"] == "performance"
    assert rows["physics.route_end_max_overshoot"] == "behavioural"
    assert rows["clock.timer_max_error_steps"] == "equal"
    # CLI compare writes artifacts
    (out_dir / "b.json").write_text(json.dumps(b), encoding="utf-8")
    assert (
        runner.main(
            [
                "compare",
                str(out_dir / "report.json"),
                str(out_dir / "b.json"),
                "--out",
                str(out_dir / "cmp"),
            ]
        )
        == 0
    )
    assert (out_dir / "cmp" / "compare.json").exists() and (
        out_dir / "cmp" / "compare.html"
    ).exists()


def test_baseline_command_and_strict_exit(smoke_report, tmp_path):
    report, out_dir, _ = smoke_report
    target = tmp_path / "baseline.json"
    assert (
        runner.main(["baseline", str(out_dir / "report.json"), "--path", str(target)])
        == 0
    )
    assert Report.load_json(target)["verdict"] == report.verdict
    failing = Report(provenance={})
    assert failing.verdict == "PASS"
    m = Metric.checked("x", 10.0, "m", tolerance=1.0)
    assert m.status == "FAIL"
    assert Metric.checked("x", 1.2, "m", tolerance=1.0).status == "WARN"
    assert Metric.checked("x", 0.9, "m", tolerance=1.0).status == "PASS"
