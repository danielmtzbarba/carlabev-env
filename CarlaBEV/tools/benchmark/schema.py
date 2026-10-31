"""Report schema for ``carlabev benchmark``.

A report is a versioned JSON document: provenance + one section per suite, each
a list of :class:`Metric` rows (``name | value | unit | tolerance | status``).
``compare`` diffs two reports metric by metric and classifies each difference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from typing import Any

SCHEMA_VERSION = "1.0"

# Differences below this (absolute) between machines are floating-point rounding
# (BLAS / FMA paths) and are classified as "platform" rather than behavioural.
PLATFORM_ROUNDING_ATOL = 1e-9

PERFORMANCE_UNITS = {"ms", "steps/s", "Hz", "s (wall)", "x"}
PERFORMANCE_SUITES = {"throughput"}


@dataclass
class Metric:
    name: str
    value: Any
    unit: str = ""
    tolerance: float | None = None
    """Absolute bound the value must satisfy (``|value| <= tolerance``) for PASS, if any."""
    lower_is_better: bool = True
    status: str = "INFO"  # PASS | WARN | FAIL | INFO
    detail: str = ""

    @staticmethod
    def checked(
        name,
        value,
        unit="",
        *,
        tolerance,
        detail="",
        warn_factor=1.5,
        lower_is_better=True,
    ):
        """Metric whose status is derived from its tolerance.

        ``lower_is_better``: PASS when ``|value| <= tolerance``, WARN up to
        ``warn_factor * tolerance``, FAIL above. Otherwise PASS when
        ``value >= tolerance``.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            status = "FAIL"
        elif lower_is_better:
            magnitude = abs(float(value))
            bound = float(tolerance)
            status = (
                "PASS"
                if magnitude <= bound
                else ("WARN" if magnitude <= warn_factor * bound else "FAIL")
            )
        else:
            status = "PASS" if float(value) >= float(tolerance) else "FAIL"
        return Metric(name, value, unit, tolerance, lower_is_better, status, detail)

    @staticmethod
    def boolean(name, ok: bool, detail=""):
        return Metric(name, bool(ok), "", None, True, "PASS" if ok else "FAIL", detail)

    @staticmethod
    def info(name, value, unit="", detail=""):
        return Metric(name, value, unit, None, True, "INFO", detail)


@dataclass
class Section:
    suite: str
    title: str
    metrics: list[Metric] = field(default_factory=list)
    tables: dict[str, list[dict]] = field(default_factory=dict)
    """Named tabular data for the HTML report (e.g. timing breakdown, convergence table)."""
    wall_time_s: float = 0.0

    def add(self, metric: Metric) -> Metric:
        self.metrics.append(metric)
        return metric

    @property
    def verdict(self) -> str:
        statuses = {m.status for m in self.metrics}
        if "FAIL" in statuses:
            return "FAIL"
        if "WARN" in statuses:
            return "WARN"
        return "PASS"


def collect_provenance(config: dict) -> dict:
    import numpy
    import scipy
    import pygame

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_sha = None
    blas = None
    try:
        cfg = numpy.show_config(mode="dicts")
        blas = (cfg.get("Build Dependencies", {}).get("blas", {}) or {}).get("name")
    except Exception:
        pass
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pygame": pygame.version.ver,
        "blas": blas,
        "git_sha": git_sha,
        "config": config,
    }


@dataclass
class Report:
    provenance: dict
    sections: list[Section] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        verdicts = {s.verdict for s in self.sections}
        if "FAIL" in verdicts:
            return "FAIL"
        if "WARN" in verdicts:
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict:
        return {
            "provenance": self.provenance,
            "verdict": self.verdict,
            "sections": [
                {
                    "suite": s.suite,
                    "title": s.title,
                    "verdict": s.verdict,
                    "wall_time_s": s.wall_time_s,
                    "metrics": [asdict(m) for m in s.metrics],
                    "tables": s.tables,
                }
                for s in self.sections
            ],
        }

    def save_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(), handle, indent=1, sort_keys=False, default=_jsonable
            )
            handle.write("\n")

    @staticmethod
    def load_json(path) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def _jsonable(value):
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def metric_index(report: dict) -> dict[str, dict]:
    return {
        f"{s['suite']}.{m['name']}": m for s in report["sections"] for m in s["metrics"]
    }


def compare_reports(a: dict, b: dict) -> list[dict]:
    """Metric-by-metric diff of two reports (``a`` = reference, ``b`` = candidate)."""
    ia, ib = metric_index(a), metric_index(b)
    rows = []
    for key in sorted(set(ia) | set(ib)):
        ma, mb = ia.get(key), ib.get(key)
        if ma is None or mb is None:
            rows.append(
                {
                    "metric": key,
                    "a": None if ma is None else ma["value"],
                    "b": None if mb is None else mb["value"],
                    "delta": None,
                    "classification": "missing",
                }
            )
            continue
        va, vb = ma["value"], mb["value"]
        if (
            isinstance(va, bool)
            or isinstance(vb, bool)
            or not isinstance(va, (int, float))
            or not isinstance(vb, (int, float))
        ):
            cls = "equal" if va == vb else "different"
            rows.append(
                {
                    "metric": key,
                    "a": va,
                    "b": vb,
                    "delta": None,
                    "classification": cls,
                    "status_a": ma["status"],
                    "status_b": mb["status"],
                }
            )
            continue
        delta = float(vb) - float(va)
        tol = ma.get("tolerance")
        if abs(delta) <= PLATFORM_ROUNDING_ATOL:
            cls = "equal" if delta == 0.0 else "platform_rounding"
        elif (
            tol is not None
            and abs(float(vb)) <= float(tol)
            and abs(float(va)) <= float(tol)
        ):
            cls = "within_tolerance"
        elif (
            ma["unit"] in PERFORMANCE_UNITS
            or key.split(".", 1)[0] in PERFORMANCE_SUITES
        ):
            cls = "performance"
        else:
            cls = "behavioural"
        rows.append(
            {
                "metric": key,
                "a": va,
                "b": vb,
                "delta": delta,
                "classification": cls,
                "status_a": ma["status"],
                "status_b": mb["status"],
            }
        )
    return rows
