"""Self-contained HTML rendering of a benchmark report.

Layout: sticky navigation, summary cards per suite, then one section per
suite with its metrics table and a *tabbed* details area. Tabs group the same
experiment run under different conditions (dt, raster size, render mode,
scene) so that they can be compared side by side. Plotly charts are embedded
(no network access needed); tabs re-layout charts when shown.
"""

from __future__ import annotations

import html
import itertools
import json
from collections import OrderedDict

STATUS_COLOR = {
    "PASS": "#2e7d32",
    "WARN": "#ef6c00",
    "FAIL": "#c62828",
    "INFO": "#546e7a",
}
SUITE_ICON = {
    "clock": "⏱",
    "physics": "⚙",
    "rendering": "🖼",
    "replay": "⟲",
    "throughput": "⚡",
}
HEADLINE_METRICS = {
    "clock": ("timer_max_error_steps", "scene_time_error_after_3s", "render_fps"),
    "physics": (
        "ego_pos_err_coarse_vs_fine",
        "ego_speed_err_coarse_vs_fine",
        "lead_decel_tracking_ratio",
    ),
    "rendering": (
        "max_state_diff_across_fps_and_modes",
        "render_fps_vs_1_over_dt",
        "video_duration_vs_sim_time_error",
    ),
    "replay": (
        "golden_max_abs_diff",
        "cross_size_max_ego_pos_err",
        "same_process_repeat_bitwise_identical",
    ),
    "throughput": (
        "ms_per_step_128_rgb_lead_brake",
        "p99_ms_per_step_128_rgb_lead_brake",
        "real_time_factor_128_rgb",
    ),
}
_ids = itertools.count()


# --------------------------------------------------------------------------- primitives
def _fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e5:
            return f"{value:.3g}"
        return (
            f"{value:.4f}".rstrip("0").rstrip(".")
            if abs(value) < 100
            else f"{value:.2f}"
        )
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return html.escape(str(value))


def _badge(status):
    return f'<span class="badge" style="background:{STATUS_COLOR.get(status, "#999")}">{html.escape(str(status))}</span>'


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _table(rows: list[dict], *, columns=None, highlight=None) -> str:
    if not rows:
        return "<p class='muted'>no data</p>"
    columns = columns or list(OrderedDict.fromkeys(k for r in rows for k in r.keys()))
    head = "".join(f"<th>{html.escape(str(c).replace('_', ' '))}</th>" for c in columns)
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            cls = " class='num'" if _is_num(v) else ""
            cells.append(f"<td{cls}>{_fmt(v)}</td>")
        accent = ""
        if highlight and highlight(r):
            accent = " class='row-accent'"
        body.append(f"<tr{accent}>{''.join(cells)}</tr>")
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _metrics_table(metrics: list[dict], baseline: dict | None) -> str:
    rows = []
    for m in metrics:
        base = baseline.get(m["name"]) if baseline else None
        delta = None
        if base is not None and _is_num(m["value"]) and _is_num(base.get("value")):
            delta = float(m["value"]) - float(base["value"])
        delta_cls = ""
        if delta is not None and delta != 0.0:
            delta_cls = " delta-up" if delta > 0 else " delta-down"
        rows.append(
            f"<tr class='st-{m['status']}'>"
            f"<td><code>{html.escape(m['name'])}</code></td>"
            f"<td class='num'>{_fmt(m['value'])}</td><td class='unit'>{html.escape(m['unit'])}</td>"
            f"<td class='num'>{_fmt(m['tolerance'])}</td><td>{_badge(m['status'])}</td>"
            f"<td class='num'>{'—' if base is None else _fmt(base['value'])}</td>"
            f"<td class='num{delta_cls}'>{'—' if delta is None else _fmt(delta)}</td>"
            f"<td class='detail'>{html.escape(m.get('detail', ''))}</td></tr>"
        )
    return (
        "<div class='tablewrap'><table class='metrics'><thead><tr><th>metric</th><th>value</th><th>unit</th>"
        "<th>tolerance</th><th>status</th><th>baseline</th><th>Δ</th><th>detail</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _tabs(tabs: list[tuple[str, str]]) -> str:
    """Dependency-free tab group. ``tabs`` = [(label, inner_html), ...]."""
    tabs = [(label, body) for label, body in tabs if body]
    if not tabs:
        return ""
    if len(tabs) == 1:
        return tabs[0][1]
    gid = f"tg{next(_ids)}"
    buttons = "".join(
        f"<button class='tab{' active' if i == 0 else ''}' data-group='{gid}' data-tab='{gid}-{i}'>{html.escape(label)}</button>"
        for i, (label, _) in enumerate(tabs)
    )
    panels = "".join(
        f"<div class='panel{' active' if i == 0 else ''}' id='{gid}-{i}' data-group='{gid}'>{body}</div>"
        for i, (_, body) in enumerate(tabs)
    )
    return f"<div class='tabgroup'><div class='tabbar'>{buttons}</div>{panels}</div>"


def _plot(fig_json: str, *, height=360) -> str:
    div_id = f"plot{next(_ids)}"
    return (
        f'<div id="{div_id}" class="plot" style="height:{height}px"></div>'
        f"<script>(function(){{var f={fig_json};f.layout=Object.assign({{margin:{{t:48,r:24,b:48,l:56}},"
        f"paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{family:'system-ui,sans-serif',size:12}}}},f.layout||{{}});"
        f'Plotly.newPlot("{div_id}",f.data,f.layout,{{responsive:true,displaylogo:false}});}})();</script>'
    )


def _group_by(rows, key):
    groups: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    return groups


def _fig():
    try:
        import plotly.graph_objects as go
    except Exception:  # pragma: no cover
        return None
    return go


# --------------------------------------------------------------------------- per-suite layouts
def _layout_clock(sec):
    timers = sec["tables"].get("timers", [])
    tabs = []
    groups = _group_by(timers, "timer")
    all_tab = _table(timers, highlight=lambda r: r.get("error_steps") not in (0, None))
    tabs.append(("All timers", all_tab))
    for label, rows in groups.items():
        tabs.append(
            (
                label,
                _table(rows, highlight=lambda r: r.get("error_steps") not in (0, None)),
            )
        )
    return [("Timers", _tabs(tabs))]


def _layout_physics(sec):
    go = _fig()
    t = sec["tables"]
    out = []
    ego = t.get("ego_convergence", [])
    if ego:
        body = ""
        if go:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[r["dt"] for r in ego],
                    y=[r["max_pos_err_m"] for r in ego],
                    mode="lines+markers",
                    name="max |Δpos| [m]",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[r["dt"] for r in ego],
                    y=[r["max_speed_err_mps"] for r in ego],
                    mode="lines+markers",
                    name="max |Δv| [m/s]",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title="Ego: error vs finer reference (first order ⇒ ∝ dt)",
                xaxis_title="physics dt [s]",
                yaxis_title="m",
                yaxis2=dict(title="m/s", overlaying="y", side="right"),
                xaxis=dict(autorange="reversed"),
            )
            body += _plot(fig.to_json())
        dt_tabs = [("All dt", _table(ego))] + [
            (f"dt = {label}", _table(rows))
            for label, rows in _group_by(ego, "dt").items()
        ]
        body += _tabs(dt_tabs)
        out.append(("Ego convergence", body))
    lead = t.get("lead_convergence", [])
    if lead:
        out.append(("Lead (NPC) convergence", _table(lead)))
    alpha = t.get("filter_alpha", [])
    if alpha:
        body = ""
        if go:
            fig = go.Figure(
                go.Bar(x=[str(r["dt"]) for r in alpha], y=[r["alpha"] for r in alpha])
            )
            fig.update_layout(
                title="Acceleration filter gain α = dt / (τ + dt)",
                xaxis_title="dt [s]",
                yaxis_title="α",
            )
            body += _plot(fig.to_json(), height=300)
        out.append(("Filter α(dt)", body + _table(alpha)))
    stop = t.get("route_end_stop", [])
    if stop:
        body = ""
        if go:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="stop offset [m]",
                    x=[str(r["speed_mps"]) for r in stop],
                    y=[r["stop_offset_m"] for r in stop],
                )
            )
            fig.add_trace(
                go.Bar(
                    name="max decel [m/s²]",
                    x=[str(r["speed_mps"]) for r in stop],
                    y=[r["max_decel_mps2"] for r in stop],
                )
            )
            fig.update_layout(
                title="Route-end stop by approach speed",
                xaxis_title="speed [m/s]",
                barmode="group",
            )
            body += _plot(fig.to_json(), height=300)
        out.append(("Route-end stop", body + _table(stop)))
    return out


def _layout_rendering(sec):
    t = sec["tables"]
    out = []
    fps = t.get("fps_independence", [])
    if fps:

        def hl(r):
            return (not r.get("digest_equal")) or r.get("max_state_diff", 0) != 0

        mode_tabs = [("All configurations", _table(fps, highlight=hl))] + [
            (f"render_mode = {label}", _table(rows, highlight=hl))
            for label, rows in _group_by(fps, "render_mode").items()
        ]
        out.append(("FPS / render-mode independence", _tabs(mode_tabs)))
    video = t.get("record_video", [])
    if video:
        out.append(("RecordVideo probe", _table(video)))
    fp = t.get("footprints_by_size", [])
    if fp:
        go = _fig()
        body = ""
        if go and len(fp) > 1:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="ego px",
                    x=[str(r["size"]) for r in fp],
                    y=[r["ego_px"] for r in fp],
                )
            )
            fig.add_trace(
                go.Bar(
                    name="vehicle px",
                    x=[str(r["size"]) for r in fp],
                    y=[r["vehicle_px"] for r in fp],
                )
            )
            fig.update_layout(
                title="Pixel footprint by raster size (metric footprint constant)",
                xaxis_title="size [px]",
                barmode="group",
            )
            body += _plot(fig.to_json(), height=300)
        size_tabs = [("All sizes", _table(fp))] + [
            (f"{label} px", _table(rows))
            for label, rows in _group_by(fp, "size").items()
        ]
        out.append(("Footprints by size", body + _tabs(size_tabs)))
    return out


def _layout_replay(sec):
    t = sec["tables"]
    out = []
    golden = t.get("golden_snapshots", [])
    if golden:

        def hl(r):
            return (r.get("max_abs_diff") or 0) > 1e-7 or r.get(
                "length_actual"
            ) != r.get("length_expected")

        tabs = [("All snapshots", _table(golden, highlight=hl))] + [
            (r["snapshot"], _table([r], highlight=hl)) for r in golden
        ]
        out.append(("Golden snapshots", _tabs(tabs)))
    cross = t.get("cross_size", [])
    if cross:
        go = _fig()
        body = ""
        if go:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="ego",
                    x=[f"{r['size']} px" for r in cross],
                    y=[r["max_ego_pos_err_m"] for r in cross],
                )
            )
            fig.add_trace(
                go.Bar(
                    name="npc",
                    x=[f"{r['size']} px" for r in cross],
                    y=[r["max_npc_pos_err_m"] for r in cross],
                )
            )
            fig.update_layout(
                title="Max position error vs the 128 px run",
                yaxis_title="m",
                barmode="group",
            )
            body += _plot(fig.to_json(), height=300)
        out.append(("Cross-size agreement", body + _table(cross)))
    return out


def _layout_throughput(sec):
    go = _fig()
    t = sec["tables"]
    out = []
    rows = t.get("throughput", [])
    if rows:
        body = ""
        if go:
            fig = go.Figure()
            for mode, group in _group_by(rows, "render_mode").items():
                fig.add_trace(
                    go.Bar(
                        name=mode,
                        x=[f"{r['scene']} @ {r['size']} px" for r in group],
                        y=[r["ms_per_step"] for r in group],
                        error_y=dict(
                            type="data",
                            array=[
                                max(
                                    0.0,
                                    (r.get("p99_ms") or r["ms_per_step"])
                                    - r["ms_per_step"],
                                )
                                for r in group
                            ],
                            visible=True,
                        ),
                    )
                )
            fig.update_layout(
                title="env.step() wall time (bar = mean, whisker = p99)",
                yaxis_title="ms / step",
                barmode="group",
            )
            body += _plot(fig.to_json())
        cols = [
            "size",
            "render_mode",
            "scene",
            "steps",
            "ms_per_step",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
            "steps_per_s",
            "real_time_x",
            "reset_s",
            "resets_in_loop",
        ]
        by_size = [
            (f"{label} px", _table(rs, columns=cols))
            for label, rs in _group_by(rows, "size").items()
        ]
        by_mode = [
            (f"render_mode = {label}", _table(rs, columns=cols))
            for label, rs in _group_by(rows, "render_mode").items()
        ]
        by_scene = [
            (f"scene = {label}", _table(rs, columns=cols))
            for label, rs in _group_by(rows, "scene").items()
        ]
        body += _tabs(
            [("All", _table(rows, columns=cols))] + by_size + by_mode + by_scene
        )
        out.append(("Step cost", body))
    breakdown = t.get("phase_breakdown", {})
    if breakdown:
        cols = [
            "phase",
            "count",
            "mean_ms",
            "std_ms",
            "min_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
            "total_ms",
        ]
        tabs = []
        for label, phases in breakdown.items():
            inner = ""
            leaf = [
                p
                for p in phases
                if p["phase"]
                not in ("env.step", "env.simulate", "env.outcome", "env.reset")
            ]
            if go and leaf:
                leaf_sorted = sorted(leaf, key=lambda p: -p["mean_ms"])
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        name="mean",
                        x=[p["phase"] for p in leaf_sorted],
                        y=[p["mean_ms"] for p in leaf_sorted],
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name="p99",
                        x=[p["phase"] for p in leaf_sorted],
                        y=[p.get("p99_ms") for p in leaf_sorted],
                    )
                )
                fig.update_layout(
                    title=f"Phase cost per step — {label}",
                    yaxis_title="ms",
                    barmode="group",
                )
                inner += _plot(fig.to_json(), height=340)
            inner += _table(sorted(phases, key=lambda p: p["phase"]), columns=cols)
            tabs.append((label, inner))
        out.append(("Phase breakdown", _tabs(tabs)))
    multi = t.get("multi_rate", [])
    if multi:
        body = ""
        if go:
            fig = go.Figure()
            labels = [f"dt={r['physics_dt']} × k={r['action_repeat']}" for r in multi]
            fig.add_trace(
                go.Bar(
                    name="decisions/s",
                    x=labels,
                    y=[r["decisions_per_s"] for r in multi],
                )
            )
            fig.add_trace(
                go.Bar(
                    name="physics steps/s",
                    x=labels,
                    y=[r["physics_steps_per_s"] for r in multi],
                )
            )
            fig.update_layout(
                title="Multi-rate presets (control step 0.1 s)",
                yaxis_title="per wall second",
                barmode="group",
            )
            body += _plot(fig.to_json(), height=340)
        body += _table(multi)
        out.append(("Multi-rate", body))
    return out


LAYOUTS = {
    "clock": _layout_clock,
    "physics": _layout_physics,
    "rendering": _layout_rendering,
    "replay": _layout_replay,
    "throughput": _layout_throughput,
}


def _generic_layout(sec):
    out = []
    for name, table in sec["tables"].items():
        if isinstance(table, dict):
            out.append(
                (name, _tabs([(label, _table(rows)) for label, rows in table.items()]))
            )
        else:
            out.append((name, _table(table)))
    return out


# --------------------------------------------------------------------------- page
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1f2328;--muted:#6b7280;--line:#e5e7eb;--accent:#1f6feb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-size:14px;line-height:1.45}
a{color:var(--accent);text-decoration:none}
header.top{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid var(--line);padding:10px 28px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header.top h1{font-size:18px;margin:0 18px 0 0;font-weight:650;display:flex;align-items:center}header.top h1 .badge{margin-left:8px}
header.top nav{display:flex;flex-wrap:wrap}header.top nav a{font-weight:500;color:#374151;display:inline-flex;align-items:center;margin-right:18px}header.top nav a .badge{margin-left:6px}
header.top .prov{margin-left:auto;color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace}
main{max-width:1480px;margin:0 auto;padding:20px 28px 60px}
.cards{display:flex;flex-wrap:wrap;margin:16px -7px 28px}
.card{flex:1 1 260px;min-width:240px;max-width:420px;margin:7px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;border-top:4px solid #999;color:inherit;display:block}
.card h3{margin:0 0 6px;font-size:15px;display:flex;justify-content:space-between;align-items:center}
.card .kv{display:flex;justify-content:space-between;flex-wrap:wrap;font-size:13px;padding:3px 0;border-top:1px dashed var(--line)}
.card .kv code{color:var(--muted);font-size:11px;overflow-wrap:anywhere;margin-right:8px}.card .kv b{font-variant-numeric:tabular-nums;white-space:nowrap;margin-left:auto}
.card .counts{color:var(--muted);font-size:12px;margin:4px 0 8px}
section.suite{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:22px 0}
section.suite h2{margin:0 0 4px;font-size:20px;display:flex;align-items:center;gap:10px}
section.suite .meta{color:var(--muted);font-size:12px;margin-bottom:12px}
h3{font-size:15px;margin:22px 0 8px;color:#374151;text-transform:uppercase;letter-spacing:.04em}
.badge{color:#fff;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;letter-spacing:.03em;vertical-align:middle}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font-size:13px;width:100%;margin:6px 0 10px}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{background:#f3f4f6;font-weight:600;color:#374151;position:sticky;top:0;white-space:nowrap}
tbody tr:nth-child(even){background:#fafafa}tbody tr:hover{background:#f0f6ff}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}td.unit{color:var(--muted)}
td.detail{color:#4b5563;font-size:12px;max-width:480px}
table.metrics tr.st-FAIL td:first-child{box-shadow:inset 4px 0 0 #c62828}
table.metrics tr.st-WARN td:first-child{box-shadow:inset 4px 0 0 #ef6c00}
table.metrics tr.st-PASS td:first-child{box-shadow:inset 4px 0 0 #2e7d32}
table.metrics tr.st-INFO td:first-child{box-shadow:inset 4px 0 0 #cbd5e1}
tr.row-accent td{background:#fff7ed}
.delta-up{color:#b45309}.delta-down{color:#047857}
code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.muted{color:var(--muted)}
.tabgroup{margin:8px 0 14px}.tabbar{display:flex;flex-wrap:wrap;border-bottom:2px solid var(--line);margin-bottom:10px}
.tab{background:none;border:none;border-bottom:2px solid transparent;margin:0 4px -2px 0;padding:7px 12px;cursor:pointer;font:inherit;font-size:13px;color:#4b5563;border-radius:6px 6px 0 0}
.tab:hover{background:#f3f4f6}.tab.active{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.panel{display:none}.panel.active{display:block}
.plot{width:100%;max-width:1000px;margin:4px 0 10px}
details.prov{margin:6px 0 10px}details.prov summary{cursor:pointer;color:var(--muted);font-size:13px}
table.provt td:first-child{font-weight:600;white-space:nowrap;width:160px}table.provt td{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.cmp-platform_rounding td{color:var(--muted)}.cmp-behavioural td{background:#fdecea}.cmp-performance td{background:#fff7ed}.cmp-within_tolerance td{background:#f0fdf4}
"""

JS = """
document.addEventListener('click',function(e){var b=e.target.closest('.tab');if(!b)return;var g=b.dataset.group;
document.querySelectorAll(".tab[data-group='"+g+"']").forEach(function(x){x.classList.remove('active')});
document.querySelectorAll(".panel[data-group='"+g+"']").forEach(function(x){x.classList.remove('active')});
b.classList.add('active');var p=document.getElementById(b.dataset.tab);p.classList.add('active');
if(window.Plotly){p.querySelectorAll('.plot').forEach(function(el){try{Plotly.Plots.resize(el)}catch(err){}})}});
"""


def _summary_cards(report: dict, baseline_index) -> str:
    cards = []
    for s in report["sections"]:
        counts = {}
        for m in s["metrics"]:
            counts[m["status"]] = counts.get(m["status"], 0) + 1
        idx = {m["name"]: m for m in s["metrics"]}
        kv = []
        for name in HEADLINE_METRICS.get(s["suite"], ()):
            m = idx.get(name)
            if m is None:
                continue
            unit = (
                f" <span class='muted'>{html.escape(m['unit'])}</span>"
                if m["unit"]
                else ""
            )
            kv.append(
                f"<div class='kv'><code>{html.escape(name)}</code><b>{_fmt(m['value'])}{unit}</b></div>"
            )
        counts_txt = " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        cards.append(
            f"<a class='card' href='#suite-{s['suite']}' style='border-top-color:{STATUS_COLOR[s['verdict']]}'>"
            f"<h3>{SUITE_ICON.get(s['suite'], '')} {html.escape(s['suite'])} {_badge(s['verdict'])}</h3>"
            f"<div class='counts'>{counts_txt} · {s['wall_time_s']:.1f} s</div>{''.join(kv)}</a>"
        )
    return f"<div class='cards'>{''.join(cards)}</div>"


def _provenance(prov: dict) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v))}</td></tr>"
        for k, v in prov.items()
    )
    return f"<details class='prov'><summary>Provenance (host, versions, configuration)</summary><table class='provt'>{rows}</table></details>"


def _compare_table(rows: list[dict]) -> str:
    counts = {}
    for r in rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    summary = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    body = "".join(
        f"<tr class='cmp-{r['classification']}'><td><code>{html.escape(r['metric'])}</code></td><td class='num'>{_fmt(r['a'])}</td>"
        f"<td class='num'>{_fmt(r['b'])}</td><td class='num'>{_fmt(r['delta'])}</td><td>{html.escape(r['classification'])}</td></tr>"
        for r in rows
        if r["classification"] != "equal"
    )
    tabs = [
        (
            "Differences",
            f"<p class='muted'>{summary}</p><div class='tablewrap'><table><thead><tr><th>metric</th><th>baseline</th><th>this run</th><th>Δ</th><th>class</th></tr></thead><tbody>{body}</tbody></table></div>",
        ),
    ]
    for cls in (
        "behavioural",
        "performance",
        "within_tolerance",
        "platform_rounding",
        "missing",
    ):
        sub = [r for r in rows if r["classification"] == cls]
        if sub:
            tabs.append(
                (
                    f"{cls} ({len(sub)})",
                    _table(sub, columns=["metric", "a", "b", "delta"]),
                )
            )
    return _tabs(tabs)


def render_html(
    report: dict,
    *,
    baseline: dict | None = None,
    compare_rows: list[dict] | None = None,
) -> str:
    prov = report["provenance"]
    baseline_index = None
    if baseline:
        baseline_index = {
            s["suite"]: {m["name"]: m for m in s["metrics"]}
            for s in baseline["sections"]
        }
    try:
        import plotly

        plotly_tag = f"<script>{plotly.offline.get_plotlyjs()}</script>"
    except Exception:  # pragma: no cover
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    nav = "".join(
        f"<a href='#suite-{s['suite']}'>{html.escape(s['suite'])} {_badge(s['verdict'])}</a>"
        for s in report["sections"]
    )
    if compare_rows:
        nav += "<a href='#compare'>compare</a>"
    prov_short = f"{prov.get('platform', '')} · {prov.get('config', {}).get('profile', '')} · {prov.get('git_sha') or 'no git'} · {prov.get('timestamp_utc', '')}"

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CarlaBEV benchmark</title>",
        f"<style>{CSS}</style>{plotly_tag}</head><body>",
        f"<header class='top'><h1>CarlaBEV benchmark {_badge(report['verdict'])}</h1><nav>{nav}</nav><div class='prov'>{html.escape(prov_short)}</div></header>",
        "<main>",
        _provenance(prov),
        _summary_cards(report, baseline_index),
    ]
    for s in report["sections"]:
        layout = LAYOUTS.get(s["suite"], _generic_layout)(s)
        parts.append(f"<section class='suite' id='suite-{s['suite']}'>")
        parts.append(
            f"<h2>{SUITE_ICON.get(s['suite'], '')} {html.escape(s['title'])} {_badge(s['verdict'])}</h2>"
        )
        parts.append(
            f"<div class='meta'>{len(s['metrics'])} metrics · {s['wall_time_s']:.1f} s wall time</div>"
        )
        parts.append("<h3>Metrics</h3>")
        parts.append(
            _metrics_table(
                s["metrics"], baseline_index.get(s["suite"]) if baseline_index else None
            )
        )
        if layout:
            parts.append("<h3>Details</h3>")
            parts.append(_tabs(layout))
        parts.append("</section>")
    if compare_rows:
        a_plat = (baseline or {}).get("provenance", {}).get("platform", "baseline")
        parts.append(
            f"<section class='suite' id='compare'><h2>Comparison vs baseline</h2><div class='meta'>baseline: {html.escape(str(a_plat))}</div>{_compare_table(compare_rows)}</section>"
        )
    parts.append(f"</main><script>{JS}</script></body></html>")
    return "".join(parts)
