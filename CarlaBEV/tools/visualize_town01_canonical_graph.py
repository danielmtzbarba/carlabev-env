from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import plotly.graph_objects as go
from PIL import Image
import numpy as np

from CarlaBEV.envs.utils import asset_path
from CarlaBEV.src.planning.town01_canonical_graph import (
    DEFAULT_GRAPH_NAME,
    DEFAULT_TOWN_NAME,
    build_town01_canonical_vehicle_graph,
    junction_centroids,
    summarize_canonical_graph,
    validate_junction_joins,
)


EDGE_STYLES = {
    ("forward", "forward_a"): {"color": "#1f77b4", "width": 1.4, "name": "Forward A"},
    ("forward", "forward_b"): {"color": "#ff7f0e", "width": 1.4, "name": "Forward B"},
    ("junction_connector", "forward_a"): {"color": "#d62728", "width": 2.2, "name": "Junction Connectors"},
    ("junction_connector", "forward_b"): {"color": "#d62728", "width": 2.2, "name": "Junction Connectors"},
    ("lane_change_candidate", "forward_a"): {"color": "#9467bd", "width": 1.2, "name": "Lane Change Candidates"},
    ("lane_change_candidate", "forward_b"): {"color": "#9467bd", "width": 1.2, "name": "Lane Change Candidates"},
}

NODE_KIND_STYLES = {
    "lane": {"color": "#111111", "size": 5, "name": "Lane Nodes"},
    "junction_entry": {"color": "#2ca02c", "size": 8, "name": "Junction Entries"},
    "junction_inner": {"color": "#17becf", "size": 8, "name": "Junction Inner"},
    "junction_exit": {"color": "#bcbd22", "size": 8, "name": "Junction Exits"},
    "dead_end": {"color": "#7f7f7f", "size": 8, "name": "Dead Ends"},
}

FLOW_ARROW_STYLES = {
    "forward_a": {"color": "#1f77b4", "name": "Flow A Arrows"},
    "forward_b": {"color": "#ff7f0e", "name": "Flow B Arrows"},
}


def _canonical_graph_path(town_name: str = DEFAULT_TOWN_NAME) -> Path:
    return Path(asset_path) / town_name / DEFAULT_GRAPH_NAME


def _load_or_build_graph(*, rebuild: bool = False):
    path = _canonical_graph_path()
    if path.exists() and not rebuild:
        with path.open("rb") as handle:
            return pickle.load(handle)
    return build_town01_canonical_vehicle_graph()


def _background_image_array(
    town_name: str = DEFAULT_TOWN_NAME,
    *,
    max_dim: int = 1400,
) -> tuple[np.ndarray, float, float] | None:
    image_path = Path(asset_path) / town_name / f"{town_name}-1024-rgb.png"
    if not image_path.exists():
        return None
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = min(1.0, max_dim / max(width, height))
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized), width / resized.size[0], height / resized.size[1]


def _edge_trace(
    graph,
    edge_kind: str,
    flow_group: str,
    *,
    junction_id: str | None = None,
) -> go.Scattergl | None:
    xs: list[float | None] = []
    ys: list[float | None] = []
    hover_text: list[str | None] = []
    count = 0
    for src, dst, data in graph.edges(data=True):
        src_data = graph.nodes[src]
        if data.get("edge_kind") != edge_kind:
            continue
        if junction_id is not None:
            src_junction = src_data.get("junction_id")
            dst_junction = graph.nodes[dst].get("junction_id")
            if str(src_junction) != junction_id and str(dst_junction) != junction_id:
                continue
        if src_data.get("flow_group") != flow_group and edge_kind != "junction_connector":
            continue
        x0, y0 = src_data["pos_raw"]
        x1, y1 = graph.nodes[dst]["pos_raw"]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        text = (
            f"{src} -> {dst}<br>"
            f"edge_kind={data['edge_kind']}<br>"
            f"turn_class={data['turn_class']}<br>"
            f"length_m={data['length_m']:.2f}<br>"
            f"confidence={data['confidence']:.2f}"
        )
        hover_text.extend([text, text, None])
        count += 1
    if count == 0:
        return None
    style = EDGE_STYLES[(edge_kind, flow_group)]
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": style["color"], "width": style["width"]},
        name=f"{style['name']} ({count})",
        hoverinfo="text",
        text=hover_text,
        opacity=0.7 if edge_kind == "forward" else 0.9,
    )


def _node_trace(graph, node_kind: str, *, junction_id: str | None = None) -> go.Scattergl | None:
    xs = []
    ys = []
    hover = []
    for node_id, data in graph.nodes(data=True):
        if data.get("node_kind") != node_kind:
            continue
        if junction_id is not None and str(data.get("junction_id")) != junction_id:
            continue
        x, y = data["pos_raw"]
        xs.append(x)
        ys.append(y)
        hover.append(
            "<br>".join(
                [
                    f"node_id={node_id}",
                    f"corridor_id={data['corridor_id']}",
                    f"flow_group={data['flow_group']}",
                    f"junction_id={data['junction_id']}",
                    f"heading_rad={data['heading_rad']}",
                    f"confidence={data['confidence']:.2f}",
                ]
            )
        )
    if not xs:
        return None
    style = NODE_KIND_STYLES[node_kind]
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="markers",
        marker={
            "size": style["size"],
            "color": style["color"],
            "line": {"width": 0.5, "color": "white"},
        },
        name=f"{style['name']} ({len(xs)})",
        hoverinfo="text",
        text=hover,
    )


def _suspicious_join_trace(graph, *, junction_id: str | None = None) -> go.Scattergl | None:
    records = [record for record in validate_junction_joins(graph) if record.is_suspicious]
    if junction_id is not None:
        records = [record for record in records if record.junction_id == junction_id]
    if not records:
        return None

    xs: list[float | None] = []
    ys: list[float | None] = []
    hover_text: list[str | None] = []
    for record in records:
        x0, y0 = graph.nodes[record.src]["pos_raw"]
        x1, y1 = graph.nodes[record.dst]["pos_raw"]
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
        text = (
            f"{record.src} -> {record.dst}<br>"
            f"junction_id={record.junction_id}<br>"
            f"src_side={record.src_side}<br>"
            f"dst_side={record.dst_side}<br>"
            f"turn_class={record.turn_class}<br>"
            f"lateral_error={record.lateral_error:.1f}<br>"
            f"reason={record.reason}"
        )
        hover_text.extend([text, text, None])

    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": "#ff00aa", "width": 3.2},
        name=f"Suspicious Joins ({len(records)})",
        hoverinfo="text",
        text=hover_text,
        opacity=1.0,
    )


def _direction_arrow_trace(
    graph,
    flow_group: str,
    *,
    junction_id: str | None = None,
    step: int = 4,
) -> go.Scattergl | None:
    xs = []
    ys = []
    hover = []

    lane_nodes = []
    for node_id, data in graph.nodes(data=True):
        if data.get("flow_group") != flow_group:
            continue
        if data.get("node_kind") == "dead_end":
            continue
        if junction_id is not None and str(data.get("junction_id")) != junction_id:
            continue
        heading = data.get("heading_rad")
        if heading is None:
            continue
        lane_nodes.append((node_id, data))

    lane_nodes.sort(
        key=lambda item: (
            str(item[1].get("corridor_id")),
            -1 if item[1].get("legacy_index") is None else int(item[1].get("legacy_index")),
        )
    )

    for idx, (node_id, data) in enumerate(lane_nodes):
        if idx % step != 0:
            continue
        x, y = data["pos_raw"]
        heading = float(data["heading_rad"])
        shaft_len = 70.0
        head_len = 24.0
        head_angle = np.deg2rad(28.0)

        tip_x = x + shaft_len * np.cos(heading)
        tip_y = y + shaft_len * np.sin(heading)
        left_x = tip_x - head_len * np.cos(heading - head_angle)
        left_y = tip_y - head_len * np.sin(heading - head_angle)
        right_x = tip_x - head_len * np.cos(heading + head_angle)
        right_y = tip_y - head_len * np.sin(heading + head_angle)

        xs.extend([x, tip_x, None, tip_x, left_x, None, tip_x, right_x, None])
        ys.extend([y, tip_y, None, tip_y, left_y, None, tip_y, right_y, None])
        hover.append(
            "<br>".join(
                [
                    f"node_id={node_id}",
                    f"flow_group={flow_group}",
                    f"corridor_id={data['corridor_id']}",
                    f"heading_deg={np.degrees(heading):.1f}",
                ]
            )
        )

    if not xs:
        return None

    style = FLOW_ARROW_STYLES[flow_group]
    repeated_hover = []
    for text in hover:
        repeated_hover.extend([text, text, None, text, text, None, text, text, None])

    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": style["color"], "width": 2.4},
        name=f"{style['name']} ({len(xs)})",
        hoverinfo="text",
        text=repeated_hover,
        opacity=0.95,
    )


def build_figure(
    *,
    rebuild: bool = False,
    show_background: bool = True,
    junction_id: str | None = None,
    show_suspicious: bool = True,
    show_arrows: bool = True,
) -> go.Figure:
    graph = _load_or_build_graph(rebuild=rebuild)
    summary = summarize_canonical_graph(graph)
    traces = []
    if show_background:
        background = _background_image_array()
        if background is not None:
            background_img, dx, dy = background
            traces.append(
                go.Image(
                    z=background_img,
                    x0=0,
                    y0=0,
                    dx=dx,
                    dy=dy,
                    opacity=0.45,
                    name="Town01 RGB Map",
                    hoverinfo="skip",
                )
            )
    for edge_kind, flow_group in (
        ("forward", "forward_a"),
        ("forward", "forward_b"),
        ("junction_connector", "forward_a"),
        ("lane_change_candidate", "forward_a"),
    ):
        trace = _edge_trace(graph, edge_kind, flow_group, junction_id=junction_id)
        if trace is not None:
            traces.append(trace)
    if show_suspicious:
        trace = _suspicious_join_trace(graph, junction_id=junction_id)
        if trace is not None:
            traces.append(trace)
    if show_arrows:
        for flow_group in ("forward_a", "forward_b"):
            trace = _direction_arrow_trace(graph, flow_group, junction_id=junction_id)
            if trace is not None:
                traces.append(trace)
    for node_kind in NODE_KIND_STYLES:
        trace = _node_trace(graph, node_kind, junction_id=junction_id)
        if trace is not None:
            traces.append(trace)

    fig = go.Figure(data=traces)
    title_suffix = f" | junction={junction_id}" if junction_id is not None else ""
    fig.update_layout(
        title=(
            "Town01 Canonical Vehicle Graph"
            f"{title_suffix}"
            f"<br><sup>nodes={summary.node_count}, edges={summary.edge_count}, "
            f"weak_components={summary.weak_component_count}, junctions={summary.junction_count}, "
            f"junction_connectors={summary.junction_connector_count}</sup>"
        ),
        template="plotly_white",
        width=1200,
        height=1000,
        legend={"itemsizing": "constant"},
        xaxis={"title": "Raw X", "scaleanchor": "y", "showgrid": False},
        yaxis={"title": "Raw Y", "autorange": "reversed", "showgrid": False},
        margin={"l": 30, "r": 30, "t": 80, "b": 30},
    )

    if junction_id is not None:
        center = junction_centroids(graph).get(junction_id)
        if center is not None:
            cx, cy = center
            fig.update_xaxes(range=[cx - 900, cx + 900])
            fig.update_yaxes(range=[cy + 900, cy - 900])
    else:
        fig.update_xaxes(range=[0, 8192])
        fig.update_yaxes(range=[10240, 0])
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly visualization of the Town01 canonical graph."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_log/town01_canonical_graph.html"),
        help="HTML file to write.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the canonical graph from legacy assets instead of loading the saved pickle.",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Disable the Town01 RGB background image overlay.",
    )
    parser.add_argument(
        "--junction-id",
        type=str,
        default=None,
        help="Focus the visualization on a single junction_id such as intersection_0.",
    )
    parser.add_argument(
        "--hide-suspicious",
        action="store_true",
        help="Hide the suspicious join overlay.",
    )
    parser.add_argument(
        "--hide-arrows",
        action="store_true",
        help="Hide directional lane arrows.",
    )
    args = parser.parse_args()

    fig = build_figure(
        rebuild=args.rebuild,
        show_background=not args.no_background,
        junction_id=args.junction_id,
        show_suspicious=not args.hide_suspicious,
        show_arrows=not args.hide_arrows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
