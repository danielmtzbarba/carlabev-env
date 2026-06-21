from CarlaBEV.src.planning.town01_canonical_graph import (
    build_town01_canonical_vehicle_graph,
    summarize_canonical_graph,
    validate_town01_canonical_vehicle_graph,
)


def test_town01_canonical_graph_structure():
    graph = build_town01_canonical_vehicle_graph()
    problems = validate_town01_canonical_vehicle_graph(graph)
    assert not problems

    summary = summarize_canonical_graph(graph)
    assert summary.node_count > 700
    assert summary.edge_count > summary.node_count
    assert summary.junction_count > 0
    assert summary.junction_connector_count > 0
    assert summary.flow_groups["forward_a"] > 0
    assert summary.flow_groups["forward_b"] > 0


def test_town01_canonical_graph_node_and_edge_schema():
    graph = build_town01_canonical_vehicle_graph()
    node_id, node_data = next(iter(graph.nodes(data=True)))
    assert node_id == node_data["node_id"]
    assert len(node_data["pos_raw"]) == 2
    assert len(node_data["pos_surface"]) == 2
    assert node_data["flow_group"] in {"forward_a", "forward_b", "unknown"}
    assert isinstance(node_data["corridor_id"], str)

    _, _, edge_data = next(iter(graph.edges(data=True)))
    assert edge_data["edge_kind"] in {
        "forward",
        "junction_connector",
        "lane_change_candidate",
    }
    assert edge_data["turn_class"] in {"straight", "left", "right", "unknown"}
    assert edge_data["length_m"] >= 0.0
    assert edge_data["cost_m"] >= 0.0
