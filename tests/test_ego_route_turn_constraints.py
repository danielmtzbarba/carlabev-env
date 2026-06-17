import networkx as nx

from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.src.scenes.utils import find_route_in_range


def _planner_from_graph(graph):
    planner = GraphPlanner(graph)
    return {
        "vehicle-canonical": planner,
    }


def test_semantic_turn_metrics_count_junction_turns():
    graph = nx.DiGraph()
    graph.add_node("A", pos_raw=(0.0, 0.0), node_kind="lane", legacy_lane="L")
    graph.add_node("B", pos_raw=(10.0, 0.0), node_kind="lane", legacy_lane="L")
    graph.add_node("C", pos_raw=(20.0, 0.0), node_kind="lane", legacy_lane="L")
    graph.add_node("D", pos_raw=(20.0, 10.0), node_kind="lane", legacy_lane="L")
    graph.add_edge("A", "B", edge_kind="forward", turn_class="straight", length_m=5.0, cost_m=5.0)
    graph.add_edge("B", "C", edge_kind="junction_connector", turn_class="left", length_m=3.0, cost_m=3.0)
    graph.add_edge("C", "D", edge_kind="forward", turn_class="straight", length_m=5.0, cost_m=5.0)

    planner = _planner_from_graph(graph)["vehicle-canonical"]
    metrics = planner.compute_path_route_metrics(["A", "B", "C", "D"])

    assert metrics["turn_count"] == 1
    assert metrics["has_left_turn"] is True
    assert metrics["has_right_turn"] is False
    assert metrics["intersection_like"] is True
    assert metrics["route_profile"] == "single_left"


def test_find_route_in_range_applies_semantic_min_turns_for_ego():
    graph = nx.DiGraph()
    for index, x in enumerate((0.0, 10.0, 20.0, 30.0)):
        graph.add_node(f"S{index}", pos_raw=(x, 0.0), node_kind="lane", legacy_lane="L")
    graph.add_node("T0", pos_raw=(0.0, 10.0), node_kind="lane", legacy_lane="L")
    graph.add_node("T1", pos_raw=(10.0, 10.0), node_kind="lane", legacy_lane="L")
    graph.add_node("T2", pos_raw=(20.0, 20.0), node_kind="lane", legacy_lane="L")
    graph.add_node("T3", pos_raw=(30.0, 20.0), node_kind="lane", legacy_lane="L")

    graph.add_edge("S0", "S1", edge_kind="forward", turn_class="straight", length_m=5.0, cost_m=5.0)
    graph.add_edge("S1", "S2", edge_kind="forward", turn_class="straight", length_m=5.0, cost_m=5.0)
    graph.add_edge("S2", "S3", edge_kind="forward", turn_class="straight", length_m=5.0, cost_m=5.0)

    graph.add_edge("T0", "T1", edge_kind="forward", turn_class="straight", length_m=5.0, cost_m=5.0)
    graph.add_edge("T1", "T2", edge_kind="junction_connector", turn_class="left", length_m=3.0, cost_m=3.0)
    graph.add_edge("T2", "T3", edge_kind="forward", turn_class="straight", length_m=5.0, cost_m=5.0)

    planners = _planner_from_graph(graph)

    class StubRng:
        def __init__(self, picks):
            self._picks = list(picks)

        def choice(self, options):
            value = self._picks.pop(0)
            assert value in options
            return value

    actor, path, _, route_metrics = find_route_in_range(
        planners,
        "agent",
        "vehicle",
        min_dist_meters=0.0,
        max_dist_meters=100.0,
        max_attempts=2,
        rng=StubRng(["S0", "S3", "T0", "T3"]),
        min_turns=1,
        planner_key="vehicle-canonical",
        node_cls="vehicle",
        turn_metric_mode="semantic",
    )

    assert actor is not None
    assert path is not None
    assert route_metrics["turn_count"] == 1
    assert route_metrics["route_profile"] == "single_left"
