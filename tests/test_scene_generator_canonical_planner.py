from CarlaBEV.src.managers.scene_generator import PlannerManager, SceneGenerator


def test_planner_manager_vehicle_aliases_share_canonical_graph():
    planners = PlannerManager("Town01")

    canonical = planners.get("vehicle-canonical")
    assert canonical is not None
    assert planners.get("vehicle-full") is canonical
    assert planners.get("vehicle") is canonical
    assert planners.get("vehicle-L") is canonical
    assert planners.get("vehicle-R") is canonical
    assert planners.get("vehicle-flow-a") is canonical
    assert planners.get("vehicle-flow-b") is canonical


def test_scene_generator_resolves_ego_route_graph_aliases():
    assert SceneGenerator._resolve_ego_route_graph("full_vehicle") == (
        "vehicle-canonical",
        "vehicle",
        "vehicle",
    )
    assert SceneGenerator._resolve_ego_route_graph("canonical") == (
        "vehicle-canonical",
        "vehicle",
        "vehicle",
    )
    assert SceneGenerator._resolve_ego_route_graph("left_lane") == (
        "vehicle-flow-a",
        "L",
        "L",
    )
    assert SceneGenerator._resolve_ego_route_graph("flow_a") == (
        "vehicle-flow-a",
        "L",
        "L",
    )
    assert SceneGenerator._resolve_ego_route_graph("right_lane") == (
        "vehicle-flow-b",
        "R",
        "R",
    )
    assert SceneGenerator._resolve_ego_route_graph("flow_b") == (
        "vehicle-flow-b",
        "R",
        "R",
    )


def test_scene_generator_can_build_random_scene_with_canonical_graph():
    generator = SceneGenerator(
        config={
            "map_size": 128,
            "map_name": "Town01",
            "traffic_enabled": True,
        }
    )

    actors, route_length = generator.generate_random(
        num_cars=1,
        dist_range=(30, 80),
        traffic_enabled=True,
        ego_route_graph="canonical",
        max_retries=10,
    )

    assert actors["agent"] is not None
    assert route_length is not None
    assert route_length >= 30.0
    assert len(actors["agent"][0]) > 1
