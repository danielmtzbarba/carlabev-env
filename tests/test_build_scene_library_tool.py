from CarlaBEV.tools.build_scene_library import _derive_structured_seed


def test_derive_structured_seed_is_deterministic():
    seed_a = _derive_structured_seed(
        study_id="PPO_NAVIGATION",
        backbone_id="traffic_fixed_train",
        study_seed=7,
        episode_index=42,
        kind="scene",
    )
    seed_b = _derive_structured_seed(
        study_id="PPO_NAVIGATION",
        backbone_id="traffic_fixed_train",
        study_seed=7,
        episode_index=42,
        kind="scene",
    )

    assert seed_a == seed_b


def test_derive_structured_seed_varies_by_kind():
    scene_seed = _derive_structured_seed(
        study_id="PPO_NAVIGATION",
        backbone_id="traffic_fixed_train",
        study_seed=7,
        episode_index=42,
        kind="scene",
    )
    route_seed = _derive_structured_seed(
        study_id="PPO_NAVIGATION",
        backbone_id="traffic_fixed_train",
        study_seed=7,
        episode_index=42,
        kind="route",
    )

    assert scene_seed != route_seed
