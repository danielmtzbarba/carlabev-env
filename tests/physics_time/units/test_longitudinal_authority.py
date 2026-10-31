"""UNITS / longitudinal: throttle, brake, resistance and filter are SI, raster-independent, configurable.

Expected: full throttle 4.5 m/s^2; brake authority 1.5 m/s^2 (behaviour-preserving
anchor) regardless of raster/window size; both configurable through EnvConfig /
EgoPhysics; resistance 0.05 + 0.002 v^2; first-order filter alpha = 0.5.
"""

from __future__ import annotations

import math

import pytest

from CarlaBEV.config import EnvConfig
from CarlaBEV.src.units import EgoPhysics, UnitContract

from physics_time_helpers import LEAD_BRAKE_OPTIONS, RASTER_SIZES, make_ego


def test_full_throttle_acceleration_is_documented_value():
    ego = make_ego(5.0)
    assert math.isclose(ego.accelerate(1.0), 4.5, rel_tol=1e-9)
    assert ego.accelerate(-1.0) == 0.0


def test_brake_authority_at_default_raster_is_preserved_value():
    ego = make_ego(10.0, window_size=128)
    assert math.isclose(ego.max_brake_mps2, 1.5, rel_tol=1e-9)
    assert math.isclose(ego.brake(1.0), 1.5, rel_tol=1e-9)


def test_brake_low_speed_factor():
    assert math.isclose(make_ego(2.0).brake(1.0), 1.5 * 0.4, rel_tol=1e-9)
    assert math.isclose(make_ego(0.5).brake(1.0), 1.5 * 0.3, rel_tol=1e-9)


def test_brake_authority_is_independent_of_raster_size():
    """U-01 (resolved): brake authority is EgoPhysics.brake_decel_mps2, not 0.6*scale*m/px."""
    values = {
        size: make_ego(10.0, window_size=size).max_brake_mps2 for size in RASTER_SIZES
    }
    assert all(math.isclose(v, 1.5, rel_tol=1e-9) for v in values.values()), values
    assert EgoPhysics().brake_decel_mps2 == 1.5


def test_longitudinal_authority_is_configurable_through_env_config():
    cfg = EnvConfig(
        render_mode="rgb_array",
        obs_mode="bev_rgb",
        action_mode="continuous",
        ego_brake_decel_mps2=4.0,
        ego_accel_mps2=3.0,
        scene_library_enabled=False,
    )
    units = UnitContract.from_env_config(cfg)
    assert units.ego.brake_decel_mps2 == 4.0 and units.ego.accel_mps2 == 3.0
    from CarlaBEV.envs.carlabev import CarlaBEV

    env = CarlaBEV(cfg)
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        hero = env.map.hero
        assert math.isclose(hero.max_brake_mps2, 4.0, rel_tol=1e-9)
        assert math.isclose(hero.accelerate(1.0), 3.0, rel_tol=1e-9)
        assert math.isclose(
            hero.brake(1.0), 4.0, rel_tol=1e-9
        )  # 10 m/s -> full authority
    finally:
        env.close()


@pytest.mark.parametrize(
    "field,value", [("ego_brake_decel_mps2", 0.0), ("ego_accel_mps2", -1.0)]
)
def test_env_config_rejects_non_positive_authority(field, value):
    with pytest.raises(ValueError):
        EnvConfig(render_mode="rgb_array", **{field: value})


def test_longitudinal_resistance_model():
    ego = make_ego(10.0)
    assert math.isclose(
        ego.longitudinal_resistance_accel(), 0.05 + 0.002 * 100.0, rel_tol=1e-9
    )
    ego.v_mps = 0.0
    assert ego.longitudinal_resistance_accel() == 0.0


def test_accel_filter_is_first_order_with_alpha_half():
    ego = make_ego(5.0)
    ego.acc = 0.0
    ego.physics_step(gas=1.0, steer=0.0, brake=0.0)
    target = 4.5 - (0.05 + 0.002 * 25.0)
    assert math.isclose(ego.acc, 0.5 * target, rel_tol=1e-9)


def test_no_raster_scale_in_ego_physics():
    """No rendering quantity may leak into dynamics."""
    ego = make_ego(5.0)
    assert not hasattr(ego, "scale")
    for name in (
        "max_accel_mps2",
        "max_brake_mps2",
        "rolling_resistance_mps2",
        "aero_drag_coeff",
    ):
        assert isinstance(getattr(ego, name), float)
