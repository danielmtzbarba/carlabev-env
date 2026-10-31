"""UNITS / metric frame: one MetricFrame defines the surface scale; footprints are metric.

Expected: all conversions derive from MetricFrame (surface 128 px <-> 40 m FOV,
0.3125 m/px, raw scale 8); the runtime refuses raster sizes for which the frame
is undefined (U-02); actor footprints are declared in meters and derived to px (U-03).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs import geometry
from CarlaBEV.envs.geometry import (
    DEFAULT_METRIC_FRAME,
    SURFACE_METERS_PER_PIXEL,
    MetricFrame,
)
from CarlaBEV.envs.utils import load_map
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.units import ActorFootprints, UnitContract

from physics_time_helpers import RASTER_SIZES, make_ego, make_env


def test_default_raster_metric_scale():
    assert math.isclose(SURFACE_METERS_PER_PIXEL, 40.0 / 128.0, rel_tol=1e-12)
    arr, _, _ = load_map("Town01", 128)
    height_px, width_px, _ = arr.shape
    assert (width_px, height_px) == (1024, 1280)


def test_metric_frame_is_single_source_of_conversion_constants():
    frame = DEFAULT_METRIC_FRAME
    assert (
        frame.surface_size == 128
        and frame.raw_size == 1024
        and frame.visible_meters == 40.0
    )
    assert geometry.SURFACE_METERS_PER_PIXEL == frame.meters_per_pixel
    assert geometry.RAW_TO_SURFACE_SCALE == frame.raw_to_surface_scale == 8.0
    assert geometry.RAW_METERS_PER_PIXEL == frame.raw_meters_per_pixel
    assert geometry.distance_meters_to_surface(1.0) == frame.pixels_per_meter == 3.2
    raw = np.array([1564.0, 8642.0])
    assert np.allclose(geometry.raw_to_meters(raw), frame.raw_to_meters(raw))
    assert np.allclose(frame.meters_to_raw(frame.raw_to_meters(raw)), raw)


def test_physical_map_extent_matches_contract_at_supported_size():
    """U-02 (resolved by contract): 1024 x 1280 px at 0.3125 m/px = 320 x 400 m."""
    env = make_env(size=128)
    try:
        assert env.map.meters_per_pixel == SURFACE_METERS_PER_PIXEL
        assert np.allclose(env.map.map_extent_meters(), (320.0, 400.0))
    finally:
        env.close()


@pytest.mark.parametrize("size", RASTER_SIZES)
def test_supported_sizes_get_a_parametric_metric_frame(size):
    """U-02 / F-01: every asset size has a consistent frame (FOV = 40 m at any size)."""
    assert MetricFrame.supports_size(size)
    frame = UnitContract.from_env_config({"size": size}).metric
    assert frame.surface_size == size
    assert math.isclose(frame.meters_per_pixel * size, 40.0)
    assert math.isclose(frame.raw_to_surface_scale * size, 1024.0)


def test_runtime_refuses_raster_sizes_without_assets():
    assert not MetricFrame.supports_size(96)
    with pytest.raises(ValueError, match="not supported"):
        EnvConfig(render_mode="rgb_array", size=96)
    with pytest.raises(ValueError, match="not supported"):
        UnitContract.from_env_config({"size": 96})


def test_metric_frame_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        MetricFrame(surface_size=0)


def test_actor_footprints_are_expressed_in_meters():
    """U-03 (resolved): footprints are declared in meters and derived to pixels.

    Since Phase 6 the default profile is ``realistic`` (4.5 x 1.8 m cars, 0.3 m
    pedestrian circles); the legacy squares remain available explicitly.
    """
    vehicle = Vehicle(map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0])
    pedestrian = Pedestrian(map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0])
    ego = make_ego(5.0)
    realistic = ActorFootprints.realistic()
    assert vehicle.length_m == realistic.vehicle_length_m == 4.5 and vehicle.width_m == 1.8
    assert pedestrian.length_m == realistic.pedestrian_size_m == 0.6
    assert (ego.length_m, ego.width_m) == (4.5, 1.8)
    assert vehicle.size == 14 and pedestrian.size == 2 and (ego.w, ego.l) == (6, 14)

    legacy = UnitContract(footprints=ActorFootprints.legacy())
    vehicle = Vehicle(map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0], units=legacy)
    pedestrian = Pedestrian(map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0], units=legacy)
    ego = make_ego(5.0, units=legacy)
    assert vehicle.length_m == ActorFootprints().vehicle_length_m == 1.25
    assert pedestrian.length_m == 0.625 and ego.length_m == ego.width_m == 1.25
    # Legacy pixel footprints preserved at the default scale: 4 px, 2 px, 4 px.
    assert vehicle.size == 4 and pedestrian.size == 2 and (ego.w, ego.l) == (4, 4)


def test_footprint_pixels_follow_metric_definition():
    units = UnitContract(
        footprints=ActorFootprints.legacy(
            vehicle_length_m=2.5, vehicle_width_m=2.5, pedestrian_size_m=1.25
        )
    )
    vehicle = Vehicle(map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0], units=units)
    pedestrian = Pedestrian(
        map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0], units=units
    )
    assert vehicle.size == 8 and pedestrian.size == 4
    assert units.ego_footprint_px == (4, 4)
