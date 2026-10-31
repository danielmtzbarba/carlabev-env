"""GEOMETRY / body (Phase 2.1): metric footprints, reference points and oriented boxes.

Expected: a Footprint declares shape, size and reference point; Body derives the
centre (reference + offset along heading), oriented corners and AABB in meters;
legacy footprints are axis-aligned squares centred on the state regardless of yaw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from CarlaBEV.src.geometry.body import Body, Footprint, OrientedBox
from CarlaBEV.src.units import ActorFootprints, UnitContract


def test_car_footprint_reference_is_rear_axle_with_centre_offset():
    fp = Footprint.car(4.5, 1.8, 2.9)
    assert fp.ref_point == "rear_axle" and fp.oriented
    # rear overhang = (4.5 - 2.9)/2 = 0.8 -> centre is 1.45 m ahead of the rear axle
    assert math.isclose(fp.center_offset_m, 1.45)
    body = Body(fp, 10.0, 5.0, 0.0)
    assert np.allclose(body.center(), (11.45, 5.0))
    body = Body(fp, 10.0, 5.0, math.pi / 2)
    assert np.allclose(body.center(), (10.0, 6.45))


def test_oriented_box_corners_and_aabb():
    box = OrientedBox(0.0, 0.0, 2.25, 0.9, 0.0)
    corners = box.corners()
    assert np.allclose(
        corners, [[2.25, 0.9], [2.25, -0.9], [-2.25, -0.9], [-2.25, 0.9]]
    )
    assert np.allclose(box.aabb(), (-2.25, -0.9, 2.25, 0.9))
    rotated = OrientedBox(0.0, 0.0, 2.25, 0.9, math.pi / 2)
    assert np.allclose(rotated.aabb(), (-0.9, -2.25, 0.9, 2.25), atol=1e-12)
    assert math.isclose(box.radius, math.hypot(2.25, 0.9))


def test_legacy_square_ignores_yaw():
    fp = Footprint.square(1.25)
    assert not fp.oriented and fp.ref_point == "center" and fp.center_offset_m == 0.0
    a = Body(fp, 0.0, 0.0, 0.0).aabb()
    b = Body(fp, 0.0, 0.0, 1.0).aabb()
    assert np.allclose(a, b) and np.allclose(a, (-0.625, -0.625, 0.625, 0.625))


def test_circle_footprint():
    fp = Footprint.circle(0.3)
    assert fp.is_circle if hasattr(fp, "is_circle") else fp.shape == "circle"
    body = Body(fp, 1.0, 2.0, 0.7)
    assert np.allclose(body.aabb(), (0.7, 1.7, 1.3, 2.3))
    assert body.is_circle


def test_footprint_validation():
    with pytest.raises(ValueError):
        Footprint(0.0, 1.0)
    with pytest.raises(ValueError):
        Footprint(1.0, 1.0, shape="hexagon")
    with pytest.raises(ValueError):
        Footprint(1.0, 1.0, ref_point="front_axle")


def test_geometry_profiles():
    legacy = ActorFootprints.legacy()
    assert legacy.profile == "legacy"
    assert (
        legacy.ego.length_m == legacy.vehicle.length_m == 1.25
        and not legacy.ego.oriented
    )
    assert legacy.pedestrian.length_m == 0.625
    realistic = ActorFootprints.realistic()
    assert realistic.profile == "realistic"
    assert (realistic.ego.length_m, realistic.ego.width_m) == (
        4.5,
        1.8,
    ) and realistic.ego.ref_point == "rear_axle"
    assert realistic.pedestrian.shape == "circle" and math.isclose(
        realistic.pedestrian.radius_m, 0.3
    )
    assert set(realistic.vehicle_classes) == {"car", "van", "bus"}
    assert realistic.vehicle_footprint("bus").length_m == 10.5
    assert realistic.vehicle_footprint("unknown") is realistic.vehicle
    with pytest.raises(ValueError):
        ActorFootprints.from_profile("cartoon")


def test_unit_contract_selects_profile_and_pixel_footprints():
    legacy = UnitContract.from_env_config({"geometry_profile": "legacy"})
    realistic = UnitContract.from_env_config({"geometry_profile": "realistic"})
    assert legacy.geometry_profile == "legacy" and legacy.ego_footprint_px == (4, 4)
    assert realistic.geometry_profile == "realistic" and realistic.ego_footprint_px == (
        6,
        14,
    )  # 1.8 m x 4.5 m at 0.3125 m/px
    assert realistic.vehicle_footprint_px == (6, 14)
    assert math.isclose(realistic.footprints.ego.center_offset_m, 1.45)
