"""GEOMETRY / collision system (Phase 2.2): SAT overlap, surface gap, swept TTC, classification.

Expected: overlap and gap computed on oriented boxes/circles in meters; TTC is the
time until footprints first overlap at constant velocity (inf if never);
contacts are classified rear_end / rear_ended / head_on / side / pedestrian.
"""

from __future__ import annotations

import math


from CarlaBEV.src.geometry.body import Body, Footprint
from CarlaBEV.src.geometry.collision import (
    CollisionSystem,
    bodies_overlap,
    box_box_overlap,
    classify_contact,
    gap_between,
    swept_overlap_time,
)

CAR = Footprint.car(4.5, 1.8, 2.9)
PED = Footprint.circle(0.3)
SQUARE = Footprint.square(1.25)


def _car(x, y, yaw=0.0, v=0.0):
    return Body(CAR, x, y, yaw, v)


def test_sat_detects_overlap_and_separation():
    a, b = (
        _car(0, 0).obb(),
        _car(4.0, 0.0).obb(),
    )  # centres 4 m apart, length 4.5 -> overlap 0.5 m
    overlapping, depth, _ = box_box_overlap(a, b)
    assert overlapping and math.isclose(depth, 0.5)
    assert not box_box_overlap(a, _car(5.0, 0.0).obb())[0]
    # rotated neighbour clipping the corner
    assert box_box_overlap(a, Body(CAR, 2.0, 1.6, math.pi / 2).obb())[0]


def test_gap_is_surface_to_surface_distance():
    assert math.isclose(gap_between(_car(0, 0), _car(10.0, 0)), 10.0 - 4.5)
    assert gap_between(_car(0, 0), _car(4.0, 0)) == 0.0
    # lateral neighbour in the next lane (centres 3.5 m apart, widths 1.8) -> 1.7 m gap
    assert math.isclose(
        gap_between(_car(0, 0), _car(0.0, 3.5)), 3.5 - 1.8, abs_tol=1e-9
    )
    # pedestrian circle 3 m ahead of the front bumper: centre offset 1.45, half length 2.25
    ped = Body(PED, 1.45 + 2.25 + 3.0 + 0.3, 0.0, 0.0)
    assert math.isclose(gap_between(_car(0, 0), ped), 3.0, abs_tol=1e-9)
    assert bodies_overlap(_car(0, 0), Body(PED, 2.0, 0.5, 0.0))


def test_legacy_squares_reproduce_axis_aligned_overlap():
    a = Body(SQUARE, 0.0, 0.0, 0.7)
    b = Body(SQUARE, 1.2, 0.0, -0.4)
    assert bodies_overlap(a, b)  # 1.25 m squares 1.2 m apart overlap regardless of yaw
    assert not bodies_overlap(a, Body(SQUARE, 1.3, 0.0, 0.0))


def test_swept_ttc_for_closing_and_diverging_bodies():
    ego = _car(0.0, 0.0, 0.0, 10.0)
    lead = _car(20.0, 0.0, 0.0, 4.0)  # gap 15.5 m, closing 6 m/s
    assert math.isclose(swept_overlap_time(ego, lead), 15.5 / 6.0, rel_tol=1e-9)
    assert (
        swept_overlap_time(ego, _car(20.0, 0.0, 0.0, 12.0)) == math.inf
    )  # pulling away
    assert (
        swept_overlap_time(ego, _car(20.0, 6.0, 0.0, 4.0)) == math.inf
    )  # other lane, never overlaps
    assert swept_overlap_time(ego, _car(4.0, 0.0)) == 0.0  # already overlapping
    # crossing pedestrian already at the lane edge: the ego front reaches it at
    # t = (12 - 0.3 - 3.7) / 10 = 0.8 s while it is inside the ego's lateral span
    ped = Body(PED, 12.0, -1.5, math.pi / 2, 1.5)
    assert math.isclose(swept_overlap_time(ego, ped), 0.8, rel_tol=1e-9)
    # a pedestrian too far from the lane reaches it only after the ego has passed
    assert swept_overlap_time(ego, Body(PED, 12.0, -6.0, math.pi / 2, 1.5)) == math.inf
    # circle-circle closed form
    t = swept_overlap_time(Body(PED, 0, 0, 0, 1.0), Body(PED, 2.6, 0, math.pi, 1.0))
    assert math.isclose(t, (2.6 - 0.6) / 2.0)


def test_contact_classification():
    ego = _car(0, 0, 0.0, 8.0)
    assert classify_contact(ego, _car(10.0, 0.0, 0.0, 5.0), "vehicle") == "rear_end"
    assert classify_contact(ego, _car(-10.0, 0.0, 0.0, 12.0), "vehicle") == "rear_ended"
    assert classify_contact(ego, _car(15.0, 0.0, math.pi, 8.0), "vehicle") == "head_on"
    assert classify_contact(ego, _car(5.0, 5.0, math.pi / 2, 8.0), "vehicle") == "side"
    assert classify_contact(ego, Body(PED, 5.0, 0.0, 0.0), "pedestrian") == "pedestrian"


def test_collision_system_contact_and_proximity():
    system = CollisionSystem(proximity_radius_m=10.9375)
    ego = _car(0, 0, 0.0, 10.0)
    contact = system.contact(ego, _car(12.0, 0.0, 0.0, 4.0), "vehicle")
    assert not contact.overlapping
    assert math.isclose(contact.gap_m, 12.0 - 4.5) and math.isclose(
        contact.center_distance_m, 12.0
    )
    assert math.isclose(contact.closing_speed_mps, 6.0) and math.isclose(
        contact.ttc_s, 7.5 / 6.0
    )
    assert contact.classification == "rear_end" and abs(contact.bearing_rad) < 1e-9
    assert not system.within_proximity(contact)  # centre distance 12 m > 10.94 m
    assert system.within_proximity(system.contact(ego, _car(10.0, 0.0), "vehicle"))
    info = contact.to_info()
    assert {
        "gap_m",
        "ttc_s",
        "classification",
        "bearing_rad",
        "closing_speed_mps",
    } <= set(info)


def test_broad_phase_candidates_filter_far_bodies():
    ego = _car(0, 0)
    near, far = _car(8.0, 0.0), _car(80.0, 0.0)
    out = CollisionSystem.candidates(ego, [("a", near), ("b", far)], margin_m=20.0)
    assert [label for label, _ in out] == ["a"]
