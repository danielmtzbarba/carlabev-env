"""Stop-line distances use the same occupied shape as physical contacts."""

import math
from types import SimpleNamespace

import pytest

from CarlaBEV.src.actors.traffic_policy import stop_line_ahead
from CarlaBEV.src.geometry.body import Body, Footprint


@pytest.mark.parametrize(
    "footprint,yaw,expected",
    [
        (Footprint.car(4.5, 1.8, 2.9), 0.0, 6.3),
        (Footprint.car(4.5, 1.8, 2.9), math.pi / 6, 6.345706005997577),
        (Footprint.car(6.0, 2.0, 4.0), 0.0, 5.0),
        (Footprint.square(1.25), math.pi / 6, 9.375),
        (Footprint.circle(0.3), 0.0, 9.7),
    ],
)
def test_stop_line_clearance_is_measured_from_occupied_front(footprint, yaw, expected):
    body = Body(footprint, 0.0, 0.0, yaw, 5.0)
    actor = SimpleNamespace(
        x_m=0.0,
        y_m=0.0,
        yaw=yaw,
        v_mps=5.0,
        length_m=footprint.length_m,
        body=lambda: body,
    )
    light = SimpleNamespace(controlled_yaw=0.0, stop_line_m=lambda: (10.0, 0.0, 3.0))
    selected, gap = stop_line_ahead(actor, [light], 20.0)
    assert selected is light
    assert gap == pytest.approx(expected)


def test_stop_line_remains_visible_during_crossing_then_disappears():
    body = Body(Footprint.car(4.5, 1.8, 2.9), 6.29, 0.0, 0.0)
    actor = SimpleNamespace(
        x_m=6.29, y_m=0.0, yaw=0.0, v_mps=5.0, length_m=4.5, body=lambda: body
    )
    light = SimpleNamespace(controlled_yaw=0.0, stop_line_m=lambda: (10.0, 0.0, 3.0))
    assert stop_line_ahead(actor, [light], 20.0)[1] == pytest.approx(0.01)
    actor.x_m = body.x_m = 6.31
    assert stop_line_ahead(actor, [light], 20.0)[1] == pytest.approx(-0.01)
    actor.x_m = body.x_m = 10.81  # rear bumper has cleared the line
    assert stop_line_ahead(actor, [light], 20.0) == (None, None)


def test_stop_line_span_intersects_vehicle_edge_even_when_reference_is_outside():
    body = Body(Footprint.car(4.5, 1.8, 2.9), 0.0, 3.5, 0.0)
    actor = SimpleNamespace(
        x_m=0.0, y_m=3.5, yaw=0.0, v_mps=5.0, length_m=4.5, body=lambda: body
    )
    light = SimpleNamespace(controlled_yaw=0.0, stop_line_m=lambda: (10.0, 0.0, 3.0))
    assert stop_line_ahead(actor, [light], 20.0)[1] == pytest.approx(6.3)
    actor.y_m = body.y_m = 4.0
    assert stop_line_ahead(actor, [light], 20.0) == (None, None)
