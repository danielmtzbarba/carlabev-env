import numpy as np

from CarlaBEV.src.control.stanley_controller import Controller


def test_controller_falls_back_to_raw_route_when_smoothed_path_leaves_drivable_mask():
    drivable_mask = np.zeros((128, 128), dtype=bool)

    route_x = [20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 76.0, 84.0, 84.0, 84.0, 84.0, 84.0, 84.0, 84.0]
    route_y = [84.0, 84.0, 84.0, 84.0, 84.0, 84.0, 84.0, 84.0, 84.0, 76.0, 68.0, 60.0, 52.0, 44.0, 36.0]
    for x, y in zip(route_x, route_y):
        drivable_mask[int(round(y)), int(round(x))] = True

    controller = Controller(target_speed=0.0)
    controller.set_route(route_x, route_y, v0=0.0, jitter_start=False, drivable_mask=drivable_mask)

    assert controller.route_smoothing_mode == "raw_fallback"
    assert controller.route_offroad_points > 0
    assert np.allclose(controller.cx, np.asarray(route_x, dtype=float))
    assert np.allclose(controller.cy, np.asarray(route_y, dtype=float))


def test_controller_keeps_smoothed_route_when_candidate_stays_drivable():
    drivable_mask = np.ones((128, 128), dtype=bool)
    route_x = [20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 76.0, 84.0, 92.0, 100.0]
    route_y = [64.0] * len(route_x)

    controller = Controller(target_speed=0.0)
    controller.set_route(route_x, route_y, v0=0.0, jitter_start=False, drivable_mask=drivable_mask)

    assert controller.route_smoothing_mode == "smoothed"
    assert controller.route_offroad_points == 0
