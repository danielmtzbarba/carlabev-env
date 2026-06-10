import math

DEFAULT_COMFORT_BOUNDS = {
    "accel_long": 2.0,
    "accel_lat": 2.0,
    "yaw_rate": 20.0,
    "jerk_long": 3.0,
    "jerk_lat": 3.0,
    "yaw_acc": 120.0,
}


def _angle_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def compute_comfort_kinematics(
    *,
    speed_px_s: float,
    prev_speed_px_s: float,
    yaw_rad: float,
    prev_yaw_rad: float,
    dt: float,
    meters_per_pixel: float,
    prev_accel_long: float | None = None,
    prev_accel_lat: float | None = None,
    prev_yaw_rate_deg: float | None = None,
):
    speed_mps = float(speed_px_s) * meters_per_pixel
    prev_speed_mps = float(prev_speed_px_s) * meters_per_pixel

    yaw_rate_rad = _angle_delta(float(yaw_rad), float(prev_yaw_rad)) / dt
    yaw_rate_deg = math.degrees(yaw_rate_rad)

    accel_long = (speed_mps - prev_speed_mps) / dt
    accel_lat = speed_mps * yaw_rate_rad

    if prev_accel_long is None:
        jerk_long = 0.0
    else:
        jerk_long = (accel_long - float(prev_accel_long)) / dt

    if prev_accel_lat is None:
        jerk_lat = 0.0
    else:
        jerk_lat = (accel_lat - float(prev_accel_lat)) / dt

    if prev_yaw_rate_deg is None:
        yaw_acc = 0.0
    else:
        yaw_acc = (yaw_rate_deg - float(prev_yaw_rate_deg)) / dt

    return {
        "speed_mps": speed_mps,
        "accel_long": accel_long,
        "accel_lat": accel_lat,
        "jerk_long": jerk_long,
        "jerk_lat": jerk_lat,
        "yaw_rate": yaw_rate_deg,
        "yaw_acc": yaw_acc,
    }


def count_comfort_violations(metrics: dict, bounds: dict | None = None):
    bounds = DEFAULT_COMFORT_BOUNDS if bounds is None else bounds
    flags = {
        name: abs(float(metrics.get(name, 0.0))) > float(limit)
        for name, limit in bounds.items()
    }
    return sum(int(flag) for flag in flags.values()), flags
