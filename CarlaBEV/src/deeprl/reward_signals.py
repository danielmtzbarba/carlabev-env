import numpy as np


def proximity_shaping(distances, safe=8.0):
    # distances: list of euclidean distances to nearest vehicles/peds
    if not distances:
        return 0.0
    d = min(distances)
    if d >= safe:
        return 0.0
    # Exponential penalty inside "safe" bubble
    return -np.exp(-(d / (safe / 2.0))) * 0.5  # ∈ [-0.5, 0)


def compute_ttc(hero_state, actors_state, ttc_threshold=8.0):
    """
    Compute minimum Time-To-Collision across all actors.
    """
    hx, hy, hyaw, hv = hero_state
    hvx = hv * np.cos(hyaw)
    hvy = hv * np.sin(hyaw)
    min_ttc = np.inf

    for actor in actors_state:
        (ax, ay) = actor["pos"]
        (avx, avy) = actor["vel"]

        # relative position and velocity
        rx, ry = ax - hx, ay - hy
        rvx, rvy = avx - hvx, avy - hvy

        rel_speed = (rvx * rx + rvy * ry) / (np.linalg.norm([rx, ry]) + 1e-6)
        if rel_speed >= 0:
            continue  # actor moving away

        ttc = abs(np.linalg.norm([rx, ry]) / rel_speed)
        min_ttc = min(min_ttc, ttc)

    if min_ttc < np.inf:
        # Exponential safety shaping (the closer the collision, the stronger the penalty)
        return -np.exp(-min_ttc / ttc_threshold)
    return 0.0


# CaRL
def compute_ttc_raw(hero_state, actors_state, dt=0.1, meters_per_pixel=0.625):
    """
    Raw Time-To-Collision (TTC) in seconds using CaRL logic.
    No shaping. Returns min TTC across all actors.
    """

    hx, hy, hyaw, hv_px = hero_state

    # Convert hero position to meters
    hx_m = hx * meters_per_pixel
    hy_m = hy * meters_per_pixel

    # Convert hero speed: px/step → m/s
    hv_m = hv_px * meters_per_pixel / dt
    hvx_m = hv_m * np.cos(hyaw)
    hvy_m = hv_m * np.sin(hyaw)

    min_ttc = np.inf

    for actor in actors_state:
        (ax_px, ay_px) = actor["pos"]
        (avx_px, avy_px) = actor["vel"]

        # Convert actor position to meters
        ax_m = ax_px * meters_per_pixel
        ay_m = ay_px * meters_per_pixel

        # Convert actor velocity to m/s
        avx_m = avx_px * meters_per_pixel / dt
        avy_m = avy_px * meters_per_pixel / dt

        # Relative position and velocity
        rx = ax_m - hx_m
        ry = ay_m - hy_m
        rvx = avx_m - hvx_m
        rvy = avy_m - hvy_m

        # Relative speed (projected onto the line of sight)
        rel_speed = (rvx * rx + rvy * ry) / (np.linalg.norm([rx, ry]) + 1e-6)

        # If moving apart or parallel → skip
        if rel_speed >= 0:
            continue

        # Constant-velocity TTC
        ttc = abs(np.linalg.norm([rx, ry]) / rel_speed)
        min_ttc = min(min_ttc, ttc)

    return min_ttc


def carl_ttc_penalty(
    hero_state, actors_state, threshold=1.0, dt=0.1, meters_per_pixel=0.625
):
    """
    CaRL soft TTC penalty:
        p_ttc = 1.0 (safe)
        p_ttc = 0.5 (TTC < threshold)
    """
    ttc = compute_ttc_raw(
        hero_state, actors_state, dt=dt, meters_per_pixel=meters_per_pixel
    )

    if ttc < threshold:
        return 0.5, ttc  # TTC violation
    else:
        return 1.0, ttc  # Safe
