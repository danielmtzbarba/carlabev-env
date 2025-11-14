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

#CaRL

def compute_ttc_raw(hero_state, actors_state):
    """
    Extract the raw minimum TTC value from your logic,
    without shaping or exponentials.
    """
    hx, hy, hyaw, hv = hero_state
    hvx = hv * np.cos(hyaw)
    hvy = hv * np.sin(hyaw)
    min_ttc = np.inf

    for actor in actors_state:
        (ax, ay) = actor["pos"]
        (avx, avy) = actor["vel"]

        rx, ry = ax - hx, ay - hy
        rvx, rvy = avx - hvx, avy - hvy

        rel_speed = (rvx * rx + rvy * ry) / (np.linalg.norm([rx, ry]) + 1e-6)
        if rel_speed >= 0:
            continue  # actor moving away

        ttc = abs(np.linalg.norm([rx, ry]) / rel_speed)
        min_ttc = min(min_ttc, ttc)

    return min_ttc

def carl_ttc_penalty(hero_state, actors_state, threshold=1.0):
    """
    Returns p_ttc ∈ {1.0, 0.5} following CaRL soft penalty rules.
    """
    ttc = compute_ttc_raw(hero_state, actors_state)

    if ttc < threshold:
        return 0.5, ttc     # TTC violation → penalty factor 0.5
    else:
        return 1.0, ttc     # Safe → no penalty reduction
