import math


LOOKAHEAD_M = 20.0
LANE_HALF_WIDTH_M = 2.25
FOLLOW_BUFFER_M = 4.0
CAUTION_GAP_M = 10.0
CRUISE_GAP_M = 18.0
TTC_BRAKE_S = 2.0
TTC_STOP_S = 1.0


def _relative_pose_metrics(actor, other_actor):
    dx = float(other_actor._controller.x_m - actor._controller.x_m)
    dy = float(other_actor._controller.y_m - actor._controller.y_m)
    cos_yaw = math.cos(actor._controller.yaw)
    sin_yaw = math.sin(actor._controller.yaw)
    forward_m = dx * cos_yaw + dy * sin_yaw
    lateral_m = -dx * sin_yaw + dy * cos_yaw
    return forward_m, lateral_m


def find_lead_actor(actor, other_actors, lookahead_m=LOOKAHEAD_M, lane_half_width_m=LANE_HALF_WIDTH_M):
    best_actor = None
    best_forward_m = None
    for other_actor in other_actors or ():
        if other_actor is actor or not hasattr(other_actor, "_controller"):
            continue
        forward_m, lateral_m = _relative_pose_metrics(actor, other_actor)
        if forward_m <= 0.0 or forward_m > lookahead_m:
            continue
        if abs(lateral_m) > lane_half_width_m:
            continue
        if best_forward_m is None or forward_m < best_forward_m:
            best_actor = other_actor
            best_forward_m = forward_m
    return best_actor, best_forward_m


def compute_follow_speed_mps(gap_m, closing_speed_mps, lead_speed_mps, cruise_speed_mps):
    if gap_m is None:
        return float(cruise_speed_mps)

    gap_m = max(0.0, float(gap_m) - FOLLOW_BUFFER_M)
    lead_speed_mps = max(0.0, float(lead_speed_mps))
    cruise_speed_mps = max(0.0, float(cruise_speed_mps))
    closing_speed_mps = max(0.0, float(closing_speed_mps))

    if closing_speed_mps > 1e-6:
        ttc_s = gap_m / closing_speed_mps
        if ttc_s <= TTC_STOP_S:
            return 0.0
        if ttc_s <= TTC_BRAKE_S:
            return min(cruise_speed_mps, lead_speed_mps * 0.5)

    if gap_m <= FOLLOW_BUFFER_M:
        return 0.0
    if gap_m <= CAUTION_GAP_M:
        return min(cruise_speed_mps, lead_speed_mps)
    if gap_m <= CRUISE_GAP_M:
        return min(cruise_speed_mps, lead_speed_mps + 1.0)
    return cruise_speed_mps


def compute_safe_target_speed_mps(actor, other_actors, nominal_target_speed_mps):
    nominal_target_speed_mps = max(0.0, float(nominal_target_speed_mps))
    lead_actor, lead_gap_m = find_lead_actor(actor, other_actors)
    debug = {
        "traffic_braking_active": False,
        "lead_gap_m": None,
        "lead_speed_mps": None,
        "lead_ttc_s": None,
        "nominal_target_speed_mps": nominal_target_speed_mps,
        "safe_target_speed_mps": nominal_target_speed_mps,
    }
    if lead_actor is None:
        return nominal_target_speed_mps, debug

    lead_speed_mps = float(lead_actor._controller.v_mps)
    closing_speed_mps = max(0.0, float(actor._controller.v_mps - lead_speed_mps))
    safe_target_speed_mps = compute_follow_speed_mps(
        lead_gap_m,
        closing_speed_mps,
        lead_speed_mps,
        nominal_target_speed_mps,
    )
    ttc_s = None
    if closing_speed_mps > 1e-6 and lead_gap_m is not None:
        ttc_s = max(0.0, float(lead_gap_m - FOLLOW_BUFFER_M)) / closing_speed_mps
    debug.update(
        {
            "traffic_braking_active": safe_target_speed_mps < nominal_target_speed_mps,
            "lead_gap_m": None if lead_gap_m is None else float(lead_gap_m),
            "lead_speed_mps": float(lead_speed_mps),
            "lead_ttc_s": None if ttc_s is None else float(ttc_s),
            "safe_target_speed_mps": float(safe_target_speed_mps),
        }
    )
    return float(safe_target_speed_mps), debug
