"""Reactive longitudinal policy for background vehicles.

Three rules produce *constraints* on the vehicle's acceleration; the most
restrictive one wins (``min``):

1. **car following** — the nearest actor ahead in the lane corridor (any
   vehicle, pedestrian or the ego) is a lead; IDM on the bumper-to-bumper gap;
2. **signals** — a red (or a yellow the vehicle can still stop for) stop line
   controlling the vehicle's heading is a virtual lead at rest;
3. **right-of-way** — the edge of a junction zone the vehicle must yield at is
   a virtual lead at rest.

Perception is body-agnostic: any object exposing a metric kinematic state
(``x_m``, ``y_m``, ``yaw``, ``v_mps``) and a ``body()`` can be a lead — background
actors through their controller and the ego directly — so traffic reacts to
the agent too.

The legacy TTC lookup table (``car_following="legacy"``) is kept for
attribution; it only produces a speed target and ignores signals/junctions.
"""

from __future__ import annotations

import math

from CarlaBEV.src.actors.car_following import IDMParams, idm_params_for
from CarlaBEV.src.geometry.collision import gap_between


LOOKAHEAD_M = 20.0
LANE_HALF_WIDTH_M = 2.25
FOLLOW_BUFFER_M = 4.0
CAUTION_GAP_M = 10.0
CRUISE_GAP_M = 18.0
TTC_BRAKE_S = 2.0
TTC_STOP_S = 1.0
SAME_APPROACH_YAW_TOL = math.radians(45.0)
SETTLE_SPEED_MPS = 0.3
SETTLE_GAP_M = 1.0


def kinematics_of(actor):
    """Metric kinematic state of an actor: its controller, or the actor itself (ego)."""
    state = getattr(actor, "_controller", None)
    if state is not None and hasattr(state, "x_m"):
        return state
    if hasattr(actor, "x_m") and hasattr(actor, "v_mps"):
        return actor
    return None


def _relative_pose_metrics(actor, other_actor):
    own, other = kinematics_of(actor), kinematics_of(other_actor)
    dx = float(other.x_m - own.x_m)
    dy = float(other.y_m - own.y_m)
    cos_yaw = math.cos(own.yaw)
    sin_yaw = math.sin(own.yaw)
    forward_m = dx * cos_yaw + dy * sin_yaw
    lateral_m = -dx * sin_yaw + dy * cos_yaw
    return forward_m, lateral_m


def find_lead_actor(
    actor, other_actors, lookahead_m=LOOKAHEAD_M, lane_half_width_m=LANE_HALF_WIDTH_M
):
    best_actor = None
    best_forward_m = None
    for other_actor in other_actors or ():
        if other_actor is actor or kinematics_of(other_actor) is None:
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


def shape_gap_m(actor, other_actor, centre_gap_m: float) -> float:
    """Bumper-to-bumper gap from the oriented bodies (falls back to centre distance minus half lengths)."""
    if hasattr(actor, "body") and hasattr(other_actor, "body"):
        return max(0.0, float(gap_between(actor.body(), other_actor.body())))
    own_half = 0.5 * float(getattr(actor, "length_m", 0.0))
    other_half = 0.5 * float(getattr(other_actor, "length_m", 0.0))
    return max(0.0, float(centre_gap_m) - own_half - other_half)


# --------------------------------------------------------------------------- legacy
def compute_follow_speed_mps(
    gap_m, closing_speed_mps, lead_speed_mps, cruise_speed_mps
):
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


def _empty_debug(nominal_target_speed_mps: float) -> dict:
    return {
        "traffic_braking_active": False,
        "lead_gap_m": None,
        "lead_speed_mps": None,
        "lead_ttc_s": None,
        "nominal_target_speed_mps": float(nominal_target_speed_mps),
        "safe_target_speed_mps": float(nominal_target_speed_mps),
        "accel_ff_mps2": None,
        "constraint": None,
        "signal_state": None,
        "signal_gap_m": None,
        "yield_gap_m": None,
        "yield_to": None,
        "junction_id": None,
        "settle": False,
    }


def compute_safe_target_speed_mps(actor, other_actors, nominal_target_speed_mps):
    """Legacy TTC-table policy: returns ``(safe_target_speed_mps, debug)``."""
    nominal_target_speed_mps = max(0.0, float(nominal_target_speed_mps))
    lead_actor, lead_gap_m = find_lead_actor(actor, other_actors)
    debug = _empty_debug(nominal_target_speed_mps)
    if lead_actor is None:
        return nominal_target_speed_mps, debug

    lead_speed_mps = float(kinematics_of(lead_actor).v_mps)
    closing_speed_mps = max(0.0, float(kinematics_of(actor).v_mps - lead_speed_mps))
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
            "constraint": "lead"
            if safe_target_speed_mps < nominal_target_speed_mps
            else None,
        }
    )
    return float(safe_target_speed_mps), debug


# --------------------------------------------------------------------------- signals
def stop_line_ahead(actor, signals, lookahead_m: float):
    """Nearest controlling stop line ahead of the vehicle: ``(light, gap_m)`` or ``(None, None)``.

    A light applies when it controls the vehicle's travel direction (within 45
    degrees), the vehicle is laterally within the strip span (measured in the
    line's frame). Signed gap is measured from the occupied front; retain the
    line until the rear clears it so crossing detection can observe its sign.
    """
    own = kinematics_of(actor)
    if own is None:
        return None, None
    best, best_gap = None, None
    half_length = 0.5 * float(getattr(actor, "length_m", 0.0))
    body = actor.body() if hasattr(actor, "body") else None
    for light in signals or ():
        yaw_c = getattr(light, "controlled_yaw", None)
        if yaw_c is None:
            continue
        dyaw = (own.yaw - yaw_c + math.pi) % (2.0 * math.pi) - math.pi
        if abs(dyaw) > SAME_APPROACH_YAW_TOL:
            continue
        lx, ly, half_span = light.stop_line_m()
        dx, dy = lx - own.x_m, ly - own.y_m
        # Measured in the stop line's own frame (controlled heading), so a
        # vehicle still aligning with the lane is not filtered out by its yaw.
        forward = dx * math.cos(yaw_c) + dy * math.sin(yaw_c)
        lateral = -dx * math.sin(yaw_c) + dy * math.cos(yaw_c)
        if body is not None:
            rear, front = body.projection(yaw_c)
            side_min, side_max = body.projection(yaw_c + math.pi / 2)
            line_forward = lx * math.cos(yaw_c) + ly * math.sin(yaw_c)
            line_side = -lx * math.sin(yaw_c) + ly * math.cos(yaw_c)
            if side_min > line_side + half_span or side_max < line_side - half_span:
                continue
            gap = line_forward - front
            cleared = line_forward < rear
        else:
            # Compatibility for point/length-only traffic-policy clients.
            if abs(lateral) > half_span:
                continue
            gap = forward - half_length
            cleared = gap < -half_length
        if cleared or gap > lookahead_m:
            continue  # already crossed, or too far
        if best_gap is None or gap < best_gap:
            best, best_gap = light, gap
    return best, best_gap


def _must_stop_for(light, gap_m: float, v_mps: float, params: IDMParams) -> bool:
    state = light.state_name
    if state == "red":
        return True
    if state == "yellow":
        # Stop on yellow only if a comfortable stop is still possible before the line.
        stopping_m = v_mps * v_mps / (2.0 * params.comfort_decel_mps2)
        return gap_m >= stopping_m
    return False


# --------------------------------------------------------------------------- right of way
def _crossing(own, other) -> bool:
    """Crossing traffic: headings differ by 45..135 degrees.

    Same-direction traffic is handled by car following; opposite traffic on the
    other lane does not conflict (turns across oncoming traffic are not
    modelled — documented simplification).
    """
    dyaw = abs((own.yaw - other.yaw + math.pi) % (2.0 * math.pi) - math.pi)
    return SAME_APPROACH_YAW_TOL < dyaw < math.pi - SAME_APPROACH_YAW_TOL


def _arrival_time_s(dist_to_edge_m: float, v_mps: float) -> float:
    if dist_to_edge_m <= 0.0:
        return 0.0
    if v_mps <= 1e-3:
        return float("inf")
    return dist_to_edge_m / v_mps


def yield_point_ahead(actor, other_actors, junctions, rules, waited_s: float):
    """Zone edge the vehicle must stop at: ``(zone, gap_m, reason)`` or ``(None, None, None)``.

    The vehicle yields when it is approaching (not inside) a zone and a
    crossing vehicle (incl. the ego) is inside the zone, or will reach it
    earlier. After ``rules.yield_max_wait_s`` of waiting only
    traffic inside the zone is yielded to (deadlock guard).
    """
    if junctions is None or len(junctions) == 0:
        return None, None, None
    own = kinematics_of(actor)
    if own is None:
        return None, None, None
    zone, dist_m = junctions.zone_ahead(
        own.x_m, own.y_m, own.yaw, rules.yield_lookahead_m
    )
    if zone is None or dist_m <= 0.0:
        return (
            None,
            None,
            None,
        )  # not approaching, or already inside: never yield inside
    own_arrival = _arrival_time_s(dist_m, max(float(own.v_mps), 0.5))
    reason = None
    for other in other_actors or ():
        if other is actor or getattr(other, "kind", "vehicle") == "pedestrian":
            continue
        kin = kinematics_of(other)
        if kin is None or not _crossing(own, kin):
            continue
        d_other = zone.distance_to_edge_m(kin.x_m, kin.y_m)
        if d_other <= 0.0:
            # Inside: yield unless it is leaving (heading away from the centre).
            to_centre = math.atan2(zone.y_m - kin.y_m, zone.x_m - kin.x_m)
            away = (
                abs((kin.yaw - to_centre + math.pi) % (2.0 * math.pi) - math.pi)
                > math.pi / 2
            )
            if away and d_other < -0.5 * zone.radius_m:
                continue
            return zone, dist_m, f"inside:{getattr(other, 'kind', 'vehicle')}"
        if waited_s >= rules.yield_max_wait_s:
            continue
        if d_other > rules.yield_lookahead_m:
            continue
        # Is the other actually heading into this zone?
        dxo, dyo = zone.x_m - kin.x_m, zone.y_m - kin.y_m
        if dxo * math.cos(kin.yaw) + dyo * math.sin(kin.yaw) <= 0.0:
            continue
        other_arrival = _arrival_time_s(d_other, float(kin.v_mps))
        if other_arrival < own_arrival or (
            other_arrival == own_arrival and (kin.x_m, kin.y_m) < (own.x_m, own.y_m)
        ):
            reason = f"arriving:{getattr(other, 'kind', 'vehicle')}"
            return zone, dist_m, reason
    return None, None, None


# --------------------------------------------------------------------------- command
def compute_traffic_command(
    actor,
    other_actors,
    nominal_target_speed_mps: float,
    *,
    behaviour_accel_mps2: float,
    signals=(),
    junctions=None,
    rules=None,
    waited_s: float = 0.0,
):
    """Acceleration command [m/s^2] after car following, signals and right-of-way.

    ``behaviour_accel_mps2`` is what the vehicle would command without traffic
    (speed loop on the nominal target incl. behaviour feed-forward). The result
    is ``min`` of that and each active constraint's IDM acceleration, so rules
    only ever *reduce* acceleration. Returns ``(accel_mps2, debug)``.
    """
    nominal_target_speed_mps = max(0.0, float(nominal_target_speed_mps))
    debug = _empty_debug(nominal_target_speed_mps)
    own = kinematics_of(actor)
    v = float(own.v_mps)
    params = idm_params_for(getattr(actor, "speed_class", "medium"))
    accel = float(behaviour_accel_mps2)
    constraint = None
    margin = float(getattr(rules, "stop_line_margin_m", 0.0))

    # 1. car following
    lead_actor, lead_centre_gap_m = find_lead_actor(actor, other_actors)
    if lead_actor is not None:
        lead_speed_mps = max(0.0, float(kinematics_of(lead_actor).v_mps))
        gap_m = shape_gap_m(actor, lead_actor, lead_centre_gap_m)
        a_follow = params.acceleration_mps2(
            v, nominal_target_speed_mps, gap_m, lead_speed_mps
        )
        closing = v - lead_speed_mps
        debug.update(
            {
                "lead_gap_m": float(gap_m),
                "lead_speed_mps": float(lead_speed_mps),
                "lead_ttc_s": float(gap_m / closing) if closing > 1e-6 else None,
            }
        )
        if a_follow < accel:
            accel, constraint = a_follow, "lead"

    # 2. signals
    if (
        rules is not None
        and rules.signals_enabled
        and getattr(actor, "obeys_signals", True)
    ):
        light, gap_m = stop_line_ahead(actor, signals, rules.signal_lookahead_m)
        if light is not None:
            debug["signal_state"] = light.state_name
            debug["signal_gap_m"] = float(gap_m)
            if _must_stop_for(light, gap_m, v, params):
                a_stop = params.acceleration_mps2(
                    v, nominal_target_speed_mps, max(gap_m - margin, 0.0), 0.0
                )
                if a_stop < accel:
                    accel, constraint = a_stop, "signal"

    # 3. right of way
    if (
        rules is not None
        and rules.right_of_way_enabled
        and getattr(actor, "yields_at_junctions", True)
    ):
        zone, gap_m, reason = yield_point_ahead(
            actor, other_actors, junctions, rules, waited_s
        )
        if zone is not None:
            debug["junction_id"] = zone.junction_id
            debug["yield_gap_m"] = float(gap_m)
            debug["yield_to"] = reason
            a_yield = params.acceleration_mps2(
                v, nominal_target_speed_mps, max(gap_m - margin, 0.0), 0.0
            )
            if a_yield < accel:
                accel, constraint = a_yield, "yield"

    # Settle: IDM approaches a standstill only asymptotically (creep). Once the
    # vehicle is nearly at rest within the standstill gap of a stopped obstacle
    # or a stop point, the actor commands a full stop instead.
    if constraint is not None:
        target_rest = constraint != "lead" or debug["lead_speed_mps"] < 0.1
        gap_key = {
            "lead": "lead_gap_m",
            "signal": "signal_gap_m",
            "yield": "yield_gap_m",
        }[constraint]
        near = float(debug[gap_key]) <= params.min_gap_m + margin + SETTLE_GAP_M
        debug["settle"] = bool(target_rest and near and v <= SETTLE_SPEED_MPS)

    debug["constraint"] = constraint
    debug["traffic_braking_active"] = constraint is not None
    debug["accel_ff_mps2"] = float(accel)
    return float(accel), debug
