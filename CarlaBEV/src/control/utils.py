import math
import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


# --------------------------------------------------------------------
# MPC
#
def pi_2_pi(angle):
    return angle_mod(angle)


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def calc_nearest_index(state, cx, cy, cyaw, pind, n_ind_search):
    dx = [state.x - icx for icx in cx[pind : (pind + n_ind_search)]]
    dy = [state.y - icy for icy in cy[pind : (pind + n_ind_search)]]

    d = [idx**2 + idy**2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def point_to_segment_distance(px, py, x1, y1, x2, y2, signed=False):
    # Vectors
    A = np.array([x1, y1])
    B = np.array([x2, y2])
    P = np.array([px, py])
    AB = B - A
    AP = P - A

    # Projection of AP onto AB, normalized
    t = np.dot(AP, AB) / np.dot(AB, AB)
    t = np.clip(t, 0.0, 1.0)  # constrain to segment

    # Closest point on segment
    closest = A + t * AB
    error = np.linalg.norm(P - closest)

    if signed:
        # cross product z-component (2D)
        cross = AB[0] * AP[1] - AB[1] * AP[0]
        error *= np.sign(cross) if cross != 0 else 1

    return error


def lateral_error(px, py, waypoints, signed=False):
    min_error = float("inf")
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        e = point_to_segment_distance(px, py, x1, y1, x2, y2, signed=signed)
        if abs(e) < abs(min_error):
            min_error = e
    return min_error


def smooth_and_compute(ax, ay, window=9, poly=3):
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    if ax.size != ay.size:
        raise ValueError("ax and ay must have same length")

    # remove consecutive duplicate points
    d = np.hypot(np.diff(ax), np.diff(ay))
    mask = np.concatenate(([True], d > 1e-9))
    ax = ax[mask]
    ay = ay[mask]

    if len(ax) < 2:
        # Fallback: create a tiny straight segment to avoid crashes
        x0, y0 = ax[0], ay[0]
        ax = np.array([x0, x0 + 1e-3])
        ay = np.array([y0, y0])

    # ensure window is valid (odd and <= len)
    if window % 2 == 0:
        window += 1
    if window > len(ax):
        window = len(ax) if len(ax) % 2 == 1 else len(ax) - 1
    if window < 3:
        window = 3

    poly = min(poly, window - 1)

    # Smooth coordinates (if path too short fallback to raw)
    if len(ax) >= window:
        cx = savgol_filter(ax, window_length=window, polyorder=poly)
        cy = savgol_filter(ay, window_length=window, polyorder=poly)
    else:
        cx, cy = ax.copy(), ay.copy()

    # compute cumulative arc-length s
    dx_local = np.diff(cx)
    dy_local = np.diff(cy)
    seg_len = np.hypot(dx_local, dy_local)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = s[-1]
    if total_len <= 1e-9:
        # degenerate: all points essentially same after smoothing
        dx_ds = np.zeros_like(cx)
        dy_ds = np.zeros_like(cy)
        cyaw = np.zeros_like(cx)
        ck = np.zeros_like(cx)
        return cx, cy, cyaw, ck, s

    # derivatives w.r.t arc-length s using numpy.gradient
    # np.gradient(y, x) approximates dy/dx
    # gradient preserves original array length
    dx_ds = np.gradient(cx, s)
    dy_ds = np.gradient(cy, s)

    # yaw is tangent direction
    cyaw = np.unwrap(np.arctan2(dy_ds, dx_ds))

    # second derivatives
    d2x_ds2 = np.gradient(dx_ds, s)
    d2y_ds2 = np.gradient(dy_ds, s)

    denom = dx_ds**2 + dy_ds**2
    # avoid division by zero (tiny denom -> set curvature to 0)
    small = denom < 1e-9
    denom_safe = np.where(small, 1.0, denom)  # avoid nan
    ck = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (denom_safe**1.5)
    ck[small] = 0.0

    return cx, cy, cyaw, ck, s
