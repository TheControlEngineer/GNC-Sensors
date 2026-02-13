"""
This module contains utility functions for mathematical operations, such as vector normalization,
rotation matrix validation, angle wrapping, linear interpolation, and measurement queue management.

These utilities are used throughout the sensor simulation pipeline to ensure consistent handling
of vector/matrix inputs, temporal interpolation of platform motion, and asynchronous measurement
scheduling with latency modeling.
"""

import numpy as np

# Numerical tolerance used throughout the module to guard against divide by zero
# and degenerate geometry edge cases (e.g., near zero vector norms, tiny time intervals).
eps = 1e-12  # [dimensionless]


def _as_vector3(value, name):
    """
    Validate and coerce input to a flat 3D float vector.

    Converts any array like input (list, tuple, np.ndarray, etc.) to a 1D numpy
    array of length 3 with float64 dtype. Raises ValueError if the input cannot
    be reshaped to exactly 3 elements.

    :param value: array like input to be coerced into a 3D vector
    :param name:  descriptive name of the parameter (used in error messages)

    :return: numpy array of shape (3,) with dtype float64
    :raises ValueError: if the input does not contain exactly 3 elements
    """
    vec = np.asarray(value, dtype=float).reshape(-1)  # flatten to 1D float array
    if vec.size != 3:
        raise ValueError(f"{name} must be a 3D vector.")
    return vec


def _as_rotation_matrix(value, name):
    """
    Validate and coerce input to a 3x3 float rotation matrix.

    Converts any array like input to a 3x3 numpy array with float64 dtype.
    Only validates shape, does NOT check orthogonality or determinant == 1.

    :param value: array like input to be coerced into a 3x3 matrix
    :param name:  descriptive name of the parameter (used in error messages)

    :return: numpy array of shape (3, 3) with dtype float64
    :raises ValueError: if the input shape is not (3, 3)
    """
    matrix = np.asarray(value, dtype=float)  # cast to float64 ndarray
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 rotation matrix.")
    return matrix

def _normalize(vec, fallback=(1.0, 0.0, 0.0)):
    """
    Normalize a vector to unit length with robust fallback for near zero vectors.

    Computes the L2 norm and divides by it. If the vector's norm is smaller than
    eps (numerical zero), the fallback vector is normalized and returned instead,
    preventing division by zero.

    :param vec:      input vector (array like, any dimension)
    :param fallback: default unit direction returned when input has near zero norm;
                     defaults to (1, 0, 0) i.e. the +x axis

    :return: unit length numpy vector
    :raises ValueError: if both input and fallback have near zero norm
    """
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)  # L2 norm of the input vector
    if norm < eps:
        # Input is numerically zero Ã¢â‚¬â€ use the fallback direction instead.
        fallback = np.asarray(fallback, dtype=float)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm < eps:
            raise ValueError("Fallback vector must be non-zero.")
        return fallback / fallback_norm  # return the normalized fallback vector
    return vec / norm  # return the normalized input vector

def _warp_angle(angle):
    """
    Wrap an angle into the canonical range [-Ãâ‚¬, Ãâ‚¬] radians.

    Uses the branch free modulo formula: output = (input + Ãâ‚¬) mod 2Ãâ‚¬ Ã¢Ë†â€™ Ãâ‚¬.
    This avoids conditional statements and works for any real valued input.

    :param angle: input angle [rad]
    :return: wrapped angle in [-Ãâ‚¬, Ãâ‚¬] [rad]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi  # modulo based wrap to [-Ãâ‚¬, Ãâ‚¬] without branching

def _interpolate_position(t_prev, pos_prev, t_curr, pos_curr, t_query):
    """
    Linearly interpolate a 3D position between two time stamped samples.

    Computes:  pos(t_query) = pos_prev + ÃŽÂ± Ã‚Â· (pos_curr Ã¢Ë†â€™ pos_prev)
    where ÃŽÂ± = clamp((t_query Ã¢Ë†â€™ t_prev) / (t_curr Ã¢Ë†â€™ t_prev), 0, 1).

    Edge cases handled:
    If t_prev or pos_prev is None (no prior sample): returns pos_curr.
    If |dt| < eps (degenerate interval): returns pos_curr to avoid /0.
    If t_query falls outside [t_prev, t_curr], ÃŽÂ± is clamped to [0, 1].

    Used by the LiDAR model to align range samples with moving platform motion.

    :param t_prev:   previous timestamp [s] (or None)
    :param pos_prev: previous 3D position [m] (or None)
    :param t_curr:   current timestamp [s]
    :param pos_curr: current 3D position [m]
    :param t_query:  desired sample epoch [s]

    :return: interpolated 3D position at t_query [m]
    """
    pos_curr = np.asarray(pos_curr, dtype=float)
    if t_prev is None or pos_prev is None:
        return pos_curr  # no prior state available; best estimate is current position
    t_prev = float(t_prev)
    t_curr = float(t_curr)
    t_query = float(t_query)
    dt = t_curr - t_prev
    if abs(dt) < eps:
        return pos_curr  # degenerate time interval; avoid divide by zero
    alpha = np.clip((t_query - t_prev) / dt, 0.0, 1.0)  # interpolation factor clamped to [0, 1]
    return pos_prev + alpha * (pos_curr - pos_prev)  # linear interpolation result

def _pop_latest_ready(pending_measurements, t_now):
    """
    Pop the latest ready measurement from an ordered pending queue.

    Measurements are stored as (available_time, data) tuples sorted by ascending
    available_time. A measurement is "ready" when available_time <= t_now + eps.
    This function finds all ready entries, extracts the newest one, and removes
    every ready entry from the front of the list (stale data is discarded).

    This ensures the consumer always receives the most recent available data and
    prevents stale measurements from accumulating in the queue.

    :param pending_measurements: list of (available_time, measurement_data) tuples,
                                  sorted ascending by available_time. Modified in place.
    :param t_now: current simulation time [s]

    :return: measurement_data of the latest ready entry, or None if nothing is ready
    """
    ready_count = 0
    for available_time, _ in pending_measurements:
        if available_time <= t_now + eps:  # measurement has arrived by now
            ready_count += 1  # tally ready entries; we will return only the newest
        else:
            break  # list is sorted, so all subsequent entries are also not ready

    if ready_count == 0:
        return None  # nothing available yet

    _, measurement = pending_measurements[ready_count - 1]  # latest (newest) ready measurement
    del pending_measurements[:ready_count]  # discard all ready entries from the queue
    return measurement  # return latest ready measurement data
