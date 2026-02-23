"""
Math Utilities Module

This module provides helper functions for common mathematical operations
needed across the sensor simulation pipeline. It covers vector normalization,
rotation matrix validation, angle wrapping, linear interpolation of positions
over time, and management of pending measurement queues with latency handling.

All functions here ensure consistent treatment of vector and matrix inputs,
smooth temporal interpolation for moving platforms, and correct scheduling
of asynchronous measurements.
"""

import numpy as np

# Small numerical tolerance to prevent division by zero and handle
# degenerate edge cases like near-zero vector norms or tiny time gaps.
eps = 1e-12  # [dimensionless]


def _as_vector3(value, name):
    """
    Validate and convert an input into a flat 3-element float vector.

    Takes any array-like input (list, tuple, numpy array, etc.) and
    converts it to a 1D numpy array of exactly 3 elements with float64
    dtype. Raises a ValueError if the result does not have exactly 3
    elements.

    :param value: Array-like input to convert into a 3D vector.
    :param name:  Human-readable parameter name, shown in error messages.

    :return: numpy array of shape (3,) with dtype float64.
    :raises ValueError: If the input does not contain exactly 3 elements.
    """
    # Convert the input to a numpy float array and flatten it to 1D
    vec = np.asarray(value, dtype=float).reshape(-1)

    # Check that the flattened array has exactly 3 elements
    if vec.size != 3:
        raise ValueError(f"{name} must be a 3D vector.")

    # Return the validated 3-element vector
    return vec


def _as_rotation_matrix(value, name):
    """
    Validate and convert an input into a 3x3 float rotation matrix.

    Takes any array-like input and converts it to a 3x3 numpy array
    with float64 dtype. Only the shape is validated here; orthogonality
    and determinant checks are not performed.

    :param value: Array-like input to convert into a 3x3 matrix.
    :param name:  Human-readable parameter name, shown in error messages.

    :return: numpy array of shape (3, 3) with dtype float64.
    :raises ValueError: If the resulting shape is not (3, 3).
    """
    # Cast the input to a float64 numpy array
    matrix = np.asarray(value, dtype=float)

    # Verify that the shape is exactly 3x3
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 rotation matrix.")

    # Return the validated 3x3 matrix
    return matrix

def _normalize(vec, fallback=(1.0, 0.0, 0.0)):
    """
    Normalize a vector to unit length, with a safe fallback for zero-length vectors.

    Computes the L2 (Euclidean) norm of the input and divides by it. If the
    norm is smaller than the module tolerance eps (effectively zero), the
    fallback vector is normalized and returned instead. This prevents
    division-by-zero errors in downstream calculations.

    :param vec:      Input vector (array-like, any dimension).
    :param fallback: Direction to return when the input has near-zero norm.
                     Defaults to (1, 0, 0), pointing along the +X axis.

    :return: Unit-length numpy vector in the same direction as the input.
    :raises ValueError: If both the input and fallback vectors have near-zero norm.
    """
    # Convert input to a float numpy array
    vec = np.asarray(vec, dtype=float)

    # Compute the Euclidean (L2) norm of the input vector
    norm = np.linalg.norm(vec)

    if norm < eps:
        # The input vector is effectively zero, so switch to the fallback direction
        fallback = np.asarray(fallback, dtype=float)

        # Compute the norm of the fallback vector
        fallback_norm = np.linalg.norm(fallback)

        # The fallback itself must be non-zero to be usable
        if fallback_norm < eps:
            raise ValueError("Fallback vector must be non-zero.")

        # Return the fallback vector scaled to unit length
        return fallback / fallback_norm

    # Return the original vector scaled to unit length
    return vec / norm

def _wrap_angle(angle):
    """
    Wrap an angle into the range [-pi, pi] radians.

    Applies a branch-free modulo formula:
        result = (angle + pi) mod (2 * pi) - pi
    This works for any real-valued input without needing conditional logic.

    :param angle: Input angle in radians.
    :return: Equivalent angle wrapped to the [-pi, pi] interval, in radians.
    """
    # Apply the modulo wrap: shift by +pi, take mod 2*pi, then shift back by -pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _interpolate_position(t_prev, pos_prev, t_curr, pos_curr, t_query):
    """
    Linearly interpolate a 3D position between two time-stamped samples.

    Computes the interpolated position at t_query using:
        pos(t_query) = pos_prev + alpha * (pos_curr - pos_prev)
    where alpha = clamp((t_query - t_prev) / (t_curr - t_prev), 0, 1).

    Edge cases handled:
        If t_prev or pos_prev is None (no prior sample), pos_curr is returned.
        If the time interval abs(dt) < eps, pos_curr is returned to avoid division by zero.
        If t_query lies outside [t_prev, t_curr], alpha is clamped to [0, 1].

    This is used by the LiDAR model to align range measurements with a
    moving platform's trajectory.

    :param t_prev:   Previous timestamp in seconds (or None if unavailable).
    :param pos_prev: Previous 3D position in meters (or None if unavailable).
    :param t_curr:   Current timestamp in seconds.
    :param pos_curr: Current 3D position in meters.
    :param t_query:  Desired query time in seconds.

    :return: Interpolated 3D position at t_query, in meters.
    """
    # Ensure current position is a float numpy array
    pos_curr = np.asarray(pos_curr, dtype=float)

    # If there is no previous sample, the best estimate is the current position
    if t_prev is None or pos_prev is None:
        return pos_curr

    # Convert all timestamps to plain floats for arithmetic
    t_prev = float(t_prev)
    t_curr = float(t_curr)
    t_query = float(t_query)

    # Compute the time difference between the two samples
    dt = t_curr - t_prev

    # If the interval is too small, return the current position to avoid division by zero
    if abs(dt) < eps:
        return pos_curr

    # Compute the interpolation factor and clamp it to [0, 1]
    alpha = np.clip((t_query - t_prev) / dt, 0.0, 1.0)

    # Perform the linear interpolation between the two positions
    return pos_prev + alpha * (pos_curr - pos_prev)

def _pop_latest_ready(pending_measurements, t_now):
    """
    Pop the most recent ready measurement from a sorted pending queue.

    Measurements are stored as (available_time, data) tuples, sorted in
    ascending order by available_time. A measurement counts as "ready"
    when its available_time <= t_now + eps. This function scans from the
    front, finds all ready entries, keeps only the newest one, and
    removes every ready entry from the list so stale data does not
    accumulate.

    :param pending_measurements: List of (available_time, measurement_data)
                                  tuples sorted ascending by available_time.
                                  This list is modified in place.
    :param t_now: Current simulation time in seconds.

    :return: The measurement_data from the latest ready entry, or None if
             no measurement is ready yet.
    """
    # Count how many measurements at the front of the queue are ready
    ready_count = 0
    for available_time, _ in pending_measurements:
        if available_time <= t_now + eps:
            # This measurement has arrived; count it as ready
            ready_count += 1
        else:
            # The list is sorted, so everything after this is also not ready
            break

    # If nothing is ready, return None
    if ready_count == 0:
        return None

    # Grab the data from the last (most recent) ready measurement
    _, measurement = pending_measurements[ready_count - 1]

    # Remove all ready entries from the front of the queue
    del pending_measurements[:ready_count]

    # Return the newest ready measurement data
    return measurement
