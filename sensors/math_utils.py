"""
This module contains utility functions for mathematical operations, such as vector normalization.

"""

import numpy as np

eps = 1e-12 # a small number to avoid divide by zero errors in normalization.

def _normalize(vec, fallback):
    """
    vector normalization with fallback for zero length vectors.
    """
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.asarray(fallback, dtype=float) # if the norm is too small, return the fallback vector instead of normalizing.
    return vec / norm  # return the normalized vector.

def _warp_angle(angle):
    """
    Warp angle to the range [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi # this formula ensures that the angle is wrapped around to the range [-pi, pi] without using conditional statements.

def _interpolate_position(t_prev,pos_prev,t_curr,pos_curr,t_query):
    """
    Linearly interpolate the position to a sample epoch 
    
    :param t_prev: previous timestamp
    :param pos_prev: previous position
    :param t_curr: current timestamp
    :param pos_curr: current position
    :param t_query: query timestamp for interpolation

    :return: interpolated position at t_query

    """
    pos_curr = np.asarray(pos_curr, dtype=float)
    if t_prev is None or pos_prev is None:
        return pos_curr # if there is no previous timestamp or position, return the current position as the interpolated position.
    t_prev = float(t_prev)
    t_curr = float(t_curr)
    t_query = float(t_query)
    dt = (t_curr - t_prev)
    if abs(dt)<eps:
        return pos_curr # if the time difference is too small, return the current position as the interpolated position to avoid divide by zero errors.
    alpha = np.clip((t_query-t_prev)/dt, 0.0,1.0) # compute the interpolation factor alpha, and clip it to the range [0, 1] 
    return pos_prev+alpha*(pos_curr-pos_prev) # return the linearly interpolated position at t_query.

def _pop_latest_ready(pending_measurements, t_now):
    """
    Pop the latest measurement that is ready (i.e., its timestamp is less than or equal to t_now) from the pending measurements list.

    :param pending_measurements: a list of pending measurements, each measurement is a tuple of (timestamp, measurement_data)
    :param t_now: current timestamp

    :return: the latest ready measurement data, or None if no measurement is ready.
    """
    ready_count = 0
    for available_time, _ in pending_measurements:
        if available_time <= t_now + eps: # check if the measurement is ready.
            ready_count +=1 # count how many measurements are ready, we will pop the latest one among them.
        else:
            break # since the pending measurements are sorted by timestamp, we can break the loop once we find a measurement that is not ready.
        if ready_count == 0:
            return None # if no measurement is ready, return None.
        
        _, measurement = pending_measurements[ready_count-1] # get the latest ready measurement data.
        del pending_measurements[:ready_count] # remove all the ready measurements from the pending list.
        return measurement # return the latest ready measurement data.
    
    return None # return None if no measurement is ready.