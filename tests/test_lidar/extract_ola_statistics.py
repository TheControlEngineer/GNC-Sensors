"""
Utilities for extracting windowed precision statistics from OLA data.
"""

import numpy as np


def compute_windowed_precision(ranges, timestamps=None, window_size=50):
    """
    Estimate precision from short windows of consecutive shots.

    :param ranges: 1D array of measured ranges in meters.
    :param timestamps: Optional 1D array of timestamps.
    :param window_size: Number of shots per window.
    :return: (window_means, window_stds)
    """
    ranges = np.asarray(ranges, dtype=float).reshape(-1)
    if timestamps is not None:
        timestamps = np.asarray(timestamps).reshape(-1)
        order = np.argsort(timestamps)
        ranges = ranges[order]

    n = int(ranges.size)
    if n < window_size:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    n_windows = n // int(window_size)
    trimmed = ranges[: n_windows * int(window_size)]
    block = trimmed.reshape(n_windows, int(window_size))
    means = np.mean(block, axis=1)
    stds = np.std(block, axis=1, ddof=1)
    return means, stds


def bin_precision_by_range(window_means, window_stds, bin_edges):
    """
    Bin window precision estimates by mean range.

    :param window_means: Mean range per window in meters.
    :param window_stds: Std range per window in meters.
    :param bin_edges: 1D edges array in meters.
    :return: (bin_centers, bin_median_stds, bin_counts)
    """
    window_means = np.asarray(window_means, dtype=float)
    window_stds = np.asarray(window_stds, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    medians = np.full(centers.shape, np.nan, dtype=float)
    counts = np.zeros(centers.shape, dtype=int)

    idx = np.digitize(window_means, bin_edges)
    for i in range(centers.size):
        m = idx == (i + 1)
        counts[i] = int(np.sum(m))
        if counts[i] > 0:
            medians[i] = float(np.median(window_stds[m]))

    return centers, medians, counts


def infer_range_meters(ranges):
    """
    Infer whether range data is in mm or m and return meters.

    :param ranges: Range array from parsed data.
    :return: Range array in meters.
    """
    r = np.asarray(ranges, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return r

    median = float(np.median(np.abs(r)))
    if median > 10000.0:
        return r / 1000.0
    return r
