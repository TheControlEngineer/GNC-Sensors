"""
Range accuracy, field-of-view boundary, and range boundary transition tests.

Validates that the LiDAR sensor correctly measures range with zero noise,
and that FoV and range gating boundaries transition at the correct angles
and distances.
"""

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.lidar import Lidar
from .helpers import attach_plot_to_html_report


@pytest.mark.test_meta(
    description="Verify zero noise boresight ranging over the full measurement envelope.",
    goal="Confirm the deterministic measurement pipeline reproduces ground truth range with no stochastic terms.",
    passing_criteria="Measured range equals true range at machine precision for every sampled distance from 0 to max range.",
)
def test_range_accuracy_zero_noise_boresight(request):
    # Skip the test automatically if matplotlib is not installed
    matplotlib = pytest.importorskip("matplotlib")
    # Use the non interactive Agg backend so no GUI window is spawned
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -- Build a perfectly noiseless sensor configuration --
    # Every stochastic and systematic error parameter is forced to zero
    # so the sensor output must exactly reproduce the geometric truth.
    class ZeroNoiseLidarConfig(LidarConfig):
        range_min = 0.0           # allow measurements starting at 0 m
        range_accuracy = 0.0      # perfect range resolution
        noise_floor_std = 0.0     # no additive Gaussian noise floor
        noise_range_coeff = 0.0   # no range dependent noise growth
        bias_init_std = 0.0       # no initial bias offset
        bias_rw_std = 0.0         # no random walk on the bias
        bias_drift_rate = 0.0     # no deterministic bias drift
        scale_factor_ppm = 0.0    # no systematic scale factor error
        scale_error_std_ppm = 0.0 # no stochastic scale factor jitter
        dropout_prob = 0.0        # dropouts are impossible
        dropout_range_coeff = 0.0 # no range dependent dropout growth
        outlier_prob = 0.0        # outliers are impossible
        outlier_std = 0.0         # (unused when prob is zero)
        outlier_bias = 0.0        # (unused when prob is zero)
        quantization_step = 0.0   # infinite ADC resolution
        latency = 0.0             # no fixed measurement delay
        sample_time_jitter_std = 0.0  # no sampling clock jitter
        latency_jitter_std = 0.0      # no latency jitter

    # Instantiate the LiDAR with the zero noise config
    lidar = Lidar(config=ZeroNoiseLidarConfig)

    # Determine the sweep step size: use the default range_accuracy, or
    # fall back to machine epsilon so the step is never exactly zero
    step = max(float(LidarConfig.range_accuracy), np.finfo(float).eps)
    # Read the maximum measurable range from the sensor instance
    range_max = float(lidar.range_max)
    # Create evenly spaced true range values from one step up to range_max
    true_ranges_sweep = np.arange(step, range_max, step, dtype=float)
    # Ensure the sweep includes the endpoint (range_max) if it was missed
    if true_ranges_sweep.size == 0 or not np.isclose(true_ranges_sweep[-1], range_max):
        true_ranges_sweep = np.append(true_ranges_sweep, range_max)

    # -- Measure at each true range along the boresight axis --
    measured_ranges_sweep = []
    for true_range in true_ranges_sweep:
        # Place the target exactly on the positive X axis (boresight)
        measurement = lidar.measure(
            rel_position=[true_range, 0.0, 0.0],
            add_noise=False,
            current_time=None,
        )
        # With zero noise and on axis geometry every measurement must be valid
        assert measurement["valid"] is True
        measured_ranges_sweep.append(float(measurement["range"]))

    # Convert the collected measurements to a NumPy array for vectorised comparison
    measured_ranges_sweep = np.asarray(measured_ranges_sweep, dtype=float)
    # Assert that measured ranges match true ranges to within machine epsilon
    np.testing.assert_allclose(
        measured_ranges_sweep,
        true_ranges_sweep,
        rtol=0.0,
        atol=np.finfo(float).eps,
    )

    # -- Diagnostic plot: measured vs true range over the full envelope --
    # Prepend a zero origin so the plot starts at (0, 0)
    true_ranges = np.concatenate(([0.0], true_ranges_sweep))
    measured_ranges = np.concatenate(([0.0], measured_ranges_sweep))

    fig, ax = plt.subplots(figsize=(8, 5))
    # An ideal sensor traces a perfect y = x line
    ax.plot(true_ranges, measured_ranges, linewidth=2.0, color="tab:blue")
    ax.set_title(f"LiDAR Range Accuracy at Zero Noise (step = {step:g} m)")
    ax.set_xlabel("True Range (m)")
    ax.set_ylabel("Measured Range (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # Embed the figure in the HTML test report and then release memory
    attach_plot_to_html_report(request, fig, name="range_accuracy_zero_noise")
    plt.close(fig)


@pytest.mark.test_meta(
    description="Sweep target angle from outside negative FoV to outside positive FoV and evaluate boundary behavior.",
    goal="Identify where measurement validity transitions between valid and out_of_fov relative to half FoV limits.",
    passing_criteria="Sweep starts invalid, center is valid, sweep ends invalid, and epsilon checks at both boundaries match expected valid/invalid transitions.",
)
def test_fov_boundary_transition(request):
    # Import matplotlib and force the non-interactive Agg backend so plots
    # render to in-memory buffers without requiring a display server.
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Custom LiDAR configuration that disables every noise and error source.
    # This isolates the field of view acceptance logic from stochastic effects,
    # ensuring that validity changes are caused purely by geometric angle checks.
    class ZeroNoiseLidarConfig(LidarConfig):
        range_accuracy = 0.0      # no constant range accuracy error
        noise_floor_std = 0.0     # no additive Gaussian noise
        noise_range_coeff = 0.0   # no range proportional noise growth
        bias_init_std = 0.0       # no initial bias scatter
        bias_rw_std = 0.0         # no random walk on the bias state
        bias_drift_rate = 0.0     # no deterministic bias drift
        scale_factor_ppm = 0.0    # no systematic scale factor offset
        scale_error_std_ppm = 0.0 # no stochastic scale factor jitter
        dropout_prob = 0.0        # no measurement dropouts
        dropout_range_coeff = 0.0 # no range dependent dropout growth
        outlier_prob = 0.0        # no outlier injections
        outlier_std = 0.0         # (unused when outlier_prob is zero)
        outlier_bias = 0.0        # (unused when outlier_prob is zero)
        quantization_step = 0.0   # infinite ADC resolution
        latency = 0.0             # no fixed measurement delay
        sample_time_jitter_std = 0.0  # no sampling clock jitter
        latency_jitter_std = 0.0      # no latency jitter

    # Instantiate the sensor with the zero noise config so only FoV gating is active
    lidar = Lidar(config=ZeroNoiseLidarConfig)

    # Small angular offset used to probe just inside/outside the FoV boundary
    epsilon = 1e-4
    # Compute the half angle of the symmetric field of view cone
    half_fov = float(lidar.fov * 0.5)
    # Choose a test range that sits comfortably within the valid range envelope
    test_range = max(float(lidar.range_min), 1.0) + 9.0

    # Three angles near the positive FoV boundary: just inside, exactly on, just outside
    pos_boundary_angles = np.array([half_fov - epsilon, half_fov, half_fov + epsilon], dtype=float)
    # Three angles near the negative FoV boundary: just outside, exactly on, just inside
    neg_boundary_angles = np.array([-half_fov - epsilon, -half_fov, -half_fov + epsilon], dtype=float)

    # Collect validity results at the positive boundary angles
    pos_valids = []
    pos_reasons = []
    for angle in pos_boundary_angles:
        # Convert the polar (range, angle) to Cartesian and measure
        measurement = lidar.measure(
            rel_position=[test_range * np.cos(angle), test_range * np.sin(angle), 0.0],
            add_noise=False,
            current_time=None,
        )
        # Record whether the sensor accepted this measurement as valid
        pos_valids.append(bool(measurement["valid"]))
        # Record the rejection reason (empty string if measurement was valid)
        pos_reasons.append(measurement.get("reason", ""))

    # Collect validity results at the negative boundary angles
    neg_valids = []
    neg_reasons = []
    for angle in neg_boundary_angles:
        # Same polar to Cartesian conversion for negative side probes
        measurement = lidar.measure(
            rel_position=[test_range * np.cos(angle), test_range * np.sin(angle), 0.0],
            add_noise=False,
            current_time=None,
        )
        neg_valids.append(bool(measurement["valid"]))
        neg_reasons.append(measurement.get("reason", ""))

    # Positive boundary: inside and on boundary are valid, just outside must be invalid
    assert pos_valids == [True, True, False]
    # The rejection reason for the outside sample must specifically say out_of_fov
    assert pos_reasons[-1] == "out_of_fov"
    # Negative boundary: just outside is invalid, on boundary and inside are valid
    assert neg_valids == [False, True, True]
    # Same rejection reason check on the negative side
    assert neg_reasons[0] == "out_of_fov"

    # Add an angular margin beyond the FoV edges to see the full transition profile
    sweep_margin = max(np.deg2rad(3.0), 8.0 * epsilon)
    # Build a dense sweep of 801 angles spanning from outside negative to outside positive FoV
    sweep_angles = np.linspace(-(half_fov + sweep_margin), half_fov + sweep_margin, 801, dtype=float)
    # Evaluate validity at each sweep angle to map the full transition curve
    sweep_valids = []
    for angle in sweep_angles:
        measurement = lidar.measure(
            rel_position=[test_range * np.cos(angle), test_range * np.sin(angle), 0.0],
            add_noise=False,
            current_time=None,
        )
        sweep_valids.append(bool(measurement["valid"]))

    # Convert boolean validity flags to numeric (0.0 / 1.0) for plotting and assertions
    sweep_valid_numeric = np.asarray(sweep_valids, dtype=float)
    # The sweep must start invalid (outside the negative FoV edge)
    assert bool(sweep_valids[0]) is False
    # The sweep must end invalid (outside the positive FoV edge)
    assert bool(sweep_valids[-1]) is False
    # The center of the sweep (boresight) must be valid
    assert bool(sweep_valids[len(sweep_valids) // 2]) is True

    # Convert sweep angles to degrees for human readable plot axes
    sweep_angles_deg = np.rad2deg(sweep_angles)
    half_fov_deg = np.rad2deg(half_fov)

    # -- Diagnostic plot: two panel figure showing spatial layout and validity profile --
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Convert sweep angles back to Cartesian target positions for the spatial panel
    target_x = test_range * np.cos(sweep_angles)
    target_y = test_range * np.sin(sweep_angles)
    # Boolean masks separating valid from invalid measurement locations
    valid_mask = sweep_valid_numeric > 0.5
    invalid_mask = ~valid_mask

    # Top panel: scatter plot of target positions in the sensor XY plane,
    # color coded by validity (red = rejected, green = accepted)
    ax_top.scatter(target_x[invalid_mask], target_y[invalid_mask], s=10, alpha=0.7, color="tab:red", label="invalid targets")
    ax_top.scatter(target_x[valid_mask], target_y[valid_mask], s=10, alpha=0.7, color="tab:green", label="valid targets")
    # Mark the sensor origin with a black cross
    ax_top.scatter([0.0], [0.0], marker="x", s=70, color="black", label="sensor")
    # Draw dashed orange lines along the positive and negative half FoV edges
    boundary_r = np.linspace(0.0, 1.05 * test_range, 100, dtype=float)
    ax_top.plot(
        boundary_r * np.cos(half_fov),
        boundary_r * np.sin(half_fov),
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="half fov boundary",
    )
    ax_top.plot(
        boundary_r * np.cos(-half_fov),
        boundary_r * np.sin(-half_fov),
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
    )
    ax_top.set_title("Target Sweep Locations in Sensor Frame")
    ax_top.set_xlabel("x (m)")
    ax_top.set_ylabel("y (m)")
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Bottom panel: validity (0 or 1) as a function of angle from boresight
    ax_bottom.scatter(sweep_angles_deg, sweep_valid_numeric, s=10, alpha=0.7, color="tab:blue", label="sweep results")
    # Overlay the six boundary check samples (three negative, three positive) as larger markers
    boundary_angles_deg = np.rad2deg(np.concatenate((neg_boundary_angles, pos_boundary_angles)))
    boundary_valids = np.asarray(neg_valids + pos_valids, dtype=float)
    ax_bottom.scatter(
        boundary_angles_deg,
        boundary_valids,
        s=40,
        color="tab:orange",
        label="boundary check samples",
        zorder=3,
    )
    # Vertical dashed lines marking the exact half FoV limits
    ax_bottom.axvline(-half_fov_deg, color="tab:red", linestyle="--", linewidth=1.5, label="half fov boundaries")
    ax_bottom.axvline(half_fov_deg, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Validity Results for the Sweep")
    ax_bottom.set_xlabel("Target Angle from Boresight (deg)")
    ax_bottom.set_ylabel("Validity")
    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_ylim(-0.1, 1.1)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Leave room on the right for the legends placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the finished figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="fov_boundary_transition")
    # Release the figure memory now that it has been captured
    plt.close(fig)


@pytest.mark.test_meta(
    description="Sweep target range from below minimum range to above maximum range and evaluate boundary behavior.",
    goal="Identify where measurement validity transitions between valid and out_of_range at range_min and range_max.",
    passing_criteria="Near range_min, minus epsilon is invalid while boundary and plus epsilon are valid; near range_max, minus epsilon and boundary are valid while plus epsilon is invalid; sweep starts invalid, is valid inside limits, and ends invalid.",
)
def test_range_boundary_transition(request):
    # Import matplotlib and force the non-interactive Agg backend for off-screen rendering
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Zero noise configuration that suppresses every stochastic and systematic error.
    # This ensures the only thing being tested is the range gating logic: whether
    # the sensor correctly accepts or rejects targets based on range_min and range_max.
    class ZeroNoiseLidarConfig(LidarConfig):
        range_accuracy = 0.0      # no constant accuracy error
        noise_floor_std = 0.0     # no additive Gaussian noise
        noise_range_coeff = 0.0   # no range proportional noise scaling
        bias_init_std = 0.0       # no initial bias offset
        bias_rw_std = 0.0         # no random walk bias evolution
        bias_drift_rate = 0.0     # no deterministic bias drift
        scale_factor_ppm = 0.0    # no scale factor offset
        scale_error_std_ppm = 0.0 # no stochastic scale jitter
        dropout_prob = 0.0        # no measurement dropouts
        dropout_range_coeff = 0.0 # no range dependent dropout growth
        outlier_prob = 0.0        # no outlier injections
        outlier_std = 0.0         # (unused when outlier_prob is zero)
        outlier_bias = 0.0        # (unused when outlier_prob is zero)
        quantization_step = 0.0   # infinite ADC resolution
        latency = 0.0             # no fixed measurement delay
        sample_time_jitter_std = 0.0  # no sampling clock jitter
        latency_jitter_std = 0.0      # no latency jitter

    # Build the sensor with the noiseless config
    lidar = Lidar(config=ZeroNoiseLidarConfig)

    # Read the minimum and maximum measurable ranges from the sensor instance
    range_min = float(lidar.range_min)
    range_max = float(lidar.range_max)
    # Small offset used to probe slightly inside and outside each boundary
    epsilon = 1e-4

    # Three probe ranges near the minimum boundary: below, exactly on, and above
    min_boundary_ranges = np.array([range_min - epsilon, range_min, range_min + epsilon], dtype=float)
    # Three probe ranges near the maximum boundary: below, exactly on, and above
    max_boundary_ranges = np.array([range_max - epsilon, range_max, range_max + epsilon], dtype=float)

    # Evaluate validity at each of the three minimum boundary probe points
    min_valids = []
    min_reasons = []
    for target_range in min_boundary_ranges:
        # Place the target on the boresight axis at the probe range
        measurement = lidar.measure(
            rel_position=[target_range, 0.0, 0.0],
            add_noise=False,
            current_time=None,
        )
        # Store the boolean validity flag for this probe
        min_valids.append(bool(measurement["valid"]))
        # Store the rejection reason string (empty if valid)
        min_reasons.append(measurement.get("reason", ""))

    # Evaluate validity at each of the three maximum boundary probe points
    max_valids = []
    max_reasons = []
    for target_range in max_boundary_ranges:
        measurement = lidar.measure(
            rel_position=[target_range, 0.0, 0.0],
            add_noise=False,
            current_time=None,
        )
        max_valids.append(bool(measurement["valid"]))
        max_reasons.append(measurement.get("reason", ""))

    # At range_min: below is invalid, on boundary is valid, above is valid
    assert min_valids == [False, True, True]
    # The below minimum sample must be rejected with the out_of_range reason
    assert min_reasons[0] == "out_of_range"
    # At range_max: below is valid, on boundary is valid, above is invalid
    assert max_valids == [True, True, False]
    # The above maximum sample must be rejected with the out_of_range reason
    assert max_reasons[-1] == "out_of_range"

    # Define a margin to extend the sweep slightly beyond both range limits
    sweep_margin = max(5.0, 10.0 * epsilon)
    # Build a dense sweep of 801 ranges spanning from below range_min to above range_max
    sweep_ranges = np.linspace(max(range_min - sweep_margin, 0.0), range_max + sweep_margin, 801, dtype=float)
    # Evaluate validity at every sweep range to map the full transition profile
    sweep_valids = []
    for target_range in sweep_ranges:
        measurement = lidar.measure(
            rel_position=[target_range, 0.0, 0.0],
            add_noise=False,
            current_time=None,
        )
        sweep_valids.append(bool(measurement["valid"]))

    # Convert boolean flags to numeric for plotting and array assertions
    sweep_valid_numeric = np.asarray(sweep_valids, dtype=float)
    # The sweep must start invalid (below minimum range)
    assert bool(sweep_valids[0]) is False
    # The sweep must end invalid (above maximum range)
    assert bool(sweep_valids[-1]) is False
    # At least some measurements in the middle must be valid
    assert np.any(sweep_valid_numeric > 0.5)

    # -- Diagnostic plot: two panel figure showing spatial layout and validity step function --
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Boolean masks for coloring valid vs invalid sweep points
    valid_mask = sweep_valid_numeric > 0.5
    invalid_mask = ~valid_mask
    # All targets lie on the boresight axis so Y coordinates are zero
    zeros = np.zeros_like(sweep_ranges)

    # Top panel: targets plotted along the boresight axis, colored by validity
    ax_top.scatter(sweep_ranges[invalid_mask], zeros[invalid_mask], s=10, alpha=0.7, color="tab:red", label="invalid targets")
    ax_top.scatter(sweep_ranges[valid_mask], zeros[valid_mask], s=10, alpha=0.7, color="tab:green", label="valid targets")
    # Vertical dashed lines at the configured range boundaries
    ax_top.axvline(range_min, color="tab:orange", linestyle="--", linewidth=1.5, label="range boundaries")
    ax_top.axvline(range_max, color="tab:orange", linestyle="--", linewidth=1.5)
    ax_top.set_title("Target Range Sweep Along Boresight")
    ax_top.set_xlabel("Target Range (m)")
    ax_top.set_ylabel("Target y (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Bottom panel: validity (0 or 1) as a function of target range
    ax_bottom.scatter(sweep_ranges, sweep_valid_numeric, s=10, alpha=0.7, color="tab:blue", label="sweep results")
    # Overlay the six explicit boundary check samples as larger markers
    boundary_ranges = np.concatenate((min_boundary_ranges, max_boundary_ranges))
    boundary_valids = np.asarray(min_valids + max_valids, dtype=float)
    ax_bottom.scatter(
        boundary_ranges,
        boundary_valids,
        s=40,
        color="tab:orange",
        label="boundary check samples",
        zorder=3,
    )
    # Vertical dashed lines at both range gate boundaries for visual reference
    ax_bottom.axvline(range_min, color="tab:red", linestyle="--", linewidth=1.5, label="range boundaries")
    ax_bottom.axvline(range_max, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Validity Results for Range Sweep")
    ax_bottom.set_xlabel("Target Range (m)")
    ax_bottom.set_ylabel("Validity")
    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_ylim(-0.1, 1.1)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Leave room on the right for the legends placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the finished figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="range_boundary_transition")
    # Release the figure memory
    plt.close(fig)

