"""
Dropout, quantization, and saturation tests.

Validates dropout probability vs range, output quantization staircase,
and saturation clipping vs invalidation behaviour.
"""

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.lidar import Lidar
from .helpers import attach_plot_to_html_report, binomial_ci


@pytest.mark.test_meta(
    description=(
        "Monte Carlo validation of dropout probability vs range. "
        "Sweep multiple ranges, collect N=10000 samples per range using the asynchronous pipeline, "
        "and verify the empirical dropout fraction matches the linear model "
        "p_drop(r) = dropout_prob + dropout_range_coeff * normalised_range."
    ),
    goal=(
        "Confirm dropout rate at close range is near the baseline, at max range is "
        "baseline + dropout_range_coeff, and the relationship is linear in normalised range."
    ),
    passing_criteria=(
        "At every test range the configured dropout probability lies within the 99 percent "
        "binomial confidence interval of the observed dropout fraction. A linear fit through "
        "the empirical dropout fractions has slope consistent with dropout_range_coeff."
    ),
)
def test_dropout_rate(request):
    # Import matplotlib, skipping the test if the package is missing
    matplotlib = pytest.importorskip("matplotlib")
    # Use the non-interactive Agg backend for off-screen rendering
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Number of samples collected at each test range for dropout statistics
    N = 10_000
    # Baseline dropout probability at range_min (constant term in the linear model)
    DROPOUT_PROB = 0.05
    # Coefficient controlling how much the dropout probability grows with
    # normalised range: p_drop(r) = DROPOUT_PROB + DROPOUT_RANGE_COEFF * norm(r)
    DROPOUT_RANGE_COEFF = 0.20
    # Boundaries of the sensor measurement envelope used to normalise range
    RANGE_MIN = 1.0
    RANGE_MAX = 1000.0
    # Set of true ranges spanning the full envelope from near to far
    TEST_RANGES = np.array([1.0, 100.0, 250.0, 500.0, 750.0, 1000.0], dtype=float)
    # High sampling rate so that N non-stale measurements are collected quickly
    SAMPLING_RATE = 10_000.0  # [Hz] high rate to collect N samples quickly

    # Accumulators for the observed dropout fractions, expected values,
    # and confidence interval bounds at each test range
    empirical_dropouts = []
    expected_dropouts = []
    ci_bounds = []

    # Iterate over each test range, creating a fresh sensor per range to
    # collect N samples and measure the empirical dropout fraction.
    for r_idx, true_range in enumerate(TEST_RANGES):

        # Configuration isolating only the dropout mechanism.
        # All noise, bias, scale, and outlier terms are zeroed so that the
        # only way a measurement becomes invalid is through a dropout event.
        # The random seed varies with r_idx for independence across ranges.
        class DropoutConfig(LidarConfig):
            range_min = RANGE_MIN         # lower boundary of the measurement envelope
            range_max = RANGE_MAX         # upper boundary of the measurement envelope
            fov = np.deg2rad(60.0)        # wide FoV preventing angular rejection
            sampling_rate = SAMPLING_RATE # high rate for rapid sample collection
            range_accuracy = 0.0          # no Gaussian range noise
            noise_floor_std = 0.0         # no noise floor
            noise_range_coeff = 0.0       # no range-proportional noise
            bias_init_std = 0.0           # no initial bias
            bias_rw_std = 0.0             # no bias random walk
            bias_drift_rate = 0.0         # no deterministic drift
            scale_factor_ppm = 0.0        # no scale offset
            scale_error_std_ppm = 0.0     # no per-sensor scale scatter
            dropout_prob = DROPOUT_PROB   # baseline dropout rate under test
            dropout_range_coeff = DROPOUT_RANGE_COEFF  # range-dependent slope under test
            outlier_prob = 0.0            # no outlier injection
            outlier_std = 0.0             # (unused)
            outlier_bias = 0.0            # (unused)
            quantization_step = 0.0       # no quantization rounding
            latency = 0.0                 # no measurement delay
            sample_time_jitter_std = 0.0  # no clock jitter
            latency_jitter_std = 0.0      # no latency jitter
            random_seed = 5000 + r_idx    # unique seed per range for independence

        # Instantiate a fresh sensor for this test range
        lidar = Lidar(config=DropoutConfig)
        # Compute the time step from the sampling rate for advancing current_time
        dt = 1.0 / SAMPLING_RATE
        # Counter tracking how many of the N samples are dropped
        n_dropout = 0

        # Collect N measurements at the current test range
        for i in range(N):
            # Advance the simulation clock by one sample period
            t = i * dt
            # Measure the boresight target with noise enabled so dropout logic runs
            m = lidar.measure(
                rel_position=[true_range, 0.0, 0.0],
                add_noise=True,
                current_time=t,
            )
            # Skip stale returns that the sampling rate gate rejected
            if m.get("stale"):
                continue
            # Count measurements that were invalidated specifically due to dropout
            if not m.get("valid", False) and m.get("reason") == "dropout":
                n_dropout += 1

        # Compute the theoretical dropout probability at this range using the
        # linear model: p_drop = baseline + coeff * normalised_range
        span = max(RANGE_MAX - RANGE_MIN, 1e-12)
        # Normalise the current range into [0, 1] relative to the sensor envelope
        norm_range = np.clip((true_range - RANGE_MIN) / span, 0.0, 1.0)
        # Evaluate the linear dropout model and clamp to valid probability [0, 1]
        p_expected = np.clip(DROPOUT_PROB + DROPOUT_RANGE_COEFF * norm_range, 0.0, 1.0)

        # Compute the observed dropout fraction and its 99% binomial confidence interval
        p_hat, ci_lo, ci_hi = binomial_ci(n_dropout, N, confidence=0.99)

        # Store results for post-loop analysis and plotting
        empirical_dropouts.append(p_hat)
        expected_dropouts.append(float(p_expected))
        ci_bounds.append((ci_lo, ci_hi))

        # The theoretical dropout probability must fall within the 99% CI;
        # failure here means the sensor dropout mechanism does not match the model
        assert ci_lo <= p_expected <= ci_hi, (
            f"range {true_range} m: expected dropout {p_expected:.4f} outside "
            f"99% CI [{ci_lo:.4f}, {ci_hi:.4f}] (observed {p_hat:.4f})"
        )

    # Convert the accumulated lists into NumPy arrays for vectorised operations
    empirical_dropouts = np.asarray(empirical_dropouts, dtype=float)
    expected_dropouts = np.asarray(expected_dropouts, dtype=float)

    # Perform a linear regression of empirical dropout fraction vs normalised
    # range to verify the fitted slope matches the configured dropout_range_coeff.
    norm_ranges = np.clip((TEST_RANGES - RANGE_MIN) / max(RANGE_MAX - RANGE_MIN, 1e-12), 0.0, 1.0)
    if len(TEST_RANGES) > 2:
        # Fit a first-degree polynomial (line) to the data
        coeffs = np.polyfit(norm_ranges, empirical_dropouts, 1)
        # The slope of the fitted line should approximate DROPOUT_RANGE_COEFF
        fitted_slope = coeffs[0]
        # Allow up to 20% relative error plus a small absolute cushion of 0.01
        # to account for finite-sample Monte Carlo noise
        assert abs(fitted_slope - DROPOUT_RANGE_COEFF) < 0.20 * DROPOUT_RANGE_COEFF + 0.01, (
            f"fitted dropout slope {fitted_slope:.4f} deviates from configured {DROPOUT_RANGE_COEFF}"
        )

    # ---- Plot ----
    # Two-panel diagnostic figure: dropout curve + side-by-side bar comparison
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: continuous theoretical dropout curve overlaid with empirical markers.
    # Build a dense grid of ranges for a smooth theoretical line
    dense_ranges = np.linspace(RANGE_MIN, RANGE_MAX, 200)
    # Normalise the dense range grid into [0, 1]
    dense_norm = (dense_ranges - RANGE_MIN) / max(RANGE_MAX - RANGE_MIN, 1e-12)
    # Evaluate the linear dropout model at every point in the dense grid
    dense_expected = np.clip(DROPOUT_PROB + DROPOUT_RANGE_COEFF * dense_norm, 0.0, 1.0)

    # Draw the theoretical dropout curve as an orange line
    ax_top.plot(dense_ranges, dense_expected, color="tab:orange", linewidth=2.0, label="theoretical")
    # Overlay the empirical dropout fractions as blue scatter points
    ax_top.scatter(TEST_RANGES, empirical_dropouts, s=50, color="tab:blue", zorder=3, label="empirical")
    # Extract the lower and upper CI bounds into arrays for vertical error bars
    ci_lo_arr = np.array([b[0] for b in ci_bounds])
    ci_hi_arr = np.array([b[1] for b in ci_bounds])
    # Draw vertical lines showing the 99% confidence interval at each test range
    ax_top.vlines(TEST_RANGES, ci_lo_arr, ci_hi_arr, color="tab:blue", linewidth=1.5, alpha=0.6, label="99% CI")
    ax_top.set_title(f"Dropout Fraction vs Range (N={N} per range)")
    ax_top.set_xlabel("True Range (m)")
    ax_top.set_ylabel("Dropout Probability")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: side-by-side bar chart comparing expected and empirical dropout
    # fractions at each discrete test range for a quick visual check.
    x_pos = np.arange(len(TEST_RANGES))
    width = 0.35
    # Orange bars for the theoretical expected dropout fractions
    ax_bottom.bar(x_pos - width / 2, expected_dropouts, width, color="tab:orange", alpha=0.8, label="expected")
    # Blue bars for the Monte Carlo observed dropout fractions
    ax_bottom.bar(x_pos + width / 2, empirical_dropouts, width, color="tab:blue", alpha=0.8, label="empirical")
    # Label the x-axis ticks with the actual range values in metres
    ax_bottom.set_xticks(x_pos)
    ax_bottom.set_xticklabels([f"{r:.0f}" for r in TEST_RANGES])
    ax_bottom.set_title("Empirical vs Expected Dropout at Each Test Range")
    ax_bottom.set_xlabel("True Range (m)")
    ax_bottom.set_ylabel("Dropout Probability")
    ax_bottom.grid(True, alpha=0.3, axis="y")
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Reserve right margin space for legends placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the figure into the HTML test report for later review
    attach_plot_to_html_report(request, fig, name="dropout_rate")
    # Release the figure memory after embedding
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output discretization and saturation tests
# ---------------------------------------------------------------------------


@pytest.mark.test_meta(
    description=(
        "Sweep true range with fine resolution through a nonzero quantization step and verify "
        "every output snaps to the nearest multiple. Confirm the quantization residual forms a "
        "sawtooth bounded by ±step/2."
    ),
    goal=(
        "Validate that the output quantization stage rounds each measurement to the nearest "
        "integer multiple of quantization_step, producing the expected staircase transfer "
        "function and bounded sawtooth residual."
    ),
    passing_criteria=(
        "Every measured range is an exact integer multiple of quantization_step. "
        "The residual (measured − true) is bounded within ±step/2 at every sample point."
    ),
)
def test_quantization_step(request):
    # Import matplotlib, skipping the test if not installed
    matplotlib = pytest.importorskip("matplotlib")
    # Select the non-interactive Agg backend for off-screen figure rendering
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Size of the quantization bin; every output must snap to a multiple of this
    QUANT_STEP = 0.5  # [m]

    # Configuration that enables only output quantization.
    # All other noise, bias, scale, dropout, and outlier terms are zeroed so the
    # measured range equals round(true_range / step) * step, producing a clean
    # staircase transfer function.
    class QuantizationConfig(LidarConfig):
        range_min = 1.0           # lower bound of the measurement window
        range_max = 100.0         # upper bound of the measurement window
        fov = np.deg2rad(60.0)    # wide FoV preventing angular rejection
        range_accuracy = 0.0      # no additive Gaussian range noise
        noise_floor_std = 0.0     # no noise floor contribution
        noise_range_coeff = 0.0   # no range-proportional noise
        bias_init_std = 0.0       # no initial bias scatter
        bias_rw_std = 0.0         # no bias random walk
        bias_drift_rate = 0.0     # no deterministic drift
        scale_factor_ppm = 0.0    # no scale factor offset
        scale_error_std_ppm = 0.0 # no per-sensor scale jitter
        dropout_prob = 0.0        # no measurement dropouts
        dropout_range_coeff = 0.0 # no range-dependent dropout growth
        outlier_prob = 0.0        # no outlier injection
        outlier_std = 0.0         # (unused when outlier_prob is zero)
        outlier_bias = 0.0        # (unused when outlier_prob is zero)
        quantization_step = QUANT_STEP  # the discretisation step under test
        saturate_output = True    # enable output clamping to [range_min, range_max]
        latency = 0.0             # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter
        random_seed = 42          # fixed seed for reproducibility

    # Instantiate a sensor with quantization as the only active effect
    lidar = Lidar(config=QuantizationConfig)

    # Sweep true range with fine resolution (much finer than the quantization step)
    # so the staircase structure and sawtooth residual are clearly resolved.
    sweep_step = QUANT_STEP / 50.0
    # Generate a dense array of true range values spanning the full sensor window
    true_ranges = np.arange(
        float(QuantizationConfig.range_min),
        float(QuantizationConfig.range_max) + sweep_step,
        sweep_step,
        dtype=float,
    )
    # Remove any values that exceed range_max to stay inside the valid envelope
    true_ranges = true_ranges[true_ranges <= float(QuantizationConfig.range_max)]

    # Preallocate the array of measured (quantized) ranges
    measured_ranges = np.empty_like(true_ranges)
    # Measure the boresight target at every fine-step range with noise disabled
    for i, r in enumerate(true_ranges):
        # Place the target on the boresight axis at distance r
        m = lidar.measure([r, 0.0, 0.0], add_noise=False, current_time=None)
        # Every measurement in this range window must be valid
        assert m["valid"] is True, f"unexpected invalid at range {r}"
        # Record the quantized output returned by the sensor
        measured_ranges[i] = float(m["range"])

    # Verify that every output is an exact integer multiple of the quantization step.
    # Dividing by the step should yield (to machine precision) a round integer.
    multiples = measured_ranges / QUANT_STEP
    np.testing.assert_allclose(
        multiples,
        np.round(multiples),
        rtol=0.0,
        atol=1e-12,
        err_msg="measured range is not an integer multiple of quantization_step",
    )

    # Compute the quantization residual: the difference between the quantized
    # output and the true input range. This should form a sawtooth wave.
    residuals = measured_ranges - true_ranges
    # The maximum possible rounding error is half the quantization step
    half_step = QUANT_STEP / 2.0
    # Assert no residual falls below the negative half-step bound
    assert np.all(residuals >= -half_step - 1e-12), (
        f"residual below -step/2: min = {residuals.min():.6f}"
    )
    # Assert no residual exceeds the positive half-step bound
    assert np.all(residuals <= half_step + 1e-12), (
        f"residual above +step/2: max = {residuals.max():.6f}"
    )

    # ---- Plot ----
    # Define a small zoom window so the staircase steps and sawtooth teeth
    # are individually visible rather than smeared together.
    zoom_lo = float(QuantizationConfig.range_min) + 5.0
    zoom_hi = zoom_lo + 5.0 * QUANT_STEP
    # Boolean mask selecting only the range samples inside the zoom window
    zoom_mask = (true_ranges >= zoom_lo) & (true_ranges <= zoom_hi)

    # Create a two-panel diagnostic figure
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: zoomed-in staircase transfer function.
    # Dashed orange line shows the ideal 1:1 response with no quantization
    ax_top.plot(true_ranges[zoom_mask], true_ranges[zoom_mask], color="tab:orange",
                linewidth=1.5, linestyle="--", label="ideal (no quantization)")
    # Solid blue line shows the actual quantized sensor output forming steps
    ax_top.plot(true_ranges[zoom_mask], measured_ranges[zoom_mask], color="tab:blue",
                linewidth=2.0, label=f"quantized (step = {QUANT_STEP} m)")
    ax_top.set_title("Quantization Staircase Transfer Function (zoomed)")
    ax_top.set_xlabel("True Range (m)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: sawtooth residual across the entire measurement range.
    # The residual oscillates between -step/2 and +step/2 in a periodic pattern.
    ax_bottom.plot(true_ranges, residuals, color="tab:blue", linewidth=0.8, label="residual")
    # Red dashed lines mark the theoretical +/- half-step bounds
    ax_bottom.axhline(half_step, color="tab:red", linestyle="--", linewidth=1.5, label=f"±step/2 = ±{half_step} m")
    ax_bottom.axhline(-half_step, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Quantization Residual (measured − true)")
    ax_bottom.set_xlabel("True Range (m)")
    ax_bottom.set_ylabel("Residual (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Reserve right margin space for legends placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="quantization_step")
    # Release figure memory after embedding
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Compare saturation clipping vs invalidation behaviour when the measured range is pushed "
        "beyond range_max (via large positive bias) and below range_min (via large negative bias). "
        "With saturate_output=True the output must clip; with saturate_output=False it must "
        "return an invalid measurement with reason='saturated'."
    ),
    goal=(
        "Confirm the two output limit modes work correctly: hard clipping to [range_min, range_max] "
        "versus outright measurement rejection when the noisy range falls outside the envelope."
    ),
    passing_criteria=(
        "Clipping mode: output equals range_max when bias pushes high, range_min when bias pushes low. "
        "Invalidation mode: measurement is invalid with reason='saturated' for both high and low cases. "
        "Nominal range (no bias push) remains valid and accurate in both modes."
    ),
)
def test_saturation_vs_invalidation(request):
    # Import matplotlib, skipping the test if the package is unavailable
    matplotlib = pytest.importorskip("matplotlib")
    # Select the non-interactive Agg backend for headless rendering
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sensor measurement envelope boundaries used by both config variants
    RANGE_MIN = 5.0
    RANGE_MAX = 200.0
    # Nominal target placed in the middle of the envelope for the baseline case
    TRUE_RANGE = 100.0  # [m] nominal target in the middle of the envelope
    # Large positive bias that pushes the measured range above RANGE_MAX
    HIGH_BIAS = 150.0   # [m] pushes measured to ~250 m (above range_max)
    # Large negative bias that pushes the measured range below RANGE_MIN
    LOW_BIAS = -120.0   # [m] pushes measured to ~-20 m (below range_min)

    # ---- Part A: saturate_output = True (clipping) ----
    # This configuration enables hard clamping: when the computed range falls
    # outside [range_min, range_max], it is clipped to the nearest boundary
    # and the measurement is still reported as valid.
    class ClippingConfig(LidarConfig):
        range_min = RANGE_MIN         # lower boundary of the measurement envelope
        range_max = RANGE_MAX         # upper boundary of the measurement envelope
        fov = np.deg2rad(60.0)        # wide FoV preventing angular rejection
        range_accuracy = 0.0          # no additive range noise
        noise_floor_std = 0.0         # no noise floor
        noise_range_coeff = 0.0       # no range-proportional noise
        bias_init_std = 0.0           # no initial bias scatter
        bias_rw_std = 0.0             # no bias random walk
        bias_drift_rate = 0.0         # no deterministic drift
        scale_factor_ppm = 0.0        # no scale factor offset
        scale_error_std_ppm = 0.0     # no per-sensor scale jitter
        dropout_prob = 0.0            # no dropouts
        dropout_range_coeff = 0.0     # no range-dependent dropout
        outlier_prob = 0.0            # no outlier injection
        outlier_std = 0.0             # (unused)
        outlier_bias = 0.0            # (unused)
        quantization_step = 0.0       # no output quantization
        saturate_output = True        # enable hard clipping to range boundaries
        latency = 0.0                 # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter
        random_seed = 42              # fixed seed for reproducibility

    # Nominal measurement with no bias override; the range is inside the
    # envelope so the output should pass through unchanged.
    lidar_clip = Lidar(config=ClippingConfig)
    m_nominal = lidar_clip.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    # The measurement must be flagged valid since the range is well within limits
    assert m_nominal["valid"] is True
    # The measured range should exactly equal the true range (no error sources active)
    np.testing.assert_allclose(float(m_nominal["range"]), TRUE_RANGE, rtol=0.0, atol=1e-12)

    # High bias case: manually inject a large positive bias so that
    # measured = true + bias = 100 + 150 = 250 m, which exceeds range_max.
    # With saturate_output=True, the output should be clipped to range_max.
    lidar_clip_hi = Lidar(config=ClippingConfig)
    # Override the internal bias state to simulate a large positive offset
    lidar_clip_hi._bias_state = HIGH_BIAS
    m_clip_hi = lidar_clip_hi.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    # The measurement should remain valid even though clamping was applied
    assert m_clip_hi["valid"] is True
    # The output must equal range_max because it was clipped from above
    np.testing.assert_allclose(float(m_clip_hi["range"]), RANGE_MAX, rtol=0.0, atol=1e-12)

    # Low bias case: inject a large negative bias so that
    # measured = true + bias = 100 + (-120) = -20 m, which is below range_min.
    # With saturate_output=True, the output should be clipped to range_min.
    lidar_clip_lo = Lidar(config=ClippingConfig)
    # Override the internal bias state to simulate a large negative offset
    lidar_clip_lo._bias_state = LOW_BIAS
    m_clip_lo = lidar_clip_lo.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    # The measurement should remain valid even though clamping was applied
    assert m_clip_lo["valid"] is True
    # The output must equal range_min because it was clipped from below
    np.testing.assert_allclose(float(m_clip_lo["range"]), RANGE_MIN, rtol=0.0, atol=1e-12)

    # ---- Part B: saturate_output = False (invalidation) ----
    # This config inherits everything from ClippingConfig but disables clamping.
    # Instead of clipping, out-of-range measurements are rejected entirely
    # and reported as invalid with reason "saturated".
    class InvalidationConfig(ClippingConfig):
        saturate_output = False

    # Nominal case: the range is well inside the envelope, so the measurement
    # should still be valid and accurate regardless of the saturation mode.
    lidar_inv = Lidar(config=InvalidationConfig)
    m_inv_nominal = lidar_inv.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    # Confirm the measurement is valid for an in-range target
    assert m_inv_nominal["valid"] is True
    # Confirm the output equals the true range with no error
    np.testing.assert_allclose(float(m_inv_nominal["range"]), TRUE_RANGE, rtol=0.0, atol=1e-12)

    # High bias case: measured = 250 m which exceeds range_max.
    # With saturate_output=False, the sensor should reject this measurement
    # rather than clipping it.
    lidar_inv_hi = Lidar(config=InvalidationConfig)
    # Inject the large positive bias to push the measurement above range_max
    lidar_inv_hi._bias_state = HIGH_BIAS
    m_inv_hi = lidar_inv_hi.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    # The measurement must be flagged invalid because the range exceeds the upper bound
    assert m_inv_hi["valid"] is False
    # The reason field should indicate saturation as the cause of rejection
    assert m_inv_hi.get("reason") == "saturated"

    # Low bias case: measured = -20 m which is below range_min.
    # With saturate_output=False, this should also be rejected as saturated.
    lidar_inv_lo = Lidar(config=InvalidationConfig)
    # Inject the large negative bias to push the measurement below range_min
    lidar_inv_lo._bias_state = LOW_BIAS
    m_inv_lo = lidar_inv_lo.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    # The measurement must be flagged invalid
    assert m_inv_lo["valid"] is False
    # The reason field confirms the rejection is due to saturation
    assert m_inv_lo.get("reason") == "saturated"

    # ---- Sweep: bias from large negative to large positive to show full behaviour ----
    # Create a fine sweep of bias values that spans well beyond the sensor
    # envelope in both directions. This sweep drives the measured range from
    # deeply negative to well above range_max, revealing the full clipping
    # and invalidation transfer functions.
    bias_sweep = np.linspace(-150.0, 200.0, 701, dtype=float)

    # Preallocate output arrays for both saturation modes
    clip_outputs = np.empty_like(bias_sweep)
    clip_valids = np.empty(bias_sweep.size, dtype=bool)
    # Invalidation outputs default to NaN for rejected measurements
    inv_outputs = np.full_like(bias_sweep, np.nan)
    inv_valids = np.empty(bias_sweep.size, dtype=bool)

    # Walk through each bias value, measuring with both config variants
    for i, bias in enumerate(bias_sweep):
        # Clipping mode: create a fresh sensor with saturate_output=True
        s_clip = Lidar(config=ClippingConfig)
        # Override the internal bias state to the current sweep value
        s_clip._bias_state = bias
        # Measure the boresight target; the sensor will clamp out-of-range results
        mc = s_clip.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
        # Record whether the measurement is valid and the output range
        clip_valids[i] = bool(mc["valid"])
        clip_outputs[i] = float(mc["range"]) if mc["valid"] else np.nan

        # Invalidation mode: create a fresh sensor with saturate_output=False
        s_inv = Lidar(config=InvalidationConfig)
        # Apply the same bias value
        s_inv._bias_state = bias
        # Measure the target; the sensor will reject out-of-range results
        mi = s_inv.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
        # Record validity and range (NaN if the measurement was rejected)
        inv_valids[i] = bool(mi["valid"])
        inv_outputs[i] = float(mi["range"]) if mi["valid"] else np.nan

    # Compute the ideal (unclamped, unrestricted) measured range for reference
    ideal_measured = TRUE_RANGE + bias_sweep  # what the range would be without limits

    # ---- Plot ----
    # Two-panel figure contrasting the clipping and invalidation transfer functions
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: clipping mode transfer function.
    # Dotted gray line shows the ideal unclamped response (true + bias)
    ax_top.plot(bias_sweep, ideal_measured, color="tab:gray", linewidth=1.0, linestyle=":", label="unclamped")
    # Solid blue line shows the actual clamped output, flat at the boundaries
    ax_top.plot(bias_sweep, clip_outputs, color="tab:blue", linewidth=2.0, label="saturate_output = True")
    # Red dashed lines mark the range_max and range_min clipping boundaries
    ax_top.axhline(RANGE_MAX, color="tab:red", linestyle="--", linewidth=1.5, label="range limits")
    ax_top.axhline(RANGE_MIN, color="tab:red", linestyle="--", linewidth=1.5)
    ax_top.set_title("Output Clipping (saturate_output = True)")
    ax_top.set_xlabel("Applied Bias (m)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: invalidation mode transfer function.
    # Dotted gray line again shows the ideal unclamped response
    ax_bottom.plot(bias_sweep, ideal_measured, color="tab:gray", linewidth=1.0, linestyle=":", label="unclamped")
    # Boolean masks separating valid and invalid measurements in the sweep
    valid_inv_mask = inv_valids
    invalid_inv_mask = ~inv_valids
    # Blue dots for measurements that remained valid (within the envelope)
    ax_bottom.scatter(bias_sweep[valid_inv_mask], inv_outputs[valid_inv_mask], s=6, color="tab:blue",
                      alpha=0.8, label="valid (saturate_output = False)")
    # Red X markers for rejected measurements plotted at their ideal position
    ax_bottom.scatter(bias_sweep[invalid_inv_mask], ideal_measured[invalid_inv_mask], s=6, color="tab:red",
                      alpha=0.5, marker="x", label="invalidated (reason = saturated)")
    # Red dashed lines marking the range boundaries where rejection kicks in
    ax_bottom.axhline(RANGE_MAX, color="tab:red", linestyle="--", linewidth=1.5, label="range limits")
    ax_bottom.axhline(RANGE_MIN, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Output Invalidation (saturate_output = False)")
    ax_bottom.set_xlabel("Applied Bias (m)")
    ax_bottom.set_ylabel("Measured / Ideal Range (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Reserve right margin space for legends placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the completed figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="saturation_vs_invalidation")
    # Release figure memory after embedding
    plt.close(fig)

