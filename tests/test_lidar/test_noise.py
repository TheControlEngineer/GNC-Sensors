"""
Gaussian noise, bias, scale factor, and outlier injection tests.

Validates the stochastic noise pipeline: Gaussian range noise statistics,
bias random walk, deterministic bias drift, scale factor (deterministic
and stochastic), and outlier injection rate and magnitude distribution.
"""

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.lidar import Lidar
from .helpers import attach_plot_to_html_report, chi2_variance_bounds, binomial_ci


@pytest.mark.test_meta(
    description=(
        "Monte Carlo validation of Gaussian range noise statistics against configured parameters. "
        "Collect N=10000 samples at fixed ranges, verify empirical standard deviation matches the "
        "RSS noise formula within a chi-squared confidence interval, and confirm near-zero mean error."
    ),
    goal=(
        "Confirm the stochastic range noise pipeline reproduces the expected distribution: "
        "zero-mean Gaussian with sigma_total(r) = sqrt(range_accuracy^2 + (noise_range_coeff * r)^2)."
    ),
    passing_criteria=(
        "At every test range the sample variance ratio s^2/sigma^2 lies within the 99 percent "
        "chi-squared confidence interval and the sample mean error is within 3*sigma/sqrt(N) of zero."
    ),
)
def test_gaussian_range_noise(request):
    # Import matplotlib and force the non-interactive Agg backend
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Monte Carlo parameters:
    N = 10_000                  # number of independent noise samples per test range
    SEED = 42                   # fixed random seed for reproducibility
    RANGE_ACCURACY = 0.10       # [m] constant accuracy term (range independent)
    NOISE_RANGE_COEFF = 0.001   # [m/m] range proportional noise growth coefficient
    # Array of test ranges spanning the operational envelope from near to far
    TEST_RANGES = np.array([10.0, 50.0, 200.0, 500.0, 800.0], dtype=float)

    # Configuration that enables only the two Gaussian noise terms:
    #   sigma_total(r) = sqrt(range_accuracy^2 + (noise_range_coeff * r)^2)
    # Every other error source (bias, scale factor, dropout, outlier, quantization,
    # latency) is disabled so the test isolates the additive Gaussian noise model.
    class GaussianNoiseConfig(LidarConfig):
        range_min = 0.5           # allow measurements starting at 0.5 m
        range_max = 1000.0        # generous max range to cover all test points
        fov = np.deg2rad(60.0)    # wide FoV to avoid rejection
        range_accuracy = RANGE_ACCURACY       # constant noise floor term
        noise_floor_std = 0.0     # no separate additive noise floor
        noise_range_coeff = NOISE_RANGE_COEFF # range proportional noise growth
        bias_init_std = 0.0       # no initial bias scatter
        bias_rw_std = 0.0         # no random walk on bias
        bias_drift_rate = 0.0     # no deterministic bias drift
        scale_factor_ppm = 0.0    # no scale factor offset
        scale_error_std_ppm = 0.0 # no stochastic scale jitter
        dropout_prob = 0.0        # no measurement dropouts
        dropout_range_coeff = 0.0 # no range dependent dropout growth
        outlier_prob = 0.0        # no outlier injections
        outlier_std = 0.0         # (unused)
        outlier_bias = 0.0        # (unused)
        quantization_step = 0.0   # infinite ADC resolution
        latency = 0.0             # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter
        random_seed = SEED        # deterministic seed for reproducible results

    # Instantiate the sensor with the Gaussian noise config
    lidar = Lidar(config=GaussianNoiseConfig)

    # Compute the chi-squared 99% confidence interval bounds for the sample
    # variance ratio s^2 / sigma^2. These bounds define the acceptable range
    # for validating that the empirical variance matches the theoretical model.
    var_lo, var_hi = chi2_variance_bounds(N, confidence=0.99)

    # Accumulators for per range statistics used later in the diagnostic plot
    empirical_stds = []
    empirical_means = []
    theoretical_stds = []

    # For each test range, collect N noisy measurements and validate the statistics
    for true_range in TEST_RANGES:
        # Preallocate an array for the range error samples (measured minus true)
        errors = np.empty(N, dtype=float)
        for i in range(N):
            # Measure the target on the boresight axis with noise enabled
            m = lidar.measure(
                rel_position=[true_range, 0.0, 0.0],
                add_noise=True,
                current_time=None,
            )
            # Every measurement must be valid since dropout_prob is zero
            assert m["valid"] is True, f"unexpected invalid at range {true_range}"
            # Store the range error: measured minus true
            errors[i] = float(m["range"]) - true_range

        # Compute sample statistics from the N error values
        sample_mean = float(np.mean(errors))
        # Unbiased sample standard deviation (Bessel correction, ddof=1)
        sample_std = float(np.std(errors, ddof=1))
        # Theoretical combined standard deviation from the RSS noise formula
        sigma_expected = float(np.sqrt(RANGE_ACCURACY**2 + (NOISE_RANGE_COEFF * true_range)**2))

        # Store the statistics for the diagnostic plot
        empirical_stds.append(sample_std)
        empirical_means.append(sample_mean)
        theoretical_stds.append(sigma_expected)

        # Mean error should be near zero: the tolerance is 3 standard errors,
        # providing approximately 99.7% coverage under the null hypothesis.
        mean_tol = 3.0 * sigma_expected / np.sqrt(N)
        assert abs(sample_mean) < mean_tol, (
            f"range {true_range} m: mean error {sample_mean:.6f} exceeds ±{mean_tol:.6f}"
        )

        # The sample variance ratio s^2/sigma^2 must lie within the chi-squared
        # 99% confidence interval, confirming the noise amplitude matches the model.
        var_ratio = (sample_std / sigma_expected) ** 2
        assert var_lo <= var_ratio <= var_hi, (
            f"range {true_range} m: variance ratio {var_ratio:.4f} outside "
            f"99% CI [{var_lo:.4f}, {var_hi:.4f}]"
        )

    # ---- Plot ----
    # Re-collect N error samples at the first test range specifically for the
    # histogram panel. A fresh Lidar instance resets the RNG seed to produce
    # an independent set of draws from the same distribution.
    hist_range = TEST_RANGES[0]
    hist_sigma = theoretical_stds[0]
    hist_lidar = Lidar(config=GaussianNoiseConfig)
    hist_errors = np.array([
        float(hist_lidar.measure([hist_range, 0.0, 0.0], add_noise=True, current_time=None)["range"]) - hist_range
        for _ in range(N)
    ])

    # Two panel diagnostic figure: error histogram + noise sigma vs range
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Top panel: normalized histogram of range errors overlaid with the
    # theoretical Gaussian PDF N(0, sigma^2) evaluated at the test range.
    bins = np.linspace(-4.0 * hist_sigma, 4.0 * hist_sigma, 81)
    ax_top.hist(hist_errors, bins=bins, density=True, color="tab:blue", alpha=0.7, label="empirical")
    # Evaluate the theoretical Gaussian PDF over a dense grid for the overlay
    x_pdf = np.linspace(bins[0], bins[-1], 300)
    pdf = np.exp(-0.5 * (x_pdf / hist_sigma) ** 2) / (hist_sigma * np.sqrt(2.0 * np.pi))
    ax_top.plot(x_pdf, pdf, color="tab:orange", linewidth=2.0, label=f"N(0, {hist_sigma:.4f}²)")
    ax_top.set_title(f"Range Error Distribution at {hist_range:.0f} m (N={N})")
    ax_top.set_xlabel("Range Error (m)")
    ax_top.set_ylabel("Probability Density")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Bottom panel: empirical std values (scatter) vs the theoretical RSS curve
    # (smooth line) across the full range of test distances.
    dense_ranges = np.linspace(TEST_RANGES[0], TEST_RANGES[-1], 200)
    dense_sigma = np.sqrt(RANGE_ACCURACY**2 + (NOISE_RANGE_COEFF * dense_ranges)**2)
    # Plot the theoretical RSS noise model as a continuous orange curve
    ax_bottom.plot(dense_ranges, dense_sigma, color="tab:orange", linewidth=2.0, label="theoretical RSS")
    # Shaded band showing the 99% chi-squared confidence interval around the curve
    ax_bottom.fill_between(
        dense_ranges,
        dense_sigma * np.sqrt(var_lo),
        dense_sigma * np.sqrt(var_hi),
        color="tab:orange",
        alpha=0.15,
        label="99% CI band",
    )
    # Overlay the empirical standard deviations as discrete scatter markers
    ax_bottom.scatter(TEST_RANGES, empirical_stds, s=50, color="tab:blue", zorder=3, label="empirical std")
    ax_bottom.set_title("Noise Standard Deviation vs Range")
    ax_bottom.set_xlabel("True Range (m)")
    ax_bottom.set_ylabel("Standard Deviation (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Leave room on the right for legends placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the finished figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="gaussian_range_noise")
    # Release the figure memory
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Ensemble Monte Carlo validation of the bias random walk process. "
        "Run N_ensemble=500 independent sensors over a long time series and verify "
        "the ensemble variance of bias(t) grows linearly as bias_rw_std^2 * t."
    ),
    goal=(
        "Confirm the bias state performs a Wiener process: Var[bias(T) - bias(0)] = sigma_rw^2 * T "
        "at several time horizons."
    ),
    passing_criteria=(
        "At every sampled time horizon the ensemble variance ratio Var/expected lies within "
        "the 99 percent chi-squared confidence interval. Sample trajectories visually exhibit "
        "random walk behavior."
    ),
)
def test_bias_random_walk(request):
    # Import matplotlib and force the non-interactive Agg backend
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ensemble Monte Carlo parameters:
    N_ENSEMBLE = 500             # number of independent sensor realisations
    BIAS_RW_STD = 0.01           # [m/sqrt(s)] random walk intensity parameter
    DT = 1.0                    # [s] time step between consecutive measurements
    T_MAX = 100.0               # [s] total simulation duration per ensemble member
    # Discrete time horizons at which the ensemble variance will be checked
    HORIZONS = np.array([5.0, 10.0, 20.0, 50.0, 100.0], dtype=float)
    N_STEPS = int(T_MAX / DT)   # number of time steps per ensemble run (100)

    # Configuration that enables only the bias random walk process.
    # All other noise, scale, dropout, and outlier terms are zeroed.
    # The sampling_rate is set to 1/DT so the sensor timestamps align with the
    # discrete time grid used in the simulation loop.
    class BiasRWConfig(LidarConfig):
        range_min = 0.5           # allow close range measurements
        range_max = 2000.0        # generous max range
        fov = np.deg2rad(60.0)    # wide FoV to avoid rejection
        sampling_rate = 1.0 / DT  # matches the discrete time step
        range_accuracy = 0.0      # no constant range noise
        noise_floor_std = 0.0     # no additive noise floor
        noise_range_coeff = 0.0   # no range proportional noise
        bias_init_std = 0.0       # bias starts exactly at zero
        bias_rw_std = BIAS_RW_STD # the random walk intensity being tested
        bias_drift_rate = 0.0     # no deterministic drift
        scale_factor_ppm = 0.0    # no scale factor offset
        scale_error_std_ppm = 0.0 # no scale jitter
        dropout_prob = 0.0        # no dropouts
        dropout_range_coeff = 0.0 # no range dependent dropout
        outlier_prob = 0.0        # no outliers
        outlier_std = 0.0         # (unused)
        outlier_bias = 0.0        # (unused)
        quantization_step = 0.0   # infinite ADC resolution
        latency = 0.0             # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter

    # Fixed target range along the boresight axis for every measurement
    TRUE_RANGE = 100.0  # [m]

    # Preallocate the bias trace matrix. Each row is one ensemble member,
    # each column is a time step. Shape: (N_ENSEMBLE, N_STEPS + 1) including t=0.
    bias_traces = np.zeros((N_ENSEMBLE, N_STEPS + 1), dtype=float)

    # Run each ensemble member with a unique random seed to produce independent
    # random walk realisations that can be used for ensemble statistics.
    for e in range(N_ENSEMBLE):

        # Create a per member config subclass that overrides only the random seed,
        # ensuring each ensemble member starts from a unique RNG state.
        class SeededConfig(BiasRWConfig):
            random_seed = 1000 + e

        # Instantiate a fresh sensor for this ensemble member
        lidar = Lidar(config=SeededConfig)

        # Step through the time series collecting the bias state at each tick
        for step in range(N_STEPS + 1):
            t = step * DT
            # Measure the fixed boresight target with noise enabled
            m = lidar.measure(
                rel_position=[TRUE_RANGE, 0.0, 0.0],
                add_noise=True,
                current_time=t,
            )
            # If the measurement is stale (e.g., sampling rate gating), carry
            # forward the previous bias value to avoid gaps in the trace.
            if m.get("stale"):
                bias_traces[e, step] = bias_traces[e, max(step - 1, 0)]
            else:
                # Record the internal bias state reported by the sensor
                bias_traces[e, step] = float(m.get("bias", 0.0))

    # Build the time axis array for indexing and plotting
    time_axis = np.arange(N_STEPS + 1) * DT
    # Compute the chi-squared 99% confidence interval bounds for the ensemble
    # variance ratio. This defines the acceptable band for Var[bias] / expected_var.
    var_lo, var_hi = chi2_variance_bounds(N_ENSEMBLE, confidence=0.99)

    # Validate that ensemble variance grows linearly at each time horizon,
    # matching the Wiener process property: Var[bias(T)] = sigma_rw^2 * T.
    horizon_vars = []
    for T in HORIZONS:
        # Convert the time horizon to an integer index into the bias trace columns
        idx = int(round(T / DT))
        # Compute bias displacement from t=0 for every ensemble member at time T
        bias_at_T = bias_traces[:, idx] - bias_traces[:, 0]
        # Unbiased sample variance across the ensemble at this time horizon
        ensemble_var = float(np.var(bias_at_T, ddof=1))
        # Theoretical Wiener variance: sigma_rw^2 * T
        expected_var = BIAS_RW_STD**2 * T
        # Store the empirical variance for the diagnostic plot
        horizon_vars.append(ensemble_var)

        # The variance ratio must lie within the chi-squared 99% CI, confirming
        # that the random walk intensity matches the configured parameter.
        if expected_var > 0.0:
            ratio = ensemble_var / expected_var
            assert var_lo <= ratio <= var_hi, (
                f"T={T:.0f}s: variance ratio {ratio:.4f} outside 99% CI [{var_lo:.4f}, {var_hi:.4f}]"
            )

    # ---- Plot ----
    # Two panel diagnostic figure: sample trajectories + variance growth
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Top panel: plot a small subset of individual bias random walk trajectories.
    # Visualising several paths gives an intuitive sense of the diffusion behavior.
    n_show = min(8, N_ENSEMBLE)
    for e in range(n_show):
        ax_top.plot(time_axis, bias_traces[e, :], linewidth=0.8, alpha=0.7)
    # Zero reference line showing the initial bias value
    ax_top.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax_top.set_title(f"Sample Bias Random Walk Trajectories ({n_show} of {N_ENSEMBLE})")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Bias State (m)")
    ax_top.grid(True, alpha=0.3)

    # Bottom panel: ensemble variance at each time step (blue) vs the
    # theoretical linear growth sigma_rw^2 * t (dashed orange).
    # Compute time series of ensemble variance after removing t=0 offset
    ensemble_var_series = np.var(bias_traces - bias_traces[:, 0:1], axis=0, ddof=1)
    # Theoretical variance: a straight line through the origin with slope sigma_rw^2
    theoretical_var = BIAS_RW_STD**2 * time_axis

    ax_bottom.plot(time_axis, ensemble_var_series, color="tab:blue", linewidth=1.5, label="ensemble variance")
    ax_bottom.plot(time_axis, theoretical_var, color="tab:orange", linewidth=2.0, linestyle="--", label="σ²_rw · t")
    # Overlay the explicitly checked horizons as red scatter markers
    ax_bottom.scatter(HORIZONS, horizon_vars, s=50, color="tab:red", zorder=3, label="checked horizons")
    ax_bottom.set_title("Bias Variance Growth vs Time")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Var[bias(t) − bias(0)]  (m²)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Leave room on the right for the external legend
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="bias_random_walk")
    # Release the figure memory
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Verify that enabling only bias_drift_rate produces a linearly growing range error "
        "whose slope matches the configured drift rate exactly."
    ),
    goal=(
        "Confirm the deterministic bias drift component: error(t) = bias_drift_rate * t "
        "with no stochastic scatter."
    ),
    passing_criteria=(
        "Measured range error at every time step equals bias_drift_rate * t within machine precision."
    ),
)
def test_bias_drift(request):
    # Import matplotlib and force off-screen rendering
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Deterministic drift parameters:
    DRIFT_RATE = 0.005  # [m/s] the constant rate at which bias grows over time
    DT = 1.0            # [s] time step between consecutive measurements
    T_MAX = 100.0        # [s] total simulation duration
    TRUE_RANGE = 100.0   # [m] fixed target range along boresight
    N_STEPS = int(T_MAX / DT)  # total number of time steps (100)

    # Configuration that enables only the deterministic bias drift rate.
    # Every stochastic term is zeroed, so the range error should grow as a
    # perfectly straight line: error(t) = DRIFT_RATE * t, with no scatter.
    class BiasDriftConfig(LidarConfig):
        range_min = 0.5           # allow close range measurements
        range_max = 2000.0        # generous max range
        fov = np.deg2rad(60.0)    # wide FoV to avoid rejection
        sampling_rate = 1.0 / DT  # matches the discrete time step
        range_accuracy = 0.0      # no constant range noise
        noise_floor_std = 0.0     # no additive noise floor
        noise_range_coeff = 0.0   # no range proportional noise
        bias_init_std = 0.0       # bias starts exactly at zero
        bias_rw_std = 0.0         # no random walk on bias
        bias_drift_rate = DRIFT_RATE  # the deterministic drift being tested
        scale_factor_ppm = 0.0    # no scale factor offset
        scale_error_std_ppm = 0.0 # no scale jitter
        dropout_prob = 0.0        # no dropouts
        dropout_range_coeff = 0.0 # no range dependent dropout
        outlier_prob = 0.0        # no outliers
        outlier_std = 0.0         # (unused)
        outlier_bias = 0.0        # (unused)
        quantization_step = 0.0   # infinite ADC resolution
        latency = 0.0             # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter
        random_seed = 42          # fixed seed (though no stochastic terms are active)

    # Instantiate the sensor with the drift only config
    lidar = Lidar(config=BiasDriftConfig)

    # Accumulators for the time stamps and range errors at each step
    times = []
    errors = []

    # Step through the full time series, measuring the fixed boresight target
    for step in range(N_STEPS + 1):
        t = step * DT
        # Measure with noise disabled (add_noise=False) so only the deterministic
        # drift component contributes to the range error
        m = lidar.measure(
            rel_position=[TRUE_RANGE, 0.0, 0.0],
            add_noise=False,
            current_time=t,
        )
        # Skip stale measurements that the sampling rate gate rejected
        if m.get("stale"):
            continue
        # Record the timestamp and the resulting range error
        times.append(t)
        errors.append(float(m["range"]) - TRUE_RANGE)

    # Convert to NumPy arrays for vectorised comparison
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)

    # The first measurement at t=0 has zero bias (drift has not yet accumulated).
    # Subsequent errors should grow as drift_rate * (t - t0).
    # The bias updates occur between consecutive measurement timestamps.
    # Expected error at time t relative to first measurement:
    #   error(t) = drift_rate * t  (since bias starts at 0 with bias_init_std=0)
    # But the first call initialises the bias clock; drift accumulates from the
    # second call onward.
    if len(times) > 1:
        t0 = times[0]
        # Build the expected linear ramp starting from zero at t0
        expected_errors = DRIFT_RATE * (times - t0)
        # Assert all errors match the linear ramp within machine precision
        np.testing.assert_allclose(errors, expected_errors, rtol=0.0, atol=1e-10)

    # Verify the slope via linear regression on the data excluding the first
    # point (which is always zero and can bias the fit intercept).
    if len(times) > 2:
        coeffs = np.polyfit(times[1:], errors[1:], 1)
        # The fitted slope must match the configured drift rate
        fitted_slope = coeffs[0]
        np.testing.assert_allclose(fitted_slope, DRIFT_RATE, rtol=1e-6)

    # ---- Plot ----
    # Single panel figure comparing measured range error against the expected
    # linear drift ramp. With no stochastic terms, these two lines should
    # overlap perfectly.
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot the measured range error over time as a solid blue line
    ax.plot(times, errors, linewidth=2.0, color="tab:blue", label="measured error")
    # Overlay the expected linear ramp as a dashed orange line
    ax.plot(times, DRIFT_RATE * (times - times[0]), linewidth=1.5, color="tab:orange", linestyle="--",
            label=f"expected drift ({DRIFT_RATE} m/s)")
    ax.set_title(f"Bias Deterministic Drift (rate = {DRIFT_RATE} m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Range Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    # Leave room on the right for the legend placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="bias_drift")
    # Release the figure memory
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Validate both deterministic and stochastic scale factor models. "
        "With a fixed scale_factor_ppm, verify measured = true * (1 + sf) at several ranges. "
        "With scale_error_std_ppm, instantiate N=10000 sensors and verify the distribution "
        "of realised scale factors matches the configured Gaussian."
    ),
    goal=(
        "Confirm the deterministic scale factor applies a precise multiplicative offset, "
        "and the stochastic per-sensor scale error is drawn from the correct N(0, sigma_ppm) distribution."
    ),
    passing_criteria=(
        "Deterministic case: measured equals true*(1+sf) within machine precision at every test range. "
        "Stochastic case: empirical mean and variance of realised scale factors lie within "
        "99 percent confidence intervals."
    ),
)
def test_scale_factor(request):
    # Import matplotlib, skipping the entire test if the package is not installed
    matplotlib = pytest.importorskip("matplotlib")
    # Select the non-interactive Agg backend so figures render off-screen
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Deterministic multiplicative offset applied uniformly in Part A
    SCALE_PPM = 50.0              # [ppm] deterministic offset
    # Standard deviation of the per-sensor random scale error tested in Part B
    SCALE_ERROR_STD_PPM = 20.0    # [ppm] per-sensor random component
    # Set of true ranges at which the deterministic scale factor is verified
    DETERMINISTIC_RANGES = np.array([10.0, 50.0, 200.0, 500.0, 1000.0], dtype=float)
    # Number of independently seeded sensor instances created for the stochastic test
    N_SENSORS = 10_000
    # Fixed range used by every stochastic sensor so the only variable is the scale draw
    STOCHASTIC_TEST_RANGE = 500.0  # [m]

    # ---- Part A: deterministic scale factor ----
    # Configuration that isolates only the deterministic scale factor.
    # Every noise, bias, dropout, and outlier term is zeroed so that the
    # measured range equals exactly true_range * (1 + scale_factor_ppm * 1e-6).
    class DeterministicScaleConfig(LidarConfig):
        range_min = 0.5           # generous lower bound to accept all test ranges
        range_max = 2000.0        # generous upper bound to accept all test ranges
        fov = np.deg2rad(60.0)    # wide field of view preventing angular rejection
        range_accuracy = 0.0      # no additive Gaussian range noise
        noise_floor_std = 0.0     # no noise floor contribution
        noise_range_coeff = 0.0   # no range-proportional noise growth
        bias_init_std = 0.0       # bias starts at exactly zero
        bias_rw_std = 0.0         # no random walk on the bias state
        bias_drift_rate = 0.0     # no deterministic drift
        scale_factor_ppm = SCALE_PPM   # the only active error source being tested
        scale_error_std_ppm = 0.0      # no per-sensor random scale scatter
        dropout_prob = 0.0        # no measurement dropouts
        dropout_range_coeff = 0.0 # no range-dependent dropout growth
        outlier_prob = 0.0        # no outlier injection
        outlier_std = 0.0         # (unused when outlier_prob is zero)
        outlier_bias = 0.0        # (unused when outlier_prob is zero)
        quantization_step = 0.0   # infinite ADC resolution, no rounding
        latency = 0.0             # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter
        random_seed = 42          # fixed seed for reproducibility

    # Instantiate a single sensor with the deterministic scale factor config
    lidar_det = Lidar(config=DeterministicScaleConfig)
    # Convert the scale factor from ppm to a dimensionless multiplier
    sf = SCALE_PPM * 1e-6  # dimensionless

    # Accumulators to hold the measured and analytically expected ranges
    det_measured = []
    det_expected = []
    # Sweep through each test range, measuring the boresight target with noise off
    for r in DETERMINISTIC_RANGES:
        # Measure along the boresight axis with noise disabled so only the
        # deterministic scale factor modifies the output
        m = lidar_det.measure([r, 0.0, 0.0], add_noise=False, current_time=None)
        # The measurement must be valid at every test range
        assert m["valid"] is True
        # Store the raw measured range returned by the sensor
        det_measured.append(float(m["range"]))
        # Compute the analytically expected range: true * (1 + sf)
        det_expected.append(r * (1.0 + sf))

    # Assert every measured range matches the expected scaled value within machine
    # precision, confirming the deterministic scale factor applies correctly
    np.testing.assert_allclose(det_measured, det_expected, rtol=0.0, atol=1e-10)

    # ---- Part B: stochastic scale factor distribution ----
    # Preallocate an array to hold the realised total scale factor from each sensor
    realised_sf = np.empty(N_SENSORS, dtype=float)
    # Loop over N_SENSORS independent sensor instances, each with a unique seed,
    # so that each sensor draws its own random scale error from the configured
    # Gaussian distribution N(0, scale_error_std_ppm^2).
    for i in range(N_SENSORS):

        # Per-sensor config: identical to Part A except scale_error_std_ppm is
        # now nonzero, introducing a per-instantiation random scale component.
        # The random_seed varies with the loop index so every sensor gets a
        # unique RNG state, producing an independent scale error draw.
        class StochasticScaleConfig(LidarConfig):
            range_min = 0.5           # generous lower bound
            range_max = 2000.0        # generous upper bound
            fov = np.deg2rad(60.0)    # wide field of view
            range_accuracy = 0.0      # no additive range noise
            noise_floor_std = 0.0     # no noise floor
            noise_range_coeff = 0.0   # no range-proportional noise
            bias_init_std = 0.0       # no initial bias scatter
            bias_rw_std = 0.0         # no bias random walk
            bias_drift_rate = 0.0     # no deterministic drift
            scale_factor_ppm = SCALE_PPM          # deterministic offset (same for all)
            scale_error_std_ppm = SCALE_ERROR_STD_PPM  # stochastic component under test
            dropout_prob = 0.0        # no dropouts
            dropout_range_coeff = 0.0 # no range-dependent dropout
            outlier_prob = 0.0        # no outliers
            outlier_std = 0.0         # (unused)
            outlier_bias = 0.0        # (unused)
            quantization_step = 0.0   # no quantization rounding
            latency = 0.0             # no delay
            sample_time_jitter_std = 0.0  # no clock jitter
            latency_jitter_std = 0.0      # no latency jitter
            random_seed = i           # unique seed per sensor for independent draws

        # Create a fresh sensor instance; the constructor draws a random scale error
        sensor = Lidar(config=StochasticScaleConfig)
        # Measure the fixed boresight target with noise disabled so the only
        # variable is the per-sensor scale factor (deterministic + random draw)
        m = sensor.measure([STOCHASTIC_TEST_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
        # Every measurement must be valid since range is well within limits
        assert m["valid"] is True
        # Recover the total scale factor: measured = true * (1 + sf_total)
        # Rearranging gives sf_total = measured / true - 1
        realised_sf[i] = float(m["range"]) / STOCHASTIC_TEST_RANGE - 1.0

    # The realised scale factors should follow a Gaussian distribution:
    #   N(scale_factor_ppm * 1e-6, (scale_error_std_ppm * 1e-6)^2)
    # The expected mean is the deterministic offset converted to dimensionless form
    expected_mean = SCALE_PPM * 1e-6
    # The expected standard deviation is the configured random spread in dimensionless form
    expected_std = SCALE_ERROR_STD_PPM * 1e-6

    # Compute the sample mean across all N_SENSORS realised scale factors
    sample_mean = float(np.mean(realised_sf))
    # Compute the unbiased sample standard deviation (ddof=1 applies Bessel correction)
    sample_std = float(np.std(realised_sf, ddof=1))

    # Mean tolerance: 3-sigma bound on the sampling error of the mean.
    # For N independent draws from N(mu, sigma^2), the standard error of
    # the sample mean is sigma / sqrt(N), so 3x that gives a 99.7% bound.
    mean_tol = 3.0 * expected_std / np.sqrt(N_SENSORS)
    # Assert the sample mean falls within the tolerance of the theoretical mean
    assert abs(sample_mean - expected_mean) < mean_tol, (
        f"scale factor mean {sample_mean:.3e} deviates from expected {expected_mean:.3e} by more than {mean_tol:.3e}"
    )

    # Compute chi-squared 99% confidence interval bounds for the variance ratio.
    # The ratio (sample_var / true_var) for N samples from a Gaussian follows a
    # scaled chi-squared distribution with N-1 degrees of freedom.
    var_lo, var_hi = chi2_variance_bounds(N_SENSORS, confidence=0.99)
    # Calculate the observed variance ratio relative to the expected variance
    var_ratio = (sample_std / expected_std) ** 2
    # Assert the variance ratio lies within the 99% chi-squared bounds
    assert var_lo <= var_ratio <= var_hi, (
        f"scale factor variance ratio {var_ratio:.4f} outside 99% CI [{var_lo:.4f}, {var_hi:.4f}]"
    )

    # ---- Plot ----
    # Create a two-panel diagnostic figure for visual inspection of results
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: deterministic scale factor validation.
    # Plot the analytically expected scaled ranges as a continuous orange line
    ax_top.plot(DETERMINISTIC_RANGES, det_expected, color="tab:orange", linewidth=2.0, label=f"true × (1 + {SCALE_PPM} ppm)")
    # Overlay the actual sensor measurements as blue scatter markers
    ax_top.scatter(DETERMINISTIC_RANGES, det_measured, s=50, color="tab:blue", zorder=3, label="measured")
    ax_top.set_title(f"Deterministic Scale Factor ({SCALE_PPM} ppm)")
    ax_top.set_xlabel("True Range (m)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    # Place the legend outside the axes to keep the data area uncluttered
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: histogram of the stochastic scale factor ensemble.
    # Convert from dimensionless to ppm for more intuitive axis labelling
    sf_ppm = realised_sf * 1e6  # convert to ppm for readability
    # Create 81 equally spaced bins spanning +/- 4 sigma around the expected mean
    bins = np.linspace(expected_mean * 1e6 - 4.0 * SCALE_ERROR_STD_PPM,
                       expected_mean * 1e6 + 4.0 * SCALE_ERROR_STD_PPM, 81)
    # Draw the normalised histogram of realised scale factors across all sensors
    ax_bottom.hist(sf_ppm, bins=bins, density=True, color="tab:blue", alpha=0.7, label="empirical")
    # Compute the theoretical Gaussian PDF for overlay comparison
    x_pdf = np.linspace(bins[0], bins[-1], 300)
    pdf = np.exp(-0.5 * ((x_pdf - SCALE_PPM) / SCALE_ERROR_STD_PPM) ** 2) / (SCALE_ERROR_STD_PPM * np.sqrt(2.0 * np.pi))
    # Plot the theoretical PDF as an orange curve on top of the histogram
    ax_bottom.plot(x_pdf, pdf, color="tab:orange", linewidth=2.0, label=f"N({SCALE_PPM}, {SCALE_ERROR_STD_PPM}²) ppm")
    ax_bottom.set_title(f"Stochastic Scale Factor Distribution (N={N_SENSORS} sensors)")
    ax_bottom.set_xlabel("Realised Scale Factor (ppm)")
    ax_bottom.set_ylabel("Probability Density")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Leave horizontal room on the right margin for legends placed outside the axes
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the finished figure into the HTML test report for later review
    attach_plot_to_html_report(request, fig, name="scale_factor")
    # Release figure memory now that the image has been saved
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Monte Carlo validation of outlier injection rate and magnitude distribution. "
        "With outlier_prob=0.10, collect N=10000 samples and verify the empirical outlier rate "
        "matches the configured probability. Confirm outlier magnitudes follow N(outlier_bias, outlier_std^2)."
    ),
    goal=(
        "Confirm the outlier injection mechanism fires at the correct rate and draws gross errors "
        "from the expected Gaussian distribution."
    ),
    passing_criteria=(
        "Empirical outlier rate lies within a 99 percent binomial confidence interval of outlier_prob. "
        "Outlier magnitude mean and standard deviation match outlier_bias and outlier_std within "
        "chi-squared and normal confidence bounds."
    ),
)
def test_outlier_injection(request):
    # Import matplotlib, skipping the test entirely if not available
    matplotlib = pytest.importorskip("matplotlib")
    # Use the non-interactive Agg backend for headless figure generation
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Total number of Monte Carlo samples to collect from a single sensor
    N = 10_000
    # Configured probability that any given measurement is replaced by an outlier
    OUTLIER_PROB = 0.10   # configured outlier probability
    # Deterministic offset added to every outlier measurement (mean of the gross error)
    OUTLIER_BIAS = 5.0    # [m] deterministic outlier offset
    # Standard deviation of the Gaussian spread around the outlier bias
    OUTLIER_STD = 2.0     # [m] outlier spread
    # Fixed boresight target range for all N measurements
    TRUE_RANGE = 100.0    # [m]
    # Reproducible random seed for the single sensor instance
    SEED = 42

    # Configuration isolating only the outlier injection mechanism.
    # All other noise, bias, scale, and dropout terms are zeroed so that
    # clean samples have exactly zero error and only outlier samples deviate.
    class OutlierConfig(LidarConfig):
        range_min = 0.5           # generous lower bound to prevent FoV rejection
        range_max = 2000.0        # generous upper bound for the test range
        fov = np.deg2rad(60.0)    # wide field of view to avoid angular rejection
        range_accuracy = 0.0      # no Gaussian range noise on clean samples
        noise_floor_std = 0.0     # no additive noise floor
        noise_range_coeff = 0.0   # no range-proportional noise
        bias_init_std = 0.0       # no initial bias scatter
        bias_rw_std = 0.0         # no bias random walk
        bias_drift_rate = 0.0     # no deterministic drift
        scale_factor_ppm = 0.0    # no scale factor offset
        scale_error_std_ppm = 0.0 # no per-sensor scale jitter
        dropout_prob = 0.0        # no dropouts so every sample is valid
        dropout_range_coeff = 0.0 # no range-dependent dropout
        outlier_prob = OUTLIER_PROB   # the outlier firing rate being validated
        outlier_std = OUTLIER_STD     # spread of the outlier gross error
        outlier_bias = OUTLIER_BIAS   # mean offset of the outlier gross error
        quantization_step = 0.0   # no output quantization
        latency = 0.0             # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter
        random_seed = SEED        # fixed seed for reproducibility

    # Instantiate a single sensor with the outlier-only configuration
    lidar = Lidar(config=OutlierConfig)

    # Preallocate arrays for range errors and per-sample outlier flags
    errors = np.empty(N, dtype=float)
    outlier_flags = np.empty(N, dtype=bool)

    # Collect N samples from the same sensor, each measuring the same target
    for i in range(N):
        # Measure with noise enabled so the outlier injection path is active
        m = lidar.measure([TRUE_RANGE, 0.0, 0.0], add_noise=True, current_time=None)
        # Every measurement should still be flagged valid (outliers are valid but erroneous)
        assert m["valid"] is True, f"unexpected invalid at sample {i}"
        # Record the range error: difference between measured and true range
        errors[i] = float(m["range"]) - TRUE_RANGE
        # Record whether the sensor internally flagged this sample as an outlier
        outlier_flags[i] = bool(m.get("outlier_applied", False))

    # ---- Outlier rate validation ----
    # Count how many of the N samples were flagged as outliers
    n_outliers = int(np.sum(outlier_flags))
    # Compute the observed outlier fraction and its 99% binomial confidence interval
    p_hat, ci_lo, ci_hi = binomial_ci(n_outliers, N, confidence=0.99)
    # The configured outlier probability must lie within the 99% CI of the
    # observed rate; otherwise the injection mechanism is mis-calibrated
    assert ci_lo <= OUTLIER_PROB <= ci_hi, (
        f"outlier_prob {OUTLIER_PROB} outside 99% CI [{ci_lo:.4f}, {ci_hi:.4f}] (observed {p_hat:.4f})"
    )

    # ---- Outlier magnitude distribution validation ----
    # Extract only the range errors from samples flagged as outliers
    outlier_errors = errors[outlier_flags]
    # There must be at least one outlier to run magnitude statistics
    assert outlier_errors.size > 0, "no outliers detected"

    # Compute the sample mean of outlier errors; should match outlier_bias
    ol_mean = float(np.mean(outlier_errors))
    # Compute the unbiased sample standard deviation of outlier errors
    ol_std = float(np.std(outlier_errors, ddof=1))
    # Store the count of outlier samples for tolerance calculations
    n_ol = outlier_errors.size

    # 3-sigma tolerance on the mean of a Gaussian sample of size n_ol
    mean_tol = 3.0 * OUTLIER_STD / np.sqrt(n_ol)
    # Assert the observed outlier mean is close to the configured outlier_bias
    assert abs(ol_mean - OUTLIER_BIAS) < mean_tol, (
        f"outlier mean {ol_mean:.4f} deviates from expected {OUTLIER_BIAS} by more than {mean_tol:.4f}"
    )

    # Check the variance of outlier magnitudes against chi-squared 99% bounds
    var_lo, var_hi = chi2_variance_bounds(n_ol, confidence=0.99)
    # Compute the ratio of observed variance to expected variance
    var_ratio = (ol_std / OUTLIER_STD) ** 2
    # The variance ratio must fall within the chi-squared confidence band
    assert var_lo <= var_ratio <= var_hi, (
        f"outlier std variance ratio {var_ratio:.4f} outside 99% CI [{var_lo:.4f}, {var_hi:.4f}]"
    )

    # ---- Non-outlier samples should have exactly zero error ----
    # Since all noise sources except outliers are disabled, clean samples
    # must return the true range with no deviation whatsoever
    clean_errors = errors[~outlier_flags]
    # Assert all clean errors are zero within floating-point tolerance
    assert np.allclose(clean_errors, 0.0, atol=1e-12), "non-outlier samples should have zero error"

    # ---- Plot ----
    # Two-panel diagnostic figure: error histogram + cumulative rate convergence
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: stacked histogram of all range errors, colour-coded by outlier flag.
    # Determine the histogram bounds to cover both the zero-spike (clean) and
    # the outlier cluster around OUTLIER_BIAS.
    all_min = min(errors.min(), -1.0)
    all_max = max(errors.max(), OUTLIER_BIAS + 4.0 * OUTLIER_STD)
    bins = np.linspace(all_min, all_max, 121)
    # Blue bars for clean (non-outlier) returns, which should cluster at zero
    ax_top.hist(errors[~outlier_flags], bins=bins, color="tab:blue", alpha=0.7, label="clean returns")
    # Red bars for outlier returns, clustered around OUTLIER_BIAS
    ax_top.hist(errors[outlier_flags], bins=bins, color="tab:red", alpha=0.7, label="outlier returns")
    ax_top.set_title(f"Range Error Distribution (N={N}, outlier_prob={OUTLIER_PROB})")
    ax_top.set_xlabel("Range Error (m)")
    ax_top.set_ylabel("Count")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: running average of the outlier rate as samples accumulate.
    # This visualises how the empirical rate converges toward the configured value.
    cumulative_rate = np.cumsum(outlier_flags) / (np.arange(N) + 1.0)
    # Plot the cumulative rate as a blue trace evolving over sample number
    ax_bottom.plot(np.arange(N) + 1, cumulative_rate, color="tab:blue", linewidth=1.0, label="cumulative rate")
    # Dashed orange line at the configured outlier probability for reference
    ax_bottom.axhline(OUTLIER_PROB, color="tab:orange", linewidth=2.0, linestyle="--", label=f"configured = {OUTLIER_PROB}")
    # Dotted red lines marking the 99% binomial confidence interval bounds
    ax_bottom.axhline(ci_lo, color="tab:red", linewidth=1.0, linestyle=":", label="99% CI bounds")
    ax_bottom.axhline(ci_hi, color="tab:red", linewidth=1.0, linestyle=":")
    ax_bottom.set_title("Cumulative Outlier Rate Convergence")
    ax_bottom.set_xlabel("Sample Number")
    ax_bottom.set_ylabel("Cumulative Outlier Rate")
    # Limit the y-axis to twice the expected rate for a cleaner view
    ax_bottom.set_ylim(0.0, 2.0 * OUTLIER_PROB)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Reserve right margin space for legends placed outside the plot area
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the completed figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="outlier_injection")
    # Free the figure memory after embedding
    plt.close(fig)

