import base64
import io

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.physics import Material, Plane, Scene, Sphere
from sensors.lidar import Lidar


def _attach_plot_to_html_report(request, fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    html_plugin = request.config.pluginmanager.getplugin("html")
    if html_plugin is not None and hasattr(html_plugin, "extras"):
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        extra = getattr(request.node, "extra", [])
        extra.append(html_plugin.extras.png(png_b64, name=name))
        request.node.extra = extra


def _chi2_variance_bounds(n, confidence=0.99):
    """
    Compute chi-squared confidence interval ratio bounds for sample variance.

    Uses the Wilson-Hilferty normal approximation for chi-squared quantiles.
    Returns (lower_ratio, upper_ratio) such that s^2 / sigma^2 should lie
    within [lower_ratio, upper_ratio] at the given confidence level.

    :param n:          number of samples
    :param confidence: two-sided confidence level (default 0.99)
    :return: (lower_ratio, upper_ratio) for s^2 / sigma^2
    """
    z_table = {0.99: 2.576, 0.95: 1.960, 0.90: 1.645}
    z = z_table.get(confidence, 2.576)

    nu = float(n - 1)  # degrees of freedom
    a = 2.0 / (9.0 * nu)
    chi2_lo = nu * (1.0 - a - z * np.sqrt(a)) ** 3  # lower chi-squared quantile
    chi2_hi = nu * (1.0 - a + z * np.sqrt(a)) ** 3  # upper chi-squared quantile
    return chi2_lo / nu, chi2_hi / nu


def _binomial_ci(k, n, confidence=0.99):
    """
    Normal approximation confidence interval for a binomial proportion.

    :param k:          number of successes
    :param n:          number of trials
    :param confidence: two-sided confidence level
    :return: (p_hat, ci_lower, ci_upper)
    """
    z_table = {0.99: 2.576, 0.95: 1.960, 0.90: 1.645}
    z = z_table.get(confidence, 2.576)

    p_hat = k / n
    margin = z * np.sqrt(p_hat * (1.0 - p_hat) / n)
    return p_hat, max(p_hat - margin, 0.0), min(p_hat + margin, 1.0)


@pytest.mark.test_meta(
    description="Verify zero noise boresight ranging over the full measurement envelope.",
    goal="Confirm the deterministic measurement pipeline reproduces ground truth range with no stochastic terms.",
    passing_criteria="Measured range equals true range at machine precision for every sampled distance from 0 to max range.",
)
def test_range_accuracy_zero_noise_boresight(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class ZeroNoiseLidarConfig(LidarConfig):
        range_min = 0.0
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    lidar = Lidar(config=ZeroNoiseLidarConfig)

    step = max(float(LidarConfig.range_accuracy), np.finfo(float).eps)
    range_max = float(lidar.range_max)
    true_ranges_sweep = np.arange(step, range_max, step, dtype=float)
    if true_ranges_sweep.size == 0 or not np.isclose(true_ranges_sweep[-1], range_max):
        true_ranges_sweep = np.append(true_ranges_sweep, range_max)

    measured_ranges_sweep = []
    for true_range in true_ranges_sweep:
        measurement = lidar.measure(
            rel_position=[true_range, 0.0, 0.0],
            add_noise=False,
            current_time=None,
        )
        assert measurement["valid"] is True
        measured_ranges_sweep.append(float(measurement["range"]))

    measured_ranges_sweep = np.asarray(measured_ranges_sweep, dtype=float)
    np.testing.assert_allclose(
        measured_ranges_sweep,
        true_ranges_sweep,
        rtol=0.0,
        atol=np.finfo(float).eps,
    )

    true_ranges = np.concatenate(([0.0], true_ranges_sweep))
    measured_ranges = np.concatenate(([0.0], measured_ranges_sweep))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(true_ranges, measured_ranges, linewidth=2.0, color="tab:blue")
    ax.set_title(f"LiDAR Range Accuracy at Zero Noise (step = {step:g} m)")
    ax.set_xlabel("True Range (m)")
    ax.set_ylabel("Measured Range (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _attach_plot_to_html_report(request, fig, name="range_accuracy_zero_noise")
    plt.close(fig)


@pytest.mark.test_meta(
    description="Sweep target angle from outside negative FoV to outside positive FoV and evaluate boundary behavior.",
    goal="Identify where measurement validity transitions between valid and out_of_fov relative to half FoV limits.",
    passing_criteria="Sweep starts invalid, center is valid, sweep ends invalid, and epsilon checks at both boundaries match expected valid/invalid transitions.",
)
def test_fov_boundary_transition(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class ZeroNoiseLidarConfig(LidarConfig):
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    lidar = Lidar(config=ZeroNoiseLidarConfig)

    epsilon = 1e-4
    half_fov = float(lidar.fov * 0.5)
    test_range = max(float(lidar.range_min), 1.0) + 9.0

    pos_boundary_angles = np.array([half_fov - epsilon, half_fov, half_fov + epsilon], dtype=float)
    neg_boundary_angles = np.array([-half_fov - epsilon, -half_fov, -half_fov + epsilon], dtype=float)

    pos_valids = []
    pos_reasons = []
    for angle in pos_boundary_angles:
        measurement = lidar.measure(
            rel_position=[test_range * np.cos(angle), test_range * np.sin(angle), 0.0],
            add_noise=False,
            current_time=None,
        )
        pos_valids.append(bool(measurement["valid"]))
        pos_reasons.append(measurement.get("reason", ""))

    neg_valids = []
    neg_reasons = []
    for angle in neg_boundary_angles:
        measurement = lidar.measure(
            rel_position=[test_range * np.cos(angle), test_range * np.sin(angle), 0.0],
            add_noise=False,
            current_time=None,
        )
        neg_valids.append(bool(measurement["valid"]))
        neg_reasons.append(measurement.get("reason", ""))

    assert pos_valids == [True, True, False]
    assert pos_reasons[-1] == "out_of_fov"
    assert neg_valids == [False, True, True]
    assert neg_reasons[0] == "out_of_fov"

    sweep_margin = max(np.deg2rad(3.0), 8.0 * epsilon)
    sweep_angles = np.linspace(-(half_fov + sweep_margin), half_fov + sweep_margin, 801, dtype=float)
    sweep_valids = []
    for angle in sweep_angles:
        measurement = lidar.measure(
            rel_position=[test_range * np.cos(angle), test_range * np.sin(angle), 0.0],
            add_noise=False,
            current_time=None,
        )
        sweep_valids.append(bool(measurement["valid"]))

    sweep_valid_numeric = np.asarray(sweep_valids, dtype=float)
    assert bool(sweep_valids[0]) is False
    assert bool(sweep_valids[-1]) is False
    assert bool(sweep_valids[len(sweep_valids) // 2]) is True

    sweep_angles_deg = np.rad2deg(sweep_angles)
    half_fov_deg = np.rad2deg(half_fov)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    target_x = test_range * np.cos(sweep_angles)
    target_y = test_range * np.sin(sweep_angles)
    valid_mask = sweep_valid_numeric > 0.5
    invalid_mask = ~valid_mask

    ax_top.scatter(target_x[invalid_mask], target_y[invalid_mask], s=10, alpha=0.7, color="tab:red", label="invalid targets")
    ax_top.scatter(target_x[valid_mask], target_y[valid_mask], s=10, alpha=0.7, color="tab:green", label="valid targets")
    ax_top.scatter([0.0], [0.0], marker="x", s=70, color="black", label="sensor")
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

    ax_bottom.scatter(sweep_angles_deg, sweep_valid_numeric, s=10, alpha=0.7, color="tab:blue", label="sweep results")
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
    ax_bottom.axvline(-half_fov_deg, color="tab:red", linestyle="--", linewidth=1.5, label="half fov boundaries")
    ax_bottom.axvline(half_fov_deg, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Validity Results for the Sweep")
    ax_bottom.set_xlabel("Target Angle from Boresight (deg)")
    ax_bottom.set_ylabel("Validity")
    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_ylim(-0.1, 1.1)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="fov_boundary_transition")
    plt.close(fig)


@pytest.mark.test_meta(
    description="Sweep target range from below minimum range to above maximum range and evaluate boundary behavior.",
    goal="Identify where measurement validity transitions between valid and out_of_range at range_min and range_max.",
    passing_criteria="Near range_min, minus epsilon is invalid while boundary and plus epsilon are valid; near range_max, minus epsilon and boundary are valid while plus epsilon is invalid; sweep starts invalid, is valid inside limits, and ends invalid.",
)
def test_range_boundary_transition(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class ZeroNoiseLidarConfig(LidarConfig):
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    lidar = Lidar(config=ZeroNoiseLidarConfig)

    range_min = float(lidar.range_min)
    range_max = float(lidar.range_max)
    epsilon = 1e-4

    min_boundary_ranges = np.array([range_min - epsilon, range_min, range_min + epsilon], dtype=float)
    max_boundary_ranges = np.array([range_max - epsilon, range_max, range_max + epsilon], dtype=float)

    min_valids = []
    min_reasons = []
    for target_range in min_boundary_ranges:
        measurement = lidar.measure(
            rel_position=[target_range, 0.0, 0.0],
            add_noise=False,
            current_time=None,
        )
        min_valids.append(bool(measurement["valid"]))
        min_reasons.append(measurement.get("reason", ""))

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

    assert min_valids == [False, True, True]
    assert min_reasons[0] == "out_of_range"
    assert max_valids == [True, True, False]
    assert max_reasons[-1] == "out_of_range"

    sweep_margin = max(5.0, 10.0 * epsilon)
    sweep_ranges = np.linspace(max(range_min - sweep_margin, 0.0), range_max + sweep_margin, 801, dtype=float)
    sweep_valids = []
    for target_range in sweep_ranges:
        measurement = lidar.measure(
            rel_position=[target_range, 0.0, 0.0],
            add_noise=False,
            current_time=None,
        )
        sweep_valids.append(bool(measurement["valid"]))

    sweep_valid_numeric = np.asarray(sweep_valids, dtype=float)
    assert bool(sweep_valids[0]) is False
    assert bool(sweep_valids[-1]) is False
    assert np.any(sweep_valid_numeric > 0.5)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    valid_mask = sweep_valid_numeric > 0.5
    invalid_mask = ~valid_mask
    zeros = np.zeros_like(sweep_ranges)

    ax_top.scatter(sweep_ranges[invalid_mask], zeros[invalid_mask], s=10, alpha=0.7, color="tab:red", label="invalid targets")
    ax_top.scatter(sweep_ranges[valid_mask], zeros[valid_mask], s=10, alpha=0.7, color="tab:green", label="valid targets")
    ax_top.axvline(range_min, color="tab:orange", linestyle="--", linewidth=1.5, label="range boundaries")
    ax_top.axvline(range_max, color="tab:orange", linestyle="--", linewidth=1.5)
    ax_top.set_title("Target Range Sweep Along Boresight")
    ax_top.set_xlabel("Target Range (m)")
    ax_top.set_ylabel("Target y (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    ax_bottom.scatter(sweep_ranges, sweep_valid_numeric, s=10, alpha=0.7, color="tab:blue", label="sweep results")
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
    ax_bottom.axvline(range_min, color="tab:red", linestyle="--", linewidth=1.5, label="range boundaries")
    ax_bottom.axvline(range_max, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Validity Results for Range Sweep")
    ax_bottom.set_xlabel("Target Range (m)")
    ax_bottom.set_ylabel("Validity")
    ax_bottom.set_yticks([0.0, 1.0])
    ax_bottom.set_ylim(-0.1, 1.1)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="range_boundary_transition")
    plt.close(fig)


@pytest.mark.test_meta(
    description="Fire a single elevation scan pattern at a planar wall perpendicular to boresight and validate beam geometry.",
    goal="Confirm hit geometry follows the expected wall intersection model, azimuth indices are ordered correctly, and azimuth spacing matches the configured linspace pattern.",
    passing_criteria="All beams return valid hits on the wall, hit points match analytic intersections, azimuth_index equals 0..N-1, and azimuth samples and spacing match linspace(az_min, az_max, N).",
)
def test_beam_pattern_geometry_planar_wall(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    az_min = np.deg2rad(-20.0)
    az_max = np.deg2rad(20.0)
    az_samples = 121
    wall_distance = 25.0

    class BeamGeometryConfig(LidarConfig):
        range_min = 0.1
        range_max = 200.0
        fov = np.deg2rad(40.0)
        scan_azimuth_min = az_min
        scan_azimuth_max = az_max
        scan_azimuth_samples = az_samples
        scan_elevation_angles = (0.0,)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    lidar = Lidar(config=BeamGeometryConfig)

    scene = Scene(
        [
            Plane(
                point=[wall_distance, 0.0, 0.0],
                normal=[-1.0, 0.0, 0.0],
                material=Material(reflectivity=1.0, retro_reflectivity=0.0, name="wall"),
                object_id="wall",
            )
        ]
    )

    frame = lidar.simulate_scene_frame(
        scene=scene,
        sensor_position=[0.0, 0.0, 0.0],
        sensor_orientation=None,
        timestamp=0.0,
        add_noise=False,
    )

    returns = frame["returns"]
    assert frame["valid"] is True
    assert frame["num_beams"] == az_samples
    assert frame["num_valid"] == az_samples
    assert len(returns) == az_samples

    expected_azimuths = np.linspace(az_min, az_max, az_samples, dtype=float)
    expected_spacing = np.diff(expected_azimuths)
    expected_ranges = wall_distance / np.cos(expected_azimuths)
    expected_hit_x = np.full_like(expected_azimuths, wall_distance, dtype=float)
    expected_hit_y = wall_distance * np.tan(expected_azimuths)
    expected_hit_z = np.zeros_like(expected_azimuths, dtype=float)

    actual_azimuths = np.asarray([float(item["azimuth"]) for item in returns], dtype=float)
    actual_indices = np.asarray([int(item["azimuth_index"]) for item in returns], dtype=int)
    actual_valids = [bool(item["valid"]) for item in returns]
    actual_measured_ranges = np.asarray([float(item["range"]) for item in returns], dtype=float)
    actual_truth_ranges = np.asarray([float(item["truth_range"]) for item in returns], dtype=float)
    actual_hit_points = np.asarray([item["hit_point"] for item in returns], dtype=float)

    assert all(actual_valids)
    np.testing.assert_array_equal(actual_indices, np.arange(az_samples, dtype=int))
    np.testing.assert_allclose(actual_azimuths, expected_azimuths, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.diff(actual_azimuths), expected_spacing, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual_truth_ranges, expected_ranges, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_measured_ranges, expected_ranges, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_hit_points[:, 0], expected_hit_x, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_hit_points[:, 1], expected_hit_y, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_hit_points[:, 2], expected_hit_z, rtol=0.0, atol=1e-12)

    az_deg = np.rad2deg(expected_azimuths)
    spacing_error_deg = np.rad2deg(np.diff(actual_azimuths) - expected_spacing)
    index_values = np.arange(az_samples, dtype=int)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    ax_top.plot(az_deg, expected_ranges, color="tab:orange", linewidth=1.5, label="expected range profile")
    ax_top.scatter(az_deg, actual_measured_ranges, s=12, alpha=0.8, color="tab:blue", label="measured range")
    ax_top.set_title("Planar Wall Beam Geometry: Range Profile Across Azimuth")
    ax_top.set_xlabel("Azimuth (deg)")
    ax_top.set_ylabel("Range to Wall (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    ax_bottom.plot(index_values, az_deg, color="tab:blue", linewidth=1.5, label="azimuth by index")
    ax_bottom.scatter(index_values, az_deg, s=12, alpha=0.8, color="tab:blue")
    ax_bottom.set_title("Azimuth Index Mapping and Spacing Consistency")
    ax_bottom.set_xlabel("Azimuth Index")
    ax_bottom.set_ylabel("Azimuth (deg)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom_twin = ax_bottom.twinx()
    ax_bottom_twin.plot(
        index_values[:-1] + 0.5,
        spacing_error_deg,
        color="tab:red",
        linewidth=1.2,
        label="spacing error",
    )
    ax_bottom_twin.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    ax_bottom_twin.set_ylabel("Spacing Error (deg)")

    handles_left, labels_left = ax_bottom.get_legend_handles_labels()
    handles_right, labels_right = ax_bottom_twin.get_legend_handles_labels()
    ax_bottom.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="beam_pattern_geometry_planar_wall")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Monte Carlo stochastic model validation tests
# ---------------------------------------------------------------------------


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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 10_000
    SEED = 42
    RANGE_ACCURACY = 0.10       # [m] constant accuracy term
    NOISE_RANGE_COEFF = 0.001   # [m/m] range proportional term
    TEST_RANGES = np.array([10.0, 50.0, 200.0, 500.0, 800.0], dtype=float)

    class GaussianNoiseConfig(LidarConfig):
        range_min = 0.5
        range_max = 1000.0
        fov = np.deg2rad(60.0)
        range_accuracy = RANGE_ACCURACY
        noise_floor_std = 0.0
        noise_range_coeff = NOISE_RANGE_COEFF
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0
        random_seed = SEED

    lidar = Lidar(config=GaussianNoiseConfig)

    # Chi-squared variance ratio bounds for a 99 % confidence interval.
    var_lo, var_hi = _chi2_variance_bounds(N, confidence=0.99)

    empirical_stds = []
    empirical_means = []
    theoretical_stds = []

    for true_range in TEST_RANGES:
        errors = np.empty(N, dtype=float)
        for i in range(N):
            m = lidar.measure(
                rel_position=[true_range, 0.0, 0.0],
                add_noise=True,
                current_time=None,
            )
            assert m["valid"] is True, f"unexpected invalid at range {true_range}"
            errors[i] = float(m["range"]) - true_range

        sample_mean = float(np.mean(errors))
        sample_std = float(np.std(errors, ddof=1))
        sigma_expected = float(np.sqrt(RANGE_ACCURACY**2 + (NOISE_RANGE_COEFF * true_range)**2))

        empirical_stds.append(sample_std)
        empirical_means.append(sample_mean)
        theoretical_stds.append(sigma_expected)

        # Mean error should be near zero: |mean| < 3 * sigma / sqrt(N).
        mean_tol = 3.0 * sigma_expected / np.sqrt(N)
        assert abs(sample_mean) < mean_tol, (
            f"range {true_range} m: mean error {sample_mean:.6f} exceeds ±{mean_tol:.6f}"
        )

        # Variance ratio must fall within the chi-squared 99 % CI.
        var_ratio = (sample_std / sigma_expected) ** 2
        assert var_lo <= var_ratio <= var_hi, (
            f"range {true_range} m: variance ratio {var_ratio:.4f} outside "
            f"99% CI [{var_lo:.4f}, {var_hi:.4f}]"
        )

    # ---- Plot ----
    # Re-collect error samples at the first test range for the histogram panel.
    hist_range = TEST_RANGES[0]
    hist_sigma = theoretical_stds[0]
    hist_lidar = Lidar(config=GaussianNoiseConfig)
    hist_errors = np.array([
        float(hist_lidar.measure([hist_range, 0.0, 0.0], add_noise=True, current_time=None)["range"]) - hist_range
        for _ in range(N)
    ])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: error histogram with Gaussian overlay.
    bins = np.linspace(-4.0 * hist_sigma, 4.0 * hist_sigma, 81)
    ax_top.hist(hist_errors, bins=bins, density=True, color="tab:blue", alpha=0.7, label="empirical")
    x_pdf = np.linspace(bins[0], bins[-1], 300)
    pdf = np.exp(-0.5 * (x_pdf / hist_sigma) ** 2) / (hist_sigma * np.sqrt(2.0 * np.pi))
    ax_top.plot(x_pdf, pdf, color="tab:orange", linewidth=2.0, label=f"N(0, {hist_sigma:.4f}²)")
    ax_top.set_title(f"Range Error Distribution at {hist_range:.0f} m (N={N})")
    ax_top.set_xlabel("Range Error (m)")
    ax_top.set_ylabel("Probability Density")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: empirical std vs range with theoretical RSS curve and CI band.
    dense_ranges = np.linspace(TEST_RANGES[0], TEST_RANGES[-1], 200)
    dense_sigma = np.sqrt(RANGE_ACCURACY**2 + (NOISE_RANGE_COEFF * dense_ranges)**2)
    ax_bottom.plot(dense_ranges, dense_sigma, color="tab:orange", linewidth=2.0, label="theoretical RSS")
    ax_bottom.fill_between(
        dense_ranges,
        dense_sigma * np.sqrt(var_lo),
        dense_sigma * np.sqrt(var_hi),
        color="tab:orange",
        alpha=0.15,
        label="99% CI band",
    )
    ax_bottom.scatter(TEST_RANGES, empirical_stds, s=50, color="tab:blue", zorder=3, label="empirical std")
    ax_bottom.set_title("Noise Standard Deviation vs Range")
    ax_bottom.set_xlabel("True Range (m)")
    ax_bottom.set_ylabel("Standard Deviation (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="gaussian_range_noise")
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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N_ENSEMBLE = 500
    BIAS_RW_STD = 0.01           # [m/sqrt(s)]
    DT = 1.0                    # [s] time step = 1 / sampling_rate
    T_MAX = 100.0               # [s] total simulation duration
    HORIZONS = np.array([5.0, 10.0, 20.0, 50.0, 100.0], dtype=float)
    N_STEPS = int(T_MAX / DT)   # 100 steps per ensemble member

    class BiasRWConfig(LidarConfig):
        range_min = 0.5
        range_max = 2000.0
        fov = np.deg2rad(60.0)
        sampling_rate = 1.0 / DT
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = BIAS_RW_STD
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    TRUE_RANGE = 100.0  # [m] fixed boresight target

    # Collect bias time series for every ensemble member.
    # Shape: (N_ENSEMBLE, N_STEPS + 1) including the t=0 initial state.
    bias_traces = np.zeros((N_ENSEMBLE, N_STEPS + 1), dtype=float)

    for e in range(N_ENSEMBLE):

        class SeededConfig(BiasRWConfig):
            random_seed = 1000 + e

        lidar = Lidar(config=SeededConfig)

        for step in range(N_STEPS + 1):
            t = step * DT
            m = lidar.measure(
                rel_position=[TRUE_RANGE, 0.0, 0.0],
                add_noise=True,
                current_time=t,
            )
            if m.get("stale"):
                bias_traces[e, step] = bias_traces[e, max(step - 1, 0)]
            else:
                bias_traces[e, step] = float(m.get("bias", 0.0))

    # Validate variance growth at each time horizon.
    time_axis = np.arange(N_STEPS + 1) * DT
    var_lo, var_hi = _chi2_variance_bounds(N_ENSEMBLE, confidence=0.99)

    horizon_vars = []
    for T in HORIZONS:
        idx = int(round(T / DT))
        bias_at_T = bias_traces[:, idx] - bias_traces[:, 0]
        ensemble_var = float(np.var(bias_at_T, ddof=1))
        expected_var = BIAS_RW_STD**2 * T
        horizon_vars.append(ensemble_var)

        if expected_var > 0.0:
            ratio = ensemble_var / expected_var
            assert var_lo <= ratio <= var_hi, (
                f"T={T:.0f}s: variance ratio {ratio:.4f} outside 99% CI [{var_lo:.4f}, {var_hi:.4f}]"
            )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: sample bias trajectories.
    n_show = min(8, N_ENSEMBLE)
    for e in range(n_show):
        ax_top.plot(time_axis, bias_traces[e, :], linewidth=0.8, alpha=0.7)
    ax_top.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax_top.set_title(f"Sample Bias Random Walk Trajectories ({n_show} of {N_ENSEMBLE})")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_ylabel("Bias State (m)")
    ax_top.grid(True, alpha=0.3)

    # Panel 2: ensemble variance vs time with theoretical line.
    ensemble_var_series = np.var(bias_traces - bias_traces[:, 0:1], axis=0, ddof=1)
    theoretical_var = BIAS_RW_STD**2 * time_axis

    ax_bottom.plot(time_axis, ensemble_var_series, color="tab:blue", linewidth=1.5, label="ensemble variance")
    ax_bottom.plot(time_axis, theoretical_var, color="tab:orange", linewidth=2.0, linestyle="--", label="σ²_rw · t")
    ax_bottom.scatter(HORIZONS, horizon_vars, s=50, color="tab:red", zorder=3, label="checked horizons")
    ax_bottom.set_title("Bias Variance Growth vs Time")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Var[bias(t) − bias(0)]  (m²)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="bias_random_walk")
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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DRIFT_RATE = 0.005  # [m/s]
    DT = 1.0            # [s]
    T_MAX = 100.0        # [s]
    TRUE_RANGE = 100.0   # [m]
    N_STEPS = int(T_MAX / DT)

    class BiasDriftConfig(LidarConfig):
        range_min = 0.5
        range_max = 2000.0
        fov = np.deg2rad(60.0)
        sampling_rate = 1.0 / DT
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = DRIFT_RATE
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0
        random_seed = 42

    lidar = Lidar(config=BiasDriftConfig)

    times = []
    errors = []

    for step in range(N_STEPS + 1):
        t = step * DT
        m = lidar.measure(
            rel_position=[TRUE_RANGE, 0.0, 0.0],
            add_noise=False,
            current_time=t,
        )
        if m.get("stale"):
            continue
        times.append(t)
        errors.append(float(m["range"]) - TRUE_RANGE)

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
        expected_errors = DRIFT_RATE * (times - t0)
        np.testing.assert_allclose(errors, expected_errors, rtol=0.0, atol=1e-10)

    # Verify the slope via linear regression.
    if len(times) > 2:
        coeffs = np.polyfit(times[1:], errors[1:], 1)
        fitted_slope = coeffs[0]
        np.testing.assert_allclose(fitted_slope, DRIFT_RATE, rtol=1e-6)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, errors, linewidth=2.0, color="tab:blue", label="measured error")
    ax.plot(times, DRIFT_RATE * (times - times[0]), linewidth=1.5, color="tab:orange", linestyle="--",
            label=f"expected drift ({DRIFT_RATE} m/s)")
    ax.set_title(f"Bias Deterministic Drift (rate = {DRIFT_RATE} m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Range Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="bias_drift")
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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SCALE_PPM = 50.0              # [ppm] deterministic offset
    SCALE_ERROR_STD_PPM = 20.0    # [ppm] per-sensor random component
    DETERMINISTIC_RANGES = np.array([10.0, 50.0, 200.0, 500.0, 1000.0], dtype=float)
    N_SENSORS = 10_000
    STOCHASTIC_TEST_RANGE = 500.0  # [m]

    # ---- Part A: deterministic scale factor ----
    class DeterministicScaleConfig(LidarConfig):
        range_min = 0.5
        range_max = 2000.0
        fov = np.deg2rad(60.0)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = SCALE_PPM
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0
        random_seed = 42

    lidar_det = Lidar(config=DeterministicScaleConfig)
    sf = SCALE_PPM * 1e-6  # dimensionless

    det_measured = []
    det_expected = []
    for r in DETERMINISTIC_RANGES:
        m = lidar_det.measure([r, 0.0, 0.0], add_noise=False, current_time=None)
        assert m["valid"] is True
        det_measured.append(float(m["range"]))
        det_expected.append(r * (1.0 + sf))

    np.testing.assert_allclose(det_measured, det_expected, rtol=0.0, atol=1e-10)

    # ---- Part B: stochastic scale factor distribution ----
    realised_sf = np.empty(N_SENSORS, dtype=float)
    for i in range(N_SENSORS):

        class StochasticScaleConfig(LidarConfig):
            range_min = 0.5
            range_max = 2000.0
            fov = np.deg2rad(60.0)
            range_accuracy = 0.0
            noise_floor_std = 0.0
            noise_range_coeff = 0.0
            bias_init_std = 0.0
            bias_rw_std = 0.0
            bias_drift_rate = 0.0
            scale_factor_ppm = SCALE_PPM
            scale_error_std_ppm = SCALE_ERROR_STD_PPM
            dropout_prob = 0.0
            dropout_range_coeff = 0.0
            outlier_prob = 0.0
            outlier_std = 0.0
            outlier_bias = 0.0
            quantization_step = 0.0
            latency = 0.0
            sample_time_jitter_std = 0.0
            latency_jitter_std = 0.0
            random_seed = i

        sensor = Lidar(config=StochasticScaleConfig)
        m = sensor.measure([STOCHASTIC_TEST_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
        assert m["valid"] is True
        # Recover the total scale factor: measured = true * (1 + sf_total)
        realised_sf[i] = float(m["range"]) / STOCHASTIC_TEST_RANGE - 1.0

    # Expected distribution of realised_sf: N(scale_factor_ppm * 1e-6, (scale_error_std_ppm * 1e-6)^2)
    expected_mean = SCALE_PPM * 1e-6
    expected_std = SCALE_ERROR_STD_PPM * 1e-6

    sample_mean = float(np.mean(realised_sf))
    sample_std = float(np.std(realised_sf, ddof=1))

    # Mean should be within 3 * expected_std / sqrt(N) of the expected mean.
    mean_tol = 3.0 * expected_std / np.sqrt(N_SENSORS)
    assert abs(sample_mean - expected_mean) < mean_tol, (
        f"scale factor mean {sample_mean:.3e} deviates from expected {expected_mean:.3e} by more than {mean_tol:.3e}"
    )

    # Variance within chi-squared 99 % CI.
    var_lo, var_hi = _chi2_variance_bounds(N_SENSORS, confidence=0.99)
    var_ratio = (sample_std / expected_std) ** 2
    assert var_lo <= var_ratio <= var_hi, (
        f"scale factor variance ratio {var_ratio:.4f} outside 99% CI [{var_lo:.4f}, {var_hi:.4f}]"
    )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: deterministic scale — measured vs true.
    ax_top.plot(DETERMINISTIC_RANGES, det_expected, color="tab:orange", linewidth=2.0, label=f"true × (1 + {SCALE_PPM} ppm)")
    ax_top.scatter(DETERMINISTIC_RANGES, det_measured, s=50, color="tab:blue", zorder=3, label="measured")
    ax_top.set_title(f"Deterministic Scale Factor ({SCALE_PPM} ppm)")
    ax_top.set_xlabel("True Range (m)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: stochastic scale factor histogram.
    sf_ppm = realised_sf * 1e6  # convert to ppm for readability
    bins = np.linspace(expected_mean * 1e6 - 4.0 * SCALE_ERROR_STD_PPM,
                       expected_mean * 1e6 + 4.0 * SCALE_ERROR_STD_PPM, 81)
    ax_bottom.hist(sf_ppm, bins=bins, density=True, color="tab:blue", alpha=0.7, label="empirical")
    x_pdf = np.linspace(bins[0], bins[-1], 300)
    pdf = np.exp(-0.5 * ((x_pdf - SCALE_PPM) / SCALE_ERROR_STD_PPM) ** 2) / (SCALE_ERROR_STD_PPM * np.sqrt(2.0 * np.pi))
    ax_bottom.plot(x_pdf, pdf, color="tab:orange", linewidth=2.0, label=f"N({SCALE_PPM}, {SCALE_ERROR_STD_PPM}²) ppm")
    ax_bottom.set_title(f"Stochastic Scale Factor Distribution (N={N_SENSORS} sensors)")
    ax_bottom.set_xlabel("Realised Scale Factor (ppm)")
    ax_bottom.set_ylabel("Probability Density")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="scale_factor")
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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 10_000
    OUTLIER_PROB = 0.10   # configured outlier probability
    OUTLIER_BIAS = 5.0    # [m] deterministic outlier offset
    OUTLIER_STD = 2.0     # [m] outlier spread
    TRUE_RANGE = 100.0    # [m]
    SEED = 42

    class OutlierConfig(LidarConfig):
        range_min = 0.5
        range_max = 2000.0
        fov = np.deg2rad(60.0)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = OUTLIER_PROB
        outlier_std = OUTLIER_STD
        outlier_bias = OUTLIER_BIAS
        quantization_step = 0.0
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0
        random_seed = SEED

    lidar = Lidar(config=OutlierConfig)

    errors = np.empty(N, dtype=float)
    outlier_flags = np.empty(N, dtype=bool)

    for i in range(N):
        m = lidar.measure([TRUE_RANGE, 0.0, 0.0], add_noise=True, current_time=None)
        assert m["valid"] is True, f"unexpected invalid at sample {i}"
        errors[i] = float(m["range"]) - TRUE_RANGE
        outlier_flags[i] = bool(m.get("outlier_applied", False))

    # ---- Outlier rate ----
    n_outliers = int(np.sum(outlier_flags))
    p_hat, ci_lo, ci_hi = _binomial_ci(n_outliers, N, confidence=0.99)
    assert ci_lo <= OUTLIER_PROB <= ci_hi, (
        f"outlier_prob {OUTLIER_PROB} outside 99% CI [{ci_lo:.4f}, {ci_hi:.4f}] (observed {p_hat:.4f})"
    )

    # ---- Outlier magnitude distribution ----
    outlier_errors = errors[outlier_flags]
    assert outlier_errors.size > 0, "no outliers detected"

    # Mean of outlier errors should be near outlier_bias.
    ol_mean = float(np.mean(outlier_errors))
    ol_std = float(np.std(outlier_errors, ddof=1))
    n_ol = outlier_errors.size

    mean_tol = 3.0 * OUTLIER_STD / np.sqrt(n_ol)
    assert abs(ol_mean - OUTLIER_BIAS) < mean_tol, (
        f"outlier mean {ol_mean:.4f} deviates from expected {OUTLIER_BIAS} by more than {mean_tol:.4f}"
    )

    # Variance of outlier errors within chi-squared 99 % CI.
    var_lo, var_hi = _chi2_variance_bounds(n_ol, confidence=0.99)
    var_ratio = (ol_std / OUTLIER_STD) ** 2
    assert var_lo <= var_ratio <= var_hi, (
        f"outlier std variance ratio {var_ratio:.4f} outside 99% CI [{var_lo:.4f}, {var_hi:.4f}]"
    )

    # ---- Non-outlier samples should have zero error ----
    clean_errors = errors[~outlier_flags]
    assert np.allclose(clean_errors, 0.0, atol=1e-12), "non-outlier samples should have zero error"

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: histogram of all range errors.
    all_min = min(errors.min(), -1.0)
    all_max = max(errors.max(), OUTLIER_BIAS + 4.0 * OUTLIER_STD)
    bins = np.linspace(all_min, all_max, 121)
    ax_top.hist(errors[~outlier_flags], bins=bins, color="tab:blue", alpha=0.7, label="clean returns")
    ax_top.hist(errors[outlier_flags], bins=bins, color="tab:red", alpha=0.7, label="outlier returns")
    ax_top.set_title(f"Range Error Distribution (N={N}, outlier_prob={OUTLIER_PROB})")
    ax_top.set_xlabel("Range Error (m)")
    ax_top.set_ylabel("Count")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: cumulative outlier rate convergence.
    cumulative_rate = np.cumsum(outlier_flags) / (np.arange(N) + 1.0)
    ax_bottom.plot(np.arange(N) + 1, cumulative_rate, color="tab:blue", linewidth=1.0, label="cumulative rate")
    ax_bottom.axhline(OUTLIER_PROB, color="tab:orange", linewidth=2.0, linestyle="--", label=f"configured = {OUTLIER_PROB}")
    ax_bottom.axhline(ci_lo, color="tab:red", linewidth=1.0, linestyle=":", label="99% CI bounds")
    ax_bottom.axhline(ci_hi, color="tab:red", linewidth=1.0, linestyle=":")
    ax_bottom.set_title("Cumulative Outlier Rate Convergence")
    ax_bottom.set_xlabel("Sample Number")
    ax_bottom.set_ylabel("Cumulative Outlier Rate")
    ax_bottom.set_ylim(0.0, 2.0 * OUTLIER_PROB)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="outlier_injection")
    plt.close(fig)


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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 10_000
    DROPOUT_PROB = 0.05
    DROPOUT_RANGE_COEFF = 0.20
    RANGE_MIN = 1.0
    RANGE_MAX = 1000.0
    TEST_RANGES = np.array([1.0, 100.0, 250.0, 500.0, 750.0, 1000.0], dtype=float)
    SAMPLING_RATE = 10_000.0  # [Hz] high rate to collect N samples quickly

    empirical_dropouts = []
    expected_dropouts = []
    ci_bounds = []

    for r_idx, true_range in enumerate(TEST_RANGES):

        class DropoutConfig(LidarConfig):
            range_min = RANGE_MIN
            range_max = RANGE_MAX
            fov = np.deg2rad(60.0)
            sampling_rate = SAMPLING_RATE
            range_accuracy = 0.0
            noise_floor_std = 0.0
            noise_range_coeff = 0.0
            bias_init_std = 0.0
            bias_rw_std = 0.0
            bias_drift_rate = 0.0
            scale_factor_ppm = 0.0
            scale_error_std_ppm = 0.0
            dropout_prob = DROPOUT_PROB
            dropout_range_coeff = DROPOUT_RANGE_COEFF
            outlier_prob = 0.0
            outlier_std = 0.0
            outlier_bias = 0.0
            quantization_step = 0.0
            latency = 0.0
            sample_time_jitter_std = 0.0
            latency_jitter_std = 0.0
            random_seed = 5000 + r_idx

        lidar = Lidar(config=DropoutConfig)
        dt = 1.0 / SAMPLING_RATE
        n_dropout = 0

        for i in range(N):
            t = i * dt
            m = lidar.measure(
                rel_position=[true_range, 0.0, 0.0],
                add_noise=True,
                current_time=t,
            )
            if m.get("stale"):
                continue
            if not m.get("valid", False) and m.get("reason") == "dropout":
                n_dropout += 1

        # Theoretical dropout probability at this range.
        span = max(RANGE_MAX - RANGE_MIN, 1e-12)
        norm_range = np.clip((true_range - RANGE_MIN) / span, 0.0, 1.0)
        p_expected = np.clip(DROPOUT_PROB + DROPOUT_RANGE_COEFF * norm_range, 0.0, 1.0)

        p_hat, ci_lo, ci_hi = _binomial_ci(n_dropout, N, confidence=0.99)

        empirical_dropouts.append(p_hat)
        expected_dropouts.append(float(p_expected))
        ci_bounds.append((ci_lo, ci_hi))

        assert ci_lo <= p_expected <= ci_hi, (
            f"range {true_range} m: expected dropout {p_expected:.4f} outside "
            f"99% CI [{ci_lo:.4f}, {ci_hi:.4f}] (observed {p_hat:.4f})"
        )

    empirical_dropouts = np.asarray(empirical_dropouts, dtype=float)
    expected_dropouts = np.asarray(expected_dropouts, dtype=float)

    # Linear fit slope should be consistent with dropout_range_coeff.
    norm_ranges = np.clip((TEST_RANGES - RANGE_MIN) / max(RANGE_MAX - RANGE_MIN, 1e-12), 0.0, 1.0)
    if len(TEST_RANGES) > 2:
        coeffs = np.polyfit(norm_ranges, empirical_dropouts, 1)
        fitted_slope = coeffs[0]
        # Slope tolerance: allow 20 % relative error due to sampling noise.
        assert abs(fitted_slope - DROPOUT_RANGE_COEFF) < 0.20 * DROPOUT_RANGE_COEFF + 0.01, (
            f"fitted dropout slope {fitted_slope:.4f} deviates from configured {DROPOUT_RANGE_COEFF}"
        )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: dropout fraction vs range with theoretical line.
    dense_ranges = np.linspace(RANGE_MIN, RANGE_MAX, 200)
    dense_norm = (dense_ranges - RANGE_MIN) / max(RANGE_MAX - RANGE_MIN, 1e-12)
    dense_expected = np.clip(DROPOUT_PROB + DROPOUT_RANGE_COEFF * dense_norm, 0.0, 1.0)

    ax_top.plot(dense_ranges, dense_expected, color="tab:orange", linewidth=2.0, label="theoretical")
    ax_top.scatter(TEST_RANGES, empirical_dropouts, s=50, color="tab:blue", zorder=3, label="empirical")
    ci_lo_arr = np.array([b[0] for b in ci_bounds])
    ci_hi_arr = np.array([b[1] for b in ci_bounds])
    ax_top.vlines(TEST_RANGES, ci_lo_arr, ci_hi_arr, color="tab:blue", linewidth=1.5, alpha=0.6, label="99% CI")
    ax_top.set_title(f"Dropout Fraction vs Range (N={N} per range)")
    ax_top.set_xlabel("True Range (m)")
    ax_top.set_ylabel("Dropout Probability")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: bar chart — empirical vs expected at each test range.
    x_pos = np.arange(len(TEST_RANGES))
    width = 0.35
    ax_bottom.bar(x_pos - width / 2, expected_dropouts, width, color="tab:orange", alpha=0.8, label="expected")
    ax_bottom.bar(x_pos + width / 2, empirical_dropouts, width, color="tab:blue", alpha=0.8, label="empirical")
    ax_bottom.set_xticks(x_pos)
    ax_bottom.set_xticklabels([f"{r:.0f}" for r in TEST_RANGES])
    ax_bottom.set_title("Empirical vs Expected Dropout at Each Test Range")
    ax_bottom.set_xlabel("True Range (m)")
    ax_bottom.set_ylabel("Dropout Probability")
    ax_bottom.grid(True, alpha=0.3, axis="y")
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="dropout_rate")
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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    QUANT_STEP = 0.5  # [m]

    class QuantizationConfig(LidarConfig):
        range_min = 1.0
        range_max = 100.0
        fov = np.deg2rad(60.0)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = QUANT_STEP
        saturate_output = True
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0
        random_seed = 42

    lidar = Lidar(config=QuantizationConfig)

    # Sweep true range with fine resolution (much finer than the quantization step).
    sweep_step = QUANT_STEP / 50.0
    true_ranges = np.arange(
        float(QuantizationConfig.range_min),
        float(QuantizationConfig.range_max) + sweep_step,
        sweep_step,
        dtype=float,
    )
    # Clip to range_max so we stay within the valid measurement envelope.
    true_ranges = true_ranges[true_ranges <= float(QuantizationConfig.range_max)]

    measured_ranges = np.empty_like(true_ranges)
    for i, r in enumerate(true_ranges):
        m = lidar.measure([r, 0.0, 0.0], add_noise=False, current_time=None)
        assert m["valid"] is True, f"unexpected invalid at range {r}"
        measured_ranges[i] = float(m["range"])

    # Every output must be an exact integer multiple of the quantization step.
    multiples = measured_ranges / QUANT_STEP
    np.testing.assert_allclose(
        multiples,
        np.round(multiples),
        rtol=0.0,
        atol=1e-12,
        err_msg="measured range is not an integer multiple of quantization_step",
    )

    # Residual must be bounded by ±step/2.
    residuals = measured_ranges - true_ranges
    half_step = QUANT_STEP / 2.0
    assert np.all(residuals >= -half_step - 1e-12), (
        f"residual below -step/2: min = {residuals.min():.6f}"
    )
    assert np.all(residuals <= half_step + 1e-12), (
        f"residual above +step/2: max = {residuals.max():.6f}"
    )

    # ---- Plot ----
    # Zoom into a small window to make the staircase and sawtooth clearly visible.
    zoom_lo = float(QuantizationConfig.range_min) + 5.0
    zoom_hi = zoom_lo + 5.0 * QUANT_STEP
    zoom_mask = (true_ranges >= zoom_lo) & (true_ranges <= zoom_hi)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: staircase transfer function (zoomed).
    ax_top.plot(true_ranges[zoom_mask], true_ranges[zoom_mask], color="tab:orange",
                linewidth=1.5, linestyle="--", label="ideal (no quantization)")
    ax_top.plot(true_ranges[zoom_mask], measured_ranges[zoom_mask], color="tab:blue",
                linewidth=2.0, label=f"quantized (step = {QUANT_STEP} m)")
    ax_top.set_title("Quantization Staircase Transfer Function (zoomed)")
    ax_top.set_xlabel("True Range (m)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: sawtooth residual over full range.
    ax_bottom.plot(true_ranges, residuals, color="tab:blue", linewidth=0.8, label="residual")
    ax_bottom.axhline(half_step, color="tab:red", linestyle="--", linewidth=1.5, label=f"±step/2 = ±{half_step} m")
    ax_bottom.axhline(-half_step, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Quantization Residual (measured − true)")
    ax_bottom.set_xlabel("True Range (m)")
    ax_bottom.set_ylabel("Residual (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="quantization_step")
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
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RANGE_MIN = 5.0
    RANGE_MAX = 200.0
    TRUE_RANGE = 100.0  # [m] nominal target in the middle of the envelope
    HIGH_BIAS = 150.0   # [m] pushes measured to ~250 m (above range_max)
    LOW_BIAS = -120.0   # [m] pushes measured to ~-20 m (below range_min)

    # ---- Part A: saturate_output = True (clipping) ----
    class ClippingConfig(LidarConfig):
        range_min = RANGE_MIN
        range_max = RANGE_MAX
        fov = np.deg2rad(60.0)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        outlier_std = 0.0
        outlier_bias = 0.0
        quantization_step = 0.0
        saturate_output = True
        latency = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0
        random_seed = 42

    # Nominal measurement — should pass through unchanged.
    lidar_clip = Lidar(config=ClippingConfig)
    m_nominal = lidar_clip.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    assert m_nominal["valid"] is True
    np.testing.assert_allclose(float(m_nominal["range"]), TRUE_RANGE, rtol=0.0, atol=1e-12)

    # High bias — measured = true + bias = 250 m, should clip to range_max.
    lidar_clip_hi = Lidar(config=ClippingConfig)
    lidar_clip_hi._bias_state = HIGH_BIAS
    m_clip_hi = lidar_clip_hi.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    assert m_clip_hi["valid"] is True
    np.testing.assert_allclose(float(m_clip_hi["range"]), RANGE_MAX, rtol=0.0, atol=1e-12)

    # Low bias — measured = true + bias = -20 m, should clip to range_min.
    lidar_clip_lo = Lidar(config=ClippingConfig)
    lidar_clip_lo._bias_state = LOW_BIAS
    m_clip_lo = lidar_clip_lo.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    assert m_clip_lo["valid"] is True
    np.testing.assert_allclose(float(m_clip_lo["range"]), RANGE_MIN, rtol=0.0, atol=1e-12)

    # ---- Part B: saturate_output = False (invalidation) ----
    class InvalidationConfig(ClippingConfig):
        saturate_output = False

    # Nominal — still valid.
    lidar_inv = Lidar(config=InvalidationConfig)
    m_inv_nominal = lidar_inv.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    assert m_inv_nominal["valid"] is True
    np.testing.assert_allclose(float(m_inv_nominal["range"]), TRUE_RANGE, rtol=0.0, atol=1e-12)

    # High bias — should be invalid with reason "saturated".
    lidar_inv_hi = Lidar(config=InvalidationConfig)
    lidar_inv_hi._bias_state = HIGH_BIAS
    m_inv_hi = lidar_inv_hi.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    assert m_inv_hi["valid"] is False
    assert m_inv_hi.get("reason") == "saturated"

    # Low bias — should be invalid with reason "saturated".
    lidar_inv_lo = Lidar(config=InvalidationConfig)
    lidar_inv_lo._bias_state = LOW_BIAS
    m_inv_lo = lidar_inv_lo.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
    assert m_inv_lo["valid"] is False
    assert m_inv_lo.get("reason") == "saturated"

    # ---- Sweep: bias from large negative to large positive to show full behaviour ----
    bias_sweep = np.linspace(-150.0, 200.0, 701, dtype=float)

    clip_outputs = np.empty_like(bias_sweep)
    clip_valids = np.empty(bias_sweep.size, dtype=bool)
    inv_outputs = np.full_like(bias_sweep, np.nan)
    inv_valids = np.empty(bias_sweep.size, dtype=bool)

    for i, bias in enumerate(bias_sweep):
        # Clipping mode.
        s_clip = Lidar(config=ClippingConfig)
        s_clip._bias_state = bias
        mc = s_clip.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
        clip_valids[i] = bool(mc["valid"])
        clip_outputs[i] = float(mc["range"]) if mc["valid"] else np.nan

        # Invalidation mode.
        s_inv = Lidar(config=InvalidationConfig)
        s_inv._bias_state = bias
        mi = s_inv.measure([TRUE_RANGE, 0.0, 0.0], add_noise=False, current_time=None)
        inv_valids[i] = bool(mi["valid"])
        inv_outputs[i] = float(mi["range"]) if mi["valid"] else np.nan

    ideal_measured = TRUE_RANGE + bias_sweep  # what the range would be without limits

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: clipping mode.
    ax_top.plot(bias_sweep, ideal_measured, color="tab:gray", linewidth=1.0, linestyle=":", label="unclamped")
    ax_top.plot(bias_sweep, clip_outputs, color="tab:blue", linewidth=2.0, label="saturate_output = True")
    ax_top.axhline(RANGE_MAX, color="tab:red", linestyle="--", linewidth=1.5, label="range limits")
    ax_top.axhline(RANGE_MIN, color="tab:red", linestyle="--", linewidth=1.5)
    ax_top.set_title("Output Clipping (saturate_output = True)")
    ax_top.set_xlabel("Applied Bias (m)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: invalidation mode.
    ax_bottom.plot(bias_sweep, ideal_measured, color="tab:gray", linewidth=1.0, linestyle=":", label="unclamped")
    valid_inv_mask = inv_valids
    invalid_inv_mask = ~inv_valids
    ax_bottom.scatter(bias_sweep[valid_inv_mask], inv_outputs[valid_inv_mask], s=6, color="tab:blue",
                      alpha=0.8, label="valid (saturate_output = False)")
    ax_bottom.scatter(bias_sweep[invalid_inv_mask], ideal_measured[invalid_inv_mask], s=6, color="tab:red",
                      alpha=0.5, marker="x", label="invalidated (reason = saturated)")
    ax_bottom.axhline(RANGE_MAX, color="tab:red", linestyle="--", linewidth=1.5, label="range limits")
    ax_bottom.axhline(RANGE_MIN, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title("Output Invalidation (saturate_output = False)")
    ax_bottom.set_xlabel("Applied Bias (m)")
    ax_bottom.set_ylabel("Measured / Ideal Range (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="saturation_vs_invalidation")
    plt.close(fig)


# ============================================================================
# Asynchronous pipeline and timing tests
# ============================================================================


@pytest.mark.test_meta(
    description=(
        "Feed measure() with current_time advancing at 1 ms resolution and verify that new "
        "measurements appear at exactly 1/sampling_rate intervals. Count the total number of "
        "non-stale measurements over a known duration and confirm the count matches the expected "
        "number of sampling periods."
    ),
    goal=(
        "Validate the internal sample clock: the sensor must produce one new measurement per "
        "sampling period regardless of how finely the caller ticks the simulation clock."
    ),
    passing_criteria=(
        "Total new measurement count equals floor(duration * sampling_rate) + 1 (initial sample). "
        "Every inter-measurement interval equals 1/sampling_rate within numerical tolerance."
    ),
)
def test_sampling_rate(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SAMPLING_RATE = 10.0   # [Hz]
    DURATION = 5.0         # [s] total simulation duration
    DT_CALL = 0.001        # [s] caller tick resolution (1 ms)
    TRUE_RANGE = 50.0      # [m] static target on boresight

    class SamplingRateConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        sampling_rate = SAMPLING_RATE
        latency = 0.0          # zero latency so measurements appear immediately
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    sensor = Lidar(config=SamplingRateConfig)
    rel_pos = [TRUE_RANGE, 0.0, 0.0]

    # Tick the simulation at 1 ms steps and record every non-stale measurement timestamp.
    new_timestamps = []
    call_times = np.arange(0.0, DURATION + DT_CALL * 0.5, DT_CALL)
    for t in call_times:
        m = sensor.measure(rel_pos, add_noise=False, current_time=float(t))
        if m.get("valid", False) and not m.get("stale", False):
            new_timestamps.append(float(m["timestamp"]))

    new_timestamps = np.array(new_timestamps)

    # Expected count: one sample at t=0, then one per period until DURATION.
    expected_period = 1.0 / SAMPLING_RATE
    expected_count = int(np.floor(DURATION * SAMPLING_RATE)) + 1
    assert len(new_timestamps) == expected_count, (
        f"expected {expected_count} measurements, got {len(new_timestamps)}"
    )

    # Inter-measurement intervals must all equal the nominal period.
    intervals = np.diff(new_timestamps)
    assert np.allclose(intervals, expected_period, atol=1e-9), (
        f"inter-measurement interval deviation: max |error| = {np.max(np.abs(intervals - expected_period)):.3e} s"
    )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: measurement timestamps as event markers along the time axis.
    ax_top.eventplot([new_timestamps], lineoffsets=0.0, linelengths=0.6, colors="tab:blue", label="new measurement")
    expected_times = np.arange(0.0, DURATION + expected_period * 0.5, expected_period)
    ax_top.eventplot([expected_times], lineoffsets=0.0, linelengths=0.3, colors="tab:orange",
                     linestyles="dashed", label="expected sample epoch")
    ax_top.set_title("Measurement Event Timeline")
    ax_top.set_xlabel("Time (s)")
    ax_top.set_yticks([])
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: inter-measurement interval vs expected period.
    sample_indices = np.arange(1, len(new_timestamps))
    ax_bottom.plot(sample_indices, intervals * 1e3, color="tab:blue", linewidth=1.5, marker=".", markersize=3,
                   label="measured interval")
    ax_bottom.axhline(expected_period * 1e3, color="tab:orange", linestyle="--", linewidth=1.5,
                      label=f"expected = {expected_period * 1e3:.1f} ms")
    ax_bottom.set_title("Inter-Measurement Interval")
    ax_bottom.set_xlabel("Sample Index")
    ax_bottom.set_ylabel("Interval (ms)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="sampling_rate")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Verify that a measurement triggered at sample epoch t does not appear in the output "
        "until current_time >= t + latency. Before that threshold the sensor must return a stale "
        "placeholder. After the latency elapses the measurement must carry the correct sample "
        "timestamp equal to the original sample epoch."
    ),
    goal=(
        "Confirm the latency pipeline delays measurement delivery by exactly the configured "
        "latency value and that the returned timestamp reflects the sample epoch, not the "
        "delivery time."
    ),
    passing_criteria=(
        "All calls with current_time < first_sample_epoch + latency return stale. "
        "The first non-stale call occurs at current_time >= first_sample_epoch + latency and "
        "carries timestamp == first_sample_epoch. Multiple latency values are swept to confirm "
        "the relationship holds across the range."
    ),
)
def test_latency(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DT_CALL = 0.001           # [s] caller tick resolution
    TRUE_RANGE = 50.0          # [m] static target on boresight
    LATENCY_VALUES = np.array([0.01, 0.05, 0.10, 0.20, 0.50])  # [s] latencies to sweep

    measured_delivery_times = []
    expected_delivery_times = []

    for latency_val in LATENCY_VALUES:

        class LatencyConfig(LidarConfig):
            range_min = 1.0
            range_max = 1000.0
            range_accuracy = 0.0
            sampling_rate = 1.0   # one sample per second so only one sample in our window
            latency = float(latency_val)
            dropout_prob = 0.0
            dropout_range_coeff = 0.0
            bias_init_std = 0.0
            bias_rw_std = 0.0
            bias_drift_rate = 0.0
            scale_factor_ppm = 0.0
            scale_error_std_ppm = 0.0
            outlier_prob = 0.0
            quantization_step = 0.0
            sample_time_jitter_std = 0.0
            latency_jitter_std = 0.0

        sensor = Lidar(config=LatencyConfig)
        rel_pos = [TRUE_RANGE, 0.0, 0.0]

        # Tick until the measurement becomes available (or a generous timeout).
        delivery_time = None
        sample_epoch = None
        timeout = latency_val + 1.0  # generous upper bound

        t = 0.0
        while t <= timeout:
            m = sensor.measure(rel_pos, add_noise=False, current_time=t)
            if m.get("valid", False) and not m.get("stale", False):
                delivery_time = t
                sample_epoch = float(m["timestamp"])
                break
            t += DT_CALL

        assert delivery_time is not None, (
            f"measurement never delivered for latency = {latency_val:.3f} s"
        )
        # The sample epoch should be 0.0 (the first sample) and delivery at >= latency.
        assert abs(sample_epoch - 0.0) < 1e-9, (
            f"sample epoch should be 0.0, got {sample_epoch:.6f}"
        )
        assert delivery_time >= latency_val - DT_CALL, (
            f"measurement delivered too early: delivery = {delivery_time:.6f}, latency = {latency_val:.3f}"
        )

        # Confirm the call just before delivery was stale.
        if delivery_time >= DT_CALL:
            sensor_check = Lidar(config=LatencyConfig)
            t_check = 0.0
            last_before = None
            while t_check < delivery_time - DT_CALL * 0.5:
                m_check = sensor_check.measure(rel_pos, add_noise=False, current_time=t_check)
                last_before = m_check
                t_check += DT_CALL
            assert last_before is not None and last_before.get("stale", False), (
                f"expected stale before delivery at latency = {latency_val:.3f} s"
            )

        measured_delivery_times.append(delivery_time)
        expected_delivery_times.append(latency_val)

    measured_delivery_times = np.array(measured_delivery_times)
    expected_delivery_times = np.array(expected_delivery_times)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(expected_delivery_times * 1e3, measured_delivery_times * 1e3, color="tab:blue",
            linewidth=2.0, marker="o", markersize=6, label="measured delivery time")
    ax.plot(expected_delivery_times * 1e3, expected_delivery_times * 1e3, color="tab:orange",
            linestyle="--", linewidth=1.5, label="ideal (delivery = latency)")
    ax.set_title("Latency Pipeline Delivery Time")
    ax.set_xlabel("Configured Latency (ms)")
    ax.set_ylabel("First Delivery Time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="latency_pipeline")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Validate both timing jitter injection paths independently. "
        "Part A: enable only sample_time_jitter_std, call measure() at mid-period cadence so the "
        "sample epoch sits in the centre of the caller interval (avoiding internal clamp truncation), "
        "and verify the sample epoch jitter std via chi-squared 99 %% CI. "
        "Part B: enable only latency_jitter_std, tick at fine resolution (0.5 ms), and verify the "
        "delivery delay jitter std via chi-squared 99 %% CI."
    ),
    goal=(
        "Confirm sample epoch jitter and latency jitter each produce the configured statistical "
        "spread in their respective output domains when tested in isolation."
    ),
    passing_criteria=(
        "Part A: sample epoch jitter std lies within 99 %% chi-squared CI around sample_time_jitter_std. "
        "Part B: delivery delay jitter std lies within 99 %% chi-squared CI around "
        "sqrt(latency_jitter_std^2 + (dt_fine^2)/12)."
    ),
)
def test_time_jitter(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SAMPLING_RATE = 10.0             # [Hz] period = 100 ms
    SAMPLE_JITTER_STD = 0.005       # [s] 5 ms sampling epoch jitter
    LATENCY_JITTER_STD = 0.005      # [s] 5 ms latency jitter
    LATENCY_NOMINAL = 0.050         # [s] 50 ms nominal latency (Part B only)
    N_SAMPLES_A = 2000              # number of samples for Part A
    N_SAMPLES_B = 500               # number of samples for Part B
    DT_FINE = 0.0005                # [s] 0.5 ms fine tick resolution (Part B)
    TRUE_RANGE = 50.0               # [m] static target on boresight
    period = 1.0 / SAMPLING_RATE    # [s] = 0.1 s

    # ================================================================
    # Part A — Sample epoch jitter (latency jitter disabled, latency = 0)
    # ================================================================
    # Strategy: anchor the sample clock at t=0, then call at (k + 0.5) * period.
    # This places each nominal sample epoch in the centre of its caller interval,
    # giving ±period/2 room for jitter before the internal clamp triggers.

    class SampleJitterConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        sampling_rate = SAMPLING_RATE
        latency = 0.0                       # immediate delivery
        sample_time_jitter_std = SAMPLE_JITTER_STD
        latency_jitter_std = 0.0            # isolate sample jitter
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        random_seed = 12345

    sensor_a = Lidar(config=SampleJitterConfig)
    rel_pos = [TRUE_RANGE, 0.0, 0.0]

    # Initial call at t=0 anchors the sample clock; sample at epoch 0 fires immediately.
    sensor_a.measure(rel_pos, add_noise=True, current_time=0.0)

    # Subsequent calls at (k + 0.5) * period.  k=0 (t=0.05 s) is stale (next epoch = 0.1 s).
    # k=1 (t=0.15 s) triggers the sample at epoch=0.1 s with clamp window [0.05, 0.15],
    # centred on the epoch — symmetric jitter preserved.
    sample_timestamps_a = []
    for k in range(N_SAMPLES_A + 1):
        t_call = (k + 0.5) * period
        m = sensor_a.measure(rel_pos, add_noise=True, current_time=t_call)
        if m.get("valid", False) and not m.get("stale", False):
            sample_timestamps_a.append(float(m["timestamp"]))

    sample_timestamps_a = np.array(sample_timestamps_a)
    n_a = len(sample_timestamps_a)
    assert n_a >= N_SAMPLES_A * 0.95, f"Part A: too few samples: {n_a}"

    # Nominal epochs for collected samples: period, 2*period, ..., n_a*period.
    nominal_epochs_a = np.arange(1, n_a + 1) * period
    sample_jitter = sample_timestamps_a - nominal_epochs_a
    sample_jitter_std_measured = float(np.std(sample_jitter, ddof=1))

    lo_ratio, hi_ratio = _chi2_variance_bounds(n_a, confidence=0.99)
    sample_jitter_lo = SAMPLE_JITTER_STD * np.sqrt(lo_ratio)
    sample_jitter_hi = SAMPLE_JITTER_STD * np.sqrt(hi_ratio)
    assert sample_jitter_lo <= sample_jitter_std_measured <= sample_jitter_hi, (
        f"Part A: sample jitter std {sample_jitter_std_measured:.6f} s outside 99% CI "
        f"[{sample_jitter_lo:.6f}, {sample_jitter_hi:.6f}] for configured {SAMPLE_JITTER_STD:.6f} s"
    )

    # ================================================================
    # Part B — Latency jitter (sample jitter disabled)
    # ================================================================
    # Tick at fine resolution so delivery time is precisely resolved.
    # delivery_delay = call_time_of_delivery − sample_timestamp ≈ latency + jitter.

    class LatencyJitterConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        sampling_rate = SAMPLING_RATE
        latency = LATENCY_NOMINAL
        sample_time_jitter_std = 0.0        # isolate latency jitter
        latency_jitter_std = LATENCY_JITTER_STD
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        random_seed = 54321

    sensor_b = Lidar(config=LatencyJitterConfig)
    duration_b = N_SAMPLES_B * period + LATENCY_NOMINAL + 1.0  # generous duration

    delivery_delays = []  # delivery_call_time − sample_timestamp for each measurement
    t = 0.0
    while t <= duration_b:
        m = sensor_b.measure(rel_pos, add_noise=True, current_time=t)
        if m.get("valid", False) and not m.get("stale", False):
            delivery_delays.append(t - float(m["timestamp"]))
        t += DT_FINE

    delivery_delays = np.array(delivery_delays)
    n_b = len(delivery_delays)
    assert n_b >= N_SAMPLES_B * 0.9, f"Part B: too few samples: {n_b}"

    # Subtract nominal latency to isolate jitter + quantisation noise.
    latency_deviations = delivery_delays - LATENCY_NOMINAL
    expected_latency_dev_std = float(np.sqrt(
        LATENCY_JITTER_STD ** 2 + (DT_FINE ** 2) / 12.0
    ))
    latency_dev_std_measured = float(np.std(latency_deviations, ddof=1))

    lo_ratio_b, hi_ratio_b = _chi2_variance_bounds(n_b, confidence=0.99)
    latency_dev_lo = expected_latency_dev_std * np.sqrt(lo_ratio_b)
    latency_dev_hi = expected_latency_dev_std * np.sqrt(hi_ratio_b)
    assert latency_dev_lo <= latency_dev_std_measured <= latency_dev_hi, (
        f"Part B: latency jitter std {latency_dev_std_measured:.6f} s outside 99% CI "
        f"[{latency_dev_lo:.6f}, {latency_dev_hi:.6f}] for expected {expected_latency_dev_std:.6f} s"
    )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: sample epoch jitter histogram (Part A).
    ax_top.hist(sample_jitter * 1e3, bins=60, density=True, color="tab:blue", alpha=0.7, label="observed")
    jitter_x = np.linspace(-4 * SAMPLE_JITTER_STD, 4 * SAMPLE_JITTER_STD, 300)
    jitter_pdf = (1.0 / (SAMPLE_JITTER_STD * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (jitter_x / SAMPLE_JITTER_STD) ** 2
    )
    ax_top.plot(jitter_x * 1e3, jitter_pdf * 1e-3, color="tab:orange", linewidth=2.0,
                label=f"N(0, {SAMPLE_JITTER_STD * 1e3:.1f} ms)")
    ax_top.axvline(sample_jitter_std_measured * 1e3, color="tab:red", linestyle="--", linewidth=1.5,
                   label=f"measured std = {sample_jitter_std_measured * 1e3:.2f} ms")
    ax_top.axvline(-sample_jitter_std_measured * 1e3, color="tab:red", linestyle="--", linewidth=1.5)
    ax_top.set_title(f"Part A — Sample Epoch Jitter (n = {n_a})")
    ax_top.set_xlabel("Jitter (ms)")
    ax_top.set_ylabel("Probability Density")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: latency deviation histogram (Part B).
    ax_bottom.hist(latency_deviations * 1e3, bins=60, density=True, color="tab:blue", alpha=0.7, label="observed")
    ld_x = np.linspace(-4 * expected_latency_dev_std, 4 * expected_latency_dev_std, 300)
    ld_pdf = (1.0 / (expected_latency_dev_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (ld_x / expected_latency_dev_std) ** 2
    )
    ax_bottom.plot(ld_x * 1e3, ld_pdf * 1e-3, color="tab:orange", linewidth=2.0,
                   label=f"N(0, {expected_latency_dev_std * 1e3:.2f} ms)")
    ax_bottom.axvline(latency_dev_std_measured * 1e3, color="tab:red", linestyle="--", linewidth=1.5,
                      label=f"measured std = {latency_dev_std_measured * 1e3:.2f} ms")
    ax_bottom.axvline(-latency_dev_std_measured * 1e3, color="tab:red", linestyle="--", linewidth=1.5)
    ax_bottom.set_title(f"Part B — Latency Jitter (n = {n_b})")
    ax_bottom.set_xlabel("Deviation from Nominal Latency (ms)")
    ax_bottom.set_ylabel("Probability Density")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="time_jitter")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Feed the sensor with a normally advancing time sequence, then inject a backward "
        "timestamp (time reversal). Verify the pipeline resets gracefully: no exception, "
        "pending measurements are cleared, and the sensor resumes correct operation from the "
        "new (earlier) time reference. A non-zero latency is used so the pending queue "
        "accumulates entries before the reversal, making the flush observable."
    ),
    goal=(
        "Confirm the time reversal detection logic clears stale pipeline state and allows "
        "the sensor to recover without crashing or producing corrupt output."
    ),
    passing_criteria=(
        "No exceptions raised during or after the reversal. The pending queue drops from a "
        "non-zero count to at most 1 entry. Subsequent ticks resume normal operation with "
        "monotonically increasing timestamps rooted at the reversal point."
    ),
)
def test_time_reversal_handling(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SAMPLING_RATE = 10.0     # [Hz] period = 100 ms
    LATENCY = 0.3            # [s] non-zero so pending queue accumulates before reversal
    DT_CALL = 0.001          # [s] tick resolution
    TRUE_RANGE = 50.0        # [m] static target on boresight
    T_FORWARD_END = 2.0      # [s] run forward until this time
    T_REVERSAL = 0.5         # [s] jump back to this time after forward phase
    T_RESUME_END = 3.0       # [s] continue forward from T_REVERSAL to this time

    class ReversalConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        sampling_rate = SAMPLING_RATE
        latency = LATENCY
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    sensor = Lidar(config=ReversalConfig)
    rel_pos = [TRUE_RANGE, 0.0, 0.0]

    # Phase 1: normal forward operation — record every output (valid, stale, or invalid).
    phase1_valid_ts = []       # (call_time, measurement_timestamp) for valid non-stale outputs
    phase1_stale_times = []    # call_times that returned stale
    t = 0.0
    while t <= T_FORWARD_END:
        m = sensor.measure(rel_pos, add_noise=False, current_time=t)
        if m.get("valid", False) and not m.get("stale", False):
            phase1_valid_ts.append((t, float(m["timestamp"])))
        elif m.get("stale", False):
            phase1_stale_times.append(t)
        t += DT_CALL

    assert len(phase1_valid_ts) > 0, "no measurements produced in forward phase"
    pending_before = len(sensor.pending_measurements)
    assert pending_before > 0, (
        "pending queue empty before reversal — use a larger latency so measurements accumulate"
    )

    # Phase 2: inject time reversal — jump back to T_REVERSAL.
    reversal_result = sensor.measure(rel_pos, add_noise=False, current_time=T_REVERSAL)
    pending_after = len(sensor.pending_measurements)

    # The pending queue must have been flushed by the reversal.
    assert pending_after <= 1, (
        f"pending queue not flushed: {pending_after} entries remain after reversal "
        f"(was {pending_before} before)"
    )

    # Phase 3: resume forward from T_REVERSAL.
    phase3_valid_ts = []
    phase3_stale_times = []
    t = T_REVERSAL + DT_CALL
    while t <= T_REVERSAL + T_RESUME_END:
        m = sensor.measure(rel_pos, add_noise=False, current_time=t)
        if m.get("valid", False) and not m.get("stale", False):
            phase3_valid_ts.append((t, float(m["timestamp"])))
        elif m.get("stale", False):
            phase3_stale_times.append(t)
        t += DT_CALL

    assert len(phase3_valid_ts) > 0, "no measurements produced after reversal"

    # Verify resumed timestamps are monotonically increasing from T_REVERSAL.
    phase3_ts = np.array([ts for _, ts in phase3_valid_ts])
    assert np.all(np.diff(phase3_ts) > 0), "timestamps not monotonic after reversal"
    assert phase3_ts[0] >= T_REVERSAL - 1e-9, (
        f"first post-reversal timestamp {phase3_ts[0]:.6f} < reversal point {T_REVERSAL}"
    )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: full output timeline — valid measurements and stale markers.
    if phase1_valid_ts:
        ct1, ts1 = zip(*phase1_valid_ts)
        ax_top.scatter(ct1, ts1, s=12, color="tab:blue", alpha=0.6, label="phase 1 valid")
    if phase1_stale_times:
        ax_top.scatter(phase1_stale_times,
                       [0.0] * len(phase1_stale_times),
                       s=4, color="tab:gray", alpha=0.15, marker=".", label="phase 1 stale")
    ax_top.axvline(T_FORWARD_END, color="tab:red", linestyle="--", linewidth=1.5, label="reversal injected")
    ax_top.axhline(T_REVERSAL, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.5,
                   label=f"reversal target = {T_REVERSAL} s")
    if phase3_valid_ts:
        ct3, ts3 = zip(*phase3_valid_ts)
        ax_top.scatter(ct3, ts3, s=12, color="tab:orange", alpha=0.6, label="phase 3 valid")
    ax_top.set_title("Measurement Timestamps Across Time Reversal")
    ax_top.set_xlabel("Call Time (s)")
    ax_top.set_ylabel("Measurement Timestamp (s)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: pending queue length and measurement counts.
    labels = ["Pending\n(before)", "Pending\n(after)", "Phase 1\nvalid", "Phase 3\nvalid"]
    counts = [pending_before, pending_after, len(phase1_valid_ts), len(phase3_valid_ts)]
    bar_colors = ["tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    bars = ax_bottom.bar(labels, counts, color=bar_colors, width=0.5)
    for bar, c in zip(bars, counts):
        ax_bottom.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
                       str(c), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_bottom.set_title("Pipeline State Before / After Reversal")
    ax_bottom.set_ylabel("Count")
    ax_bottom.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="time_reversal_handling")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Move a target linearly from position A to position B over one sampling period. "
        "A deliberately coarse caller tick (0.3 s) is used with a 2 Hz sensor so the sample "
        "epoch falls mid-interval rather than on a call boundary, exercising the internal "
        "linear interpolation. For each velocity profile the measured range is compared "
        "against the true position at the sample epoch and against the naive call-time "
        "position to demonstrate the interpolation benefit."
    ),
    goal=(
        "Validate the intra-period position interpolation logic that aligns each range sample "
        "with the target's true position at the sample epoch rather than at the caller's clock."
    ),
    passing_criteria=(
        "For each velocity profile the measured range equals the Euclidean distance to the "
        "linearly interpolated position at the sample epoch within 1e-6 m precision. "
        "The naive call-time range error is visibly larger for non-zero velocities."
    ),
)
def test_position_interpolation(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SAMPLING_RATE = 2.0      # [Hz] period = 0.5 s
    LATENCY = 0.0            # [s] zero latency for immediate delivery
    DT_CALL = 0.3            # [s] deliberately misaligned with the 0.5 s period so the
                             #     sample epoch falls mid-interval (alpha != 1)

    class InterpConfig(LidarConfig):
        range_min = 1.0
        range_max = 10000.0
        range_accuracy = 0.0
        sampling_rate = SAMPLING_RATE
        latency = LATENCY
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        sample_time_jitter_std = 0.0
        latency_jitter_std = 0.0

    # The target moves along +x at constant velocity.
    # Calls happen at: 0, 0.3, 0.6, 0.9, ...
    # Sample clock anchors to 0, then next epoch = 0.5 s.
    # The sample at epoch=0.5 s fires during the call at t=0.6 s (first call >= 0.5).
    # Interpolation: alpha = (0.5 − 0.3) / (0.6 − 0.3) = 2/3.
    # Interpolated position = pos(0.3) + 2/3 * (pos(0.6) − pos(0.3)) = pos(0.5).
    POS_START = 100.0        # [m] initial x-distance
    VELOCITIES = np.array([-30.0, -15.0, 0.0, 15.0, 30.0, 50.0])  # [m/s] along +x
    period = 1.0 / SAMPLING_RATE  # [s] = 0.5 s

    measured_ranges = []
    expected_ranges = []
    naive_ranges = []       # range at the call-time position (what you'd get without interp.)

    for vel in VELOCITIES:
        sensor = Lidar(config=InterpConfig)

        # Tick forward, recording the measurement at the second sample epoch.
        collected = None
        call_time_pos_x = None
        t = 0.0
        while t <= period + 1.0:
            pos_t = [POS_START + vel * t, 0.0, 0.0]
            m = sensor.measure(pos_t, add_noise=False, current_time=t)
            if m.get("valid", False) and not m.get("stale", False):
                ts = float(m["timestamp"])
                if abs(ts - period) < 1e-6:
                    collected = m
                    call_time_pos_x = POS_START + vel * t   # position fed at this call
            t += DT_CALL

        assert collected is not None, (
            f"no measurement at sample epoch for velocity = {vel:.1f} m/s"
        )

        measured_range = float(collected["range"])
        expected_pos_x = POS_START + vel * period           # true position at sample epoch
        expected_range = abs(expected_pos_x)
        naive_range = abs(call_time_pos_x)                  # range if call-time pos were used

        measured_ranges.append(measured_range)
        expected_ranges.append(expected_range)
        naive_ranges.append(naive_range)

        assert abs(measured_range - expected_range) < 1e-6, (
            f"velocity = {vel:.1f} m/s: measured range {measured_range:.6f} != "
            f"expected {expected_range:.6f} (interpolated position at sample epoch)"
        )

    measured_ranges = np.array(measured_ranges)
    expected_ranges = np.array(expected_ranges)
    naive_ranges = np.array(naive_ranges)

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: measured, expected (interpolated), and naive (call-time) ranges vs velocity.
    ax_top.plot(VELOCITIES, expected_ranges, color="tab:orange", linewidth=2.0,
                linestyle="--", marker="s", markersize=6, label="expected (interpolated pos)")
    ax_top.plot(VELOCITIES, measured_ranges, color="tab:blue", linewidth=2.0,
                marker="o", markersize=6, label="measured")
    ax_top.plot(VELOCITIES, naive_ranges, color="tab:red", linewidth=1.5,
                linestyle=":", marker="x", markersize=6, label="naive (call-time pos)")
    ax_top.set_title("Range at Sample Epoch vs Target Velocity")
    ax_top.set_xlabel("Target Velocity (m/s)")
    ax_top.set_ylabel("Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: error comparison — interpolation error (≈ 0) vs naive call-time error.
    interp_errors = measured_ranges - expected_ranges
    naive_errors = naive_ranges - expected_ranges
    bar_width = 4.0
    ax_bottom.bar(VELOCITIES - bar_width / 2, interp_errors, width=bar_width,
                  color="tab:blue", label="interpolation error")
    ax_bottom.bar(VELOCITIES + bar_width / 2, naive_errors, width=bar_width,
                  color="tab:red", alpha=0.7, label="call-time error (no interp.)")
    ax_bottom.axhline(0.0, color="black", linestyle="-", linewidth=0.5)
    ax_bottom.set_title("Range Error: Interpolated vs Naive Call-Time Position")
    ax_bottom.set_xlabel("Target Velocity (m/s)")
    ax_bottom.set_ylabel("Error (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="position_interpolation")
    plt.close(fig)


# ============================================================================
# Physics-based detection chain tests
# ============================================================================


@pytest.mark.test_meta(
    description=(
        "Sweep target range with atmosphere disabled, Lambertian reflectivity, and normal "
        "incidence. Received power should follow a 1/r^4 law (1/r^2 geometric falloff "
        "divided by footprint area proportional to r^2). A linear fit on the log-log plot "
        "must recover slope = -4 within tight tolerance."
    ),
    goal=(
        "Validate the geometric and footprint area terms of the lidar link budget equation "
        "in isolation from atmospheric and angular effects."
    ),
    passing_criteria=(
        "log-log slope of received power vs range equals -4.0 within +-0.01. "
        "Absolute power at every range matches the closed-form prediction within 1e-12 relative error."
    ),
)
def test_received_power_vs_range(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REFLECTIVITY = 0.7
    E_TX = 1.0
    A_RX = 1.0
    BEAM_DIV = np.deg2rad(0.2)   # [rad] full angle divergence

    class PowerConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        pulse_energy = E_TX
        receiver_aperture_area = A_RX
        beam_divergence = BEAM_DIV
        atmosphere_extinction_coeff = 0.0
        min_detectable_power = 1e-30       # effectively disable detection gating
        intensity_noise_std = 0.0

    sensor = Lidar(config=PowerConfig)
    material = Material(reflectivity=REFLECTIVITY, retro_reflectivity=0.0)

    ranges = np.logspace(np.log10(5.0), np.log10(500.0), 60)  # [m]
    powers = np.array([sensor._received_power(r, 1.0, material) for r in ranges])

    # Closed-form prediction: P = E * rho * A_rx / (4 pi^2 r^4 tan^2(div/2))
    tan_half = np.tan(BEAM_DIV * 0.5)
    predicted = E_TX * REFLECTIVITY * A_RX / (4.0 * np.pi ** 2 * ranges ** 4 * tan_half ** 2)
    np.testing.assert_allclose(powers, predicted, rtol=1e-12)

    # Linear regression on log-log data to recover slope.
    log_r = np.log10(ranges)
    log_p = np.log10(powers)
    slope, intercept = np.polyfit(log_r, log_p, 1)
    assert abs(slope - (-4.0)) < 0.01, f"log-log slope = {slope:.6f}, expected -4.0"

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    ax_top.loglog(ranges, powers, color="tab:blue", linewidth=2.0, marker=".", markersize=4, label="measured")
    ax_top.loglog(ranges, predicted, color="tab:orange", linewidth=1.5, linestyle="--", label="theory (1/r⁴)")
    ax_top.set_title(f"Received Power vs Range (slope = {slope:.4f})")
    ax_top.set_xlabel("Range (m)")
    ax_top.set_ylabel("Received Power (arb)")
    ax_top.grid(True, alpha=0.3, which="both")
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    residual_pct = (powers - predicted) / predicted * 100.0
    ax_bottom.semilogx(ranges, residual_pct, color="tab:blue", linewidth=1.5)
    ax_bottom.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    ax_bottom.set_title("Relative Residual (measured − theory)")
    ax_bottom.set_xlabel("Range (m)")
    ax_bottom.set_ylabel("Residual (%)")
    ax_bottom.grid(True, alpha=0.3, which="both")

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="received_power_vs_range")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Enable atmosphere_extinction_coeff while keeping all other factors fixed. "
        "Sweep range and verify received power decays as exp(-2 alpha r) on top of the "
        "geometric 1/r^4 baseline. Recover the extinction coefficient from the data and "
        "compare to the configured value."
    ),
    goal=(
        "Validate the Beer-Lambert two-way atmospheric attenuation term in isolation."
    ),
    passing_criteria=(
        "Recovered extinction coefficient matches the configured value within 1%%. "
        "Attenuation ratio P_atm / P_no_atm equals exp(-2 alpha r) within 1e-12 at every range."
    ),
)
def test_atmosphere_extinction(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ALPHA = 0.005  # [1/m] extinction coefficient
    REFLECTIVITY = 0.5

    class AtmConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        atmosphere_extinction_coeff = ALPHA
        beam_divergence = np.deg2rad(0.2)
        pulse_energy = 1.0
        receiver_aperture_area = 1.0
        min_detectable_power = 1e-30
        intensity_noise_std = 0.0

    class NoAtmConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        atmosphere_extinction_coeff = 0.0
        beam_divergence = np.deg2rad(0.2)
        pulse_energy = 1.0
        receiver_aperture_area = 1.0
        min_detectable_power = 1e-30
        intensity_noise_std = 0.0

    sensor_atm = Lidar(config=AtmConfig)
    sensor_no_atm = Lidar(config=NoAtmConfig)
    material = Material(reflectivity=REFLECTIVITY, retro_reflectivity=0.0)

    ranges = np.linspace(10.0, 400.0, 80)  # [m]
    power_atm = np.array([sensor_atm._received_power(r, 1.0, material) for r in ranges])
    power_no_atm = np.array([sensor_no_atm._received_power(r, 1.0, material) for r in ranges])

    # Attenuation ratio = exp(-2 alpha r).
    attenuation_ratio = power_atm / power_no_atm
    expected_ratio = np.exp(-2.0 * ALPHA * ranges)
    np.testing.assert_allclose(attenuation_ratio, expected_ratio, rtol=1e-12)

    # Recover alpha: ln(ratio) = -2 alpha r  →  slope = -2 alpha.
    ln_ratio = np.log(attenuation_ratio)
    slope, _ = np.polyfit(ranges, ln_ratio, 1)
    alpha_recovered = -slope / 2.0
    assert abs(alpha_recovered - ALPHA) / ALPHA < 0.01, (
        f"recovered alpha = {alpha_recovered:.6f}, expected {ALPHA:.6f}"
    )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    ax_top.plot(ranges, attenuation_ratio, color="tab:blue", linewidth=2.0, label="measured ratio")
    ax_top.plot(ranges, expected_ratio, color="tab:orange", linewidth=1.5, linestyle="--",
                label=f"exp(−2·{ALPHA}·r)")
    ax_top.set_title(f"Atmospheric Attenuation (α = {ALPHA} 1/m, recovered = {alpha_recovered:.5f})")
    ax_top.set_xlabel("Range (m)")
    ax_top.set_ylabel("P_atm / P_no_atm")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    ax_bottom.plot(ranges, ln_ratio, color="tab:blue", linewidth=2.0, label="ln(ratio)")
    ax_bottom.plot(ranges, -2.0 * ALPHA * ranges, color="tab:orange", linewidth=1.5, linestyle="--",
                   label=f"−2α·r (α = {ALPHA})")
    ax_bottom.set_title("Log Attenuation vs Range (linear fit)")
    ax_bottom.set_xlabel("Range (m)")
    ax_bottom.set_ylabel("ln(P_atm / P_no_atm)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="atmosphere_extinction")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Fix range and atmosphere, sweep the surface incidence angle from near-normal to "
        "near-grazing. With retro_reflectivity = 0 the Lambertian model predicts received "
        "power proportional to cos(incidence angle). Verify proportionality and plot the "
        "angular dependence."
    ),
    goal=(
        "Validate the diffuse (Lambertian) reflectance term: power scales linearly with "
        "cos(theta) when retro-reflectivity is zero."
    ),
    passing_criteria=(
        "Normalized power P(theta)/P(0) equals cos(theta) within 1e-12 at every tested angle."
    ),
)
def test_incidence_angle_dependence(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DISTANCE = 50.0          # [m] fixed range
    REFLECTIVITY = 0.6

    class AngleConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        atmosphere_extinction_coeff = 0.0
        beam_divergence = np.deg2rad(0.2)
        pulse_energy = 1.0
        receiver_aperture_area = 1.0
        min_detectable_power = 1e-30
        intensity_noise_std = 0.0

    sensor = Lidar(config=AngleConfig)
    material = Material(reflectivity=REFLECTIVITY, retro_reflectivity=0.0)

    # Sweep incidence angle from 0 (normal) to 85 degrees.
    angles_deg = np.linspace(0.0, 85.0, 60)
    cos_angles = np.cos(np.deg2rad(angles_deg))

    powers = np.array([sensor._received_power(DISTANCE, float(ca), material) for ca in cos_angles])
    power_normal = powers[0]  # power at normal incidence (theta = 0)

    # Normalized power should equal cos(theta).
    normalized = powers / power_normal
    np.testing.assert_allclose(normalized, cos_angles, rtol=0.0, atol=1e-12)

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    ax_top.plot(angles_deg, normalized, color="tab:blue", linewidth=2.0, label="measured P/P₀")
    ax_top.plot(angles_deg, cos_angles, color="tab:orange", linewidth=1.5, linestyle="--",
                label="cos(θ)")
    ax_top.set_title("Normalized Power vs Incidence Angle (Lambertian)")
    ax_top.set_xlabel("Incidence Angle (deg)")
    ax_top.set_ylabel("P(θ) / P(0)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    residual = (normalized - cos_angles) * 1e12
    ax_bottom.plot(angles_deg, residual, color="tab:blue", linewidth=1.5)
    ax_bottom.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    ax_bottom.set_title("Residual (measured − cos θ)")
    ax_bottom.set_xlabel("Incidence Angle (deg)")
    ax_bottom.set_ylabel("Residual (×10⁻¹²)")
    ax_bottom.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="incidence_angle_dependence")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Set retro_reflectivity > 0 on a material and sweep the incidence angle. "
        "The retro component adds a constant power offset independent of theta, so "
        "P_retro(theta) - P_base(theta) should be constant across all angles."
    ),
    goal=(
        "Confirm the retro-reflective term contributes a fixed, angle-independent boost "
        "to received power as specified by the link budget equation."
    ),
    passing_criteria=(
        "The absolute power difference (P_retro - P_base) is constant across all incidence "
        "angles within 1e-12 relative tolerance. The expected boost equals the power that "
        "would result from replacing rho_d*cos(theta) with rho_r in the link budget."
    ),
)
def test_retro_reflectivity(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DISTANCE = 50.0          # [m] fixed range
    REFLECTIVITY = 0.5       # Lambertian component
    RETRO = 0.3              # retro component

    class RetroConfig(LidarConfig):
        range_min = 1.0
        range_max = 1000.0
        range_accuracy = 0.0
        atmosphere_extinction_coeff = 0.0
        beam_divergence = np.deg2rad(0.2)
        pulse_energy = 1.0
        receiver_aperture_area = 1.0
        min_detectable_power = 1e-30
        intensity_noise_std = 0.0

    sensor = Lidar(config=RetroConfig)
    mat_retro = Material(reflectivity=REFLECTIVITY, retro_reflectivity=RETRO)
    mat_base = Material(reflectivity=REFLECTIVITY, retro_reflectivity=0.0)

    angles_deg = np.linspace(5.0, 80.0, 50)
    cos_angles = np.cos(np.deg2rad(angles_deg))

    p_retro = np.array([sensor._received_power(DISTANCE, float(ca), mat_retro) for ca in cos_angles])
    p_base = np.array([sensor._received_power(DISTANCE, float(ca), mat_base) for ca in cos_angles])

    # The difference should be constant = K * rho_r, where K is the common scale factor.
    # K = E_tx * A_rx / (4 pi^2 r^4 tan^2(div/2))  (for zero atmosphere)
    tan_half = np.tan(sensor.beam_divergence * 0.5)
    K = sensor.pulse_energy * sensor.receiver_aperture_area / (
        4.0 * np.pi ** 2 * DISTANCE ** 4 * tan_half ** 2
    )
    expected_boost = K * RETRO

    power_diff = p_retro - p_base
    np.testing.assert_allclose(power_diff, expected_boost, rtol=1e-12)

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    ax_top.plot(angles_deg, p_retro, color="tab:blue", linewidth=2.0, label="with retro")
    ax_top.plot(angles_deg, p_base, color="tab:orange", linewidth=2.0, linestyle="--", label="Lambertian only")
    ax_top.set_title(f"Received Power: Retro ({RETRO}) vs Lambertian Only")
    ax_top.set_xlabel("Incidence Angle (deg)")
    ax_top.set_ylabel("Received Power (arb)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    ax_bottom.plot(angles_deg, power_diff, color="tab:blue", linewidth=2.0, label="P_retro − P_base")
    ax_bottom.axhline(expected_boost, color="tab:orange", linestyle="--", linewidth=1.5,
                      label=f"expected boost = {expected_boost:.4e}")
    ax_bottom.set_title("Retro Power Boost (should be constant)")
    ax_bottom.set_xlabel("Incidence Angle (deg)")
    ax_bottom.set_ylabel("Power Difference (arb)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="retro_reflectivity")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Analytically compute the range at which detection_probability = 50 %% from the "
        "link budget equation (P_rx = P_min at r_50). Then run a Monte Carlo sweep over "
        "range using a single-beam scanner aimed at a perpendicular plane and verify the "
        "empirical detection rate crosses 50 %% at the predicted range."
    ),
    goal=(
        "End-to-end validation of the SNR-based detection model: the sigmoid detection "
        "probability P_rx / (P_rx + P_min) must produce the expected detection statistics "
        "as a function of range."
    ),
    passing_criteria=(
        "Empirical 50 %% crossing range (linear interpolation) lies within 5 %% of the "
        "analytically predicted r_50. At close range (r << r_50) detection rate is near "
        "100 %%; at far range (r >> r_50) it drops toward 0 %%."
    ),
)
def test_snr_detection_threshold(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REFLECTIVITY = 0.5
    E_TX = 1.0
    A_RX = 1.0
    P_MIN = 1e-6
    BEAM_DIV = np.deg2rad(0.2)
    N_MC = 500   # trials per range

    class SnrConfig(LidarConfig):
        range_min = 1.0
        range_max = 600.0
        fov = np.deg2rad(10.0)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        pulse_energy = E_TX
        receiver_aperture_area = A_RX
        beam_divergence = BEAM_DIV
        atmosphere_extinction_coeff = 0.0
        min_detectable_power = P_MIN
        intensity_noise_std = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        scan_azimuth_samples = 1
        scan_elevation_angles = (0.0,)

    # Analytic r_50: P_rx(r_50) = P_min
    # P_rx = E * rho * A_rx / (4 pi^2 r^4 tan^2(div/2))
    tan_half = np.tan(BEAM_DIV * 0.5)
    r_50_analytic = (E_TX * REFLECTIVITY * A_RX / (4.0 * np.pi ** 2 * P_MIN * tan_half ** 2)) ** 0.25

    # Sweep ranges around r_50.
    test_ranges = np.linspace(max(r_50_analytic * 0.2, 5.0), min(r_50_analytic * 2.0, 590.0), 30)
    detection_rates = np.empty(test_ranges.size)
    material = Material(reflectivity=REFLECTIVITY, retro_reflectivity=0.0)

    for i, r in enumerate(test_ranges):
        scene = Scene([
            Plane(
                point=[float(r), 0.0, 0.0],
                normal=[-1.0, 0.0, 0.0],
                material=material,
                object_id="target",
            )
        ])
        detections = 0
        for trial in range(N_MC):

            class SeededSnr(SnrConfig):
                random_seed = 100000 + i * N_MC + trial

            sensor = Lidar(config=SeededSnr)
            frame = sensor.simulate_scene_frame(
                scene=scene,
                sensor_position=[0.0, 0.0, 0.0],
                timestamp=0.0,
                add_noise=True,
            )
            if frame["returns"][0].get("valid", False):
                detections += 1
        detection_rates[i] = detections / N_MC

    # Find the empirical 50% crossing by linear interpolation.
    crossed = False
    r_50_empirical = r_50_analytic  # fallback
    for j in range(len(test_ranges) - 1):
        if detection_rates[j] >= 0.5 >= detection_rates[j + 1]:
            # Linear interpolation between the two bracketing points.
            frac = (0.5 - detection_rates[j]) / (detection_rates[j + 1] - detection_rates[j])
            r_50_empirical = test_ranges[j] + frac * (test_ranges[j + 1] - test_ranges[j])
            crossed = True
            break

    assert crossed, "empirical detection rate never crossed 50%"
    assert abs(r_50_empirical - r_50_analytic) / r_50_analytic < 0.05, (
        f"empirical r_50 = {r_50_empirical:.2f} m vs analytic = {r_50_analytic:.2f} m "
        f"(error = {abs(r_50_empirical - r_50_analytic) / r_50_analytic * 100:.1f}%%)"
    )

    # Theoretical detection probability curve.
    theory_power = E_TX * REFLECTIVITY * A_RX / (4.0 * np.pi ** 2 * test_ranges ** 4 * tan_half ** 2)
    theory_det = theory_power / (theory_power + P_MIN)

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    ax_top.plot(test_ranges, detection_rates * 100, color="tab:blue", linewidth=2.0, marker=".", markersize=5,
                label=f"Monte Carlo (N = {N_MC})")
    ax_top.plot(test_ranges, theory_det * 100, color="tab:orange", linewidth=1.5, linestyle="--",
                label="theory: P/(P+P_min)")
    ax_top.axvline(r_50_analytic, color="tab:red", linestyle="--", linewidth=1.5,
                   label=f"r_50 analytic = {r_50_analytic:.1f} m")
    ax_top.axvline(r_50_empirical, color="tab:red", linestyle=":", linewidth=1.5,
                   label=f"r_50 empirical = {r_50_empirical:.1f} m")
    ax_top.axhline(50.0, color="tab:gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_top.set_title("Detection Rate vs Range (SNR Threshold)")
    ax_top.set_xlabel("Range (m)")
    ax_top.set_ylabel("Detection Rate (%)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    ax_bottom.semilogy(test_ranges, theory_power, color="tab:blue", linewidth=2.0, label="P_rx(r)")
    ax_bottom.axhline(P_MIN, color="tab:red", linestyle="--", linewidth=1.5, label=f"P_min = {P_MIN:.1e}")
    ax_bottom.axvline(r_50_analytic, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_bottom.set_title("Received Power vs Range (link budget)")
    ax_bottom.set_xlabel("Range (m)")
    ax_bottom.set_ylabel("Received Power (arb)")
    ax_bottom.grid(True, alpha=0.3, which="both")
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="snr_detection_threshold")
    plt.close(fig)


# ============================================================================
# Scene scanning integration tests
# ============================================================================


@pytest.mark.test_meta(
    description=(
        "Scan a sphere centered on boresight. Valid returns should form a circular cluster "
        "in azimuth/elevation space bounded by the angular subtent of the sphere. "
        "Reconstruct a point cloud from the hit points and verify every point lies on the "
        "sphere surface within machine precision."
    ),
    goal=(
        "End-to-end validation of the full-frame scan pipeline: beam generation, ray casting "
        "against a sphere, and geometric consistency of the returned point cloud."
    ),
    passing_criteria=(
        "All hit points satisfy |hit - center| = radius within 1e-6 m. "
        "The angular extent of valid returns matches the geometric subtent of the sphere. "
        "No valid return lies outside the sphere's angular footprint."
    ),
)
def test_full_frame_sphere_consistency(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SPHERE_DISTANCE = 50.0   # [m] centre along +x
    SPHERE_RADIUS = 5.0      # [m]
    AZ_SAMPLES = 121
    ELEV_ANGLES = np.deg2rad(np.array([-6.0, -3.0, 0.0, 3.0, 6.0]))  # five channels

    class SphereConfig(LidarConfig):
        range_min = 1.0
        range_max = 200.0
        fov = np.deg2rad(30.0)
        scan_azimuth_min = np.deg2rad(-15.0)
        scan_azimuth_max = np.deg2rad(15.0)
        scan_azimuth_samples = AZ_SAMPLES
        scan_elevation_angles = tuple(ELEV_ANGLES.tolist())
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        atmosphere_extinction_coeff = 0.0
        min_detectable_power = 1e-30
        intensity_noise_std = 0.0

    sensor = Lidar(config=SphereConfig)
    scene = Scene([
        Sphere(
            center=[SPHERE_DISTANCE, 0.0, 0.0],
            radius=SPHERE_RADIUS,
            material=Material(reflectivity=0.8, name="sphere"),
            object_id="sphere",
        )
    ])

    frame = sensor.simulate_scene_frame(
        scene=scene,
        sensor_position=[0.0, 0.0, 0.0],
        timestamp=0.0,
        add_noise=False,
    )

    assert frame["valid"] is True
    returns = frame["returns"]

    # Separate valid and invalid returns.
    valid_returns = [r for r in returns if r.get("valid", False)]
    assert len(valid_returns) > 0, "no valid returns from sphere scan"

    # Verify every hit point lies on the sphere surface.
    center = np.array([SPHERE_DISTANCE, 0.0, 0.0])
    hit_points = np.array([r["hit_point"] for r in valid_returns])
    distances_from_center = np.linalg.norm(hit_points - center, axis=1)
    np.testing.assert_allclose(
        distances_from_center, SPHERE_RADIUS, atol=1e-6,
        err_msg="hit points do not lie on the sphere surface",
    )

    # Valid returns should be confined within the angular subtent of the sphere.
    angular_subtent = float(np.arctan2(SPHERE_RADIUS, SPHERE_DISTANCE))
    valid_az = np.array([float(r["azimuth"]) for r in valid_returns])
    valid_el = np.array([float(r["elevation"]) for r in valid_returns])
    assert np.all(np.abs(valid_az) <= angular_subtent + np.deg2rad(1.0)), (
        "valid return outside azimuth subtent"
    )
    assert np.all(np.abs(valid_el) <= angular_subtent + np.deg2rad(1.0)), (
        "valid return outside elevation subtent"
    )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: azimuth/elevation scatter of valid vs invalid returns.
    all_az = np.array([float(r["azimuth"]) for r in returns])
    all_el = np.array([float(r["elevation"]) for r in returns])
    all_valid = np.array([r.get("valid", False) for r in returns])

    ax_top.scatter(np.rad2deg(all_az[~all_valid]), np.rad2deg(all_el[~all_valid]),
                   s=6, color="tab:gray", alpha=0.4, label="miss")
    ax_top.scatter(np.rad2deg(valid_az), np.rad2deg(valid_el),
                   s=12, color="tab:blue", alpha=0.8, label="hit")
    # Draw the angular subtent circle.
    circle_theta = np.linspace(0, 2 * np.pi, 200)
    ax_top.plot(np.rad2deg(angular_subtent) * np.cos(circle_theta),
                np.rad2deg(angular_subtent) * np.sin(circle_theta),
                color="tab:orange", linewidth=1.5, linestyle="--", label="subtent boundary")
    ax_top.set_title(f"Scan Pattern Hits ({len(valid_returns)} / {len(returns)} beams)")
    ax_top.set_xlabel("Azimuth (deg)")
    ax_top.set_ylabel("Elevation (deg)")
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: distance of each hit point from sphere centre (should all = radius).
    ax_bottom.plot(distances_from_center, color="tab:blue", linewidth=1.5, marker=".", markersize=3)
    ax_bottom.axhline(SPHERE_RADIUS, color="tab:orange", linestyle="--", linewidth=1.5,
                      label=f"radius = {SPHERE_RADIUS} m")
    ax_bottom.set_title("Hit Point Distance from Sphere Centre")
    ax_bottom.set_xlabel("Valid Return Index")
    ax_bottom.set_ylabel("Distance (m)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="full_frame_sphere_consistency")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Place a small sphere in front of a large wall along the sensor boresight. "
        "Beams that intersect the sphere must report the sphere's range, not the wall's. "
        "Beams that miss the sphere should hit the wall at the expected range."
    ),
    goal=(
        "Verify the scene ray caster returns the closest intersection when multiple objects "
        "lie along the same ray direction."
    ),
    passing_criteria=(
        "Every beam that hits the sphere reports object_id = sphere and range consistent "
        "with the sphere geometry. Every beam that misses the sphere reports object_id = wall "
        "and range consistent with the wall geometry. No beam detects the far wall through "
        "the sphere."
    ),
)
def test_multi_object_occlusion(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SPHERE_DIST = 15.0       # [m] sphere centre along +x
    SPHERE_RADIUS = 2.0      # [m]
    WALL_DIST = 50.0         # [m] wall perpendicular to +x
    AZ_SAMPLES = 181

    class OcclusionConfig(LidarConfig):
        range_min = 1.0
        range_max = 200.0
        fov = np.deg2rad(30.0)
        scan_azimuth_min = np.deg2rad(-15.0)
        scan_azimuth_max = np.deg2rad(15.0)
        scan_azimuth_samples = AZ_SAMPLES
        scan_elevation_angles = (0.0,)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        atmosphere_extinction_coeff = 0.0
        min_detectable_power = 1e-30
        intensity_noise_std = 0.0

    sensor = Lidar(config=OcclusionConfig)
    scene = Scene([
        Sphere(
            center=[SPHERE_DIST, 0.0, 0.0],
            radius=SPHERE_RADIUS,
            material=Material(reflectivity=0.6, name="sphere"),
            object_id="sphere",
        ),
        Plane(
            point=[WALL_DIST, 0.0, 0.0],
            normal=[-1.0, 0.0, 0.0],
            material=Material(reflectivity=0.9, name="wall"),
            object_id="wall",
        ),
    ])

    frame = sensor.simulate_scene_frame(
        scene=scene,
        sensor_position=[0.0, 0.0, 0.0],
        timestamp=0.0,
        add_noise=False,
    )
    returns = frame["returns"]

    azimuths = np.array([float(r["azimuth"]) for r in returns])
    measured_ranges = np.array([float(r["range"]) if r.get("valid") else np.nan for r in returns])
    object_ids = [r.get("object_id", "") for r in returns]

    # Angular subtent of the sphere from the sensor.
    sphere_subtent = float(np.arctan2(SPHERE_RADIUS, SPHERE_DIST))

    sphere_beams = 0
    wall_beams = 0
    for i, r in enumerate(returns):
        assert r.get("valid", False), f"beam {i} invalid — both objects should be reachable"
        az = azimuths[i]
        oid = object_ids[i]
        meas_r = measured_ranges[i]
        if abs(az) < sphere_subtent - np.deg2rad(0.5):
            # Beam is well within the sphere — must hit sphere, not wall.
            assert oid == "sphere", (
                f"beam {i} at az={np.rad2deg(az):.2f}° hit '{oid}' instead of sphere"
            )
            assert meas_r < WALL_DIST, (
                f"beam {i} reported range {meas_r:.2f} > wall distance — seeing through sphere"
            )
            sphere_beams += 1
        elif abs(az) > sphere_subtent + np.deg2rad(0.5):
            # Beam clearly misses the sphere — must hit wall.
            assert oid == "wall", (
                f"beam {i} at az={np.rad2deg(az):.2f}° hit '{oid}' instead of wall"
            )
            expected_wall_range = WALL_DIST / np.cos(az)
            assert abs(meas_r - expected_wall_range) < 1e-6, (
                f"beam {i} wall range mismatch: {meas_r:.6f} vs {expected_wall_range:.6f}"
            )
            wall_beams += 1

    assert sphere_beams > 0, "no beams hit the sphere"
    assert wall_beams > 0, "no beams hit the wall"

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    az_deg = np.rad2deg(azimuths)
    sphere_mask = np.array([oid == "sphere" for oid in object_ids])
    wall_mask = np.array([oid == "wall" for oid in object_ids])

    # Panel 1: range profile coloured by object.
    ax_top.scatter(az_deg[sphere_mask], measured_ranges[sphere_mask], s=12, color="tab:blue",
                   label=f"sphere ({sphere_mask.sum()} beams)")
    ax_top.scatter(az_deg[wall_mask], measured_ranges[wall_mask], s=12, color="tab:orange",
                   label=f"wall ({wall_mask.sum()} beams)")
    ax_top.axhline(SPHERE_DIST, color="tab:blue", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_top.axhline(WALL_DIST, color="tab:orange", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_top.axvline(np.rad2deg(sphere_subtent), color="tab:red", linestyle="--", linewidth=1.0, alpha=0.5,
                   label="sphere subtent")
    ax_top.axvline(-np.rad2deg(sphere_subtent), color="tab:red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_top.set_title("Range Profile: Sphere Occludes Wall")
    ax_top.set_xlabel("Azimuth (deg)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: top-down view of hit points.
    all_hits = np.array([r["hit_point"] for r in returns if r.get("valid")])
    ax_bottom.scatter(all_hits[sphere_mask, 1], all_hits[sphere_mask, 0], s=8, color="tab:blue", label="sphere hits")
    ax_bottom.scatter(all_hits[wall_mask, 1], all_hits[wall_mask, 0], s=8, color="tab:orange", label="wall hits")
    ax_bottom.plot(0.0, 0.0, "k^", markersize=8, label="sensor")
    ax_bottom.set_title("Top-Down Hit Point Map (x vs y)")
    ax_bottom.set_xlabel("y (m)")
    ax_bottom.set_ylabel("x (m)")
    ax_bottom.set_aspect("equal")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="multi_object_occlusion")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Apply a known yaw rotation to sensor_orientation and scan a target on the world "
        "+x axis. With zero rotation the target sits at azimuth = 0. With a yaw of +theta "
        "the sensor's boresight rotates, so the target should appear at azimuth = -theta in "
        "the scan pattern. Verify the peak return shifts by the expected number of azimuth "
        "indices."
    ),
    goal=(
        "Confirm the sensor_orientation rotation matrix correctly transforms beam directions "
        "from the sensor body frame to the world frame."
    ),
    passing_criteria=(
        "The azimuth index of the closest return to the target shifts by exactly the number "
        "of indices corresponding to the applied yaw angle. The measured range is consistent "
        "with the rotated geometry."
    ),
)
def test_rotated_sensor(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    WALL_DIST = 30.0             # [m] wall perpendicular to +x
    YAW_DEG = 10.0               # [deg] yaw rotation around z axis
    YAW_RAD = np.deg2rad(YAW_DEG)
    AZ_SAMPLES = 181
    AZ_MIN = np.deg2rad(-15.0)
    AZ_MAX = np.deg2rad(15.0)

    class RotatedConfig(LidarConfig):
        range_min = 1.0
        range_max = 200.0
        fov = np.deg2rad(30.0)
        scan_azimuth_min = AZ_MIN
        scan_azimuth_max = AZ_MAX
        scan_azimuth_samples = AZ_SAMPLES
        scan_elevation_angles = (0.0,)
        range_accuracy = 0.0
        noise_floor_std = 0.0
        noise_range_coeff = 0.0
        bias_init_std = 0.0
        bias_rw_std = 0.0
        bias_drift_rate = 0.0
        scale_factor_ppm = 0.0
        scale_error_std_ppm = 0.0
        dropout_prob = 0.0
        dropout_range_coeff = 0.0
        outlier_prob = 0.0
        quantization_step = 0.0
        atmosphere_extinction_coeff = 0.0
        min_detectable_power = 1e-30
        intensity_noise_std = 0.0

    material = Material(reflectivity=0.8)
    scene = Scene([
        Plane(
            point=[WALL_DIST, 0.0, 0.0],
            normal=[-1.0, 0.0, 0.0],
            material=material,
            object_id="wall",
        )
    ])

    # Scan 1: no rotation (identity orientation).
    sensor_no_rot = Lidar(config=RotatedConfig)
    frame_no_rot = sensor_no_rot.simulate_scene_frame(
        scene=scene,
        sensor_position=[0.0, 0.0, 0.0],
        sensor_orientation=None,
        timestamp=0.0,
        add_noise=False,
    )

    # Scan 2: yaw rotation of +YAW_RAD around z axis.
    cos_y, sin_y = np.cos(YAW_RAD), np.sin(YAW_RAD)
    R_yaw = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ])

    sensor_rot = Lidar(config=RotatedConfig)
    frame_rot = sensor_rot.simulate_scene_frame(
        scene=scene,
        sensor_position=[0.0, 0.0, 0.0],
        sensor_orientation=R_yaw,
        timestamp=0.0,
        add_noise=False,
    )

    # Extract range profiles.
    ranges_no_rot = np.array([float(r["range"]) if r.get("valid") else np.nan
                              for r in frame_no_rot["returns"]])
    ranges_rot = np.array([float(r["range"]) if r.get("valid") else np.nan
                           for r in frame_rot["returns"]])
    azimuths = np.linspace(AZ_MIN, AZ_MAX, AZ_SAMPLES)

    # Without rotation the minimum range (perpendicular hit) is at azimuth = 0 (centre index).
    min_idx_no_rot = int(np.nanargmin(ranges_no_rot))
    centre_idx = AZ_SAMPLES // 2
    assert min_idx_no_rot == centre_idx, (
        f"unrotated minimum at index {min_idx_no_rot}, expected centre {centre_idx}"
    )

    # With yaw rotation, the wall's perpendicular direction in sensor frame is at azimuth = -YAW.
    # The closest beam to az = -YAW should have the minimum range.
    expected_shifted_az = -YAW_RAD
    expected_shifted_idx = int(np.argmin(np.abs(azimuths - expected_shifted_az)))
    min_idx_rot = int(np.nanargmin(ranges_rot))
    assert abs(min_idx_rot - expected_shifted_idx) <= 1, (
        f"rotated minimum at index {min_idx_rot}, expected ~{expected_shifted_idx} "
        f"(az = {np.rad2deg(expected_shifted_az):.1f}°)"
    )

    # At the shifted index the range should equal WALL_DIST / cos(0) = WALL_DIST
    # (beam perpendicular to wall in world frame).
    assert abs(ranges_rot[min_idx_rot] - WALL_DIST) < 0.1, (
        f"rotated perpendicular range {ranges_rot[min_idx_rot]:.4f} != wall distance {WALL_DIST}"
    )

    # ---- Plot ----
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    az_deg = np.rad2deg(azimuths)
    ax_top.plot(az_deg, ranges_no_rot, color="tab:blue", linewidth=2.0, label="no rotation")
    ax_top.plot(az_deg, ranges_rot, color="tab:orange", linewidth=2.0, label=f"yaw = +{YAW_DEG}°")
    ax_top.axvline(0.0, color="tab:blue", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_top.axvline(-YAW_DEG, color="tab:orange", linestyle=":", linewidth=1.0, alpha=0.5,
                   label=f"expected shift = {-YAW_DEG}°")
    ax_top.set_title("Range Profile Shift Under Sensor Yaw Rotation")
    ax_top.set_xlabel("Scan Azimuth (deg, sensor frame)")
    ax_top.set_ylabel("Measured Range (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Panel 2: index shift.
    shift = min_idx_rot - min_idx_no_rot
    az_step = (AZ_MAX - AZ_MIN) / (AZ_SAMPLES - 1)
    expected_shift_indices = int(round(-YAW_RAD / az_step))

    labels_bar = ["Expected", "Measured"]
    values_bar = [expected_shift_indices, shift]
    bar_colors = ["tab:orange", "tab:blue"]
    bars = ax_bottom.bar(labels_bar, values_bar, color=bar_colors, width=0.4)
    for bar, v in zip(bars, values_bar):
        ax_bottom.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.2,
                       str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_bottom.set_title("Azimuth Index Shift (minimum range beam)")
    ax_bottom.set_ylabel("Index Shift")
    ax_bottom.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="rotated_sensor")
    plt.close(fig)


@pytest.mark.test_meta(
    description=(
        "Scan an empty scene (no geometry objects). Every beam should return an invalid "
        "measurement with reason = 'no_hit'. No crashes, no valid returns."
    ),
    goal=(
        "Confirm the scanner handles the degenerate case of an empty scene gracefully "
        "and produces the expected diagnostic output for every beam."
    ),
    passing_criteria=(
        "num_valid == 0. Every beam has valid == False and reason == 'no_hit'. "
        "Total beam count matches scan_azimuth_samples * len(scan_elevation_angles)."
    ),
)
def test_empty_scene(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    AZ_SAMPLES = 61
    ELEV_ANGLES = np.deg2rad(np.array([-3.0, 0.0, 3.0]))

    class EmptyConfig(LidarConfig):
        range_min = 1.0
        range_max = 200.0
        fov = np.deg2rad(30.0)
        scan_azimuth_min = np.deg2rad(-15.0)
        scan_azimuth_max = np.deg2rad(15.0)
        scan_azimuth_samples = AZ_SAMPLES
        scan_elevation_angles = tuple(ELEV_ANGLES.tolist())
        range_accuracy = 0.0

    sensor = Lidar(config=EmptyConfig)
    scene = Scene()  # empty — no objects

    frame = sensor.simulate_scene_frame(
        scene=scene,
        sensor_position=[0.0, 0.0, 0.0],
        timestamp=0.0,
        add_noise=False,
    )

    expected_beams = AZ_SAMPLES * len(ELEV_ANGLES)
    assert frame["valid"] is True
    assert frame["num_beams"] == expected_beams
    assert frame["num_valid"] == 0, f"expected 0 valid returns, got {frame['num_valid']}"

    returns = frame["returns"]
    assert len(returns) == expected_beams

    reasons = []
    for i, r in enumerate(returns):
        assert r["valid"] is False, f"beam {i} unexpectedly valid in empty scene"
        reason = r.get("reason", "")
        assert reason == "no_hit", (
            f"beam {i} reason = '{reason}', expected 'no_hit'"
        )
        reasons.append(reason)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 5))

    # Bar chart: all beams report "no_hit".
    reason_counts = {"no_hit": reasons.count("no_hit"), "other": len(reasons) - reasons.count("no_hit")}
    bar_colors = ["tab:blue", "tab:red"]
    bars = ax.bar(list(reason_counts.keys()), list(reason_counts.values()), color=bar_colors, width=0.4)
    for bar, v in zip(bars, reason_counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title(f"Empty Scene: {expected_beams} Beams, 0 Valid Returns")
    ax.set_ylabel("Beam Count")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    _attach_plot_to_html_report(request, fig, name="empty_scene")
    plt.close(fig)
