import base64
import io

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.physics import Material, Plane, Scene
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
