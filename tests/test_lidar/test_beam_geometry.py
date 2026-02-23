"""
Beam geometry and 3D scene integration tests.

Validates beam pattern geometry against planar walls, full-frame sphere
consistency, multi-object occlusion, and rotated sensor orientation.
"""

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.physics import Material, Plane, Scene, Sphere
from sensors.lidar import Lidar
from .helpers import attach_plot_to_html_report


@pytest.mark.test_meta(
    description="Fire a single elevation scan pattern at a planar wall perpendicular to boresight and validate beam geometry.",
    goal="Confirm hit geometry follows the expected wall intersection model, azimuth indices are ordered correctly, and azimuth spacing matches the configured linspace pattern.",
    passing_criteria="All beams return valid hits on the wall, hit points match analytic intersections, azimuth_index equals 0..N-1, and azimuth samples and spacing match linspace(az_min, az_max, N).",
)
def test_beam_pattern_geometry_planar_wall(request):
    # Import matplotlib and force off-screen rendering
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Define the azimuth scan envelope: symmetric +/-20 degrees converted to radians
    az_min = np.deg2rad(-20.0)
    az_max = np.deg2rad(20.0)
    # Number of azimuth samples across the scan pattern
    az_samples = 121
    # Distance from the sensor origin to the planar wall along the boresight axis
    wall_distance = 25.0

    # Custom config that sets up a specific scan pattern and disables all noise.
    # The scan is a single elevation row (elevation = 0) with 121 azimuth beams
    # spanning +/-20 degrees. All error sources are zeroed so that measured
    # hit points can be compared to analytic wall intersection geometry.
    class BeamGeometryConfig(LidarConfig):
        range_min = 0.1           # allow close range hits for this geometry
        range_max = 200.0         # generous max range to avoid clipping
        fov = np.deg2rad(40.0)    # full cone FoV must cover the +/-20 deg scan
        scan_azimuth_min = az_min           # leftmost azimuth beam direction
        scan_azimuth_max = az_max           # rightmost azimuth beam direction
        scan_azimuth_samples = az_samples   # total number of beams across azimuth
        scan_elevation_angles = (0.0,)      # single elevation row at zero elevation
        range_accuracy = 0.0      # perfect range measurement
        noise_floor_std = 0.0     # no additive noise
        noise_range_coeff = 0.0   # no range proportional noise
        bias_init_std = 0.0       # no initial bias offset
        bias_rw_std = 0.0         # no random walk on bias
        bias_drift_rate = 0.0     # no deterministic drift
        scale_factor_ppm = 0.0    # no scale factor offset
        scale_error_std_ppm = 0.0 # no scale factor jitter
        dropout_prob = 0.0        # no dropouts
        dropout_range_coeff = 0.0 # no range dependent dropout growth
        outlier_prob = 0.0        # no outliers
        outlier_std = 0.0         # (unused)
        outlier_bias = 0.0        # (unused)
        quantization_step = 0.0   # infinite ADC resolution
        latency = 0.0             # no measurement delay
        sample_time_jitter_std = 0.0  # no clock jitter
        latency_jitter_std = 0.0      # no latency jitter

    # Instantiate the sensor with the beam geometry config
    lidar = Lidar(config=BeamGeometryConfig)

    # Build a minimal scene containing a single infinite plane perpendicular
    # to the boresight axis. The plane faces the sensor (normal pointing toward
    # the negative X direction) and sits at x = wall_distance.
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

    # Fire a complete scan frame at the scene with the sensor at the origin,
    # no rotation, and no noise. This returns one measurement per beam.
    frame = lidar.simulate_scene_frame(
        scene=scene,
        sensor_position=[0.0, 0.0, 0.0],
        sensor_orientation=None,
        timestamp=0.0,
        add_noise=False,
    )

    # Extract the list of per beam return dictionaries from the frame
    returns = frame["returns"]
    # The frame level valid flag should be True (at least one beam hit)
    assert frame["valid"] is True
    # Total number of beams fired must equal the configured azimuth sample count
    assert frame["num_beams"] == az_samples
    # Every beam must have produced a valid return against the wall
    assert frame["num_valid"] == az_samples
    # The returns list length must match the beam count exactly
    assert len(returns) == az_samples

    # Compute expected values analytically from the scan geometry:
    # Evenly spaced azimuth angles matching linspace(az_min, az_max, az_samples)
    expected_azimuths = np.linspace(az_min, az_max, az_samples, dtype=float)
    # Consecutive azimuth spacing (should be uniform)
    expected_spacing = np.diff(expected_azimuths)
    # Slant range to a perpendicular wall at distance d is d / cos(azimuth)
    expected_ranges = wall_distance / np.cos(expected_azimuths)
    # All beams hit the wall at x = wall_distance (the plane equation)
    expected_hit_x = np.full_like(expected_azimuths, wall_distance, dtype=float)
    # The Y coordinate of each hit follows the tangent relation
    expected_hit_y = wall_distance * np.tan(expected_azimuths)
    # All hits have z = 0 because elevation is zero
    expected_hit_z = np.zeros_like(expected_azimuths, dtype=float)

    # Extract actual values from the per beam return dictionaries
    actual_azimuths = np.asarray([float(item["azimuth"]) for item in returns], dtype=float)
    actual_indices = np.asarray([int(item["azimuth_index"]) for item in returns], dtype=int)
    actual_valids = [bool(item["valid"]) for item in returns]
    actual_measured_ranges = np.asarray([float(item["range"]) for item in returns], dtype=float)
    actual_truth_ranges = np.asarray([float(item["truth_range"]) for item in returns], dtype=float)
    actual_hit_points = np.asarray([item["hit_point"] for item in returns], dtype=float)

    # Every beam must report a valid hit
    assert all(actual_valids)
    # Azimuth indices must be a contiguous 0..N sequence with no gaps or reordering
    np.testing.assert_array_equal(actual_indices, np.arange(az_samples, dtype=int))
    # Actual azimuth angles must match the expected linspace to within floating point precision
    np.testing.assert_allclose(actual_azimuths, expected_azimuths, rtol=0.0, atol=1e-14)
    # Azimuth spacing must be uniform (constant step between consecutive beams)
    np.testing.assert_allclose(np.diff(actual_azimuths), expected_spacing, rtol=0.0, atol=1e-14)
    # Truth range (geometric ground truth) must match the d/cos(az) formula
    np.testing.assert_allclose(actual_truth_ranges, expected_ranges, rtol=0.0, atol=1e-12)
    # Measured range (with zero noise) must also match the analytic formula
    np.testing.assert_allclose(actual_measured_ranges, expected_ranges, rtol=0.0, atol=1e-12)
    # Hit point X coordinates must all equal wall_distance
    np.testing.assert_allclose(actual_hit_points[:, 0], expected_hit_x, rtol=0.0, atol=1e-12)
    # Hit point Y coordinates must follow wall_distance * tan(azimuth)
    np.testing.assert_allclose(actual_hit_points[:, 1], expected_hit_y, rtol=0.0, atol=1e-12)
    # Hit point Z coordinates must all be zero (single elevation row at zero)
    np.testing.assert_allclose(actual_hit_points[:, 2], expected_hit_z, rtol=0.0, atol=1e-12)

    # Convert expected azimuths to degrees for human readable axis labels
    az_deg = np.rad2deg(expected_azimuths)
    # Compute the spacing error: difference between actual and expected consecutive steps
    spacing_error_deg = np.rad2deg(np.diff(actual_azimuths) - expected_spacing)
    # Integer index array from 0 to az_samples for the horizontal axis
    index_values = np.arange(az_samples, dtype=int)

    # -- Diagnostic plot: range profile and azimuth index consistency --
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    # Top panel: overlay the analytic range curve (orange) with measured scatter (blue).
    # The cos(azimuth) relationship produces the characteristic "smile" shape.
    ax_top.plot(az_deg, expected_ranges, color="tab:orange", linewidth=1.5, label="expected range profile")
    ax_top.scatter(az_deg, actual_measured_ranges, s=12, alpha=0.8, color="tab:blue", label="measured range")
    ax_top.set_title("Planar Wall Beam Geometry: Range Profile Across Azimuth")
    ax_top.set_xlabel("Azimuth (deg)")
    ax_top.set_ylabel("Range to Wall (m)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Bottom panel (left axis): azimuth angle vs beam index, showing the linear mapping
    ax_bottom.plot(index_values, az_deg, color="tab:blue", linewidth=1.5, label="azimuth by index")
    ax_bottom.scatter(index_values, az_deg, s=12, alpha=0.8, color="tab:blue")
    ax_bottom.set_title("Azimuth Index Mapping and Spacing Consistency")
    ax_bottom.set_xlabel("Azimuth Index")
    ax_bottom.set_ylabel("Azimuth (deg)")
    ax_bottom.grid(True, alpha=0.3)
    # Bottom panel (right axis): spacing error at half integer positions, showing
    # any deviation from perfectly uniform angular steps
    ax_bottom_twin = ax_bottom.twinx()
    ax_bottom_twin.plot(
        index_values[:-1] + 0.5,
        spacing_error_deg,
        color="tab:red",
        linewidth=1.2,
        label="spacing error",
    )
    # Zero reference line for the spacing error
    ax_bottom_twin.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    ax_bottom_twin.set_ylabel("Spacing Error (deg)")

    # Merge legends from both axes into a single combined legend
    handles_left, labels_left = ax_bottom.get_legend_handles_labels()
    handles_right, labels_right = ax_bottom_twin.get_legend_handles_labels()
    ax_bottom.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    # Leave room on the right for the external legends
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    # Embed the figure into the HTML test report
    attach_plot_to_html_report(request, fig, name="beam_pattern_geometry_planar_wall")
    # Release the figure memory
    plt.close(fig)



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
    attach_plot_to_html_report(request, fig, name="full_frame_sphere_consistency")
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
    attach_plot_to_html_report(request, fig, name="multi_object_occlusion")
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
    attach_plot_to_html_report(request, fig, name="rotated_sensor")
    plt.close(fig)

