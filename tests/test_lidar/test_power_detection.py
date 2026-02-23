"""
Received power, atmospheric extinction, incidence angle, retro-reflectivity,
SNR detection threshold, and empty scene tests.

Validates the physics-based detection chain: 1/r^4 power law, Beer-Lambert
atmospheric attenuation, Lambertian cosine dependence, retro-reflective
boost, SNR-based detection probability, and empty scene edge case.
"""

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.physics import Material, Plane, Scene, Sphere
from sensors.lidar import Lidar
from .helpers import attach_plot_to_html_report


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
    attach_plot_to_html_report(request, fig, name="received_power_vs_range")
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
    attach_plot_to_html_report(request, fig, name="atmosphere_extinction")
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
    attach_plot_to_html_report(request, fig, name="incidence_angle_dependence")
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
    attach_plot_to_html_report(request, fig, name="retro_reflectivity")
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
    attach_plot_to_html_report(request, fig, name="snr_detection_threshold")
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
    attach_plot_to_html_report(request, fig, name="empty_scene")
    plt.close(fig)
