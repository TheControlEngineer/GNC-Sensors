"""
Validation test for OLA LELT specification reproduction.
"""

import numpy as np
import pytest

from sensors.lidar import Lidar
from sensors.physics import Material, Scene, Sphere
from .helpers import attach_plot_to_html_report, chi2_variance_bounds
from .ola_config import OLA_LELT_Config


def _gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


@pytest.mark.test_meta(
    description=(
        "OLA LELT spec sheet validation. Configure the model with published "
        "OSIRIS Rex OLA LELT parameters and verify range noise, detection rate, "
        "and gating boundaries match the expected instrument envelope."
    ),
    goal=(
        "Confirm the LiDAR model can reproduce key published OLA LELT behavior "
        "with the parameterized truth model."
    ),
    passing_criteria=(
        "Range noise sigma is consistent with 0.06 m at 99 percent confidence. "
        "Detection rate is above 95 percent at 700 m on a dark target. "
        "Range gating rejects values outside 500 to 1200 m."
    ),
)
def test_ola_lelt_specsheet(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 10_000
    TEST_RANGES = np.array([550.0, 700.0, 900.0, 1100.0], dtype=float)
    PUBLISHED_SIGMA = 0.06

    class OLA_NoDropout(OLA_LELT_Config):
        dropout_prob = 0.0
        random_seed = 42

    sensor_noise = Lidar(config=OLA_NoDropout)
    expected_bias = float(sensor_noise._bias_state)

    std_emp = []
    n_per_range = []
    means_emp = []
    error_700 = None

    for true_range in TEST_RANGES:
        errors = np.empty(N, dtype=float)
        for i in range(N):
            m = sensor_noise.measure([true_range, 0.0, 0.0], add_noise=True, current_time=None)
            assert m["valid"] is True
            errors[i] = float(m["range"]) - true_range

        sample_mean = float(np.mean(errors))
        sample_std = float(np.std(errors, ddof=1))
        n_valid = int(errors.size)

        means_emp.append(sample_mean)
        std_emp.append(sample_std)
        n_per_range.append(n_valid)

        mean_tol = 3.0 * PUBLISHED_SIGMA / np.sqrt(n_valid)
        assert abs(sample_mean - expected_bias) <= mean_tol, (
            f"range {true_range:.1f} m mean {sample_mean:.6f} not within tolerance "
            f"around expected bias {expected_bias:.6f}"
        )

        lo, hi = chi2_variance_bounds(n_valid, confidence=0.99)
        ratio = (sample_std / PUBLISHED_SIGMA) ** 2
        assert lo <= ratio <= hi, (
            f"range {true_range:.1f} m variance ratio {ratio:.4f} outside 99% CI [{lo:.4f}, {hi:.4f}]"
        )

        if np.isclose(true_range, 700.0):
            error_700 = errors.copy()

    assert error_700 is not None

    class OLA_Detection(OLA_LELT_Config):
        scan_azimuth_samples = 3
        scan_elevation_angles = (0.0,)
        random_seed = 43
        dropout_prob = 0.0
        min_detectable_power = 1e-15

    sensor_det = Lidar(config=OLA_Detection)
    scene = Scene(
        [
            Sphere(
                center=[700.0, 0.0, 0.0],
                radius=5.0,
                material=Material(reflectivity=0.044, retro_reflectivity=0.0, name="bennu_like"),
                object_id="bennu_like",
            )
        ]
    )

    detections = 0
    for i in range(N):
        frame = sensor_det.simulate_scene_frame(
            scene=scene,
            sensor_position=[0.0, 0.0, 0.0],
            timestamp=float(i),
            add_noise=True,
        )
        if frame.get("num_valid", 0) > 0:
            detections += 1
    det_rate = detections / N
    assert det_rate > 0.95, f"detection rate at 700 m is too low: {det_rate:.4f}"

    class OLA_Boundary(OLA_LELT_Config):
        dropout_prob = 0.0
        random_seed = 44

    sensor_gate = Lidar(config=OLA_Boundary)
    below = sensor_gate.measure([499.0, 0.0, 0.0], add_noise=False, current_time=None)
    above = sensor_gate.measure([1201.0, 0.0, 0.0], add_noise=False, current_time=None)
    assert below.get("valid", False) is False and below.get("reason") == "out_of_range"
    assert above.get("valid", False) is False and above.get("reason") == "out_of_range"

    std_emp = np.asarray(std_emp, dtype=float)
    n_per_range = np.asarray(n_per_range, dtype=int)
    means_emp = np.asarray(means_emp, dtype=float)
    n_min = int(np.min(n_per_range))
    lo_ratio, hi_ratio = chi2_variance_bounds(max(n_min, 2), confidence=0.99)
    std_lo = PUBLISHED_SIGMA * np.sqrt(lo_ratio)
    std_hi = PUBLISHED_SIGMA * np.sqrt(hi_ratio)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 8))

    bins = np.linspace(expected_bias - 0.35, expected_bias + 0.35, 90)
    ax_top.hist(error_700, bins=bins, density=True, color="tab:blue", alpha=0.7, label="empirical at 700 m")
    x_pdf = np.linspace(bins[0], bins[-1], 400)
    ax_top.plot(x_pdf, _gaussian_pdf(x_pdf, expected_bias, PUBLISHED_SIGMA), color="tab:orange", linewidth=2.0, label="N(bias, 0.06^2)")
    ax_top.set_title("OLA LELT range error distribution at 700 m")
    ax_top.set_xlabel("Range error [m]")
    ax_top.set_ylabel("Probability density")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best")

    ax_bottom.scatter(TEST_RANGES, std_emp, s=60, color="tab:blue", zorder=3, label="empirical sigma")
    ax_bottom.plot(TEST_RANGES, np.full(TEST_RANGES.shape, PUBLISHED_SIGMA), color="tab:orange", linewidth=2.0, linestyle="--", label="published sigma = 0.06 m")
    ax_bottom.fill_between(TEST_RANGES, std_lo, std_hi, color="tab:orange", alpha=0.2, label="99% CI band")
    ax_bottom.set_title("Empirical sigma vs range under OLA LELT configuration")
    ax_bottom.set_xlabel("True range [m]")
    ax_bottom.set_ylabel("Sigma [m]")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="best")

    fig.tight_layout()
    attach_plot_to_html_report(request, fig, name="ola_lelt_specsheet")
    plt.close(fig)
