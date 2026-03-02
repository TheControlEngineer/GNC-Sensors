"""
Cross validation test against OLA flight data from PDS.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from sensors.lidar import Lidar
from .extract_ola_statistics import bin_precision_by_range, compute_windowed_precision, infer_range_meters
from .helpers import attach_plot_to_html_report, binomial_ci, chi2_variance_bounds
from .ola_config import OLA_LELT_Config
from .pds4_parser import load_ola_dataset


def _find_data_pair(data_dir):
    data_dir = Path(data_dir)
    xml_files = sorted(data_dir.rglob("*.xml"))
    if not xml_files:
        return None, None

    for xml in xml_files:
        dat = xml.with_suffix(".dat")
        if dat.exists():
            return xml, dat

    dat_files = sorted(data_dir.rglob("*.dat"))
    if dat_files:
        return xml_files[0], dat_files[0]
    return xml_files[0], None


def _pick_column_name(dtype_names, candidates):
    if dtype_names is None:
        return None
    lookup = {str(n).lower(): str(n) for n in dtype_names}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def _as_numeric(values):
    arr = np.asarray(values)
    if arr.dtype.kind in ("i", "u", "f"):
        return arr.astype(float)
    parsed = np.full(arr.shape, np.nan, dtype=float)
    for i, v in enumerate(arr):
        try:
            parsed[i] = float(str(v).strip())
        except Exception:
            parsed[i] = np.nan
    return parsed


def _load_flight_data(data_dir, max_records):
    xml_path, dat_path = _find_data_pair(data_dir)
    if xml_path is None:
        raise FileNotFoundError("No OLA xml label found.")

    arr = load_ola_dataset(xml_path, dat_path=dat_path, max_records=max_records)
    names = arr.dtype.names
    if names is None:
        raise ValueError("Loaded dataset has no named columns.")

    range_col = _pick_column_name(
        names,
        [
            "range_m",
            "range",
            "range_mm",
            "calibrated_range",
            "calibrated_range_m",
        ],
    )
    if range_col is None:
        raise ValueError("Could not find a range column in flight data.")

    flag_col = _pick_column_name(
        names,
        [
            "flag",
            "flag_status",
            "quality_flag",
            "data_quality_flag",
            "valid_flag",
        ],
    )
    time_col = _pick_column_name(
        names,
        [
            "sclk",
            "timestamp",
            "utc",
            "time",
            "ephemeris_time",
        ],
    )

    ranges_all = infer_range_meters(_as_numeric(arr[range_col]))
    finite_mask = np.isfinite(ranges_all)
    ranges_all = ranges_all[finite_mask]

    if time_col is None:
        time_all = np.arange(ranges_all.size, dtype=float)
    else:
        raw_time = _as_numeric(arr[time_col])[finite_mask]
        if np.all(~np.isfinite(raw_time)):
            time_all = np.arange(ranges_all.size, dtype=float)
        else:
            time_all = raw_time

    if flag_col is None:
        valid_mask = np.ones(ranges_all.shape, dtype=bool)
    else:
        raw_flag = _as_numeric(arr[flag_col])[finite_mask]
        valid_mask = np.isfinite(raw_flag) & (raw_flag == 0.0)

    ranges_valid = ranges_all[valid_mask]
    time_valid = time_all[valid_mask]

    return {
        "xml_path": xml_path,
        "dat_path": dat_path,
        "ranges_all": ranges_all,
        "time_all": time_all,
        "valid_mask": valid_mask,
        "ranges_valid": ranges_valid,
        "time_valid": time_valid,
        "has_flag": flag_col is not None,
    }


@pytest.mark.test_meta(
    description=(
        "Cross validation of the OLA LELT configured model against PDS flight data. "
        "Compares precision, distribution shape, and detection fraction across range bins."
    ),
    goal=(
        "Verify model statistics are consistent with real OLA measurements at matched "
        "operating ranges."
    ),
    passing_criteria=(
        "Model precision is within chi squared 99 percent confidence of flight precision per range bin. "
        "Model detection fraction is within binomial 99 percent confidence when flight quality flags are present."
    ),
)
def test_ola_flight_data_crossvalidation(request):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env_dir = os.getenv("OLA_DATA_DIR", "").strip()
    default_dir = Path(__file__).resolve().parent / "data" / "ola_orbit_b"
    data_dir = Path(env_dir) if env_dir else default_dir
    if not data_dir.exists():
        pytest.skip(f"OLA data directory not found: {data_dir}")

    max_records = int(os.getenv("OLA_MAX_RECORDS", "250000"))
    try:
        flight = _load_flight_data(data_dir, max_records=max_records)
    except Exception as exc:
        pytest.skip(f"OLA data could not be loaded: {exc}")

    if flight["ranges_valid"].size < 1000:
        pytest.skip("Not enough valid OLA rows for cross validation.")

    range_min = float(OLA_LELT_Config.range_min)
    range_max = float(OLA_LELT_Config.range_max)

    valid_in_envelope = (flight["ranges_valid"] >= range_min) & (flight["ranges_valid"] <= range_max)
    if int(np.sum(valid_in_envelope)) < 1000:
        pytest.skip(
            f"Dataset has insufficient valid rows in {range_min:.0f} to {range_max:.0f} m envelope."
        )

    ranges_valid = flight["ranges_valid"][valid_in_envelope]
    time_valid = flight["time_valid"][valid_in_envelope]

    window_means, window_stds = compute_windowed_precision(
        ranges_valid,
        timestamps=time_valid,
        window_size=50,
    )
    if window_means.size < 20:
        pytest.skip("Not enough flight windows for precision statistics.")

    bin_edges = np.arange(range_min, range_max + 100.0, 100.0, dtype=float)
    centers, flight_precision, flight_counts = bin_precision_by_range(window_means, window_stds, bin_edges)
    if int(np.sum(np.isfinite(flight_precision) & (flight_counts >= 5))) < 2:
        pytest.skip("Not enough populated range bins for precision comparison.")

    flight_det = np.full(centers.shape, np.nan, dtype=float)
    flight_det_n = np.zeros(centers.shape, dtype=int)
    flight_det_k = np.zeros(centers.shape, dtype=int)
    all_in_envelope = (flight["ranges_all"] >= range_min) & (flight["ranges_all"] <= range_max)
    ranges_all_env = flight["ranges_all"][all_in_envelope]
    valid_mask_env = flight["valid_mask"][all_in_envelope]
    idx_all = np.digitize(ranges_all_env, bin_edges)
    for i in range(centers.size):
        m = idx_all == (i + 1)
        n = int(np.sum(m))
        if n > 0:
            k = int(np.sum(valid_mask_env[m]))
            flight_det[i] = k / n
            flight_det_n[i] = n
            flight_det_k[i] = k

    class OLA_Crossval_Config(OLA_LELT_Config):
        random_seed = 42
        dropout_prob = 0.005
        outlier_prob = 0.0
        quantization_step = 0.0

    model = Lidar(config=OLA_Crossval_Config)
    N_MODEL = int(os.getenv("OLA_MODEL_SAMPLES", "5000"))

    model_precision = np.full(centers.shape, np.nan, dtype=float)
    model_mean = np.full(centers.shape, np.nan, dtype=float)
    model_det = np.full(centers.shape, np.nan, dtype=float)
    model_n_valid = np.zeros(centers.shape, dtype=int)
    model_errors_by_bin = {}

    for i, r in enumerate(centers):
        errs = np.empty(N_MODEL, dtype=float)
        valid = 0
        for j in range(N_MODEL):
            m = model.measure([float(r), 0.0, 0.0], add_noise=True, current_time=None)
            if m.get("valid", False):
                errs[valid] = float(m["range"]) - float(r)
                valid += 1
        model_det[i] = valid / float(N_MODEL)
        if valid >= 2:
            e = errs[:valid]
            model_errors_by_bin[i] = e.copy()
            model_n_valid[i] = valid
            model_precision[i] = float(np.std(e, ddof=1))
            model_mean[i] = float(np.mean(e))

    for i in range(centers.size):
        fp = flight_precision[i]
        mp = model_precision[i]
        nv = int(model_n_valid[i])
        if not np.isfinite(fp) or not np.isfinite(mp) or nv < 10:
            continue
        lo, hi = chi2_variance_bounds(nv, confidence=0.99)
        ratio = (mp / fp) ** 2
        assert lo <= ratio <= hi, (
            f"bin {centers[i]:.1f} m precision ratio {ratio:.4f} outside 99% CI [{lo:.4f}, {hi:.4f}]"
        )

        mean_tol = 3.0 * mp / np.sqrt(nv)
        assert abs(model_mean[i]) <= mean_tol, (
            f"bin {centers[i]:.1f} m model mean error {model_mean[i]:.6f} exceeds {mean_tol:.6f}"
        )

    if flight["has_flag"]:
        for i in range(centers.size):
            n = int(flight_det_n[i])
            if n < 50 or not np.isfinite(model_det[i]):
                continue
            k = int(flight_det_k[i])
            _, ci_lo, ci_hi = binomial_ci(k, n, confidence=0.99)
            assert ci_lo <= model_det[i] <= ci_hi, (
                f"bin {centers[i]:.1f} m model detection {model_det[i]:.4f} outside 99% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
            )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    ax1, ax2, ax3 = axes

    ax1.scatter(centers, flight_precision, color="tab:blue", s=45, label="flight precision")
    ax1.scatter(centers, model_precision, color="tab:red", s=45, marker="s", label="model precision")
    ax1.axhline(0.06, color="tab:orange", linestyle="--", linewidth=1.8, label="published sigma 0.06 m")
    ax1.set_title("Precision vs range bin")
    ax1.set_xlabel("Range bin center [m]")
    ax1.set_ylabel("Window precision [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ref_idx = int(np.argmin(np.abs(centers - 700.0)))
    m_ref = (window_means >= bin_edges[ref_idx]) & (window_means < bin_edges[ref_idx + 1])
    flight_ref = window_stds[m_ref]
    model_ref = model_errors_by_bin.get(ref_idx, np.asarray([], dtype=float))
    if flight_ref.size >= 5 and model_ref.size >= 10:
        ax2.hist(flight_ref, bins=40, density=True, color="tab:blue", alpha=0.6, label="flight window std")
        ax2.hist(np.abs(model_ref), bins=40, density=True, color="tab:red", alpha=0.45, label="model abs error")
        ax2.set_title("Distribution near 700 m bin")
        ax2.set_xlabel("Metric value [m]")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "Insufficient samples for 700 m histogram", ha="center", va="center")
        ax2.set_axis_off()

    ax3.scatter(centers, model_det, color="tab:red", s=45, marker="s", label="model detection fraction")
    if np.any(np.isfinite(flight_det)):
        ax3.scatter(centers, flight_det, color="tab:blue", s=45, label="flight detection fraction")
    ax3.set_title("Detection fraction vs range bin")
    ax3.set_xlabel("Range bin center [m]")
    ax3.set_ylabel("Detection fraction")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    fig.tight_layout()
    attach_plot_to_html_report(request, fig, name="ola_flight_crossvalidation")
    plt.close(fig)
