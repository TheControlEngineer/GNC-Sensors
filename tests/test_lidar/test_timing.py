"""
Timing, sampling rate, latency, jitter, and position interpolation tests.

Validates the asynchronous measurement pipeline: sampling rate clock,
latency delay, timing jitter injection, time reversal handling, and
intra-period position interpolation.
"""

import numpy as np
import pytest

from sensors.Config import LidarConfig
from sensors.lidar import Lidar
from .helpers import attach_plot_to_html_report, chi2_variance_bounds


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
    attach_plot_to_html_report(request, fig, name="sampling_rate")
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
    attach_plot_to_html_report(request, fig, name="latency_pipeline")
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

    lo_ratio, hi_ratio = chi2_variance_bounds(n_a, confidence=0.99)
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

    lo_ratio_b, hi_ratio_b = chi2_variance_bounds(n_b, confidence=0.99)
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
    attach_plot_to_html_report(request, fig, name="time_jitter")
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
    attach_plot_to_html_report(request, fig, name="time_reversal_handling")
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
    attach_plot_to_html_report(request, fig, name="position_interpolation")
    plt.close(fig)

