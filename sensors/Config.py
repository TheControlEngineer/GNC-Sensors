import numpy as np


class LidarConfig:
    # Core instrument envelope
    range_min = 1.0  # [m]
    range_max = 1000.0  # [m]
    fov = np.deg2rad(30.0)  # [rad]
    sampling_rate = 5.0  # [Hz]
    latency = 0.08  # [s]

    # Legacy Gaussian noise term (kept for backward compatibility)
    range_accuracy = 0.1  # [m], interpreted as 1 sigma in the truth model

    # Truth model toggles
    include_metadata = True  # expose truth model terms in each output
    random_seed = None  # optional deterministic seed

    # Range error model:
    # sigma_total(range) = sqrt(range_accuracy^2 + noise_floor_std^2 + (noise_range_coeff * range)^2)
    noise_floor_std = 0.0  # [m]
    noise_range_coeff = 0.0  # [m/m]

    # Bias and scale model:
    # measured = true * (1 + scale_factor) + bias + noise + outlier
    bias_init_std = 0.0  # [m]
    bias_rw_std = 0.0  # [m/sqrt(s)] random walk diffusion
    bias_drift_rate = 0.0  # [m/s] deterministic drift
    scale_factor_ppm = 0.0  # [ppm], fixed deterministic offset
    scale_error_std_ppm = 0.0  # [ppm], sampled once at sensor init

    # Output discretization/saturation
    quantization_step = 0.0  # [m], 0 disables quantization
    saturate_output = True  # clip to [range_min, range_max] instead of invalidating

    # Data quality model
    dropout_prob = 0.005  # baseline dropout probability per valid return
    dropout_range_coeff = 0.0  # extra dropout term vs normalized range [0..1]
    outlier_prob = 0.0  # probability of gross error on valid return
    outlier_std = 0.0  # [m], Gaussian outlier spread
    outlier_bias = 0.0  # [m], deterministic outlier offset

    # Timing model
    sample_time_jitter_std = 0.0  # [s], sampling epoch jitter
    latency_jitter_std = 0.0  # [s], latency jitter

    # Scene scan pattern model
    scan_azimuth_min = None  # [rad], if None uses -fov/2
    scan_azimuth_max = None  # [rad], if None uses +fov/2
    scan_azimuth_samples = 181  # number of azimuth beams per elevation channel
    scan_elevation_angles = (0.0,)  # [rad], tuple/list of channel elevations

    # LiDAR return physics
    beam_divergence = np.deg2rad(0.2)  # [rad], full angle divergence
    pulse_energy = 1.0  # [arb], emitted pulse energy scaling
    receiver_aperture_area = 1.0  # [arb], effective receiver aperture
    atmosphere_extinction_coeff = 0.0  # [1/m], Beer Lambert attenuation coefficient
    min_detectable_power = 1e-6  # [arb], detection threshold scale
    intensity_noise_std = 0.0  # [arb], additive intensity noise std
