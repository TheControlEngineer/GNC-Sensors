import numpy as np

from sensors.Config import LidarConfig


class OLA_LELT_Config(LidarConfig):
    """
    OSIRIS Rex Laser Altimeter, Low Energy Laser Transmitter configuration.

    Source:
        Daly et al. (2017), Space Science Reviews, 212(1-2), 899-924
        NSSDCA catalog, experiment ID 2016-055A-02
    """

    range_min = 500.0
    range_max = 1200.0
    fov = np.deg2rad(60.0)
    sampling_rate = 10_000.0

    range_accuracy = 0.06
    noise_floor_std = 0.0
    noise_range_coeff = 0.0

    bias_init_std = 0.011
    bias_rw_std = 0.0
    bias_drift_rate = 0.0
    scale_factor_ppm = 0.0
    scale_error_std_ppm = 0.0

    latency = 0.0
    sample_time_jitter_std = 0.0
    latency_jitter_std = 0.0

    dropout_prob = 0.005
    dropout_range_coeff = 0.0
    outlier_prob = 0.0
    outlier_std = 0.0
    outlier_bias = 0.0

    quantization_step = 0.0
    saturate_output = True

    beam_divergence = 0.1e-3
    pulse_energy = 10e-6
    receiver_aperture_area = 0.0044
    atmosphere_extinction_coeff = 0.0
    min_detectable_power = 1e-12
    intensity_noise_std = 0.0

    scan_azimuth_min = np.deg2rad(-15.0)
    scan_azimuth_max = np.deg2rad(15.0)
    scan_azimuth_samples = 181
    scan_elevation_angles = (0.0,)

    include_metadata = True
    random_seed = None
