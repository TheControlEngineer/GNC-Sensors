"""
LiDAR Sensor Truth Model

This module implements a high-fidelity LiDAR sensor simulator covering
range measurement with configurable noise, bias, scale factor, and outlier
injection, field-of-view gating, dropout, output quantization and saturation,
asynchronous sampling with latency, jitter, and platform-motion interpolation,
full 3D scan-pattern generation with per-beam ray casting into a Scene, and
radiometric detection probability based on material reflectance, beam divergence,
atmospheric extinction, and receiver SNR threshold.

The primary entry points are:
    Lidar.measure()       Single-beam range measurement (relative position input).
    Lidar.scan_scene()    Full scan frame against a Scene of geometry primitives.
"""

import numpy as np

from .Config import LidarConfig
from .math_utils import _as_rotation_matrix, _as_vector3, _interpolate_position, _normalize, _pop_latest_ready, eps


class Lidar:
    """
    Configurable LiDAR sensor truth model.

    Built from a LidarConfig (or any object exposing the same attributes).
    Maintains internal RNG state, bias random-walk state, and asynchronous
    measurement queues for both single-beam and scene-scan modes.
    """
    def __init__(self, config=LidarConfig, boresight=None):
        """
        Initialize the LiDAR model from a configuration object.

        :param config:    Class or instance with LidarConfig-compatible attributes.
        :param boresight: Optional 3D unit vector for the sensor boresight direction
                          in the body frame. Defaults to +X = [1, 0, 0].
        """
        # ---------- core instrument envelope ----------
        self.range_min = float(config.range_min)            # [m] minimum detectable range
        self.range_max = float(config.range_max)            # [m] maximum detectable range
        self.range_accuracy = float(config.range_accuracy)  # [m] 1-sigma Gaussian range noise
        self.fov = float(config.fov)                        # [rad] full-cone field of view
        if self.range_min < 0.0:
            raise ValueError("range_min must be >= 0.")
        if self.range_max <= self.range_min:
            raise ValueError("range_max must be greater than range_min.")
        if self.range_accuracy < 0.0:
            raise ValueError("range_accuracy must be >= 0.")
        if not (0.0 < self.fov <= np.pi):
            raise ValueError("fov must be in the range (0, pi].")
        self._cos_half_fov = float(np.cos(self.fov * 0.5))  # precomputed cosine for fast FoV gating

        # ---------- timing model ----------
        self.sampling_rate = float(config.sampling_rate)     # [Hz] nominal sample rate
        if self.sampling_rate < 0.0:
            raise ValueError("sampling_rate must be >= 0.")
        # Period between consecutive samples; infinite when rate is zero (single-shot mode).
        self.sampling_period = float("inf") if self.sampling_rate <= eps else 1.0 / self.sampling_rate
        self.latency = max(float(getattr(config, "latency", 0.0)), 0.0)  # [s] output latency

        # ---------- general flags ----------
        self.include_metadata = bool(getattr(config, "include_metadata", True))  # attach truth terms to output
        self.random_seed = getattr(config, "random_seed", None)                  # optional deterministic seed
        self.rng = np.random.default_rng(self.random_seed)                       # numpy RNG instance

        # ---------- range noise model ----------
        # sigma_total(r) = sqrt(range_accuracy^2 + noise_floor_std^2 + (noise_range_coeff * r)^2)
        self.noise_floor_std = max(float(getattr(config, "noise_floor_std", 0.0)), 0.0)    # [m]
        self.noise_range_coeff = max(float(getattr(config, "noise_range_coeff", 0.0)), 0.0) # [m/m]

        # ---------- bias and scale model ----------
        # measured = true * (1 + scale_factor) + bias + noise + outlier
        self.bias_init_std = max(float(getattr(config, "bias_init_std", 0.0)), 0.0)        # [m] initial bias 1-sigma
        self.bias_rw_std = max(float(getattr(config, "bias_rw_std", 0.0)), 0.0)            # [m/sqrt(s)] random walk diffusion
        self.bias_drift_rate = float(getattr(config, "bias_drift_rate", 0.0))               # [m/s] deterministic drift
        self.scale_factor_ppm = float(getattr(config, "scale_factor_ppm", 0.0))             # [ppm] fixed scale offset
        self.scale_error_std_ppm = max(float(getattr(config, "scale_error_std_ppm", 0.0)), 0.0)  # [ppm] random scale 1-sigma

        # ---------- output discretization / saturation ----------
        self.quantization_step = max(float(getattr(config, "quantization_step", 0.0)), 0.0)  # [m] 0 = disabled
        self.saturate_output = bool(getattr(config, "saturate_output", True))                # clip vs. invalidate

        # ---------- data quality model ----------
        self.dropout_prob = float(np.clip(getattr(config, "dropout_prob", 0.0), 0.0, 1.0))   # baseline dropout probability
        self.dropout_range_coeff = float(getattr(config, "dropout_range_coeff", 0.0))         # extra dropout vs. normalized range
        self.outlier_prob = float(np.clip(getattr(config, "outlier_prob", 0.0), 0.0, 1.0))    # gross error probability
        self.outlier_std = max(float(getattr(config, "outlier_std", 0.0)), 0.0)               # [m] outlier spread
        self.outlier_bias = float(getattr(config, "outlier_bias", 0.0))                       # [m] deterministic outlier offset

        # ---------- timing jitter ----------
        self.sample_time_jitter_std = max(float(getattr(config, "sample_time_jitter_std", 0.0)), 0.0)  # [s]
        self.latency_jitter_std = max(float(getattr(config, "latency_jitter_std", 0.0)), 0.0)          # [s]

        # ---------- scan pattern model ----------
        scan_azimuth_min = getattr(config, "scan_azimuth_min", None)
        scan_azimuth_max = getattr(config, "scan_azimuth_max", None)
        self.scan_azimuth_min = -0.5 * self.fov if scan_azimuth_min is None else float(scan_azimuth_min)  # [rad]
        self.scan_azimuth_max = 0.5 * self.fov if scan_azimuth_max is None else float(scan_azimuth_max)    # [rad]
        if self.scan_azimuth_max <= self.scan_azimuth_min:
            raise ValueError("scan_azimuth_max must be > scan_azimuth_min.")
        self.scan_azimuth_samples = int(getattr(config, "scan_azimuth_samples", 181))  # number of azimuth beams
        if self.scan_azimuth_samples < 1:
            raise ValueError("scan_azimuth_samples must be >= 1.")
        raw_elevations = getattr(config, "scan_elevation_angles", (0.0,))
        if np.isscalar(raw_elevations):
            raw_elevations = (float(raw_elevations),)
        self.scan_elevation_angles = np.asarray(raw_elevations, dtype=float).reshape(-1)  # [rad] channel elevations
        if self.scan_elevation_angles.size < 1:
            raise ValueError("scan_elevation_angles must contain at least one angle.")

        # ---------- LiDAR return physics ----------
        self.beam_divergence = max(float(getattr(config, "beam_divergence", np.deg2rad(0.2))), 0.0)  # [rad] full angle
        self.pulse_energy = max(float(getattr(config, "pulse_energy", 1.0)), 0.0)                     # [arb] emitted energy
        self.receiver_aperture_area = max(float(getattr(config, "receiver_aperture_area", 1.0)), eps)  # [arb] effective aperture
        self.atmosphere_extinction_coeff = max(float(getattr(config, "atmosphere_extinction_coeff", 0.0)), 0.0)  # [1/m] Beer-Lambert
        self.min_detectable_power = max(float(getattr(config, "min_detectable_power", 1e-6)), eps)     # [arb] detection threshold
        self.intensity_noise_std = max(float(getattr(config, "intensity_noise_std", 0.0)), 0.0)        # [arb] additive noise

        # ---------- boresight direction ----------
        default_boresight = np.array([1.0, 0.0, 0.0], dtype=float)  # default: +x axis
        raw_boresight = default_boresight if boresight is None else _as_vector3(boresight, "boresight")
        self.boresight = _normalize(raw_boresight, fallback=default_boresight)  # [unit] sensor boresight

        # ---------- single beam pipeline state ----------
        self.next_sample_time = 0.0              # [s] next scheduled sample epoch
        self.pending_measurements = []           # queue of (available_time, measurement) tuples
        self.last_input_time = None              # [s] timestamp of previous measure() call
        self.last_rel_position = None            # [m] relative position from previous call
        self._last_available_time = -float("inf")  # [s] monotonicity guard for available times

        # ---------- scene scan pipeline state ----------
        self.next_scene_sample_time = 0.0        # [s] next scheduled scene scan epoch
        self.pending_scene_frames = []           # queue of (available_time, frame) tuples
        self.last_scene_input_time = None        # [s] timestamp of previous scan_scene() call
        self._last_scene_available_time = -float("inf")  # [s] monotonicity guard

        # ---------- bias / scale internal state ----------
        # Initial bias is sampled once from N(0, bias_init_std).
        self._bias_state = float(self.rng.normal(0.0, self.bias_init_std)) if self.bias_init_std > 0.0 else 0.0
        # Total scale factor = deterministic ppm + random ppm (sampled once at init).
        scale_random = float(self.rng.normal(0.0, self.scale_error_std_ppm * 1e-6)) if self.scale_error_std_ppm > 0.0 else 0.0
        self._scale_factor = self.scale_factor_ppm * 1e-6 + scale_random  # [dimensionless]
        self._last_bias_update_time = None  # [s] last time bias RW was stepped
        self._beam_pattern_cache = None     # lazily built scan pattern

    def _build_invalid(self, timestamp, reason=None, stale=False, truth_range=None):
        """
        Construct a dictionary representing an invalid (no-detection) measurement.

        :param timestamp:   Sample epoch [s].
        :param reason:      Human-readable cause string (e.g. "out_of_range", "dropout").
        :param stale:       True when returning a placeholder because no new data is ready.
        :param truth_range: Ground truth range [m], attached when include_metadata is True.
        :return: Dict with valid=False and optional diagnostics.
        """
        measurement = {"valid": False, "timestamp": float(timestamp)}
        if reason is not None:
            measurement["reason"] = str(reason)
        if stale:
            measurement["stale"] = True
        if self.include_metadata and truth_range is not None:
            measurement["truth_range"] = float(truth_range)
        return measurement

    def _build_stale_frame(self, timestamp):
        """
        Construct a stale (no new data) scene-scan frame placeholder.

        Returned by scan_scene() when the latency pipeline has not yet delivered
        a new frame at the queried time.

        :param timestamp: Query epoch [s].
        :return: Dict with valid=False, stale=True, and an empty returns list.
        """
        return {
            "valid": False,
            "type": "frame",
            "timestamp": float(timestamp),
            "stale": True,
            "returns": [],
        }

    def _update_bias_state(self, sample_time, stochastic=True):
        """
        Advance the bias random walk and deterministic drift to a new sample epoch.

        Bias model: bias(t) = bias(t_prev) + drift_rate * dt + N(0, rw_std * sqrt(dt))

        Called once per measurement so that the bias evolves continuously.

        :param sample_time: Current sample epoch [s].
        :param stochastic:  If False, suppress the random-walk component (deterministic mode).
        """
        if self._last_bias_update_time is None:
            self._last_bias_update_time = float(sample_time)  # first call -- initialize clock
            return

        dt = max(float(sample_time) - self._last_bias_update_time, 0.0)  # [s] elapsed time
        if dt <= eps:
            return  # no time has passed; nothing to update

        self._bias_state += self.bias_drift_rate * dt  # deterministic drift [m]
        if stochastic and self.bias_rw_std > 0.0:
            # Random walk diffusion: increment ~ N(0, sigma_rw * sqrt(dt)) [m]
            self._bias_state += float(self.rng.normal(0.0, self.bias_rw_std * np.sqrt(dt)))
        self._last_bias_update_time = float(sample_time)

    def _noise_std(self, true_range):
        """
        Compute the composite 1-sigma range noise at a given true range.

        sigma_total(r) = sqrt(range_accuracy^2 + noise_floor_std^2 + (noise_range_coeff * r)^2)

        :param true_range: Ground truth range to target [m].
        :return: Total 1-sigma noise standard deviation [m].
        """
        return float(np.sqrt(
            self.range_accuracy * self.range_accuracy            # constant accuracy term
            + self.noise_floor_std * self.noise_floor_std        # additive noise floor term
            + (self.noise_range_coeff * true_range) * (self.noise_range_coeff * true_range)  # range-proportional term
        ))

    def _dropout_probability(self, true_range):
        """
        Compute the total dropout (missed detection) probability for a given range.

        p_drop(r) = clamp( dropout_prob + dropout_range_coeff * normalized_range, 0, 1 )
        where normalized_range = (r - range_min) / (range_max - range_min)

        :param true_range: Ground truth range [m].
        :return: Dropout probability [0..1].
        """
        span = max(self.range_max - self.range_min, eps)  # full range span, guarded against zero
        range_ratio = np.clip((true_range - self.range_min) / span, 0.0, 1.0)  # normalized range [0..1]
        return float(np.clip(self.dropout_prob + self.dropout_range_coeff * range_ratio, 0.0, 1.0))

    def _apply_output_limits(self, measured_range, timestamp, truth_range):
        """
        Apply quantization and saturation/clipping to a measured range value.

        1. If quantization_step > 0, round to the nearest quantization step.
        2. If saturate_output is True, clamp to [range_min, range_max].
           Otherwise, invalidate the measurement if it falls outside the range.

        :param measured_range: Noisy measured range [m].
        :param timestamp:      Sample epoch [s] (for invalid measurement construction).
        :param truth_range:    Ground truth range [m] (for metadata).
        :return: (clamped_range, None) if valid, or (None, invalid_dict) if saturated out.
        """
        if self.quantization_step > eps:
            # Round to the nearest quantization step.
            measured_range = round(measured_range / self.quantization_step) * self.quantization_step

        if self.saturate_output:
            return float(np.clip(measured_range, self.range_min, self.range_max)), None  # clamp and accept

        # In non-saturate mode, out of range values produce an invalid measurement.
        if measured_range < self.range_min or measured_range > self.range_max:
            return None, self._build_invalid(timestamp, reason="saturated", truth_range=truth_range)
        return float(measured_range), None  # within limits

    def _range_measurement_model(self, true_range, timestamp=0.0, add_noise=True):
        """
        Full range measurement truth model: bias + scale + noise + outlier.

        measured = true * (1 + scale_factor) + bias + gaussian_noise + outlier

        The result is then passed through _apply_output_limits for quantization/saturation.

        :param true_range: Ground truth range [m].
        :param timestamp:  Sample epoch [s].
        :param add_noise:  If False, suppress all stochastic terms.
        :return: Measurement dict with valid/range/timestamp and optional truth metadata.
        """
        stochastic = bool(add_noise)
        self._update_bias_state(timestamp, stochastic=stochastic)  # advance bias RW

        # Gaussian range noise.
        noise_std = self._noise_std(true_range) if stochastic else 0.0
        gaussian_noise = float(self.rng.normal(0.0, noise_std)) if noise_std > 0.0 else 0.0

        # Outlier injection (gross error): applied with probability outlier_prob.
        outlier = 0.0
        outlier_applied = False
        if stochastic and self.outlier_prob > 0.0 and float(self.rng.random()) < self.outlier_prob:
            outlier_applied = True
            outlier = self.outlier_bias  # deterministic outlier offset [m]
            if self.outlier_std > 0.0:
                outlier += float(self.rng.normal(0.0, self.outlier_std))  # random outlier spread [m]

        # Combine all error terms into the measured range.
        measured_range = true_range * (1.0 + self._scale_factor) + self._bias_state + gaussian_noise + outlier

        # Apply quantization and saturation limits.
        measured_range, invalid_measurement = self._apply_output_limits(measured_range, timestamp, truth_range=true_range)
        if invalid_measurement is not None:
            # Measurement was invalidated by saturation; attach diagnostics and return.
            if self.include_metadata:
                invalid_measurement["scale_factor"] = float(1.0 + self._scale_factor)
                invalid_measurement["bias"] = float(self._bias_state)
                invalid_measurement["noise_std"] = float(noise_std)
                invalid_measurement["outlier_applied"] = outlier_applied
            return invalid_measurement

        # Build valid measurement output dictionary.
        measurement = {"valid": True, "range": measured_range, "timestamp": float(timestamp)}
        if self.include_metadata:
            measurement["truth_range"] = float(true_range)            # [m] ground truth
            measurement["scale_factor"] = float(1.0 + self._scale_factor)  # applied scale
            measurement["bias"] = float(self._bias_state)             # current bias state [m]
            measurement["noise_std"] = float(noise_std)               # 1-sigma noise used [m]
            measurement["outlier_applied"] = outlier_applied           # was an outlier injected?
        return measurement

    def _instant_measure(self, rel_position, add_noise=True, timestamp=0.0):
        """
        Single-shot range measurement from a relative position vector.

        Checks range limits and FoV gating before forwarding to the range
        measurement model. No sampling or latency scheduling is performed.

        :param rel_position: Target position relative to the sensor [m].
        :param add_noise:    Enable stochastic error terms.
        :param timestamp:    Sample epoch [s].
        :return: Measurement dict.
        """
        rel_position = _as_vector3(rel_position, "rel_position")
        true_range = float(np.linalg.norm(rel_position))  # [m] Euclidean distance

        # Range gate: reject targets outside [range_min, range_max].
        if true_range < self.range_min or true_range > self.range_max:
            return self._build_invalid(timestamp, reason="out_of_range", truth_range=true_range)

        if true_range <= eps:
            return self._range_measurement_model(0.0, timestamp=timestamp, add_noise=add_noise)

        # Field of view gate: reject targets outside the sensor cone.
        los = rel_position / true_range  # line-of-sight unit vector
        cos_angle = float(np.clip(np.dot(los, self.boresight), -1.0, 1.0))  # cosine of off-boresight angle
        if cos_angle < self._cos_half_fov:
            return self._build_invalid(timestamp, reason="out_of_fov", truth_range=true_range)

        return self._range_measurement_model(true_range, timestamp=timestamp, add_noise=add_noise)

    def _schedule_measurement(self, sample_time, measurement, add_noise):
        """
        Push a measurement into the latency pipeline.

        Computes available_time = sample_time + latency (+ jitter) and appends
        the measurement to pending_measurements. A monotonicity guard ensures
        that available_time is strictly increasing.

        :param sample_time:  Sample epoch [s].
        :param measurement:  Completed measurement dict.
        :param add_noise:    If True, add latency jitter.
        """
        latency_value = self.latency  # [s] nominal latency
        if add_noise and self.latency_jitter_std > 0.0:
            latency_value += float(self.rng.normal(0.0, self.latency_jitter_std))  # jitter [s]
        latency_value = max(latency_value, 0.0)  # latency cannot be negative

        available_time = sample_time + latency_value  # [s] when the measurement becomes available
        # Enforce strict monotonicity so queue ordering is maintained.
        if available_time <= self._last_available_time:
            available_time = self._last_available_time + eps
        self._last_available_time = available_time
        self.pending_measurements.append((available_time, measurement))

    def measure(self, rel_position, add_noise=True, current_time=None):
        """
        Primary single-beam measurement interface.

        If current_time is None, an immediate (instantaneous) measurement is
        performed with no sampling or latency scheduling.

        If current_time is provided, the asynchronous sampling pipeline runs:
        1. Align next_sample_time to the first call's timestamp.
        2. Detect time reversals and reset the pipeline if needed.
        3. Generate all samples that fall within [last_input_time, current_time],
           interpolating the target position to each sample epoch.
        4. Apply dropout to valid measurements.
        5. Schedule each through the latency pipeline.
        6. Pop and return the latest measurement whose latency has elapsed.

        :param rel_position: Target position relative to the sensor [m].
        :param add_noise:    Enable stochastic terms (noise, dropout, jitter, outlier).
        :param current_time: Simulation clock [s]. None = instantaneous mode.
        :return: Measurement dict.
        """
        if current_time is None:
            measurement = self._instant_measure(rel_position, add_noise=add_noise, timestamp=0.0)
            if measurement.get("valid", False):
                truth_range = float(measurement.get("truth_range", np.linalg.norm(_as_vector3(rel_position, "rel_position"))))
                if bool(add_noise) and float(self.rng.random()) < self._dropout_probability(truth_range):
                    measurement = self._build_invalid(0.0, reason="dropout", truth_range=truth_range)
            return measurement

        rel_position = _as_vector3(rel_position, "rel_position")
        t = float(current_time)

        # On first call, synchronize the sample clock to the simulation clock.
        if self.last_input_time is None and not self.pending_measurements and self.next_sample_time == 0.0:
            self.next_sample_time = t

        # Detect time reversal (caller rewound the clock) -- reset the pipeline.
        if self.last_input_time is not None and t + eps < self.last_input_time:
            self.pending_measurements.clear()
            self.next_sample_time = t
            self.last_input_time = None
            self.last_rel_position = None
            self._last_available_time = -float("inf")
            self._last_bias_update_time = None

        stochastic = bool(add_noise)

        # Generate every sample that falls within the current input interval.
        while t + eps >= self.next_sample_time:
            sample_time = self.next_sample_time
            # Apply sampling epoch jitter.
            if stochastic and self.sample_time_jitter_std > 0.0:
                sample_time += float(self.rng.normal(0.0, self.sample_time_jitter_std))
            # Clamp sample_time to the valid input interval.
            if self.last_input_time is not None:
                sample_time = float(np.clip(sample_time, self.last_input_time, t))
            else:
                sample_time = min(float(sample_time), t)

            # Interpolate target position to the (possibly jittered) sample epoch.
            sample_position = _interpolate_position(
                self.last_input_time,
                self.last_rel_position,
                t,
                rel_position,
                sample_time,
            )

            # Produce a range measurement at this sample epoch.
            measurement = self._instant_measure(sample_position, add_noise=stochastic, timestamp=sample_time)

            # Apply dropout (missed detection) to valid returns.
            if measurement.get("valid", False):
                truth_range = float(measurement.get("truth_range", np.linalg.norm(sample_position)))
                if stochastic and float(self.rng.random()) < self._dropout_probability(truth_range):
                    measurement = self._build_invalid(sample_time, reason="dropout", truth_range=truth_range)

            # Push into the latency pipeline.
            self._schedule_measurement(sample_time, measurement, add_noise=stochastic)
            self.next_sample_time += self.sampling_period  # advance to the next sample epoch

        # Store current input for interpolation on the next call.
        self.last_input_time = t
        self.last_rel_position = rel_position

        # Pop the latest measurement whose latency has elapsed.
        measurement = _pop_latest_ready(self.pending_measurements, t)
        if measurement is not None:
            return measurement

        return self._build_invalid(t, stale=True)  # no data ready yet

    def _beam_directions_sensor_frame(self):
        """
        Build (and cache) the full scan pattern in the sensor body frame.

        Generates a grid of beam directions from the azimuth x elevation scan
        parameters. Each entry is a tuple:
            (elev_idx, az_idx, elevation, azimuth, direction_unit_vector)

        Direction vector in sensor frame:
            x = cos(elev) * cos(az)
            y = cos(elev) * sin(az)
            z = sin(elev)

        The result is cached in _beam_pattern_cache because the pattern does
        not change between frames for a given sensor configuration.

        :return: List of (elev_idx, az_idx, elevation, azimuth, direction) tuples.
        """
        if self._beam_pattern_cache is not None:
            return self._beam_pattern_cache  # return cached pattern

        azimuth_values = np.linspace(self.scan_azimuth_min, self.scan_azimuth_max, self.scan_azimuth_samples)  # [rad]
        pattern = []
        for elev_idx, elevation in enumerate(self.scan_elevation_angles):
            cos_el = float(np.cos(elevation))
            sin_el = float(np.sin(elevation))
            for az_idx, azimuth in enumerate(azimuth_values):
                # Spherical to Cartesian conversion for beam direction.
                direction = np.array(
                    [cos_el * np.cos(azimuth), cos_el * np.sin(azimuth), sin_el],
                    dtype=float,
                )
                direction = _normalize(direction)  # ensure unit length
                pattern.append((elev_idx, az_idx, float(elevation), float(azimuth), direction))

        self._beam_pattern_cache = pattern  # cache for subsequent frames
        return pattern

    def _received_power(self, distance, incidence_cos, material):
        """
        Compute received optical power for a single LiDAR return.

        Radiometric model (simplified lidar equation):
            P_rx = E_tx * (rho_d * cos(theta) + rho_r) * A_rx / (4 * pi * r^2)
                   * exp(-2 * alpha * r) / A_footprint

        Where:
            E_tx          = pulse_energy                [arb]
            rho_d         = Lambertian reflectivity     [0..1]
            rho_r         = retro-reflectivity          [0..1]
            cos(theta)    = incidence_cos               [0..1]
            A_rx          = receiver_aperture_area      [arb]
            r             = distance                    [m]
            alpha         = atmosphere_extinction_coeff [1/m]
            A_footprint   = pi * (r * tan(div/2))^2     [m^2]

        :param distance:       Range to target [m].
        :param incidence_cos:  Cosine of angle between incoming ray and surface normal [0..1].
        :param material:       Material dataclass with reflectivity attributes.
        :return: Received power [arb].
        """
        reflectivity = float(np.clip(getattr(material, "reflectivity", 0.5), 0.0, 1.0))
        retro = float(np.clip(getattr(material, "retro_reflectivity", 0.0), 0.0, 1.0))

        beam_radius = max(distance * np.tan(self.beam_divergence * 0.5), 1e-6)  # [m] beam footprint radius
        footprint_area = np.pi * beam_radius * beam_radius                       # [m^2] illuminated area
        geometric_term = self.receiver_aperture_area / max(4.0 * np.pi * distance * distance, eps)  # 1/r^2 geometric falloff
        reflectance_term = reflectivity * incidence_cos + retro                   # combined surface reflectance
        attenuation = np.exp(-2.0 * self.atmosphere_extinction_coeff * distance)  # two-way Beer-Lambert extinction
        return float(self.pulse_energy * reflectance_term * geometric_term * attenuation / max(footprint_area, eps))

    def _hit_to_beam_return(self, hit, direction_world, timestamp, add_noise):
        """
        Convert a RayHit into a single-beam measurement dictionary.

        Applies range gating, grazing-angle rejection, radiometric SNR-based
        detection probability, and dropout. Attaches hit metadata when enabled.

        :param hit:             RayHit from scene.cast_ray().
        :param direction_world: Beam direction in world frame [unit].
        :param timestamp:       Sample epoch [s].
        :param add_noise:       Enable stochastic terms.
        :return: Measurement dict.
        """
        true_range = float(hit.distance)  # [m] ground truth range to hit

        # Range gate: reject hits outside [range_min, range_max].
        if true_range < self.range_min or true_range > self.range_max:
            return self._build_invalid(timestamp, reason="out_of_range", truth_range=true_range)

        # Incidence angle: cosine of angle between incoming ray and surface normal.
        incidence_cos = float(np.clip(np.dot(-direction_world, hit.normal), 0.0, 1.0))
        if incidence_cos <= eps:
            return self._build_invalid(timestamp, reason="grazing_angle", truth_range=true_range)  # near-parallel hit

        # Compute received power via the radiometric model.
        received_power = self._received_power(true_range, incidence_cos, hit.material)
        if add_noise and self.intensity_noise_std > 0.0:
            received_power += float(self.rng.normal(0.0, self.intensity_noise_std))  # additive intensity noise
        received_power = max(received_power, 0.0)  # physical floor

        # Detection probability: sigmoid-like function of SNR = P_rx / (P_rx + P_min).
        detection_probability = float(np.clip(received_power / (received_power + self.min_detectable_power), 0.0, 1.0))
        if add_noise:
            if float(self.rng.random()) > detection_probability:
                # Failed to detect -- below SNR threshold.
                measurement = self._build_invalid(timestamp, reason="below_snr", truth_range=true_range)
            else:
                # Detected -- produce a range measurement and apply dropout.
                measurement = self._range_measurement_model(true_range, timestamp=timestamp, add_noise=True)
                if measurement.get("valid", False) and float(self.rng.random()) < self._dropout_probability(true_range):
                    measurement = self._build_invalid(timestamp, reason="dropout", truth_range=true_range)
        else:
            if received_power < self.min_detectable_power:
                measurement = self._build_invalid(timestamp, reason="below_snr", truth_range=true_range)
            else:
                measurement = self._range_measurement_model(true_range, timestamp=timestamp, add_noise=False)

        # Attach per-beam metadata (radiometric and geometric diagnostics).
        if self.include_metadata:
            measurement["incidence_cos"] = incidence_cos                              # surface incidence cosine
            measurement["received_power"] = float(received_power)                    # [arb] received power
            measurement["detection_probability"] = detection_probability              # detection probability [0..1]
            measurement["object_id"] = getattr(hit, "object_id", "")                 # which object was hit
            measurement["hit_point"] = np.asarray(hit.point, dtype=float).tolist()    # [m] world space hit point
            measurement["hit_normal"] = np.asarray(hit.normal, dtype=float).tolist()  # surface normal at hit

        return measurement

    def simulate_scene_frame(self, scene, sensor_position, sensor_orientation=None, timestamp=0.0, add_noise=True):
        """
        Produce a complete scan frame by ray casting through a Scene.

        Iterates over every beam in the scan pattern, transforms it from sensor
        frame to world frame, casts it into the scene, and collects per-beam
        measurement dictionaries.

        :param scene:              Scene object with a cast_ray() method.
        :param sensor_position:    Sensor origin in world space [m].
        :param sensor_orientation: 3x3 rotation matrix (sensor -> world). None = identity.
        :param timestamp:          Frame epoch [s].
        :param add_noise:          Enable stochastic error terms.
        :return: Frame dict with type="frame", returns list, and summary statistics.
        """
        if not hasattr(scene, "cast_ray"):
            raise TypeError("scene must provide a cast_ray(origin, direction, max_range, min_range) method.")

        origin_world = _as_vector3(sensor_position, "sensor_position")  # [m] sensor origin in world frame
        rotation = np.eye(3, dtype=float) if sensor_orientation is None else _as_rotation_matrix(
            sensor_orientation, "sensor_orientation"
        )  # 3x3 rotation matrix (sensor body -> world)

        beam_pattern = self._beam_directions_sensor_frame()  # cached scan pattern
        frame_returns = []
        valid_count = 0

        for beam_id, (elev_idx, az_idx, elevation, azimuth, dir_sensor) in enumerate(beam_pattern):
            # Transform beam direction from sensor body frame to world frame.
            dir_world = _normalize(rotation @ dir_sensor)

            # Cast ray into the scene geometry.
            hit = scene.cast_ray(
                origin_world,
                dir_world,
                max_range=self.range_max,
                min_range=max(self.range_min, eps),
            )

            # Convert hit (or miss) into a measurement dictionary.
            if hit is None:
                beam_measurement = self._build_invalid(timestamp, reason="no_hit")
            else:
                beam_measurement = self._hit_to_beam_return(hit, dir_world, timestamp=timestamp, add_noise=bool(add_noise))

            # Tag each beam with its scan pattern indices.
            beam_measurement["beam_id"] = beam_id            # sequential beam counter
            beam_measurement["channel_id"] = int(elev_idx)   # elevation channel index
            beam_measurement["azimuth_index"] = int(az_idx)  # azimuth step index
            if self.include_metadata:
                beam_measurement["elevation"] = float(elevation)               # [rad]
                beam_measurement["azimuth"] = float(azimuth)                   # [rad]
                beam_measurement["direction_sensor"] = dir_sensor.tolist()      # beam dir in sensor frame
                beam_measurement["direction_world"] = dir_world.tolist()        # beam dir in world frame

            if beam_measurement.get("valid", False):
                valid_count += 1
            frame_returns.append(beam_measurement)

        # Assemble the frame-level output dictionary.
        frame = {
            "valid": True,
            "type": "frame",
            "timestamp": float(timestamp),
            "num_beams": len(frame_returns),    # total beams in this frame
            "num_valid": int(valid_count),       # beams that produced a valid range
            "returns": frame_returns,
        }
        if self.include_metadata:
            frame["sensor_position"] = origin_world.tolist()   # [m] sensor origin
            frame["sensor_orientation"] = rotation.tolist()     # 3x3 rotation used
        return frame

    def _schedule_scene_frame(self, sample_time, frame, add_noise):
        """
        Push a scene-scan frame into the latency pipeline.

        Analogous to _schedule_measurement but operates on complete scan frames.

        :param sample_time: Frame sample epoch [s].
        :param frame:       Completed frame dict.
        :param add_noise:   If True, add latency jitter.
        """
        latency_value = self.latency  # [s] nominal latency
        if add_noise and self.latency_jitter_std > 0.0:
            latency_value += float(self.rng.normal(0.0, self.latency_jitter_std))  # jitter [s]
        latency_value = max(latency_value, 0.0)  # clamp to non-negative

        available_time = sample_time + latency_value  # [s] when this frame becomes available
        # Enforce strict monotonicity on available times.
        if available_time <= self._last_scene_available_time:
            available_time = self._last_scene_available_time + eps
        self._last_scene_available_time = available_time
        self.pending_scene_frames.append((available_time, frame))

    def scan_scene(self, scene, sensor_position, sensor_orientation=None, add_noise=True, current_time=None):
        """
        Primary scene-scan measurement interface (full scan pattern with latency pipeline).

        If current_time is None, an instantaneous (no-latency) frame is produced.

        If current_time is provided, the asynchronous sampling pipeline runs:
        1. Align the sample clock on the first call.
        2. Detect time reversals and reset the scene pipeline if needed.
        3. Generate all frames that fall within [last_scene_input_time, current_time].
        4. Schedule each frame through the latency pipeline.
        5. Pop and return the latest frame whose latency has elapsed.

        :param scene:              Scene object with a cast_ray() method.
        :param sensor_position:    Sensor origin in world space [m].
        :param sensor_orientation: 3x3 rotation matrix (sensor -> world). None = identity.
        :param add_noise:          Enable stochastic terms.
        :param current_time:       Simulation clock [s]. None = instantaneous mode.
        :return: Frame dict.
        """
        if current_time is None:
            return self.simulate_scene_frame(
                scene,
                sensor_position,
                sensor_orientation=sensor_orientation,
                timestamp=0.0,
                add_noise=add_noise,
            )

        t = float(current_time)

        # On first call, synchronize the scene scan clock to the simulation clock.
        if self.last_scene_input_time is None and not self.pending_scene_frames and self.next_scene_sample_time == 0.0:
            self.next_scene_sample_time = t

        # Detect time reversal -- reset the scene pipeline.
        if self.last_scene_input_time is not None and t + eps < self.last_scene_input_time:
            self.pending_scene_frames.clear()
            self.next_scene_sample_time = t
            self.last_scene_input_time = None
            self._last_scene_available_time = -float("inf")

        stochastic = bool(add_noise)

        # Generate all frames that fall within the current input interval.
        while t + eps >= self.next_scene_sample_time:
            sample_time = self.next_scene_sample_time
            # Apply sampling epoch jitter.
            if stochastic and self.sample_time_jitter_std > 0.0:
                sample_time += float(self.rng.normal(0.0, self.sample_time_jitter_std))
            # Clamp to valid input interval.
            if self.last_scene_input_time is not None:
                sample_time = float(np.clip(sample_time, self.last_scene_input_time, t))
            else:
                sample_time = min(float(sample_time), t)

            # Produce a full scan frame at this sample epoch.
            frame = self.simulate_scene_frame(
                scene,
                sensor_position=sensor_position,
                sensor_orientation=sensor_orientation,
                timestamp=sample_time,
                add_noise=stochastic,
            )
            # Push into the latency pipeline.
            self._schedule_scene_frame(sample_time, frame, add_noise=stochastic)
            self.next_scene_sample_time += self.sampling_period  # advance to next frame epoch

        # Store current input time for the next call.
        self.last_scene_input_time = t

        # Pop the latest frame whose latency has elapsed.
        frame = _pop_latest_ready(self.pending_scene_frames, t)
        if frame is not None:
            return frame
        return self._build_stale_frame(t)  # no frame ready yet
