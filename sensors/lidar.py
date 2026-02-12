"""
This module contains the LiDAR sensors of GNC application.

"""

import numpy as np 
from .Config import LidarConfig
from . math_utils import _normalize, _warp_angle, _interpolate_position, _pop_latest_ready, eps

class Lidar:
    def __init__(self, config = LidarConfig, boresight =None):
        self.range_min = config.range_min
        self.range_max = config.range_max
        self.range_accuracy = config.range_accuracy
        self.fov = config.fov
        self.sampling_rate = config.sampling_rate
        self.sampling_period = 1.0/ max(config.sampling_rate, 1e-12) # did this to avoid divide by zero error.

        self.latency = float(getattr(config, "latency", 0.0)) # some config may not have latency defined, so we set it to 0.0 by default.
        self.dropout_prob = float(getattr(config,"dropout_prob",0.0)) # some config may not have drop out defined, so we set it to 0.0 by default.
        self.boresight = _normalize(np.array([1.0, 0.0, 0.0])) if boresight is None else _normalize(np.array(boresight)) # default boresight is along x-axis.
        self.next_sample_time = 0.0 
        self.pending_measurements = [] # list of pending measurements, each element is a tuple of (measurement_time, measurement_value)
        self.last_input_time = None # last time the sensor received input, used for interpolation.
        self.last_rel_position = None # last relative position, used for interpolation.

    def _instant_measure(self, rel_position, add_noise = True, timestamp =0.0 ):
        """
        Generate one lidar sample without update rate and latency scheduling.
        
        :param self: variable of type Lidar
        :param rel_position: relative position of the target in the sensor frame, should be a 3D vector.
        :param add_noise: option to add noise to the measurement, default is True.
        :param time_stamp: time stamp of the measurement, default is 0.0.

        """

        rel_position = np.array(rel_position, dtype=float)
        range_val = np.linalg.norm(rel_position) # calculate the range value.
        if range_val < self.range_min or range_val > self.range_max:
            return {"valid": False, "timestamp": float(timestamp)} # if the range value is out of the sensor's range, return invalid measurement.
        
        los = rel_position / range_val # Line of sight unit vector.
        if np.arccos(np.clip(np.dot(los, self.boresight), -1.0, 1.0)) > self.fov / 2.0: # check if the target is within the field of view.
            return {"valid": False, "timestamp": float(timestamp)} # if the target is out of the field of view, return invalid measurement.
        
        measured_range = range_val #
        if add_noise:
            measured_range += np.random.normal(0,self.range_accuracy) #
        return {"valid":True, "range": measured_range, "timestamp": float(timestamp)} #return the measurement as a dictionary, including validity, range value, and timestamp.
    
    def measure(self, rel_position, add_noise = True, current_time=None):
        """
        Generate LIDAR range measurement.
        
        :param self: variable of type Lidar
        :param rel_position: relative position of the target in the sensor frame, should be a 3D vector.
        :param add_noise: option to add noise to the measurement, default is True.
        :param current_time: current time of the measurement, if None, it will be set to 0.0.

        return : a dictionary containing the measurement result, including validity, range value, and timestamp. The timestamp is the time when the measurement is available, which is the current time plus the latency of the sensor.

        """

        if current_time is None:
            return self._instant_measure(rel_position, add_noise=add_noise, timestamp=0.0) # if current time is not provided, we assume it is 0.0 and return the measurement immediately without scheduling.
        rel_position = np.array(rel_position, dtype=float)
        t = float (current_time)
        if self.last_input_time is not None and t + eps < self.last_input_time:
            self.pending_measurements.clear() # if the current time is earlier than the last input time, we clear the pending measurements to avoid confusion.
            self.pending_measurements.clear()# reset the pending measurements to avoid confusion.
            self.next_sample_time = t # reset the next sample time to the current time to avoid confusion.

        while t + _EPS >= self.next_sample_time:# if the current time is greater than or equal to the next sample time, we need to generate a new measurement.
            sample_time = self.next_sample_time # the time when the measurement is taken, which is the next sample time.
            sample_position = _interpolate_position(
                self.last_input_time,
                self.last_rel_position,
                t,
                rel_position,
                sample_time,
            ) # interpolate the relative position at the sample time based on the last input time, last relative position, current time, and current relative position.
            measurement = self._instant_measure(sample_position, add_noise=add_noise, timestamp=sample_time) # generate the measurement at the sample time based on the interpolated relative position.
            if measurement.get("valid", False) and add_noise and np.random.rand() < self.dropout_prob: # if the measurement is valid and we are adding noise, we check for dropout based on the dropout probability.
                measurement = {"valid": False, "timestamp": float(sample_time), "reason": "dropout"} # if the measurement is dropped out, we set it to invalid and add a reason for dropout.

            available_time = sample_time + self.latency # the time when the measurement is available, which is the sample time plus the latency of the sensor.
            self.pending_measurements.append((available_time, measurement)) # add the measurement to the pending measurements list with its available time.
            self.next_sample_time += self.sample_period

        self.last_input_time = t
        self.last_rel_position = rel_position

        measurement = _pop_latest_ready(self.pending_measurements, t) # check if there is any measurement that is ready to be returned based on the current time and the available time of the pending measurements. If there is a measurement that is ready, we return it; otherwise, we return an invalid measurement indicating that the measurement is stale.
        if measurement is not None:
            return measurement

        return {"valid": False, "timestamp": t, "stale": True} # if there is no measurement that is ready, we return an invalid measurement indicating that the measurement is stale.

        
        



