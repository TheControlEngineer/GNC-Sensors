import numpy as np 

class LidarConfig:
    range_min = 1.0 # [m]
    range_max = 1000.0 # [m]
    range_accuracy = 0.1 # [m]  minimum measurable distance change.
    fov = np.deg2rad(30.0) # [rad]  field of view in radians.
    sampling_rate = 5.0 # [Hz] some sensors call it update rate.
    latency = 0.08 # [s]  time delay between measurement and availability.
    dropout_prob = 0.005 # probability of measurement drop out
    