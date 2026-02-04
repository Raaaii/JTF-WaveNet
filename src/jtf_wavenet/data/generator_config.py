import numpy as np

CONFIG = {
    "total_points": 512 * 8,
    "sw_ppm": 16e-6,
    "field": 14.1,
    "gamma_h": 267.52218744e6 / (2 * np.pi),
    "n_peaks": 150,
    "max_couplings": 6,
    "vib_max": 80,
    "vib_freq_max": 150,
    "echo_times": [0.0],
}
