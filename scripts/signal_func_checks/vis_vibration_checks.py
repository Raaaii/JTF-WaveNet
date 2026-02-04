import tensorflow as tf
import numpy as np

from jtf_wavenet.vis.signal_debug_plots import plot_vibration_diagnostics


def main():
    # Example shapes
    n_peaks = 30
    n_coup = 3
    vib_max = 40
    time_len = 4096

    times = tf.linspace(0.0, 0.4, time_len)

    # Metabolite params
    frequency = tf.complex(tf.random.normal([n_peaks], stddev=100.0), tf.zeros([n_peaks]))
    phase = tf.complex(tf.random.normal([n_peaks], stddev=0.2), tf.zeros([n_peaks]))
    r2 = tf.complex(tf.random.uniform([n_peaks], 1.0, 20.0), tf.zeros([n_peaks]))
    r2_inhom = tf.complex(tf.fill([n_peaks], 0.7), tf.zeros([n_peaks]))
    scalar_coupling = tf.complex(
        tf.random.uniform([n_peaks, n_coup], 0.0, 10.0), tf.zeros([n_peaks, n_coup])
    )
    peak_amplitude_main = tf.complex(tf.random.uniform([n_peaks], 1.0, 50.0), tf.zeros([n_peaks]))

    # Vibration params
    vibration_scaling = tf.cast(tf.random.uniform([], 0.01, 0.07), tf.complex64)
    vib_amps = tf.complex(tf.random.uniform([vib_max], 1.0, 10.0), tf.zeros([vib_max]))
    vib_frequencies = tf.complex(tf.random.uniform([vib_max], -150.0, 150.0), tf.zeros([vib_max]))
    vib_phase = tf.constant(0.0, tf.complex64)
    vib_time = tf.constant(0.0, tf.complex64)
    vib_r2 = tf.complex(tf.random.uniform([vib_max], 0.0, 50.0), tf.zeros([vib_max]))

    plot_vibration_diagnostics(
        frequency=frequency,
        phase=phase,
        r2=r2,
        r2_inhom=r2_inhom,
        scalar_coupling=scalar_coupling,
        times=times,
        echo_time=tf.constant(0.0),
        k=tf.constant(0.0),
        peak_amplitude_main=peak_amplitude_main,
        vibration_scaling=vibration_scaling,
        vib_amps=vib_amps,
        vib_phase=vib_phase,
        vib_frequencies=vib_frequencies,
        vib_time=vib_time,
        vib_r2=vib_r2,
        vib_level=0.1,
        title="FID clean vs vibrated (and vibration internals)",
    )


if __name__ == "__main__":
    main()
