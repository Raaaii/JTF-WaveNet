import tensorflow as tf
import numpy as np

from jft_wavenet.vis.signal_checks import quick_signal_plot


def main():
    n_peaks = 20
    n_coup = 5
    time_len = 4096

    times = tf.linspace(0.0, 0.5, time_len)
    example_kwargs = dict(
        frequency=tf.complex(tf.random.normal([n_peaks]), tf.zeros([n_peaks])),
        phase=tf.complex(tf.random.normal([n_peaks]), tf.zeros([n_peaks])),
        r2=tf.complex(tf.random.uniform([n_peaks], 1.0, 10.0), tf.zeros([n_peaks])),
        r2_inhom=tf.complex(tf.random.uniform([n_peaks], 0.1, 2.0), tf.zeros([n_peaks])),
        scalar_couplings=tf.complex(
            tf.random.uniform([n_peaks, n_coup], 0.0, 10.0), tf.zeros([n_peaks, n_coup])
        ),
        times=times,
        T=tf.constant(0.0),
        peak_amplitude_main=tf.complex(
            tf.random.uniform([n_peaks], 1.0, 10.0), tf.zeros([n_peaks])
        ),
        k=tf.constant(0.0),
    )

    quick_signal_plot(example_kwargs)


if __name__ == "__main__":
    main()
