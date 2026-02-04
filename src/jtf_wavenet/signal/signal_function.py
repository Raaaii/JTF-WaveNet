import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from jtf_wavenet.utils import tf_funcs

tfd = tfp.distributions


def signal(
    frequency,  # (n_peaks,)
    phase,  # (n_peaks,)
    r2,  # (n_peaks,)
    r2_inhom,  # (n_peaks,) or scalar-ish; applied per-peak then reduced effectively
    scalar_couplings,  # (n_peaks, n_couplings) OR (n_peaks,)
    times,  # (time_len,)
    T,  # scalar echo time
    peak_amplitude_main,  # (n_peaks,)
    k=0.0,  # scalar roofing factor
):
    """
    Returns complex signal (time_len,).

    Per-peak:
      exp(i*(2π f t + phase) - r2 t) * Π_j [cos(π J (t+T)) + i*k*sin(π J (t+T))]

    Then applies inhomogeneous decay factor exp(-r2_inhom * t).
    """
    # Cast
    times = tf.cast(times, tf.complex64)  # (time_len,)
    times_expanded = times[tf.newaxis, ...]  # (1, time_len)
    T = tf.cast(T, tf.complex64)
    T = tf.reshape(T, (1, 1))  # (1,1)

    frequency = tf.cast(frequency, tf.complex64)[..., tf.newaxis]  # (n_peaks,1)
    phase = tf.cast(phase, tf.complex64)[..., tf.newaxis]  # (n_peaks,1)
    r2 = tf.cast(r2, tf.complex64)[..., tf.newaxis]  # (n_peaks,1)
    peak_amplitude_main = tf.cast(peak_amplitude_main, tf.complex64)[..., tf.newaxis]  # (n_peaks,1)

    r2_inhom = tf.cast(r2_inhom, tf.complex64)[..., tf.newaxis]  # (n_peaks,1)

    # Couplings: ensure (n_peaks, n_couplings, 1)
    scalar_couplings = tf.convert_to_tensor(scalar_couplings)
    if scalar_couplings.shape.ndims == 1:
        scalar_couplings = tf.expand_dims(scalar_couplings, axis=-1)  # (n_peaks,1)
    scalar_couplings = tf.cast(scalar_couplings, tf.complex64)[
        ..., tf.newaxis
    ]  # (n_peaks,n_coup,1)

    # Base exponential: (n_peaks, time_len)
    base_expo = tf.exp(
        1j * (2.0 * np.pi * frequency * times_expanded + phase) - r2 * times_expanded
    )

    # Coupling factors: (n_peaks, n_coup, time_len)
    coupling_factors = tf.cos(np.pi * scalar_couplings * (times_expanded + T)) + 1j * tf.cast(
        k, tf.complex64
    ) * tf.sin(np.pi * scalar_couplings * (times_expanded + T))

    # Product over couplings -> (n_peaks, time_len)
    coupling_product = tf.reduce_prod(coupling_factors, axis=1)

    # Sum peaks -> (time_len,)
    signals_each_peak = peak_amplitude_main * base_expo * coupling_product
    signal_sum = tf.reduce_sum(signals_each_peak, axis=0)

    # Inhomogeneous decay:
    r2_inhom_global = tf.reduce_mean(r2_inhom)  # scalar complex
    signal_sum = signal_sum * tf.exp(-r2_inhom_global * times)

    return signal_sum


def calculate_vibration(
    times,
    vib_amps,
    vib_frequencies,
    vib_phase,
    vib_time,
    vib_r2,
    vibration_scaling,
):
    """
    Returns vibration term (time_len,) complex.
    """
    times = tf.reshape(times, [-1])  # (time_len,)
    times_expanded = times[..., tf.newaxis]  # (time_len,1)
    vib_r2_expanded = vib_r2[tf.newaxis, ...]  # (1,vib_max)
    vib_frequencies_expanded = vib_frequencies[tf.newaxis, ...]  # (1,vib_max)

    total_vibration_amp = 1 - vibration_scaling
    vib_amps_normalized = total_vibration_amp * vib_amps / tf.reduce_sum(vib_amps)
    vib_amps_normalized = vib_amps_normalized[tf.newaxis, ...]  # (1,vib_max)

    vib_dampening = tf.exp(-1 * times_expanded * vib_r2_expanded)  # (time_len,vib_max)
    vib_inside = 2 * vib_frequencies_expanded * np.pi * (times_expanded + vib_time)

    vibration_term = vib_amps_normalized * (-1j) * tf.sin(vib_inside + vib_phase) * vib_dampening
    vibration_term = tf.reduce_sum(vibration_term, axis=1)  # (time_len,)
    return vibration_term


def combine_signal_and_vibration(
    frequency,
    phase,
    r2,
    times,
    vibration_scaling,
    vib_amps,
    vib_phase,
    vib_frequencies,
    vib_time,
    vib_r2,
    peak_amplitude_main,
    vib_level=0.1,
    r2_inhom=None,
    scalar_coupling=None,
    k=0.0,
    echo_time=0.0,
    water_params=None,
):
    """
    Returns:
      total:  (time_len,) complex
      target: (time_len,) complex
    """
    if r2_inhom is None:
        r2_inhom = tf.zeros_like(r2)
    if scalar_coupling is None:
        scalar_coupling = tf.zeros_like(r2)

    target = signal(
        frequency=frequency,
        phase=phase,
        r2=r2,
        r2_inhom=r2_inhom,
        scalar_couplings=scalar_coupling,
        times=times,
        T=echo_time,
        peak_amplitude_main=peak_amplitude_main,
        k=k,
    )

    # Optional water
    if water_params is not None:
        water_peak_amplitude = water_params["water_peak_amplitude"]
        water_frequency = water_params["water_frequency"]
        water_r2 = water_params["water_r2"]

        times_c = tf.cast(times, tf.complex64)
        echo_time_c = tf.cast(echo_time, tf.complex64)
        water_signal = water_peak_amplitude * tf.exp(
            1j * (2 * np.pi * water_frequency * (times_c - echo_time_c))
            - water_r2 * (times_c - echo_time_c)
        )
        target = target + water_signal

    vib_term = calculate_vibration(
        tf.cast(times, tf.complex64),
        tf.cast(vib_amps, tf.complex64),
        tf.cast(vib_frequencies, tf.complex64),
        vib_phase,
        vib_time,
        tf.cast(vib_r2, tf.complex64),
        vibration_scaling,
    )

    # Normalize vib term using FFT peak
    ft_vib = tf.signal.fft(vib_term)
    ft_vib_abs = tf.math.abs(ft_vib)
    vib_freq_max_amp = tf.reduce_max(ft_vib_abs)
    vib_freq_max_amp = tf.cast(vib_freq_max_amp, tf.complex64)

    signal_length = tf.cast(tf.shape(target)[0], tf.complex64)
    vib_term = signal_length * vib_level * vib_term / vib_freq_max_amp

    beta = 1 - vibration_scaling / 2
    alpha = vibration_scaling / 2
    vibration_factor = beta + alpha * vib_term

    total = target * vibration_factor
    return total, target
