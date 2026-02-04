import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from jtf_wavenet.vis.style import set_science_style
from jtf_wavenet.signal.signal_function import calculate_vibration, combine_signal_and_vibration


def _fftshift_fft(x: tf.Tensor) -> tf.Tensor:
    """fftshift(fft(x))"""
    x = tf.cast(x, tf.complex64)
    X = tf.signal.fft(x)
    return tf.signal.fftshift(X, axes=0)


def _mag(x: tf.Tensor) -> tf.Tensor:
    return tf.abs(tf.cast(x, tf.complex64))


def _safe_np(x: tf.Tensor) -> np.ndarray:
    return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)


def plot_vibration_diagnostics(
    *,
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
    title="Vibration diagnostics",
):
    """
    Generates:
      - target (clean) and total (vibrated)
      - vib_term (raw vibration term)
      - vibration_factor = beta + alpha * scaled_vib_term
    And plots time + frequency domain comparisons.

    IMPORTANT: No randomness here. This only visualizes given inputs.
    """
    set_science_style()

    # --- compute signals ---
    total, target = combine_signal_and_vibration(
        frequency=frequency,
        phase=phase,
        r2=r2,
        times=times,
        vibration_scaling=vibration_scaling,
        vib_amps=vib_amps,
        vib_phase=vib_phase,
        vib_frequencies=vib_frequencies,
        vib_time=vib_time,
        vib_r2=vib_r2,
        peak_amplitude_main=peak_amplitude_main,
        vib_level=vib_level,
        r2_inhom=r2_inhom,
        scalar_coupling=scalar_coupling,
        k=k,
        echo_time=echo_time,
        water_params=water_params,
    )

    vib_term = calculate_vibration(
        tf.cast(times, tf.complex64),
        tf.cast(vib_amps, tf.complex64),
        tf.cast(vib_frequencies, tf.complex64),
        vib_phase,
        vib_time,
        tf.cast(vib_r2, tf.complex64),
        vibration_scaling,
    )

    # --- reproduce the scaling used in combine_signal_and_vibration ---
    ft_vib = tf.signal.fft(vib_term)
    vib_freq_max_amp = tf.reduce_max(tf.abs(ft_vib))
    vib_freq_max_amp = tf.cast(vib_freq_max_amp, tf.complex64)

    signal_length = tf.cast(tf.shape(target)[0], tf.complex64)
    vib_term_scaled = signal_length * tf.cast(vib_level, tf.complex64) * vib_term / vib_freq_max_amp

    beta = 1 - vibration_scaling / 2
    alpha = vibration_scaling / 2
    vibration_factor = beta + alpha * vib_term_scaled

    # --- time-domain arrays ---
    t_np = _safe_np(times)
    target_np = _safe_np(target)
    total_np = _safe_np(total)
    vib_np = _safe_np(vib_term)
    vib_scaled_np = _safe_np(vib_term_scaled)
    vf_np = _safe_np(vibration_factor)

    residual_time = total - target
    residual_time_np = _safe_np(residual_time)

    # --- frequency-domain arrays ---
    TARGET_F = _fftshift_fft(target)
    TOTAL_F = _fftshift_fft(total)
    VIB_F = _fftshift_fft(vib_term)
    VF_F = _fftshift_fft(vibration_factor)
    RES_F = TOTAL_F - TARGET_F

    TARGET_F_np = _safe_np(TARGET_F)
    TOTAL_F_np = _safe_np(TOTAL_F)
    VIB_F_np = _safe_np(VIB_F)
    VF_F_np = _safe_np(VF_F)
    RES_F_np = _safe_np(RES_F)

    # =======================
    # PLOTS
    # =======================
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title)

    # --- (1) Target vs Total (time) ---
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(np.real(target_np), label="target real")
    ax1.plot(np.real(total_np), label="total real", alpha=0.7)
    ax1.set_title("Time domain (real): target vs total")
    ax1.legend()

    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(np.imag(target_np), label="target imag")
    ax2.plot(np.imag(total_np), label="total imag", alpha=0.7)
    ax2.set_title("Time domain (imag): target vs total")
    ax2.legend()

    # --- (2) Vibration term time ---
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(np.real(vib_np), label="vib_term real")
    ax3.plot(np.real(vib_scaled_np), label="vib_term_scaled real", alpha=0.7)
    ax3.set_title("Vibration term (real): raw vs scaled")
    ax3.legend()

    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(np.imag(vib_np), label="vib_term imag")
    ax4.plot(np.imag(vib_scaled_np), label="vib_term_scaled imag", alpha=0.7)
    ax4.set_title("Vibration term (imag): raw vs scaled")
    ax4.legend()

    # --- (3) Vibration factor time ---
    ax5 = plt.subplot(4, 2, 5)
    ax5.plot(np.real(vf_np), label="vibration_factor real")
    ax5.set_title("Vibration factor (real) = beta + alpha*vib_term_scaled")
    ax5.legend()

    ax6 = plt.subplot(4, 2, 6)
    ax6.plot(np.imag(vf_np), label="vibration_factor imag")
    ax6.set_title("Vibration factor (imag)")
    ax6.legend()

    # --- (4) Residual time ---
    ax7 = plt.subplot(4, 2, 7)
    ax7.plot(np.real(residual_time_np), label="(total-target) real")
    ax7.plot(np.imag(residual_time_np), label="(total-target) imag", alpha=0.7)
    ax7.set_title("Residual in time domain")
    ax7.legend()

    # --- (5) FFT magnitudes (overlay) ---
    ax8 = plt.subplot(4, 2, 8)
    ax8.plot(np.real(TARGET_F_np), label="|FFT(target)|")
    ax8.plot(np.real(TOTAL_F_np), label="|FFT(total)|", alpha=0.7)
    ax8.plot(np.real(VIB_F_np), label="|FFT(vib_term)|", alpha=0.7)
    ax8.plot(np.real(RES_F_np), label="|FFT(residual)|", alpha=0.7)
    ax8.set_title("Frequency domain magnitudes (fftshift)")
    ax8.legend()

    plt.tight_layout()
    plt.show()
