import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from jtf_wavenet.vis.style import set_science_style
from jtf_wavenet.data.generator_core import generate_all_signals_once, create_common_aux_vars
from jtf_wavenet.data.parameter_sampling import random_parameter_gen_with_HF


def make_ft_complex(fid: tf.Tensor) -> np.ndarray:
    """
    Mirrors your old make_ft style: exponential window + zero pad by same length,
    then fftshift(fft()).
    """
    fid_np = fid.numpy()
    points = fid_np.shape[0]

    window = np.exp(-5 * np.linspace(0.0, 1.0, points))
    windowed = fid_np * window

    zp = np.pad(windowed, (0, points), "constant")
    zp[0] /= 2.0
    return np.fft.fftshift(np.fft.fft(zp))

def plot_all_three_in_one_pass(config):
    """
    EXACT intent of your old function, but now fully JSON-driven:
      - sample sw from JSON
      - generate HF/NF params once
      - generate mask/window once
      - generate HF, NF_vib, NF_clean once
      - plot FFTs
    """
    set_science_style()

    # --- spectrum config ---
    spec = config["spectrum"]

    # sample SW from JSON (no numbers in src)
    sw_lo, sw_hi = spec["sw_ppm_range"]
    sw_ppm = tf.random.uniform((), sw_lo, sw_hi)

    spec["sw_ppm"] = sw_ppm
    spec["sw_hz"] = spec["field_t"] * spec["gamma_h"] * sw_ppm

    total_points = spec["total_points"]
    sw_hz = spec["sw_hz"]
    config["sw_ppm"] = spec["sw_ppm"]
    config["sw_hz"] = spec["sw_hz"]
    config["total_points"] = spec["total_points"]
    acquisition_time = config["total_points"] / config["sw_hz"]
    config["echo_times"] = spec["echo_times"]
    config["_cfg"] = config


    # params (JSON-driven)
    hf_params, nf_params, vib_params = random_parameter_gen_with_HF(config)

    # mask/window
    fid_mask, window = create_common_aux_vars(config, total_points, acquisition_time)

    # signals once
    hf_fid, nf_vib_fid, nf_clean_fid, times = generate_all_signals_once(
        config, hf_params, nf_params, vib_params, fid_mask, window
    )

    # FFTs
    fft_hf = make_ft_complex(hf_fid)
    fft_nf_vib = make_ft_complex(nf_vib_fid)
    fft_nf_clean = make_ft_complex(nf_clean_fid)

    freqs = np.fft.fftshift(
        np.fft.fftfreq(total_points * 2, d=float(times[1] - times[0]))
    )

    hf_r = np.real(fft_hf)
    nf_vib_r = np.real(fft_nf_vib)
    nf_clean_r = np.real(fft_nf_clean)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(freqs, hf_r, linewidth=1.5, label="High Field")
    axes[0].plot(freqs, nf_vib_r, linewidth=1.5, label="Normal w/ Vibes")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Input Spectra (HF & NF)")
    axes[0].legend()

    axes[1].plot(freqs, nf_clean_r, linewidth=1.5, label="Target Clean")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Target Spectrum")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
