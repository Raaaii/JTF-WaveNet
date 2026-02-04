import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path

from jtf_wavenet.utils import tf_funcs
from jtf_wavenet.data.parameter_sampling import (
    random_parameter_gen_with_HF,
    build_dist,
    load_config,
    to_flat_core_config,
)
from jtf_wavenet.signal.signal_function import combine_signal_and_vibration

tfd = tfp.distributions


# -----------------------------------------------------------------------------
# Helpers: acquisition config access
# -----------------------------------------------------------------------------
def _acq(config: dict) -> dict:
    """Return acquisition config dict (may be empty)."""
    return config.get("_cfg", {}).get("acquisition", {}) or {}


def _get(acq: dict, key: str, default):
    v = acq.get(key, default)
    return default if v is None else v


def _as_bool(x) -> bool:
    return bool(x) if isinstance(x, (bool, int)) else False


# -----------------------------------------------------------------------------
# Common aux vars: mask + window (JSON-driven)
# -----------------------------------------------------------------------------
def create_common_aux_vars(config: dict, total_points: int, acquisition_time: tf.Tensor):
    acq = _acq(config)

    # --- active points mask ---
    ap_cfg = _get(acq, "active_points", {"enabled": False})
    ap_enabled = _as_bool(ap_cfg.get("enabled", False))

    if ap_enabled:
        ap_min = int(ap_cfg["min"])
        ap_max = int(ap_cfg["max"])
        active_points = tf.random.uniform((), ap_min, ap_max + 1, dtype=tf.int32)
        fid_mask_f = tf.concat(
            [tf.ones([active_points], tf.float32), tf.zeros([total_points - active_points], tf.float32)],
            axis=0,
        )
        fid_mask = tf.cast(fid_mask_f, tf.complex64)
    else:
        fid_mask = tf.ones([total_points], tf.complex64)

    # --- window ---
    win_cfg = _get(acq, "window", {"enabled": False})
    win_enabled = _as_bool(win_cfg.get("enabled", False))
    win_type = str(win_cfg.get("type", "cos2")).lower()

    if not win_enabled:
        window = tf.ones([total_points], tf.complex64)
    else:
        times_temp = tf.linspace(0.0, acquisition_time, total_points)
        if win_type == "cos2":
            window_f = tf.math.cos(np.pi * times_temp / (2.0 * acquisition_time)) ** 2
        else:
            raise ValueError(f"Unknown window.type='{win_type}' (supported: 'cos2')")
        window = tf.cast(window_f, tf.complex64)

    return fid_mask, window


# -----------------------------------------------------------------------------
# Signal generation: water/noise/baseline all JSON-driven
# -----------------------------------------------------------------------------
def generate_all_signals_once(config, hf_params, nf_params, vib_params, fid_mask, window):
    acq = _acq(config)

    sw_hz = config["sw_hz"]
    total_points = config["total_points"]
    times = tf.linspace(0.0, total_points / sw_hz, total_points)

    # --- baseline slope k (mixture) ---
    base_cfg = _get(acq, "baseline", {})
    k_cfg = (base_cfg.get("k", {}) or {}).get("mixture", None)

    if k_cfg is None:
        # If you want strict JSON-only control, replace with: raise KeyError(...)
        # Here we keep behavior: default to the old mixture (but without hardcoding numbers)
        raise KeyError("acquisition.baseline.k.mixture missing in JSON")

    p = float(k_cfg["p"])
    a = build_dist(k_cfg["a"]).sample(())
    b = build_dist(k_cfg["b"]).sample(())
    switch = tf.random.uniform(()) < p
    k_val_float = tf.where(switch, a, b)
    k_val = tf.cast(k_val_float, tf.complex64)

    # --- water ---
    water_cfg = _get(acq, "water", {"enabled_prob": 0.0})
    water_p = float(water_cfg.get("enabled_prob", 0.0))
    water_enabled = tf.random.uniform((1,), 0.0, 1.0, dtype=tf.float32) < water_p
    water_enabled_c = tf.cast(water_enabled, tf.complex64)

    # distributions (required if enabled_prob > 0)
    water_frequency = tf.cast(build_dist(water_cfg["frequency_hz"]).sample((1,)), tf.complex64)
    water_r2 = tf.cast(build_dist(water_cfg["r2"]).sample((1,)), tf.complex64)
    water_int_factor = build_dist(water_cfg["intensity_factor"]).sample(())
    water_nf_scale = tf.cast(build_dist(water_cfg["nf_scale"]).sample(()), tf.complex64)

    apply_to_hf = _as_bool(water_cfg.get("apply_to_hf", True))
    apply_to_nf = _as_bool(water_cfg.get("apply_to_nf", True))

    base_water_peak_amp = tf.cast(water_int_factor * hf_params["max_peak_amp"], tf.complex64) * water_enabled_c

    # NOTE: we can multiply by *0.0, which disabled water completely, do it in JSON via enabled_prob=0.0.
    water_params = {
        "water_peak_amplitude": base_water_peak_amp,
        "water_frequency": water_frequency,
        "water_r2": water_r2,
        "water_phase": nf_params["phase"][0:1],
    }

    def _build_signal(peak_params, vib_level, add_mask, add_window, water_scale_c):
        water_params_local = dict(water_params)
        water_params_local["water_peak_amplitude"] *= water_scale_c

        total_sig, _ = combine_signal_and_vibration(
            frequency=peak_params["frequency"],
            phase=peak_params["phase"],
            r2=peak_params["r2"],
            times=times,
            vibration_scaling=vib_params["vibration_scaling"],
            vib_amps=vib_params["vib_amps"],
            vib_phase=vib_params["vib_phase"],
            vib_frequencies=vib_params["vib_frequencies"],
            vib_time=vib_params["vib_time"],
            vib_r2=vib_params["vib_r2"],
            peak_amplitude_main=peak_params["peak_amplitude_main"],
            vib_level=vib_level,
            r2_inhom=peak_params["r2_inhom"],
            scalar_coupling=peak_params["scalar_coupling"],
            k=-k_val,
            echo_time=config["echo_times"][0],
            water_params=water_params_local,
        )

        total_sig = tf.reshape(total_sig, [total_points])
        if add_mask:
            total_sig *= fid_mask
        if add_window:
            total_sig *= window
        return total_sig

    # --- noise ---
    noise_cfg = _get(acq, "noise", None)
    if noise_cfg is None:
        raise KeyError("acquisition.noise missing in JSON")
    noise_std = build_dist(noise_cfg["std"]).sample(())
    shared = _as_bool(noise_cfg.get("shared_hf_nf", True))

    def _noise_vec(std):
        nr = tf.random.normal((total_points,), stddev=std)
        ni = tf.random.normal((total_points,), stddev=std)
        return tf.complex(nr, ni)

    noise_hf = _noise_vec(noise_std)
    noise_nf = _noise_vec(noise_std if shared else build_dist(noise_cfg["std"]).sample(()))

    # --- water scaling per channel ---
    # HF: typically 1.0 if enabled, or 0.0 if apply_to_hf is false
    hf_water_scale = tf.cast(1.0 if apply_to_hf else 0.0, tf.complex64)

    # NF: either 0.0 (disabled), or sampled scale
    nf_water_scale = water_nf_scale if apply_to_nf else tf.cast(0.0, tf.complex64)

    # signals
    hf_fid = _build_signal(hf_params, vib_level=0.0, add_mask=True, add_window=False, water_scale_c=hf_water_scale)
    nf_vib_fid = _build_signal(nf_params, vib_level=1.0, add_mask=True, add_window=False, water_scale_c=nf_water_scale)
    nf_clean_fid = _build_signal(nf_params, vib_level=0.0, add_mask=False, add_window=False, water_scale_c=nf_water_scale)

    hf_fid = hf_fid + noise_hf
    nf_vib_fid = nf_vib_fid + noise_nf

    return hf_fid, nf_vib_fid, nf_clean_fid, times


# -----------------------------------------------------------------------------
# Main generator
# -----------------------------------------------------------------------------
def generator(config):
    # 1) JSON path -> structured
    if isinstance(config, (str, Path)):
        config = load_config(config)

    # 2) structured dict -> flat core dict with "_cfg"
    if isinstance(config, dict) and "_cfg" not in config:
        config = to_flat_core_config(config)

    if not isinstance(config, dict):
        raise TypeError("generator(...) expects a dict, or a JSON path")

    required = [
        "total_points", "field", "gamma_h", "echo_times",
        "n_peaks", "max_couplings", "vib_max", "vib_freq_max",
        "_cfg",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"generator config missing keys: {missing}")

    total_points = config["total_points"]

    # sw_ppm range comes from JSON (no numbers in src)
    sw_lo, sw_hi = config["_cfg"]["spectrum"]["sw_ppm_range"]
    sw_ppm = tf.random.uniform([], minval=sw_lo, maxval=sw_hi, dtype=tf.float32)

    config["_cfg"]["spectrum"]["sw_ppm"] = float(sw_ppm.numpy())
    config["_cfg"]["spectrum"]["sw_hz"] = config["field"] * config["gamma_h"] * sw_ppm

    config["sw_ppm"] = sw_ppm
    config["sw_hz"] = config["_cfg"]["spectrum"]["sw_hz"]

    sw_hz = config["sw_hz"]
    acquisition_time = total_points / sw_hz

    fid_mask, window = create_common_aux_vars(config, total_points, acquisition_time)

    while True:
        # parameter sampling uses structured cfg
        hf_params, nf_params, vib_params = random_parameter_gen_with_HF(config["_cfg"])

        hf_fid, nf_vib_fid, nf_clean_fid, times = generate_all_signals_once(
            config, hf_params, nf_params, vib_params, fid_mask, window
        )

        hf_real = tf_funcs.complex_to_real_reshape(hf_fid)
        nf_vib_real = tf_funcs.complex_to_real_reshape(nf_vib_fid)
        nf_clean_real = tf_funcs.complex_to_real_reshape(nf_clean_fid)

        hf_norm = tf.reduce_max(tf.abs(hf_real))
        hf_real = hf_real / hf_norm
        nf_vib_real = nf_vib_real / hf_norm
        nf_clean_real = nf_clean_real / hf_norm

        hf_real = tf.transpose(hf_real)       # (2, points)
        nf_vib_real = tf.transpose(nf_vib_real)
        nf_clean_real = tf.transpose(nf_clean_real)

        input_signal = tf.stack([hf_real, nf_vib_real], axis=2)  # (2, points, 2)
        target_signal = nf_clean_real                            # (2, points)

        # FFT target
        target_signal_complex = tf.complex(target_signal[:, 0], target_signal[:, 1])
        dc_concat = [target_signal_complex[:1] / 2, target_signal_complex[1:]]
        target_signal_complex = tf.concat(dc_concat, axis=0)

        target_signal_complex_freq = tf.signal.fft(target_signal_complex)
        target_signal_complex_freq = tf.signal.fftshift(target_signal_complex_freq, axes=0)

        target_signal_real = tf.math.real(target_signal_complex_freq)
        target_signal_imag = tf.math.imag(target_signal_complex_freq)

        target_signal = tf.stack([target_signal_real, target_signal_imag], axis=1)
        target_signal = tf.cast(target_signal, tf.float32)

        yield input_signal, target_signal
