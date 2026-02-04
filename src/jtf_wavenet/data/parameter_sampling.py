import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# --- Distribution builders ----------------------------------------------------

def build_dist(spec: dict):
    name = spec["dist"].lower()

    if name == "uniform":
        return tfd.Uniform(low=spec["min"], high=spec["max"])

    if name == "normal":
        return tfd.Normal(loc=spec["mean"], scale=spec["std"])

    if name == "gamma":
        return tfd.Gamma(concentration=spec["concentration"], rate=spec["rate"])

    if name == "randint":
        # inclusive max in JSON; tf.random.uniform uses exclusive max
        lo = int(spec["min"])
        hi = int(spec["max_inclusive"]) + 1
        # Use a simple callable instead of tfd to control dtype easily
        def _sample(shape, dtype=tf.int32):
            return tf.random.uniform(shape, minval=lo, maxval=hi, dtype=dtype)
        return _sample

    raise ValueError(f"Unknown dist: {name}")



def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    return json.loads(path.read_text())

def to_flat_core_config(cfg: dict) -> dict:
    """Project structured JSON into the flat keys generator_core expects."""
    return {
        "total_points": int(cfg["spectrum"]["total_points"]),
        "field": float(cfg["spectrum"]["field_t"]),
        "gamma_h": float(cfg["spectrum"]["gamma_h"]),
        "echo_times": cfg["spectrum"]["echo_times"],

        "n_peaks": int(cfg["peaks"]["n_peaks"]),
        "max_couplings": int(cfg["peaks"]["max_couplings"]),

        "vib_max": int(cfg["vibration"]["vib_max"]),
        "vib_freq_max": float(cfg["vibration"]["vib_freq_max"]),

        # keep full structured cfg for parameter sampling + sw_ppm range
        "_cfg": cfg,
    }



# --- Generator core -----------------------------------------------------------

def random_parameter_gen(config: dict):
    sw_hz = float(config["spectrum"]["sw_hz"])
    n_peaks = int(config["peaks"]["n_peaks"])
    max_couplings = int(config["peaks"]["max_couplings"])

    vib_cfg = config["vibration"]
    vib_max = int(vib_cfg["vib_max"])
    vib_freq_max = float(vib_cfg["vib_freq_max"])

    # --- Frequency generation (same idea, now driven by JSON) ---
    freq_cfg = config["peaks"]["frequency"]
    condensed_fraction = float(freq_cfg["condensed_fraction"])

    n_condensed = int(round(condensed_fraction * n_peaks))
    n_non_condensed = n_peaks - n_condensed

    c_lo, c_hi = freq_cfg["condensed_center_range"]
    condensed_center = tf.random.uniform(
        (n_condensed,), -sw_hz * float(c_hi), sw_hz * float(c_hi), dtype=tf.float32
    )

    s_lo, s_hi = freq_cfg["condensed_spread_range"]
    condensed_spread = tf.random.uniform((n_non_condensed,), s_lo, s_hi, dtype=tf.float32)

    condensed_frequencies = tf.random.normal(
        (n_condensed,), mean=condensed_center, stddev=condensed_spread * sw_hz
    )

    non_std = float(freq_cfg["non_condensed_std_fraction"]) * sw_hz
    non_condensed_frequencies = tf.random.normal((n_non_condensed,), mean=0.0, stddev=non_std)

    combined = tf.concat([condensed_frequencies, non_condensed_frequencies], axis=0)
    frequency = tf.math.tanh((2.0 / sw_hz) * combined) * (sw_hz / 2.0)
    frequency = tf.random.shuffle(frequency)
    frequency = tf.cast(frequency, tf.complex64)

    # --- R2 ---
    r2_cfg = config["peaks"]["r2"]
    use_constant = tf.random.uniform((), 0.0, 1.0) < float(r2_cfg["use_constant_prob"])

    const_dist = build_dist(r2_cfg["constant_value"])
    constant_r2_value = const_dist.sample(())  # scalar

    r2_sel = tf.random.uniform((n_peaks,), 0.0, 1.0)
    uniform_mask = r2_sel < float(r2_cfg["uniform_fraction"])

    n_uni = tf.reduce_sum(tf.cast(uniform_mask, tf.int32))
    n_gam = n_peaks - n_uni

    uni_dist = build_dist(r2_cfg["uniform"])
    gam_dist = build_dist(r2_cfg["gamma"])

    r2_uni = uni_dist.sample((n_uni,))
    r2_gam = gam_dist.sample((n_gam,))
    r2_random = tf.random.shuffle(tf.concat([r2_uni, r2_gam], axis=0))

    r2_vals = tf.where(use_constant, tf.fill([n_peaks], constant_r2_value), r2_random)
    r2_vals = tf.cast(r2_vals, tf.complex64)

    # --- r2_inhom ---
    r2i_dist = build_dist(config["peaks"]["r2_inhom"]["scalar"])
    r2_inhom_scalar = r2i_dist.sample(())
    r2_inhom_vals = tf.fill([n_peaks], tf.cast(r2_inhom_scalar, tf.complex64))

    # --- Couplings ---
    coup_cfg = config["peaks"]["couplings"]
    n_coup_sampler = build_dist(coup_cfg["n_couplings"])
    n_couplings = n_coup_sampler((n_peaks,), dtype=tf.int32)

    mask = tf.sequence_mask(n_couplings, maxlen=max_couplings, dtype=tf.float32)

    j_dist = build_dist(coup_cfg["j_hz"])
    couplings = j_dist.sample((n_peaks, max_couplings))
    scalar_coupling = tf.cast(couplings * mask, tf.complex64)

    # --- Amplitude ---
    amp_cfg = config["peaks"]["amplitude"]
    raw_amp = build_dist(amp_cfg["raw"]).sample((n_peaks,))
    nspin_sampler = build_dist(amp_cfg["nspin"])
    nspin = nspin_sampler((1,), dtype=tf.int32)

    selector = tf.concat(
        [tf.ones(nspin, tf.float32), tf.zeros((n_peaks - nspin[0],), tf.float32)], axis=0
    )
    selector = tf.random.shuffle(selector)
    hf_amp = raw_amp * selector

    nf_frac = build_dist(amp_cfg["nf_fraction"]).sample((n_peaks,))
    zero_out_fraction = build_dist(amp_cfg["zero_out_fraction"]).sample(())
    n_zeros = tf.cast(zero_out_fraction * tf.cast(n_peaks, tf.float32), tf.int32)

    dropout_mask = tf.random.shuffle(
        tf.concat([tf.ones((n_peaks - n_zeros,), tf.float32), tf.zeros((n_zeros,), tf.float32)], axis=0)
    )
    nf_amp = hf_amp * (nf_frac * dropout_mask)

    # --- Phase ---
    ph_cfg = config["peaks"]["phase"]
    p0_std = float(ph_cfg["phase0_deg_std"]) * np.pi / 180.0
    p1_std = float(ph_cfg["phase1_deg_std"]) * np.pi / 180.0

    phase_0 = tf.random.normal((1,), stddev=p0_std, dtype=tf.float32)
    phase_1 = tf.random.normal((1,), stddev=p1_std, dtype=tf.float32)
    phase = tf.cast(phase_0, tf.complex64) + tf.cast(phase_1, tf.complex64) * (
        frequency / tf.cast(sw_hz, tf.complex64)
    )

    peak_params = {
        "frequency": frequency,
        "phase": phase,
        "r2": r2_vals,
        "r2_inhom": r2_inhom_vals,
        "scalar_coupling": scalar_coupling,
        "peak_amplitude_main": tf.cast(nf_amp, tf.complex64),
        "hf_amp": hf_amp,
        "max_peak_amp": tf.reduce_max(hf_amp),
    }

    # --- Vibrations ---
    vib_scaling = build_dist(vib_cfg["vibration_scaling"]).sample((1,))
    vibration_scaling = tf.cast(vib_scaling, tf.complex64)

    vib_n_sampler = build_dist(vib_cfg["vib_freq_number"])
    vib_freq_number = vib_n_sampler((1,), dtype=tf.int32)

    vib_amp_dist = build_dist(vib_cfg["vib_amp"])
    vib_amps = tf.cast(vib_amp_dist.sample((vib_max,)), tf.complex64)

    selector_vib = tf.random.shuffle(
        tf.concat([tf.ones(vib_freq_number, tf.complex64), tf.zeros((vib_max - vib_freq_number[0],), tf.complex64)], axis=0)
    )
    vib_amps = vib_amps * selector_vib

    vib_freqs = tf.random.uniform((vib_max,), -vib_freq_max, vib_freq_max, dtype=tf.float32)
    vib_freqs = tf.cast(vib_freqs, tf.complex64)

    vib_r2 = tf.cast(build_dist(vib_cfg["vib_r2"]).sample((vib_max,)), tf.complex64)

    vib_params = {
        "vib_amps": vib_amps,
        "vib_phase": tf.constant(0.0, tf.complex64),
        "vib_frequencies": vib_freqs,
        "vib_time": tf.constant(0.0, tf.complex64),
        "vib_r2": vib_r2,
        "vibration_scaling": vibration_scaling,
    }

    return peak_params, vib_params


def random_parameter_gen_with_HF(config: dict):
    nf_peak_params, vib_params = random_parameter_gen(config)
    hf_amp = nf_peak_params.pop("hf_amp")

    sw_hz = float(config["spectrum"]["sw_hz"])
    hf_shift = build_dist(config["peaks"]["hf"]["freq_shift_hz"]).sample(())
    shift = tf.cast(hf_shift, tf.complex64)

    # separate HF phase
    ph_cfg = config["peaks"]["phase"]
    p0_std = float(ph_cfg["phase0_deg_std"]) * np.pi / 180.0
    p1_std = float(ph_cfg["phase1_deg_std"]) * np.pi / 180.0

    hf_phase_0 = tf.random.normal((1,), stddev=p0_std, dtype=tf.float32)
    hf_phase_1 = tf.random.normal((1,), stddev=p1_std, dtype=tf.float32)

    new_hf_phase = tf.cast(hf_phase_0, tf.complex64) + tf.cast(hf_phase_1, tf.complex64) * (
        nf_peak_params["frequency"] / tf.cast(sw_hz, tf.complex64)
    )

    hf_peak_params = {
        "frequency": nf_peak_params["frequency"] + shift,
        "phase": new_hf_phase,
        "r2": nf_peak_params["r2"],
        "r2_inhom": nf_peak_params["r2_inhom"],
        "scalar_coupling": nf_peak_params["scalar_coupling"],
        "peak_amplitude_main": tf.cast(hf_amp, tf.complex64),
        "max_peak_amp": tf.reduce_max(hf_amp),
    }
    return hf_peak_params, nf_peak_params, vib_params
