#!/usr/bin/env python3
"""
scripts/eval.py

Unified evaluation + plotting for JTF-WaveNet.

Features
- Loads latest checkpoint (stage1 or stage2) and runs inference on one batch.
- Saves arrays + figures into runs/<timestamp>/{arrays,plots}
-  FFT helpers: make_ft_inp / make_ft_pt / make_ft_err
- Soft/Hard binning switch for sigma calibration + per-bin z hist grid

Run
  pip install -e .
  python scripts/eval.py --stage stage2 --binning soft --ckpt checkpoints/

Or (without editable install):
  PYTHONPATH=src python scripts/eval.py --stage stage2 --binning soft --ckpt checkpoints/stage2
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.stats import norm as _norm

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp  # for hard σ-quantile binning
from jtf_wavenet.data.parameter_sampling import load_config


# =====================
# Optional repo style
# =====================
def _apply_repo_style():
    """
    Tries to apply project style from jtf_wavenet.vis.style.
    Supports common patterns:
      - apply_style()
      - use_style()
      - set_style()
      - style() / configure()
    If none exist, uses a safe matplotlib default.
    """
    try:
        from jtf_wavenet.vis import style as vis_style  # type: ignore

        for fn_name in ("apply_style", "use_style", "set_style", "style", "configure"):
            if hasattr(vis_style, fn_name):
                getattr(vis_style, fn_name)()
                return
        # If module import works but no known entrypoint, do nothing.
        return
    except Exception:
        # fallback style
        plt.rcParams["text.usetex"] = False
        plt.rcParams["svg.fonttype"] = "none"
        return


_apply_repo_style()


# =====================
# Project imports
# =====================
from jtf_wavenet.data.generator_core import generator
from jtf_wavenet.model.builders import build_jtfwavenet
from jtf_wavenet.model.jtf_wavenet import JTFWaveNetConfig
from jtf_wavenet.training.callbacks import CustomLearningSchedule, get_checkpoint_objects

# Optional: if your model has loss_total
# from jtf_wavenet.losses.sbece import SmoothBinnedECE
# from jtf_wavenet.losses.xi_momenta import XiMomentaBase


# =====================
# Palette + constants
# =====================
COL_BLUE = (0.42, 0.60, 0.85)  # HF
COL_GREEN = (0.40, 0.75, 0.40)  # Vib
COL_PURPLE = (0.65, 0.50, 0.82)  # Pred
COL_GREY = (0.25, 0.25, 0.25)  # Target
COL_BAND = COL_PURPLE

RES_OFFSET = -5.0  # residual/band offset in panel (a)


def colorize_legend(legend_obj):
    for handle, txt in zip(legend_obj.legend_handles, legend_obj.get_texts()):
        col = None
        if hasattr(handle, "get_color"):
            col = handle.get_color()
        elif hasattr(handle, "get_facecolor"):
            fc = handle.get_facecolor()
            if fc is not None:
                fc = np.array(fc)
                col = tuple(fc[0]) if fc.ndim > 1 else tuple(np.array(fc).flatten())
        if isinstance(col, (list, tuple, np.ndarray)):
            txt.set_color(tuple(np.array(col).ravel()))


# =====================
# Config
# =====================
@dataclass
class EvalConfig:
    POINTS: int = 512 * 8
    BATCH_SIZE: int = 10
    CHECKPOINT_PATH: str = "checkpoints"
    RUNS_DIR: str = "runs"
    INTERACTIVE: bool = True
    ASK_TO_SAVE: bool = False

    MODEL_STAGE: str = "stage2"  # {"stage1","stage2"}
    BINNING_MODE: str = "soft"  # {"soft","hard"}

    HIST_RANGE: Tuple[float, float] = (-20.0, 20.0)
    HIST_BINS: int = 100
    BINS_RMSD: int = 10
    BINS_HIST: int = 10
    TAU: float = 0.05
    EPS: float = 1e-12


# =====================
# Utilities
# =====================
def cm_to_inches(cm: float) -> float:
    return cm * 0.393701


def make_run_dirs(base: str) -> Dict[str, Path]:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = Path(base) / ts
    plots = run / "plots"
    arrays = run / "arrays"
    plots.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)
    return {"run": run, "plots": plots, "arrays": arrays}


def save_json_obj(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2))


def ask_to_save(cfg: EvalConfig, fig: plt.Figure, default_name: str, plots_dir: Path) -> None:
    if cfg.INTERACTIVE:
        plt.show(block=True)

    if not cfg.ASK_TO_SAVE:
        plt.close(fig)
        return

    ans = input(f"Save as '{default_name}.svg'? [y/N or custom-name]: ").strip()
    if ans.lower() in ("y", "yes", ""):
        fn = f"{default_name}.svg"
    elif ans.lower() in ("n", "no"):
        plt.close(fig)
        return
    else:
        fn = ans if ans.endswith(".svg") else f"{ans}.svg"

    out = plots_dir / fn
    fig.savefig(out, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def create_dataset(points: int, batch_size: int, gen_config_path: str) -> tf.data.Dataset:
    output_sig = (
        tf.TensorSpec(shape=(points, 2, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(points, 2), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: generator(gen_config_path),  # generator_core can load JSON path now
        output_signature=output_sig,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()


def build_model(points: int) -> keras.Model:
    dilations = np.geomspace(1, 100, num=17, dtype=int)
    cfg = JTFWaveNetConfig(
        points=points,
        filter_count=48,
        dilations=tuple(int(d) for d in dilations),
        blocks=3,
        convolution_kernel=(4, 2),
        separate_activation=True,
        use_dropout=False,
        use_custom_padding=True,
        scale_factor_ft=1.0,
        initial_factor=1.0,
    )
    model = build_jtfwavenet(cfg=cfg)
    model.build(input_shape=(None, points, 2, 2))
    return model


def load_model(cfg: EvalConfig) -> keras.Model:
    schedule = CustomLearningSchedule(warmup_steps=20000, d_model=10)
    opt = keras.optimizers.legacy.Adam(learning_rate=schedule)

    model = build_model(cfg.POINTS)

    ckpt, mgr = get_checkpoint_objects(model, opt, cfg.CHECKPOINT_PATH)
    if not mgr.latest_checkpoint:
        raise RuntimeError(f"No checkpoint found in: {cfg.CHECKPOINT_PATH}")
    ckpt.restore(mgr.latest_checkpoint).expect_partial()
    print(f"Restored from {mgr.latest_checkpoint}")
    return model


def extract_mean_sigma(pred: np.ndarray):
    """Split model output [B,P,2,2] -> (mean[B,P,2], sigma[B,P,2])."""
    return pred[..., 0], pred[..., 1]


# =====================
# Your EXACT FFT helpers
# =====================
def make_ft_inp(inp, spec):
    points = inp.shape[0]
    sw_ppm = float(spec["sw_ppm"]) if "sw_ppm" in spec else float(spec["sw_ppm_range"][1])
    sw_hz = float(spec["field_t"]) * float(spec["gamma_h"]) * sw_ppm
    acquisition_time = points / sw_hz
    times = np.linspace(0.0, acquisition_time, points)
    real = inp[:, 0]
    imag = inp[:, 1]
    total = tf.complex(real, imag).numpy()
    window = np.exp(-10 * times)
    windowed_signal = total * window
    _ = np.pad(windowed_signal, (0, points), "constant")
    windowed_signal[0] /= 2
    return np.fft.fftshift(np.fft.fft(windowed_signal))


def make_ft_pt(inp, spec):
    points = inp.shape[0]
    sw_ppm = float(spec["sw_ppm"]) if "sw_ppm" in spec else float(spec["sw_ppm_range"][1])
    sw_hz = float(spec["field_t"]) * float(spec["gamma_h"]) * sw_ppm
    acquisition_time = points / sw_hz
    times = np.linspace(0.0, acquisition_time, points)

    real = inp[:, 0]
    imag = inp[:, 1]
    total = tf.complex(real, imag).numpy()
    total_ifft = np.fft.ifft(total)
    window = np.exp(-10 * times)
    windowed_signal = total_ifft * window
    _ = np.pad(windowed_signal, (0, points), "constant")
    return np.fft.fft(windowed_signal)

def make_ft_err(inp, spec):
    points = inp.shape[0]
    sw_ppm = float(spec["sw_ppm"]) if "sw_ppm" in spec else float(spec["sw_ppm_range"][1])
    sw_hz = float(spec["field_t"]) * float(spec["gamma_h"]) * sw_ppm
    acquisition_time = points / sw_hz
    times = np.linspace(0.0, acquisition_time, points)
    real = inp[:, 0]
    imag = inp[:, 1]
    total = tf.complex(real, imag).numpy()
    _ = np.fft.ifft(total)
    _ = np.exp(-10 * np.linspace(0.0, acquisition_time, points))
    _ = np.pad(total, (0, points), "constant")
    return total  # keep your behavior


# =====================
# Binning back-ends
# =====================
def _softbin_like_training_safe(
    cfg: EvalConfig, target, pred_mean, pred_sigma, num_bins: int, tau: float
):
    dtype = pred_sigma.dtype
    eps = tf.cast(cfg.EPS, dtype)

    t_r, t_i = tf.unstack(target, axis=-1)
    m_r, m_i = tf.unstack(pred_mean, axis=-1)
    s_r, s_i = tf.unstack(pred_sigma, axis=-1)

    t_cat = tf.concat([t_r, t_i], axis=1)
    m_cat = tf.concat([m_r, m_i], axis=1)
    s_cat = tf.concat([s_r, s_i], axis=1)

    diff = tf.reshape(t_cat - m_cat, (-1, 1))
    sigma = tf.reshape(tf.maximum(s_cat, eps), (-1, 1))
    z = diff / sigma
    z = tf.where(tf.math.is_finite(z), z, tf.zeros_like(z))

    residuals = tf.abs(diff)
    residuals = tf.where(tf.math.is_finite(residuals), residuals, tf.zeros_like(residuals))

    rmax = tf.reduce_max(residuals)
    rmax = tf.where(tf.math.is_finite(rmax) & (rmax > 0), rmax, tf.cast(1.0, dtype))

    centers = tf.linspace(tf.cast(0.0, dtype), tf.cast(1.0, dtype), num_bins)[tf.newaxis, :] * rmax
    T = tf.cast(tau, dtype) * tf.maximum(rmax * rmax, eps)

    g = -tf.square(residuals - centers) / T
    W = tf.nn.softmax(g, axis=1)
    counts = tf.reduce_sum(W, axis=0, keepdims=True) + eps
    return z, residuals, W, counts, centers


def _hardbin_like_training(target, pred_mean, pred_sigma, num_bins: int):
    eps = tf.constant(1e-12, pred_sigma.dtype)

    t_r, t_i = tf.unstack(target, axis=-1)
    m_r, m_i = tf.unstack(pred_mean, axis=-1)
    s_r, s_i = tf.unstack(pred_sigma, axis=-1)

    t = tf.concat([t_r, t_i], axis=1)
    m = tf.concat([m_r, m_i], axis=1)
    s = tf.concat([s_r, s_i], axis=1)

    diff = tf.reshape(t - m, (-1, 1))
    sigm = tf.reshape(tf.maximum(s, eps), (-1, 1))
    z = diff / sigm
    z = tf.where(tf.math.is_finite(z), z, tf.zeros_like(z))
    residual = tf.abs(diff)

    edges = tfp.stats.quantiles(sigm, num_bins + 1, axis=0, interpolation="linear")
    lo, hi = edges[:-1], edges[1:]

    in_bin = (sigm >= tf.transpose(lo)) & (sigm < tf.transpose(hi))
    last = in_bin[:, -1] | (sigm[:, 0] >= hi[-1, 0])
    in_bin = tf.concat([in_bin[:, :-1], last[:, None]], axis=1)
    in_bin = tf.cast(in_bin, sigm.dtype)

    counts = tf.reduce_sum(in_bin, axis=0, keepdims=True) + eps
    return z, residual, sigm, in_bin, counts


# =====================
# Plots
# =====================
def plot_sigma_calibration(
    cfg: EvalConfig, target, pred_mean, pred_sigma, ax, *, mode: str, bins_rmsd: int
):
    mode = mode.lower()

    t_tf = tf.convert_to_tensor(target)
    m_tf = tf.convert_to_tensor(pred_mean)
    s_tf = tf.convert_to_tensor(pred_sigma)

    dtype = s_tf.dtype
    eps = tf.cast(1e-12, dtype)

    if mode == "hard":
        _, residual, sigm, in_bin, _ = _hardbin_like_training(t_tf, m_tf, s_tf, num_bins=bins_rmsd)

        res_tiles = tf.tile(residual, [1, tf.shape(in_bin)[1]])
        sum_res = tf.reduce_sum(res_tiles * in_bin, axis=0, keepdims=True)

        sig_tiles = tf.tile(tf.maximum(sigm, eps), [1, tf.shape(in_bin)[1]])
        sum_sig = tf.reduce_sum(sig_tiles * in_bin, axis=0, keepdims=True)

        x = sum_sig.numpy().ravel()
        y = sum_res.numpy().ravel()

    elif mode == "soft":
        _, residuals, W, _, _ = _softbin_like_training_safe(
            cfg, t_tf, m_tf, s_tf, num_bins=bins_rmsd, tau=cfg.TAU
        )

        sum_res = tf.reduce_sum(W * residuals, axis=0, keepdims=True)

        s_r, s_i = tf.unstack(s_tf, axis=-1)
        s_cat = tf.concat([s_r, s_i], axis=1)
        sig_vec = tf.reshape(tf.maximum(s_cat, eps), (-1, 1))

        sum_sig = tf.reduce_sum(W * sig_vec, axis=0, keepdims=True)

        x = (sum_sig.numpy().ravel()) / cfg.POINTS
        y = (sum_res.numpy().ravel()) / cfg.POINTS

    else:
        raise ValueError("mode must be 'soft' or 'hard'")

    xy_min = float(np.nanmin([x.min(), y.min()]))
    xy_max = float(np.nanmax([x.max(), y.max()]))
    ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="-", label="x = y")
    ax.scatter(x, y, alpha=0.9)
    ax.set_xlabel("Per-bin ∑σ")
    ax.set_ylabel("Per-bin ∑|residual|")
    ax.legend(frameon=False)


def plot_z_hist_grid(
    cfg: EvalConfig, target, pred_mean, pred_sigma, ax_host, *, mode: str, bins_hist: int
):
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    mode = mode.lower()
    if mode == "hard":
        z_tf, _, _, in_bin_tf, _ = _hardbin_like_training(
            tf.convert_to_tensor(target),
            tf.convert_to_tensor(pred_mean),
            tf.convert_to_tensor(pred_sigma),
            num_bins=bins_hist,
        )
        z = z_tf.numpy().ravel()
        in_bin = in_bin_tf.numpy()

        def bin_mask(b):
            return in_bin[:, b].astype(bool)

    elif mode == "soft":
        z_tf, _, W_tf, _, _ = _softbin_like_training_safe(
            cfg,
            tf.convert_to_tensor(target),
            tf.convert_to_tensor(pred_mean),
            tf.convert_to_tensor(pred_sigma),
            num_bins=bins_hist,
            tau=cfg.TAU,
        )
        z = z_tf.numpy().ravel()
        W = W_tf.numpy()

        def bin_mask(b, thr=1e-6):
            return W[:, b] > thr

    else:
        raise ValueError("mode must be 'soft' or 'hard'")

    cols = int(np.ceil(np.sqrt(bins_hist)))
    rows = int(np.ceil(bins_hist / cols))
    subgrid = GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=ax_host.get_subplotspec(), wspace=0.15, hspace=0.25
    )

    lo, hi = cfg.HIST_RANGE
    edges_hist = np.linspace(lo, hi, cfg.HIST_BINS + 1)
    x_line = np.linspace(lo, hi, 800)
    ideal_pdf = _norm.pdf(x_line, 0.0, 1.0)

    b = 0
    for r in range(rows):
        for c in range(cols):
            ax = ax_host.figure.add_subplot(subgrid[r, c])
            if b < bins_hist:
                zz = z[bin_mask(b)]
                if zz.size:
                    mu = float(np.mean(zz))
                    sd = float(np.std(zz, ddof=1)) if zz.size > 1 else 0.0
                    ax.hist(zz, bins=edges_hist, density=True, alpha=0.75)
                    ax.plot(x_line, ideal_pdf, linestyle="--", linewidth=1.0)
                    ax.set_xlim(lo, hi)
                    ax.set_title(
                        f"bin {b} | n={zz.size} | μ≈{mu:.2f}, σ≈{sd:.2f}", fontsize=8, pad=2
                    )
                else:
                    ax.hist([], bins=edges_hist)
                    ax.set_title(f"bin {b} (empty)", fontsize=8, pad=2)

                if r < rows - 1:
                    ax.tick_params(labelbottom=False)
                if c > 0:
                    ax.tick_params(labelleft=False)
            else:
                ax.axis("off")
            b += 1

    ax_host.set_frame_on(False)
    ax_host.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    title = "Per-bin z histograms (" + (
        "hard σ-quantiles)" if mode == "hard" else "soft residual bins)"
    )
    ax_host.set_title(title, fontsize=12, pad=6)


def plot_batch_normalized_errors_histogram(cfg: EvalConfig, predictions, targets):
    mean, sigma = extract_mean_sigma(predictions)
    err = (targets[..., 0] - mean[..., 0]) / np.maximum(sigma[..., 0], 1e-6)
    err = err[np.isfinite(err)]

    mu, sd = _norm.fit(err) if err.size else (0.0, 1.0)
    x = np.linspace(cfg.HIST_RANGE[0], cfg.HIST_RANGE[1], 1000)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.hist(err, bins=cfg.HIST_BINS, range=cfg.HIST_RANGE, density=True, alpha=0.6, label="Errors")
    ax.plot(x, _norm.pdf(x, mu, sd), lw=2, label=f"Fit: μ={mu:.2f}, σ={sd:.2f}")
    ax.plot(x, _norm.pdf(x, 0, 1), lw=2, linestyle="--", label="Ideal N(0,1)")
    ax.set_xlabel("Normalized Error (target − pred) / σ  (real)")
    ax.set_ylabel("Density")
    ax.set_title("Normalized Error Histogram (Batch)")
    ax.legend(frameon=False)
    return fig

def plot_batch_normalized_errors_histogram_on_ax(cfg: EvalConfig, predictions, targets, ax):
    mean, sigma = extract_mean_sigma(predictions)

    # real-part normalized error
    err = (targets[..., 0] - mean[..., 0]) / np.maximum(sigma[..., 0], 1e-6)
    err = err[np.isfinite(err)]

    ax.hist(
        err,
        bins=cfg.HIST_BINS,
        range=cfg.HIST_RANGE,
        density=True,
        alpha=0.65,
        label="Batch errors",
    )

    # optional: ideal N(0,1) overlay (uncomment if you want)
    x = np.linspace(cfg.HIST_RANGE[0], cfg.HIST_RANGE[1], 1000)
    ax.plot(x, _norm.pdf(x, 0, 1), lw=2, linestyle="--", label="Ideal N(0,1)")

    ax.set_title("Normalized Error Histogram (Batch)", fontsize=12, fontfamily="Arial")
    # ax.set_xlabel("Normalized Error (target − pred) / σ  (real)", fontsize=12, fontfamily="Arial")
    ax.set_ylabel("Density", fontsize=12, fontfamily="Arial")
    ax.legend(frameon=False)


def plot_4panel_mosaic_one(
    cfg: EvalConfig,
    current_input_hf,
    current_input_vib,
    current_target,
    current_prediction,
    predictions_batch,
    targets_batch,
    *,
    spec: dict,
):
    fig, ax = plt.subplot_mosaic(
        [["(a)", "(a)", "(c)"], ["(b)", "(b)", "(d)"]],
        layout="constrained",
        figsize=(cm_to_inches(32), cm_to_inches(18)),
    )

    ax["(b)"].sharex(ax["(a)"])
    ax["(a)"].tick_params(labelbottom=False)

    for label, ax_i in ax.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax_i.text(
            0.0,
            1.0,
            label,
            transform=ax_i.transAxes + trans,
            fontsize=12,
            va="bottom",
            fontfamily="Arial",
        )

    inp_ft_hf = make_ft_inp(current_input_hf, spec=spec)
    inp_ft_vib = make_ft_inp(current_input_vib, spec=spec)
    targ_ft = make_ft_pt(current_target, spec=spec)
    pred_ft = make_ft_pt(current_prediction[:, :, 0], spec=spec)

    ax["(a)"].plot(np.real(pred_ft), label="Prediction", color=COL_PURPLE, lw=5.2, alpha=0.55)
    ax["(a)"].plot(np.real(inp_ft_hf), label="High Field", color="red", lw=1.0, alpha=1.0)
    ax["(a)"].plot(np.real(inp_ft_vib), label="Vibrated", color=COL_GREEN, lw=2.2, alpha=1.0)
    ax["(a)"].plot(np.real(targ_ft), label="Target", color="blue", lw=1.3, alpha=1.0)

    residual_fd = np.real(targ_ft) - np.real(pred_ft)
    pred_std_fd = np.real(make_ft_err(current_prediction[:, :, 1], spec=spec))

    x_freq = np.arange(residual_fd.shape[0])
    band_top = RES_OFFSET + pred_std_fd
    band_bot = RES_OFFSET - pred_std_fd

    ax["(a)"].fill_between(
        x_freq, band_bot, band_top, color=COL_BAND, alpha=0.22, label="±σ (offset)"
    )
    ax["(a)"].plot(
        x_freq,
        RES_OFFSET + residual_fd,
        color=COL_PURPLE,
        lw=1.2,
        alpha=0.95,
        label="Residual (offset)",
    )

    ax["(a)"].set_ylabel("Amplitude (a.u.)", fontsize=12, fontfamily="Arial")
    ax["(a)"].set_xlabel("Frequency (arb.)", fontsize=12, fontfamily="Arial")
    legA = ax["(a)"].legend(
        frameon=False, fontsize=12, prop={"family": "Arial"}, ncol=3, loc="upper right"
    )
    colorize_legend(legA)

    ax["(a)"].text(
        0.01,
        0.02,
        f"Residual & band offset: {RES_OFFSET:+.2f} a.u.",
        transform=ax["(a)"].transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=COL_PURPLE,
    )

    ax["(b)"].set_axis_off()

    pred_mean = current_prediction[:, 0, 0]
    pred_std = np.maximum(current_prediction[:, 0, 1], 1e-12)
    norm_err = (current_target[:, 0] - pred_mean) / pred_std
    norm_err = norm_err[np.isfinite(norm_err)]

    # x_hist = np.linspace(cfg.HIST_RANGE[0], cfg.HIST_RANGE[1], 1000)
    # ax["(c)"].hist(
    #     norm_err, bins=cfg.HIST_BINS, range=cfg.HIST_RANGE, density=True, alpha=0.65, label="Errors"
    # )
    # ax["(c)"].plot(x_hist, _norm.pdf(x_hist, 0, 1), color=COL_PURPLE, lw=2, label="Ideal N(0,1)")
    # ax["(c)"].legend(frameon=False, fontsize=12, prop={"family": "Arial"})
    # ax["(c)"].set_xlabel("Normalized error", fontsize=12, fontfamily="Arial")
    # ax["(c)"].set_ylabel("Density", fontsize=12, fontfamily="Arial")
    #ax c should be batch normalized error histogram
    plot_batch_normalized_errors_histogram_on_ax(cfg, predictions_batch, targets_batch, ax["(c)"])




    pred_mean_batch, pred_sigma_batch = extract_mean_sigma(predictions_batch)
    plot_sigma_calibration(
        cfg,
        targets_batch,
        pred_mean_batch,
        pred_sigma_batch,
        ax=ax["(d)"],
        mode=cfg.BINNING_MODE,
        bins_rmsd=cfg.BINS_RMSD,
    )
    ax["(d)"].set_xlabel("Prediction Uncertainty", fontsize=12, fontfamily="Arial")
    ax["(d)"].set_ylabel("RMSD proxy", fontsize=12, fontfamily="Arial")

    for ax_i in ax.values():
        ax_i.tick_params(axis="both", labelsize=12)

    return fig


# =====================
# CLI + main
# =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-config", default="configs/default_generator.json",
               help="Path to generator JSON config")
    p.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    p.add_argument("--binning", choices=["soft", "hard"], default="soft")
    p.add_argument("--ckpt", default="checkpoints", help="Checkpoint directory")
    p.add_argument("--runs", default="runs", help="Runs output directory")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--points", type=int, default=None)
    p.add_argument("--no-interactive", action="store_true")
    p.add_argument("--ask-save", action="store_true")
    p.add_argument("--bins-rmsd", type=int, default=20)
    p.add_argument("--bins-hist", type=int, default=10)
    p.add_argument("--hist-bins", type=int, default=100)
    p.add_argument("--hist-lo", type=float, default=-10.0)
    p.add_argument("--hist-hi", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.05)
    return p.parse_args()


def main():

    args = parse_args()

    gen_cfg = load_config(args.gen_config)
    # save_json_obj(gen_cfg, dirs["run"] / "generator_config.json")
    spec = gen_cfg["spectrum"]


    json_points = int(gen_cfg["spectrum"]["total_points"])
    if args.points is None:
        args.points = json_points
    elif int(args.points) != json_points:
        raise ValueError(f"--points={args.points} but JSON total_points={json_points} (must match)")

    cfg = EvalConfig(
        POINTS=args.points,
        BATCH_SIZE=args.batch,
        CHECKPOINT_PATH=args.ckpt,
        RUNS_DIR=args.runs,
        INTERACTIVE=not args.no_interactive,
        ASK_TO_SAVE=args.ask_save,
        MODEL_STAGE=args.stage,
        BINNING_MODE=args.binning,
        HIST_RANGE=(args.hist_lo, args.hist_hi),
        HIST_BINS=args.hist_bins,
        BINS_RMSD=args.bins_rmsd,
        BINS_HIST=args.bins_hist,
        TAU=args.tau,
    )

    if not cfg.INTERACTIVE:
        matplotlib.use("Agg")

    dirs = make_run_dirs(base=cfg.RUNS_DIR)
    run_cfg = {
        "points": cfg.POINTS,
        "batch_size": cfg.BATCH_SIZE,
        "hist_range": cfg.HIST_RANGE,
        "hist_bins": cfg.HIST_BINS,
        "bins_rmsd": cfg.BINS_RMSD,
        "bins_hist": cfg.BINS_HIST,
        "tau": cfg.TAU,
        "binning_mode": cfg.BINNING_MODE,
        "model_stage": cfg.MODEL_STAGE,
        "checkpoint_path": cfg.CHECKPOINT_PATH,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_json_obj(run_cfg, dirs["run"] / "run_config.json")

    model = load_model(cfg)
    ds = create_dataset(cfg.POINTS, cfg.BATCH_SIZE, args.gen_config)
    it = iter(ds)
    inputs, targets = next(it)

    preds = model(inputs, training=False, stage=cfg.MODEL_STAGE).numpy()

    np.save(dirs["arrays"] / "inputs.npy", inputs.numpy())
    np.save(dirs["arrays"] / "targets.npy", targets.numpy())
    np.save(dirs["arrays"] / "predictions.npy", preds)

    # If stage1: sigma is random -> skip sigma-dependent diagnostics by default
    sigma_ok = cfg.MODEL_STAGE == "stage2"

    # fig_hist = plot_batch_normalized_errors_histogram(cfg, preds, targets.numpy())
    # ask_to_save(cfg, fig_hist, "batch_normalized_errors_hist", dirs["plots"])

    # if sigma_ok:
    #     mean_b, sigma_b = extract_mean_sigma(preds)
    #     fig_z = plt.figure(figsize=(cm_to_inches(32), cm_to_inches(20)), constrained_layout=True)
    #     ax_host = fig_z.add_subplot(111)
    #     plot_z_hist_grid(
    #         cfg,
    #         targets.numpy(),
    #         mean_b,
    #         sigma_b,
    #         ax_host=ax_host,
    #         mode=cfg.BINNING_MODE,
    #         bins_hist=cfg.BINS_HIST,
    #     )
    #     ask_to_save(cfg, fig_z, "z_hist_grid", dirs["plots"])
    # else:
    #     print(
    #         "Stage1 selected: skipping sigma calibration + z-hist grid (sigma is synthetic in stage1)."
    #     )

    for i in range(min(cfg.BATCH_SIZE, inputs.shape[0])):
        inp_hf = inputs[i, :, :, 0].numpy()
        inp_vib = inputs[i, :, :, 1].numpy()
        targ = targets[i, :, :].numpy()
        pred = preds[i, :, :, :]

        fig_combo = plot_4panel_mosaic_one(cfg, inp_hf, inp_vib, targ, pred, preds, targets.numpy(), spec=spec)
        ask_to_save(cfg, fig_combo, f"combined_4panel_{i:02d}", dirs["plots"])

    print(f"\nSaved arrays to: {dirs['arrays']}")
    print(f"Saved plots  to: {dirs['plots']}")


if __name__ == "__main__":
    main()
