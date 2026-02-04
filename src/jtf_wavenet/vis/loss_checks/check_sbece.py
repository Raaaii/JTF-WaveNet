"""
SB-ECE sanity checks + plots.

Runs SmoothBinnedECE on synthetic data and plots:
  1) Residual magnitude vs sigma (scatter)
  2) A simple histogram comparison (optional)

This is NOT training code — only diagnostics.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from jtf_wavenet.losses.sbece import SmoothBinnedECE


def _make_synthetic_batch(batch_size: int, points: int, seed: int = 0):
    """
    Match your model/loss conventions:
      real: (B, P, 2)          -> RI ground truth
      pred: (B, P, 2, 2)       -> [..., 0]=mean (RI), [..., 1]=sigma (RI-like)
    """
    tf.random.set_seed(seed)

    # Ground-truth RI signal
    real = tf.random.normal((batch_size, points, 2), mean=0.0, stddev=1.0)

    # Make mean close to real + controlled error
    mean = real + 0.1 * tf.random.normal((batch_size, points, 2))

    # Make sigma correlated with |residual| (plus a floor)
    residual = tf.abs(mean - real)
    sigma = 0.05 + 0.5 * residual + 0.05 * tf.random.uniform((batch_size, points, 2))

    pred = tf.stack([mean, sigma], axis=-1)  # (B, P, 2, 2)
    return real, pred


def _plot_residual_vs_sigma(real: tf.Tensor, pred: tf.Tensor, title: str):
    mean = pred[..., 0]  # (B,P,2)
    sigma = pred[..., 1]  # (B,P,2)
    residual = tf.abs(mean - real)

    # Flatten everything for a single scatter
    r = tf.reshape(residual, (-1,)).numpy()
    s = tf.reshape(sigma, (-1,)).numpy()

    plt.figure(figsize=(7, 5))
    plt.scatter(r, s, s=5, alpha=0.25)
    plt.xlabel("|mean - real|")
    plt.ylabel("sigma")
    plt.title(title)
    plt.grid(True, alpha=0.3)


def run_check_sbece(
    batch_size: int = 4,
    points: int = 4096,
    num_bins: int = 10,
    temp: float = 0.01,
    seed: int = 0,
    show_plots: bool = True,
):
    real, pred = _make_synthetic_batch(batch_size, points, seed=seed)

    sb_ece = SmoothBinnedECE(weight=1.0, temp=temp, num_bins=num_bins)
    loss_val = sb_ece.compute(real, pred)

    print("\n[SB-ECE CHECK]")
    print(f"  real shape: {real.shape}")
    print(f"  pred shape: {pred.shape}")
    print(f"  SB-ECE loss: {float(loss_val.numpy()):.6g}")

    _plot_residual_vs_sigma(real, pred, title="SB-ECE check: residual vs sigma")

    if show_plots:
        plt.show()

    return loss_val


if __name__ == "__main__":
    run_check_sbece()
