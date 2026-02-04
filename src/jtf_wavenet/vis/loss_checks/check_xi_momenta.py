"""
Xi-momenta sanity checks + lightweight plots.

Runs XiMomentaBase on synthetic data and plots:
  1) histogram of chi = (real-mean)/sigma
  2) prints basic stats
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from jtf_wavenet.losses.xi_momenta import XiMomentaBase


def _make_synthetic_batch(batch_size: int, points: int, seed: int = 0):
    """
    Match your conventions:
      real: (B, P, 2)
      pred: (B, P, 2, 2)  with last dim [mean, sigma]
    """
    tf.random.set_seed(seed)

    real = tf.random.normal((batch_size, points, 2), mean=0.0, stddev=1.0)
    mean = real + 0.15 * tf.random.normal((batch_size, points, 2))

    # sigma positive
    residual = tf.abs(mean - real)
    sigma = 0.05 + 0.7 * residual + 0.05 * tf.random.uniform((batch_size, points, 2))

    pred = tf.stack([mean, sigma], axis=-1)
    return real, pred


def _plot_chi_hist(real: tf.Tensor, pred: tf.Tensor, title: str):
    mean = pred[..., 0]
    sigma = pred[..., 1]
    chi = (real - mean) / (sigma + 1e-8)

    chi_flat = tf.reshape(chi, (-1,)).numpy()

    plt.figure(figsize=(7, 5))
    plt.hist(chi_flat, bins=120, alpha=0.9)
    plt.xlabel("chi = (real - mean) / sigma")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)


def run_check_xi_momenta(
    batch_size: int = 2,
    points: int = 4096,
    xi_weights=(1.0, 1.0, 1.0, 1.0),
    seed: int = 0,
    show_plots: bool = True,
):
    real, pred = _make_synthetic_batch(batch_size, points, seed=seed)

    xi = XiMomentaBase(weight=1.0, xi_momenta_weights=tf.constant(xi_weights, tf.float32))
    loss_val = xi.compute(real, pred)

    mean = pred[..., 0]
    sigma = pred[..., 1]
    chi = (real - mean) / (sigma + 1e-8)

    print("\n[XI-MOMENTA CHECK]")
    print(f"  real shape: {real.shape}")
    print(f"  pred shape: {pred.shape}")
    print(f"  Xi-momenta loss: {float(loss_val.numpy()):.6g}")
    print(f"  chi mean: {float(tf.reduce_mean(chi).numpy()):.6g}")
    print(f"  chi std : {float(tf.math.reduce_std(chi).numpy()):.6g}")

    _plot_chi_hist(real, pred, title="Xi-momenta check: chi histogram")

    if show_plots:
        plt.show()

    return loss_val


if __name__ == "__main__":
    run_check_xi_momenta()
