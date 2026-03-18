import tensorflow as tf
import numpy as np

from base import LossComponent


class XiMomentaBase(LossComponent):
    """
    A class for computing smooth residual weights and analytic Gaussian reference moments
    for XiMomenta loss calculations.

    Args:
        weight (float): Weight for the loss component.
        xi_momenta_weights (list or np.ndarray): Weights for the XiMomenta moments.
        num_bins (int, optional): Number of bins for smooth residual-space binning. Defaults to 20.
        temp (float, optional): Temperature parameter for smoothing. Defaults to 0.01.
    """

    def __init__(self, weight, xi_momenta_weights, num_bins=20, temp=0.01):
        super().__init__(weight)
        self.xi_momenta_weights = xi_momenta_weights
        self.num_bins = num_bins
        self.T = temp

    def _analytic_normal_moments_complex(self, powers):
        """
        Compute analytic raw moments E[Z^p] for Z ~ N(0,1) using the principal complex branch.

        Args:
            powers (tf.Tensor): Complex128 tensor of shape (P,) representing the powers.

        Returns:
            tf.Tensor: Complex128 tensor of shape (P,) containing the computed moments.
        """
        powers_real = tf.math.real(powers)

        two = tf.constant(2.0, dtype=tf.float64)
        pi = tf.constant(np.pi, dtype=tf.float64)

        prefactor_real = (
            tf.pow(two, powers_real / 2.0)
            * tf.exp(tf.math.lgamma((powers_real + 1.0) / 2.0))
            / tf.sqrt(pi)
        )
        prefactor = tf.cast(prefactor_real, tf.complex128)

        phase = tf.exp(1j * tf.cast(np.pi, tf.complex128) * powers)
        symmetry_factor = 0.5 * (1.0 + phase)

        return prefactor * symmetry_factor

    def _smooth_residual_weights(self, real, pred_value):
        """
        Compute smooth residual weights and counts for residual-space binning.

        Args:
            real (tf.Tensor): Ground truth values.
            pred_value (tf.Tensor): Predicted values.

        Returns:
            tuple: A tuple containing:
                - weights (tf.Tensor): Tensor of shape (B, N, 1) representing the weights.
                - counts (tf.Tensor): Tensor of shape (B, 1) representing the counts.
        """
        residuals = tf.abs(pred_value - real)
        residuals = tf.reshape(residuals, (-1, 1))  # (N,1)

        scale = tf.reduce_max(residuals)
        eps = tf.cast(1e-12, residuals.dtype)
        scale = tf.maximum(scale, eps)

        bin_centers = tf.linspace(
            tf.cast(0.0, real.dtype),
            tf.cast(1.0, real.dtype),
            self.num_bins
        )
        bin_centers = bin_centers[:, tf.newaxis, tf.newaxis]  # (B,1,1)
        bin_centers = bin_centers * scale

        T = tf.cast(self.T, real.dtype) * (scale ** 2)
        T = tf.maximum(T, eps)

        residuals = residuals[tf.newaxis, :, :]  # (1,N,1)

        g = -tf.square(residuals - bin_centers) / T   # (B,N,1)
        weights = tf.nn.softmax(g, axis=0)            # (B,N,1)

        counts = tf.reduce_sum(weights, axis=1)       # (B,1)

        return weights, counts

    def compute(self, real, pred):
        """
        Compute the XiMomenta loss with smooth residual-space binning and analytic Gaussian reference moments.

        Args:
            real (tf.Tensor): Ground truth values.
            pred (tf.Tensor): Predicted values with shape (..., 2), where the last dimension contains
                              predicted values and predicted standard deviations.

        Returns:
            tf.Tensor: The computed XiMomenta loss.
        """
        pred_value = pred[..., 0]
        pred_sigma = pred[..., 1]

        diff_list = real - pred_value
        sigma_list = pred_sigma

        # flatten
        diff_list = tf.reshape(diff_list, (-1, 1))
        sigma_list = tf.reshape(sigma_list, (-1, 1))

        # avoid division by zero
        eps = tf.cast(1e-12, sigma_list.dtype)
        sigma_list = tf.maximum(sigma_list, eps)

        # compute chi2 and mean chi2 for moment centering
        chi2 = diff_list / sigma_list
        chi2 = tf.cast(chi2, tf.float64)
        
        # compute mean chi2 for moment centering
        mean_chi2 = tf.reduce_mean(chi2)
        chi_diff = chi2 - mean_chi2

        # smooth weights from residual-space binning
        weights, counts = self._smooth_residual_weights(real, pred_value)
        # weights: (B,N,1), counts: (B,1)

        # complex arithmetic to preserve original fractional signed-power behavior
        chi_diff = tf.cast(chi_diff, tf.complex128)                # (N,1)
        chi_diff = chi_diff[tf.newaxis, :, :]                      # (1,N,1)

        powers = tf.constant([0.5, 2.0, 3.0, 3.5], dtype=tf.complex128)
        w_c = tf.cast(self.xi_momenta_weights, tf.complex128)      # (P,)

        # same signed fractional powers 
        powers_reshape = tf.reshape(powers, (1, 1, -1))            # (1,1,P)
        sigma_loss_3c = tf.pow(chi_diff, powers_reshape)           # (1,N,P)

        # apply moment-selection weights
        sigma_loss_3c = sigma_loss_3c * w_c[tf.newaxis, tf.newaxis, :]  # (1,N,P)

        # broadcast smooth bin weights
        weights_c = tf.cast(weights, tf.complex128)                # (B,N,1)
        counts_c = tf.cast(counts, tf.complex128)                  # (B,1)

        sigma_loss_3c = sigma_loss_3c * weights_c                  # (B,N,P)
        sigma_loss_3c = tf.reduce_sum(sigma_loss_3c, axis=1)       # (B,P)
        sigma_loss_3c = sigma_loss_3c / counts_c                   # (B,P)

        # analytic Gaussian reference moments instead of Monte Carlo
        normal_momenta = self._analytic_normal_moments_complex(powers)   # (P,)
        normal_momenta = normal_momenta * w_c                            # (P,)
        this_normal_momenta = normal_momenta[tf.newaxis, :]              # (1,P)

        # L1 blend with L2, with a smooth transition around sigma_loss_3c ~ 0.5
        sigma_loss_3c = tf.abs(this_normal_momenta - sigma_loss_3c)
        sigma_loss_3c = tf.cast(sigma_loss_3c, pred_sigma.dtype)

        # blending
        sigma_loss_3c_l1 = 2.0 * (tf.sqrt(1.0 + sigma_loss_3c) - 1.0)
        sigma_loss_3c_l2 = tf.square(sigma_loss_3c)

        scaling = tf.math.sigmoid((sigma_loss_3c - 0.5) * 10.0)

        scaled_l2 = sigma_loss_3c_l2 * (1.0 - scaling)
        scaled_l1 = (sigma_loss_3c_l1 - 0.2) * scaling
        sigma_loss_3c = scaled_l2 + scaled_l1

        sigma_loss_3c = tf.reduce_mean(sigma_loss_3c)

        return sigma_loss_3c