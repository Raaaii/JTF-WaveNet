"""
This is a module for functions based on the smooth binned
loss functions desrbibed in

Soft Calibration Objectives for Neural Networks
Archit Karandikar, Nicholas Cain, Dustin Tran, Balaji
Lakshminarayanan, Jonathon Shlens, Michael C. Mozer, Becca Roelofs

https://doi.org/10.48550/arXiv.2108.00106
"""

import tensorflow as tf
from jtf_wavenet.losses.base import LossComponent


class SmoothBinnedECEBase(LossComponent):
    def __init__(self, weight, temp, num_bins=15):
        """
        Base class for computing Smooth Binned Expected Calibration Error (ECE) in regression tasks.

        This loss component computes the calibration error between predicted uncertainties and actual residuals
        using soft binning, as introduced in:

        *Kuleshov et al., "Soft Calibration Objectives for Neural Networks," NeurIPS 2021.*

        Args:
            weight (float): Weight of this loss component in the overall loss.
            num_bins (int): Number of soft bins used to partition the error space.
            temp (float) : Tempreture factor determining how steep the transition between bins is
        """
        super().__init__(weight)
        self.num_bins = num_bins
        self.T = temp

    def sb_ece_base(self, real, pred):
        """
        Compute the Smooth Binned Expected Calibration Error (ECE) for regression.

        This method measures how well the predicted standard deviations align with the actual errors (residuals)
        between the predicted means and true values, using soft binning over the error distribution.

        Args:
            real (Tensor): Ground truth target values, shape (N,).
            pred (Tensor): Model predictions, shape (N, 2), where:
                - pred[..., 0] contains predicted means
                - pred[..., 1] contains predicted standard deviations

        Returns:
            Tuple[Tensor, Tensor]:
                - weighted_residuals (Tensor): Expected residual per example, softly binned.
                - weighted_stds (Tensor): Expected predicted std per example, softly binned.

        Notes:
            - The bin centers are distributed linearly between 0 and 1 and scaled to match the residuals' scale.
            - Binning uses a softmax-weighted Gaussian kernel around bin centers, parameterized by temperature `self.T`.
        """

        print("pred", pred.shape)
        print("real", real.shape)
        y_pred = pred[..., 0]
        # predicted errors
        y_std = pred[..., 1]
        y_true = real

        # temp
        T = self.T
        # self.num_bins = self.num_bins

        # residuals, {batch, points}
        residuals = tf.abs(y_pred - y_true)
        scale = tf.reduce_max(residuals)
        residuals = residuals[tf.newaxis, :]

        # get the weightings for each bin
        bin_centers = tf.linspace(0, 1, self.num_bins)
        bin_centers = tf.cast(bin_centers, real.dtype)

        # shape should be (bin centers, batch, points, )
        bin_centers = bin_centers[:, tf.newaxis, tf.newaxis, tf.newaxis]

        # rescale the bins and the tempreture
        bin_centers = bin_centers * scale
        T = T * (scale**2)

        print("residuals", residuals.shape)
        print("bin_centers", bin_centers.shape)
        # quit()
        g = -((residuals - bin_centers) ** 2) / T
        weights = tf.nn.softmax(g, axis=0)
        # print(sum(weights[:,1])) # this should be one.

        weighted_residuals = residuals * weights
        weighted_residuals = tf.reduce_sum(weighted_residuals, axis=1)
        # print(weighted_residuals.shape)

        weighted_stds = y_std * weights
        weighted_stds = tf.reduce_sum(weighted_stds, axis=1)

        return weighted_residuals, weighted_stds


class SmoothBinnedECE(SmoothBinnedECEBase):
    def __init__(self, weight, temp, num_bins=15):
        """
        This loss component computes the calibration error between predicted uncertainties and actual residuals
        using soft binning, as introduced in:

        *Kuleshov et al., "Soft Calibration Objectives for Neural Networks," NeurIPS 2021.*

        Args:
            weight (float): Weight of this loss component in the overall loss.
            num_bins (int): Number of soft bins used to partition the error space.
            temp (float) : Tempreture factor determining how steep the transition between bins is
        """
        super().__init__(weight, temp, num_bins=num_bins)

    def compute(self, real, pred):
        weighted_residuals, weighted_stds = self.sb_ece_base(real, pred)

        # now determine the mse between the two
        square_diff = tf.square(weighted_residuals - weighted_stds)
        loss = tf.reduce_mean(square_diff)
        return loss


class SmoothBinnedECENormalised(SmoothBinnedECEBase):
    def __init__(self, weight, temp, num_bins=15):
        """
        This loss component computes the normalised calibration error between predicted uncertainties and actual residuals
        using soft binning, as introduced in:

        *Kuleshov et al., "Soft Calibration Objectives for Neural Networks," NeurIPS 2021.*

        Args:
            weight (float): Weight of this loss component in the overall loss.
            num_bins (int): Number of soft bins used to partition the error space.
            temp (float) : Tempreture factor determining how steep the transition between bins is
        """
        super().__init__(weight, temp, num_bins=num_bins)

    def compute(self, real, pred, epsilon=1e-3, use_epsilon=True):

        weighted_residuals, weighted_stds = self.sb_ece_base(real, pred)

        # to avoid devision by values very close to 0.
        if use_epsilon:
            weighted_residuals = weighted_residuals + epsilon
            weighted_stds = weighted_stds + epsilon

        # now determine the mse between the two
        normalised_residuals = (weighted_residuals - weighted_stds) / (weighted_residuals)
        square_diff = tf.square(normalised_residuals)
        loss = tf.reduce_mean(square_diff)
        return loss
