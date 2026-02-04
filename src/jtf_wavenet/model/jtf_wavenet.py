from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math as m
from dataclasses import dataclass
from typing import Tuple

from jtf_wavenet.losses.sbece import SmoothBinnedECE
from jtf_wavenet.losses.xi_momenta import XiMomentaBase
from jtf_wavenet.losses.baseline import BaselineSTD


tfd = tfp.distributions


@dataclass(frozen=True)
class JTFWaveNetConfig:
    points: int
    filter_count: int
    dilations: Tuple[int, ...]
    blocks: int = 3
    convolution_kernel: Tuple[int, int] = (4, 2)
    separate_activation: bool = True
    use_dropout: bool = False
    use_custom_padding: bool = True
    scale_factor_ft: float = 1.0
    initial_factor: float = 1.0


class JTFWaveNet(keras.Model):
    """
    Two-path WaveNet-style model:
      - Mean branch predicts mean spectrum
      - Error branch predicts sigma spectrum

    Expected input shape:  (B, 2, points, 2)
      - axis=1 -> [real, imag]
      - axis=2 -> time points
      - axis=3 -> [HF, NF_vib] channels

    Output shape: (B, points, 2, 2) after _post_process:
      - axis=-1 -> [mean, sigma]
      - the preceding axes are whatever _post_process returns
        (currently: fftshifted frequency domain with real/imag channels)
    """

    def __init__(
        self,
        cfg: JTFWaveNetConfig,
        baseline_std=None,
        sb_ece=None,
        xi_mom=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.points = cfg.points
        self.filter_count = cfg.filter_count
        self.dilations = cfg.dilations
        self.blocks = cfg.blocks
        self.convolution_kernal = cfg.convolution_kernel  # keep your old spelling used everywhere
        self.separate_activation = cfg.separate_activation
        self.use_dropout = cfg.use_dropout
        self.use_custom_padding = cfg.use_custom_padding
        self.scale_factor_ft = cfg.scale_factor_ft

        # Trainable scalars
        self.error_offset = tf.Variable(0.05, dtype=tf.float32, trainable=True, name="error_offset")
        self.factor = tf.Variable(
            cfg.initial_factor, trainable=False, dtype=tf.float32, name="dynamic_factor"
        )
        self.error_softplus_log_beta = tf.Variable(
            1.0, dtype=tf.float32, trainable=True, name="error_softplus_log_beta"
        )

        # Optional calibration helpers (can be None)
        self.baseline_std = baseline_std
        self.sb_ece = sb_ece
        self.xi_mom = xi_mom

        # ---------------------------
        # Common initial expansion layer
        # ---------------------------
        self.x_dense = keras.layers.Dense(
            2 * self.filter_count, activation="tanh", name="x_dense_mean"
        )
        self.x_dense_error = keras.layers.Dense(
            2 * self.filter_count, activation="tanh", name="x_dense_eror"
        )

        # ---------------------------
        # Create two independent sets of WaveNet blocks:
        # one for the mean branch and one for the error branch.
        # Each branch has separate time and frequency layers.
        # ---------------------------
        self.mean_time_layers = []
        self.mean_freq_layers = []
        self.error_time_layers = []
        self.error_freq_layers = []

        for block_num in range(self.blocks):
            mean_time_block = []
            mean_freq_block = []
            error_time_block = []
            error_freq_block = []
            for dilation_idx, dilation in enumerate(self.dilations):
                # Create layers for the mean branch.
                mean_time_block.append(
                    self._create_wavenet_layer(dilation, "time", block_num, dilation_idx)
                )
                mean_freq_block.append(
                    self._create_wavenet_layer(dilation, "freq", block_num, dilation_idx)
                )
                # Create layers for the error branch.
                error_time_block.append(
                    self._create_wavenet_layer(dilation, "time", block_num, dilation_idx)
                )
                error_freq_block.append(
                    self._create_wavenet_layer(dilation, "freq", block_num, dilation_idx)
                )
            self.mean_time_layers.append(mean_time_block)
            self.mean_freq_layers.append(mean_freq_block)
            self.error_time_layers.append(error_time_block)
            self.error_freq_layers.append(error_freq_block)

        # ---------------------------
        # Final trunk layers for each branch
        # ---------------------------
        self.mean_final_conv = keras.layers.Conv2D(
            self.filter_count,
            kernel_size=self.convolution_kernal,
            activation="relu",
            name="mean_final_shared_representation",
            padding="valid" if self.use_custom_padding else "same",
        )
        self.error_final_conv = keras.layers.Conv2D(
            self.filter_count,
            kernel_size=self.convolution_kernal,
            activation="relu",
            name="error_final_shared_representation",
            padding="valid" if self.use_custom_padding else "same",  # match mean
        )

        # ---------------------------
        # Not used - dropouts
        # ---------------------------
        if self.use_dropout:
            self.mean_final_dropout = keras.layers.Dropout(0.3, name="mean_final_dropout")

        if self.use_dropout:
            self.error_final_dropout = keras.layers.Dropout(0.3, name="error_final_dropout")

        # ---------------------------
        # Relu conv for both
        # ---------------------------

        self.mean_shared_trunk = keras.layers.Conv2D(
            self.filter_count,
            kernel_size=(4, 2),
            activation="relu",
            name="mean_post_trunk_conv",
            padding="same",
        )
        self.error_shared_trunk = keras.layers.Conv2D(
            self.filter_count,
            kernel_size=(4, 2),
            activation="relu",
            name="error_post_trunk_conv",
            padding="same",
        )

        # ---------------------------
        # Heads for each branch
        # ---------------------------

        ### CHANGED: Replaced tanh head with conv
        self.mean_head_conv = keras.layers.Conv2D(
            self.filter_count,
            kernel_size=(4, 2),
            activation="tanh",  # Maybe to use relu?
            name="mean_head_conv",
            padding="same",
        )

        self.error_head = keras.layers.Conv2D(
            self.filter_count,
            kernel_size=(4, 2),
            activation="tanh",
            name="error_head",
            padding="same",
        )

        # ---------------------------
        # Last dense for each branch
        # ---------------------------

        self.mean_head_dense = keras.layers.Dense(
            1, activation=None, name="mean_head_dense"  # No activation: linear output
        )
        self.error_head_dense = keras.layers.Dense(
            1, activation=None, name="error_head_dense"  # No activation: linear output
        )

    def build(self, input_shape):
        super().build(input_shape)

    def _create_wavenet_layer(self, dilation, channel, block_num, dilation_idx):
        """
        Create a set of convolution layers for a single WaveNet block (for either the time or freq channel).
        """

        def apply_padding(inputs, kernel_size, dilation_rate=None):
            if not self.use_custom_padding:
                return inputs
            # Example custom padding logic; adjust as needed.
            kernel_shape_1 = kernel_size[1] + 1
            kernel_shape_0 = kernel_size[0] - 3
            pad_before = 0
            pad_after = dilation_rate * kernel_shape_1 if dilation_rate else kernel_shape_1
            paddings = [[0, 0], [pad_before, pad_after], [0, kernel_shape_0], [0, 0]]
            return tf.pad(inputs, paddings)

        y1_conv = keras.layers.Conv2D(
            filters=self.filter_count,
            kernel_size=self.convolution_kernal,
            padding=("valid" if (channel == "time" and self.use_custom_padding) else "same"),
            dilation_rate=[dilation, 1],
            name=f"{channel}_block{block_num}_dilation{dilation_idx}_y1_conv",
        )
        y2_conv = keras.layers.Conv2D(
            filters=self.filter_count,
            kernel_size=self.convolution_kernal,
            padding=("valid" if (channel == "time" and self.use_custom_padding) else "same"),
            dilation_rate=[dilation, 1],
            name=f"{channel}_block{block_num}_dilation{dilation_idx}_y2_conv",
        )
        z_conv = keras.layers.Conv2D(
            filters=self.filter_count * 2,
            kernel_size=self.convolution_kernal,
            padding=("valid" if (channel == "time" and self.use_custom_padding) else "same"),
            name=f"{channel}_block{block_num}_dilation{dilation_idx}_z_conv",
        )
        return {
            "apply_padding": apply_padding if channel == "time" else None,
            "y1_conv": y1_conv,
            "y2_conv": y2_conv,
            "z_conv": z_conv,
        }

    ########################################################################
    # Forward and Inverse FFT (same as before)
    ########################################################################
    def forwards_ft(self, signal):
        real_part = tf.cast(signal[..., 0, :], tf.float32)
        imag_part = tf.cast(signal[..., 1, :], tf.float32)
        complex_signal = tf.complex(real_part, imag_part)
        complex_signal = tf.transpose(complex_signal, perm=[0, 2, 1])
        ft_complex = tf.signal.fft(complex_signal, name="forwards_ft")
        ft_complex = tf.transpose(ft_complex, perm=[0, 2, 1])
        real = tf.cast(tf.math.real(ft_complex), tf.float32)
        imag = tf.cast(tf.math.imag(ft_complex), tf.float32)
        signal_real = tf.stack([real, imag], name="ft_final_stack")
        return tf.transpose(signal_real, perm=[1, 2, 0, 3])

    def backwards_ft(self, signal):
        real_part = tf.cast(signal[..., 0, :], tf.float32)
        imag_part = tf.cast(signal[..., 1, :], tf.float32)
        complex_signal = tf.complex(real_part, imag_part)
        complex_signal = tf.transpose(complex_signal, perm=[0, 2, 1])
        ift_complex = tf.signal.ifft(complex_signal, name="backwards_ft")
        ift_complex = tf.transpose(ift_complex, perm=[0, 2, 1])
        real = tf.cast(tf.math.real(ift_complex), tf.float32)
        imag = tf.cast(tf.math.imag(ift_complex), tf.float32)
        signal_real = tf.stack([real, imag], name="ift_final_stack")
        return tf.transpose(signal_real, perm=[1, 2, 0, 3])

    def _variable_softplus(self, x):
        """
        Variable (parametric) softplus:
            vsp(x; beta) = softplus(beta * x) / beta
        We keep beta > 0 via beta = softplus(log_beta) + eps for stability.
        """
        beta = tf.math.square(self.error_softplus_log_beta) + 1e-6
        beta = tf.cast(beta, x.dtype)
        return tf.nn.softplus(beta * x) / beta

    def get_norm(self, x):
        abs_x = tf.math.abs(x)
        norm = tf.reduce_max(abs_x, axis=(1, 2, 3), keepdims=True)
        return norm

    def window_function(self, inputs):
        shape = tf.shape(inputs)
        fractions = tf.cast(tf.range(shape[1]), tf.float32) / tf.cast(shape[1], tf.float32)
        window = tf.math.cos(tf.constant(m.pi) * fractions / 2.0) ** 2
        window = tf.reshape(window, [1, shape[1], 1, 1])
        outputs = inputs * window
        return outputs

    def _apply_final_padding(self, inputs, kernel_size):
        # Example final padding logic; adjust as needed.
        kernel_shape_1 = kernel_size[0] - 1
        kernel_shape_0 = kernel_size[0] - 3
        paddings = [[0, 0], [0, kernel_shape_1], [0, kernel_shape_0], [0, 0]]
        return tf.pad(inputs, paddings)

    def _do_fourier_transform(self, x):
        x = self.forwards_ft(x)
        return x

    def _post_process(self, x):
        # Dc correction
        x = tf.concat([x[:, :1, :, :] / 2.0, x[:, 1:, :, :]], axis=1)
        # ft
        x = self._do_fourier_transform(x)  # shapes (10, 4096, 2, 1)
        # fft shift
        x = tf.signal.fftshift(x, axes=1)  # shift along P
        return x

    def _process_branch(self, x, time_layers, freq_layers):
        """
        Process a branch (mean or error) through its WaveNet blocks.
        Runs the time and frequency paths in parallel, then combines them.
        """
        # Frequency-domain processing
        x_fft = self.forwards_ft(x)
        ft_norm = self.get_norm(x_fft)
        x_fft = x_fft / ft_norm

        skip_time = []
        skip_freq = []
        x_time = x  # local copy for time-domain processing

        for block_num in range(self.blocks):
            for dilation_idx, dilation in enumerate(self.dilations):
                # ----- TIME CHANNEL -----
                layer_time = time_layers[block_num][dilation_idx]
                if layer_time["apply_padding"]:
                    xpad = layer_time["apply_padding"](x_time, self.convolution_kernal, dilation)
                else:
                    xpad = x_time
                y1 = layer_time["y1_conv"](xpad)
                y2 = layer_time["y2_conv"](xpad)
                y1 = keras.activations.tanh(y1)
                y2 = keras.activations.sigmoid(y2)
                z = y1 * y2
                if layer_time["apply_padding"]:
                    zpad = layer_time["apply_padding"](z, self.convolution_kernal)
                else:
                    zpad = z
                z_out = layer_time["z_conv"](zpad)
                x_time = x_time + z_out
                skip_time.append(z_out)

                # ----- FREQUENCY CHANNEL -----
                layer_freq = freq_layers[block_num][dilation_idx]
                y1_fft = layer_freq["y1_conv"](x_fft)
                y2_fft = layer_freq["y2_conv"](x_fft)
                y1_fft = keras.activations.tanh(y1_fft)
                y2_fft = keras.activations.sigmoid(y2_fft)
                z_fft = y1_fft * y2_fft
                z_fft = layer_freq["z_conv"](z_fft)
                x_fft = x_fft + z_fft
                skip_freq.append(z_fft)

        if self.separate_activation:
            out_time = keras.activations.relu(tf.add_n(skip_time))
            out_freq = keras.activations.relu(tf.add_n(skip_freq))
            out_freq = out_freq * ft_norm
            out_freq = self.backwards_ft(out_freq)
            out = out_time + out_freq
        else:
            out_time = tf.add_n(skip_time)
            out_freq = tf.add_n(skip_freq)
            out_freq = out_freq * ft_norm
            out_freq = self.backwards_ft(out_freq)
            out = keras.activations.relu(out_time + out_freq)
        return out

    ########################################################################
    # Forward Pass
    ########################################################################

    def call(
        self,
        inputs,
        training=False,
        stage="stage2",
        return_intermediate=False,
        target=None,
    ):
        """
        Forward pass through both branches.

        In stage1 the common features (from x_dense) and error branch are not updated
        (error branch skipped entirely) so that only the mean branch is trained with MSE.
        In stage2, the common layer and mean branch outputs are frozen (using stop_gradient)
        so that only the error branch is updated with the full loss.
        """
        # Common expansion
        x_common = self.x_dense(inputs)
        # print(x_common.shape, "shape in moel")
        # if stage == "stage2":
        #     x_common = tf.stop_gradient(x_common)

        # -----------------------
        # Mean Branch
        # -----------------------
        x_mean = self._process_branch(x_common, self.mean_time_layers, self.mean_freq_layers)

        if self.use_custom_padding:
            x_mean = self._apply_final_padding(x_mean, self.convolution_kernal)

        x_mean = self.mean_final_conv(x_mean)
        x_mean = self.mean_shared_trunk(x_mean)

        x_mean = self.mean_head_conv(x_mean)
        mean_out = self.mean_head_dense(x_mean)
        mean_out = self._post_process(mean_out)

        # -----------------------
        # Error Branch
        # -----------------------
        if stage == "stage2":
            x_common = self.x_dense_error(inputs)
            x_error = self._process_branch(x_common, self.error_time_layers, self.error_freq_layers)

            if self.use_custom_padding:
                x_error = self._apply_final_padding(x_error, self.convolution_kernal)

            x_error = self.error_final_conv(x_error)  # relu
            x_error = self.error_shared_trunk(x_error)  # relu

            error_out = self.error_head(x_error)  # tanh
            error_out = self.error_head_dense(error_out)

            error_out = self._post_process(error_out)  # does the FT, FFTshift, dc correction
            error_out = self._variable_softplus(error_out)
            error_out = error_out / 10 + self.error_offset  # (512*16*100*1000000*1000000)

        else:
            # Stage 1: skip error branch entirely
            x_error = None
            error_out = tf.random.normal(tf.shape(mean_out), mean=0.05, stddev=0.01)

        if stage == "stage2":
            mean_out = tf.stop_gradient(mean_out)

        out = tf.concat([mean_out, error_out], axis=-1)

        if return_intermediate:
            intermediate = {
                "inputs": inputs,
                "mean_features": x_mean,
                "error_features": x_error,
                "mean_out": mean_out,
                "error_out": error_out,
            }
            if target is not None:
                intermediate["error_residual"] = target - mean_out
            return out, intermediate
        else:
            return out

    # =========================
    # Loss helpers (model API)
    # =========================
    def calculate_mse(self, true_vals, predictions):
        return tf.reduce_mean(tf.square(true_vals - predictions))

    def mse_only(self, true_vals, predictions):
        # predictions[..., 0] is mean
        pred_mean = predictions[..., 0]
        return tf.reduce_mean(tf.square(true_vals - pred_mean))

    def loss_total(self, real, pred, sb_ece=None, xi_mom=None, q_sigma=10, q_xi=10):
        """
        real: [..., 2] (RI)  (your target)
        pred: [..., 2] last channel = [mean, sigma]
        """
        mean = pred[..., 0]
        sigma = pred[..., 1]  # not used directly here, but present

        mse = self.calculate_mse(real, mean)

        # prefer injected objects, fallback to self.sb_ece / self.xi_mom if set
        sb = sb_ece if sb_ece is not None else getattr(self, "sb_ece", None)
        xm = xi_mom if xi_mom is not None else getattr(self, "xi_mom", None)

        if sb is None:
            raise ValueError("sb_ece is None. Pass sb_ece to loss_total(...) or set self.sb_ece.")
        if xm is None:
            raise ValueError("xi_mom is None. Pass xi_mom to loss_total(...) or set self.xi_mom.")

        sigma_loss_val = sb.compute(real, pred)
        xi_val = xm.compute(real, pred)

        factor = self.factor
        total_loss = sigma_loss_val + xi_val * factor

        return total_loss, mse, sigma_loss_val, xi_val * factor

    def update_factor(self, new_factor: float):
        self.factor.assign(new_factor)
