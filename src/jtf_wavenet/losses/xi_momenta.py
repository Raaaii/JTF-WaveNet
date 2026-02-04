import tensorflow as tf
import tensorflow_probability as tfp

from jtf_wavenet.losses.base import LossComponent


class XiMomentaBase(LossComponent):
    def __init__(self, weight, xi_momenta_weights, num_bins=10, tau=0.01):
        super().__init__(weight)
        self.xi_momenta_weights = xi_momenta_weights

    def compute(self, real, pred):
        """This function calculates the loss based on the momenta of the xi function
        (y-pred)/sigma and the normal distribution.

        In parctice we minimize the differences between 1/2, 1, 2, 3, 7/2 momentum of
        xi distribution and an ideal normal distribution.

        here pred contains prediction at pred[..., 0]
        and error estimations at pred[..., 1]

        Args:
            real (tf.tensor) : ground truth
            pred (tf.tensor) : model prediction

        Returns:
            tf.tensor: loss

        Notes: we have hard coded 20 bins/this should be changed at some point

        """
        pred_value = pred[..., 0]
        pred_sigma = pred[..., 1]  # sigma

        sigma_list = tf.math.scalar_mul(1.0, pred_sigma)  # (15,400,512)
        diff_list = real - pred_value

        # Reshape
        sigma_list = tf.reshape(sigma_list, (-1, 1))
        diff_list = tf.reshape(diff_list, (-1, 1))

        # number of quantiles of sigma.
        # This makes sure that both small and large sigmas will have a chi2
        # distribution that is Gaussian

        sigma_parts = tf.reshape(tfp.stats.quantiles(sigma_list, 20), (1, -1))
        condition_one = sigma_parts[:, :-1] < sigma_list
        condition_two = sigma_parts[:, 1:] > sigma_list
        joint_condition = condition_one & condition_two
        ones = tf.constant(1.0, dtype=tf.float64)
        zeros = tf.constant(0.0, dtype=tf.float64)

        # Indices with sigma parts, (5*400*512,10)
        idxs = tf.where(joint_condition, ones, zeros)

        # YAK - we need to work in float64 ..
        chi2 = diff_list / sigma_list
        chi2 = tf.cast(chi2, tf.float64)  # (5*400*512,1)

        chi2 = tf.tile(chi2, (1, sigma_parts.shape[-1] - 1))
        mean_chi2 = tf.math.reduce_mean(chi2)

        counts = tf.math.reduce_sum(idxs, axis=0, keepdims=True)  # (1,10)
        counts = tf.expand_dims(tf.expand_dims(counts, axis=0), axis=-1)
        idxs = tf.expand_dims(tf.expand_dims(idxs, axis=0), axis=-1)

        chi2 = tf.complex(chi2, zeros)
        mean_chi2 = tf.complex(mean_chi2, zeros)
        idxs = tf.complex(idxs, zeros)
        counts = tf.complex(counts, zeros)

        powers = tf.constant([0.5, 2.0, 3.0, 3.5], dtype=tf.complex128)

        # where do these magic numbers come from ? 16*400*512,1
        normal_momenta = tf.random.normal(mean=0.0, stddev=1.0, shape=(16 * 400 * 512, 1))
        normal_momenta = tf.cast(normal_momenta, tf.complex128)
        reshaped_momental = tf.pow(normal_momenta, tf.reshape(powers, (1, -1)))

        # print('momenta shape', reshaped_momental.shape)

        # select the momenta we want
        w_c = tf.cast(self.xi_momenta_weights, tf.complex128)
        reshaped_momental = reshaped_momental * w_c[tf.newaxis, :]

        normal_momenta = tf.math.reduce_mean(reshaped_momental, axis=0)
        this_normal_momenta = tf.expand_dims(normal_momenta, axis=0)

        chi_diff = chi2 - mean_chi2
        chi_diff = tf.expand_dims(chi_diff, axis=-1)
        powers_reshape = tf.reshape(powers, (1, 1, -1))
        sigma_loss_3c = tf.pow(chi_diff, powers_reshape)

        # select the momenta we want
        sigma_loss_3c = sigma_loss_3c * w_c[tf.newaxis, :]

        sigma_loss_3c = tf.expand_dims(sigma_loss_3c, axis=0)  # (1,:,sigma,powers)
        sigma_loss_3c = sigma_loss_3c * idxs
        sigma_loss_3c = tf.math.reduce_sum(sigma_loss_3c / counts, axis=(0, 1))  # (sigma,powers)
        #
        # this should be more robust - using soft l1 norm
        sigma_loss_3c = tf.abs(this_normal_momenta - sigma_loss_3c)
        sigma_loss_3c = tf.cast(sigma_loss_3c, pred_sigma.dtype)

        sigma_loss_3c_l1 = 2.0 * (tf.math.sqrt(1.0 + sigma_loss_3c) - 1.0)  # soft L1
        sigma_loss_3c_l2 = tf.square(sigma_loss_3c)  # L2

        # soft change between L1 and L2 at ~ 0.5
        scaling = tf.math.sigmoid((sigma_loss_3c - 0.5) * 10.0)

        scaled_l2 = sigma_loss_3c_l2 * (1.0 - scaling)
        scaled_l1 = (sigma_loss_3c_l1 - 0.2) * scaling
        sigma_loss_3c = scaled_l2 + scaled_l1

        sigma_loss_3c = tf.math.reduce_mean(sigma_loss_3c)

        # we also add a small push on sigma to not be too big
        return sigma_loss_3c
