import tensorflow as tf


class JTFWaveNetLosses:
    """
    Orchestrates all loss components used by JTF-WaveNet.

    Holds:
    - SB-ECE calibration loss
    - Xi-momenta loss
    - dynamic scaling factor
    """

    def __init__(self, sb_ece, xi_mom, initial_factor=1.0):
        self.sb_ece = sb_ece
        self.xi_mom = xi_mom

        self.factor = tf.Variable(
            initial_factor,
            trainable=False,
            dtype=tf.float32,
            name="dynamic_factor",
        )

    # ------------------------
    # Atomic losses
    # ------------------------

    @staticmethod
    def calculate_mse(true_vals, predictions):
        return tf.reduce_mean(tf.square(true_vals - predictions))

    @staticmethod
    def mse_only(true_vals, predictions):
        pred_mean = predictions[..., 0]
        return tf.reduce_mean(tf.square(true_vals - pred_mean))

    # ------------------------
    # Full loss
    # ------------------------

    def loss_total(self, real, pred):
        """
        real: [..., 2]   (RI)
        pred: [..., 2]   [mean, sigma]
        """

        mean = pred[..., 0]

        mse = self.calculate_mse(real, mean)
        sigma_loss = self.sb_ece.compute(real, pred)
        xi_loss = self.xi_mom.compute(real, pred)

        total = sigma_loss + self.factor * xi_loss

        return {
            "total": total,
            "mse": mse,
            "sigma": sigma_loss,
            "xi": self.factor * xi_loss,
        }

    # ------------------------
    # State update
    # ------------------------

    def update_factor(self, new_factor: float):
        self.factor.assign(new_factor)
