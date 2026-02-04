"""
contains the base class for the loss classes after refactoring
"""

import tensorflow as tf


class LossComponent(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight

    def compute(self, real, pred):
        raise NotImplementedError()

    def loss(self, real, pred, weight=False):
        value = self.compute(real, pred)
        if weight:
            return self.weight * value
        else:
            return value
