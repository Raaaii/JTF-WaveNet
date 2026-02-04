import tensorflow as tf


class StageWrapper(tf.keras.Model):
    """
    Keras fit() won't pass stage=... into your model.
    This wrapper forces calling the core model with a fixed stage.
    """

    def __init__(self, core_model, stage: str):
        super().__init__()
        self.core = core_model
        self.stage = stage

    def call(self, inputs, training=False):
        return self.core(inputs, training=training, stage=self.stage)
