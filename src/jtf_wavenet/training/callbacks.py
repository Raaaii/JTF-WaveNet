import os
import numpy as np
import tensorflow as tf


class CustomLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        d_model=20 * 20 * 13,
        warmup_steps=200000,
        baseline_learning_rate=2.0e-8,
        warmup_exponent=-1.5,
        value_scalar=100,
        max_learning_rate=9e-5,
    ):
        super().__init__()
        self.float_type = tf.float32
        self.d_model = tf.cast(d_model, self.float_type)
        self.d_model_decay = tf.math.rsqrt(self.d_model)
        self.warmup_steps = tf.cast(warmup_steps, self.float_type)
        self.warmup_exponent = tf.cast(warmup_exponent, self.float_type)
        self.baseline_learning_rate = tf.cast(baseline_learning_rate, self.float_type)
        self.value_scalar = tf.cast(value_scalar, self.float_type)
        self.max_learning_rate = tf.constant(max_learning_rate, dtype=self.float_type)

    def calc_base_line_rate(self, step):
        return tf.math.sigmoid(step - self.warmup_steps) * self.baseline_learning_rate

    def calc_warmup_phase_value(self, step):
        return step * (self.warmup_steps**self.warmup_exponent)

    def calc_decay_phase_value(self, step):
        return tf.math.rsqrt(step)

    def calc_decay_phase_rate(self, step):
        return self.d_model_decay * self.calc_decay_phase_value(step) / self.value_scalar

    def calc_warmup_phase_rate(self, step):
        return self.d_model_decay * self.calc_warmup_phase_value(step) / self.value_scalar

    def __call__(self, step):
        step = tf.cast(step, self.float_type)
        warmup_rate = self.calc_warmup_phase_rate(step)
        decay_rate = self.calc_decay_phase_rate(step)
        val = tf.math.minimum(warmup_rate, decay_rate)
        baseline = self.calc_base_line_rate(step)
        val = tf.math.maximum(val, baseline)
        val = tf.math.minimum(val, self.max_learning_rate)
        return val

    def get_config(self):
        return {
            "d_model": float(self.d_model.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "baseline_learning_rate": float(self.baseline_learning_rate.numpy()),
            "warmup_exponent": float(self.warmup_exponent.numpy()),
            "value_scalar": float(self.value_scalar.numpy()),
            "max_learning_rate": float(self.max_learning_rate.numpy()),
        }


class MetricLogger:
    def __init__(self, working_directory, tags="all"):
        self.working_directory = working_directory
        self.tags = tags
        os.makedirs(working_directory, exist_ok=True)

    def _file_name(self, tag):
        return f"{self.working_directory}/{tag}.log"

    def _write_line(self, batch_count, value, f):
        f.write(f"{batch_count},{value}\n")
        f.flush()

    def write_to_file(self, tag, batches, value):
        name = self._file_name(tag)

        if not os.path.isfile(name):
            with open(name, "w") as f:
                self._write_line(batches, value, f)
            return

        with open(name, "r") as f:
            last_line = f.readlines()[-1]
            last_batch_count = int(last_line.split(",")[0])

        with open(name, "a") as f:
            new_batch_count = last_batch_count + batches
            self._write_line(new_batch_count, value, f)

    def nan_check(self, History, tag="loss"):
        values = History.history[tag]
        nan_check = np.isnan(np.sum(values))
        if nan_check:
            print("WARNING: NaN in loss")
        return nan_check

    def __call__(self, History, batches):
        keys = History.history.keys() if self.tags == "all" else self.tags
        for tag in keys:
            value = History.history[tag][0]
            self.write_to_file(tag, batches, value)
        self.nan_check(History)


def get_checkpoint_objects(model, optimiser, checkpoint_path, check_point_count=2):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimiser)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=check_point_count)
    return ckpt, manager


class LearningRateTracker(tf.keras.callbacks.Callback):
    """Log learning rate each batch to a CSV file."""

    def __init__(self, log_file="learning_rate_log.csv"):
        super().__init__()
        self.log_file = log_file
        self.global_step = 0
        with open(self.log_file, "w") as f:
            f.write("callback_step,optimizer_step,batch,learning_rate\n")

    def on_train_batch_end(self, batch, logs=None):
        opt_step = int(self.model.optimizer.iterations)
        lr_t = self.model.optimizer.learning_rate(opt_step)
        lr_val = float(lr_t.numpy())
        with open(self.log_file, "a") as f:
            f.write(f"{self.global_step},{opt_step},{batch},{lr_val:.12f}\n")
        self.global_step += 1
