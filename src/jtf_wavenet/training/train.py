import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from jtf_wavenet.config import load_default_config
from jtf_wavenet.data.generator_core import generator

from jtf_wavenet.model.builders import build_jtfwavenet
from jtf_wavenet.model.jtf_wavenet import JTFWaveNetConfig

from jtf_wavenet.losses.sbece import SmoothBinnedECE
from jtf_wavenet.losses.xi_momenta import XiMomentaBase

from jtf_wavenet.training.callbacks import (
    CustomLearningSchedule,
    MetricLogger,
    get_checkpoint_objects,
    LearningRateTracker,
)

from jtf_wavenet.training.losses_wrappers import (
    mse_only_wrapper,
    total_loss_wrapper,
    mse_metric_wrapper,
    sigma_loss_metric_wrapper,
    xi_mom_loss_metric_wrapper,
)

from jtf_wavenet.training.freeze_utils import (
    set_mean_branch_trainable,
    set_error_branch_trainable,
)

from jtf_wavenet.training.stage_wrapper import StageWrapper


def save_model_summary(model, file_path):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def make_dataset(cfg, batch_size):
    points = cfg["spectrum"]["total_points"]
    output_signature = (
        tf.TensorSpec(shape=(points, 2, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(points, 2), dtype=tf.float32),
    )

    ds = (
        tf.data.Dataset.from_generator(
            lambda: generator(cfg),
            output_signature=output_signature,
        )
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
        .repeat()
    )
    return ds

    return ds


def build_model(points):
    dilations = np.geomspace(1, 100, num=17, dtype=int)

    cfg = JTFWaveNetConfig(
        points=points,
        filter_count=48,
        dilations=tuple(int(d) for d in dilations),
        blocks=3,
        convolution_kernel=(4, 2),
        separate_activation=True,
        use_dropout=False,
        use_custom_padding=True,
        scale_factor_ft=1.0,
        initial_factor=1e-4,
    )

    model = build_jtfwavenet(cfg=cfg)
    model.build(input_shape=(None, points, 2, 2))
    return model


def main(
    stage: str = "stage1",
    checkpoint_root: str = "checkpoints",
    batch_size: int = 10,
    steps_per_epoch: int = 1000,
    warmup_epochs: int = 2,
    rounds: int = 100,
    check_point_count: int = 2,
):


    cfg = load_default_config()
    points = cfg["spectrum"]["total_points"]
    ds = make_dataset(cfg, batch_size)


    # def main(stage="stage1"):
    tf.get_logger().setLevel("ERROR")

    points = cfg["spectrum"]["total_points"]
    batch_size = 10

    steps_per_epoch = 1000
    warmup_epochs = 2
    rounds = 100
    steps_schedule = np.full(warmup_epochs + rounds, steps_per_epoch, dtype=int)

  

    # core model
    core = build_model(points)

    # external losses (kept outside the model)
    sb_ece = SmoothBinnedECE(weight=1.0, temp=0.01, num_bins=10)
    xi_mom = XiMomentaBase(
        weight=1.0, xi_momenta_weights=tf.constant([1.0, 1.0, 1.0, 1.0], tf.float32)
    )

    # optimizer
    learning_schedule = CustomLearningSchedule(warmup_steps=20000, d_model=10)
    optimiser = keras.optimizers.legacy.Adam(learning_rate=learning_schedule)

    # checkpoints
    checkpoint_path = os.path.join(checkpoint_root)

    ckpt, manager = get_checkpoint_objects(core, optimiser, checkpoint_path, check_point_count=2)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"Restored from {manager.latest_checkpoint}")

    metric_logger = MetricLogger(checkpoint_path)
    lr_logger = LearningRateTracker(log_file="learning_rate_log.csv")

    # -------------------------
    # STAGE 1
    # -------------------------
    if stage in ("stage1", "both"):
        set_mean_branch_trainable(core, True)
        set_error_branch_trainable(core, False, train_error_scalars=False)

        save_model_summary(core, "./final_net_stage1.txt")
        print("Stage1 trainable params:", sum(int(tf.size(v)) for v in core.trainable_variables))
        train_model = StageWrapper(core, stage="stage1")

        train_model.compile(
            optimizer=optimiser,
            loss=mse_only_wrapper(core),
            metrics=[
                mse_metric_wrapper(core),
                sigma_loss_metric_wrapper(core, sb_ece),
                xi_mom_loss_metric_wrapper(core, xi_mom),
            ],
        )

        print("=== Trainable variables in STAGE 1 ===")
        for v in train_model.trainable_variables:
            print(v.name)

        for steps in steps_schedule:
            history = train_model.fit(
                ds, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[lr_logger]
            )
            manager.save()
            metric_logger(history, steps)
            print(f"[Stage1] checkpoint saved at step {int(optimiser.iterations.numpy())}")

    # -------------------------
    # STAGE 2
    # -------------------------
    if stage in ("stage2", "both"):
        set_mean_branch_trainable(core, False)
        set_error_branch_trainable(core, True, train_error_scalars=True)

        save_model_summary(core, "./final_net_stage2.txt")
        print("Stage2 trainable params:", sum(int(tf.size(v)) for v in core.trainable_variables))

        train_model = StageWrapper(core, stage="stage2")
        train_model.compile(
            optimizer=optimiser,
            loss=total_loss_wrapper(core, sb_ece, xi_mom),
            metrics=[
                mse_metric_wrapper(core),
                sigma_loss_metric_wrapper(core, sb_ece),
                xi_mom_loss_metric_wrapper(core, xi_mom),
            ],
        )

        print("=== Trainable variables in STAGE 2 ===")
        for v in train_model.trainable_variables:
            print(v.name)

        for steps in steps_schedule:
            history = train_model.fit(
                ds, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[lr_logger]
            )
            manager.save()
            metric_logger(history, steps)
            print(f"[Stage2] checkpoint saved at step {int(optimiser.iterations.numpy())}")


if __name__ == "__main__":
    # options: "stage1", "stage2", "both"
    main(stage="stage1")
