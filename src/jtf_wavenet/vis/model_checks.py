from __future__ import annotations

import tensorflow as tf

from jtf_wavenet.model.jtf_wavenet import JTFWaveNetConfig
from jtf_wavenet.model.builders import build_jtfwavenet


def _assert_no_nan_inf(x: tf.Tensor, name: str) -> None:
    tf.debugging.assert_all_finite(x, f"{name} contains NaN/Inf")


def run_model_shape_smoke_test(
    *,
    points: int = 4096,
    batch_size: int = 2,
    stage: str = "stage2",
    # config knobs
    filter_count: int = 32,
    dilations=(1, 2, 4, 8, 16),
    blocks: int = 3,
    convolution_kernel=(4, 2),
    use_custom_padding: bool = True,
    separate_activation: bool = True,
    use_dropout: bool = False,
) -> None:
    """
    Smoke test: build model, run forward pass, validate finiteness and output shape.

    Assumes input tensor shape: (B, 2, points, 2)
    """
    cfg = JTFWaveNetConfig(
        points=points,
        filter_count=filter_count,
        dilations=tuple(dilations),
        blocks=blocks,
        convolution_kernel=convolution_kernel,
        use_custom_padding=use_custom_padding,
        separate_activation=separate_activation,
        use_dropout=use_dropout,
    )

    model = build_jtfwavenet(cfg=cfg)

    x = tf.random.normal([batch_size, points, 2, 2])
    y = model(x, training=False, stage=stage)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    _assert_no_nan_inf(y, "model output")

    tf.debugging.assert_equal(
        tf.shape(y)[-1],
        2,
        message="Expected last axis size 2 (mean, sigma).",
    )

    print("✅ model shape smoke test passed")
