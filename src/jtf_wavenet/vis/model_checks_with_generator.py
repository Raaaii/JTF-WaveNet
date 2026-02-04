from __future__ import annotations

import tensorflow as tf

from jtf_wavenet.config import load_default_config
from jtf_wavenet.data.generator_core import generator
from jtf_wavenet.model.builders import build_jtfwavenet
from jtf_wavenet.model.jtf_wavenet import JTFWaveNetConfig


def _assert_no_nan_inf(x: tf.Tensor, name: str) -> None:
    tf.debugging.assert_all_finite(x, f"{name} contains NaN/Inf")


def _to_model_layout(x: tf.Tensor, points: int) -> tf.Tensor:
    """
    Convert generator output to the model's expected layout.

    Model expects: (B, points, 2, 2)

    Supported generator layouts:
      - (2, points, 2)   -> treat as (ri, points, channels)
      - (points, 2, 2)   -> already (points, ri, channels)
      - (B, points, 2, 2)-> already batched
    """
    x = tf.convert_to_tensor(x)

    if x.shape.rank == 4:
        # Assume already (B, points, 2, 2)
        tf.debugging.assert_equal(tf.shape(x)[1], points, message="Expected points at axis=1")
        return x

    if x.shape.rank != 3:
        raise ValueError(f"Unsupported x rank={x.shape.rank}, shape={x.shape}")

    # rank=3: could be (2, points, 2) or (points, 2, 2)
    s0 = tf.shape(x)[0]
    s1 = tf.shape(x)[1]
    s2 = tf.shape(x)[2]

    # Case A: (2, points, 2)  -> transpose to (points, 2, 2)
    # (ri, points, ch) -> (points, ri, ch)
    def case_a():
        x2 = tf.transpose(x, perm=[1, 0, 2])
        return x2

    # Case B: (points, 2, 2) already
    def case_b():
        return x

    x_pts = tf.cond(tf.equal(s0, 2), true_fn=case_a, false_fn=case_b)

    # Now ensure (points, 2, 2)
    tf.debugging.assert_equal(
        tf.shape(x_pts)[0], points, message="Expected points at axis=0 after conversion"
    )
    tf.debugging.assert_equal(tf.shape(x_pts)[1], 2, message="Expected real/imag at axis=1")
    tf.debugging.assert_equal(tf.shape(x_pts)[2], 2, message="Expected channels at axis=2")

    # Add batch dim -> (B, points, 2, 2)
    return tf.expand_dims(x_pts, axis=0)


def run_generator_shapes_only() -> None:
    """Just print generator output shapes (no model)."""
    gen = generator(load_default_config())
    x, target = next(gen)
    print("x shape:", getattr(x, "shape", None))
    print("target shape:", getattr(target, "shape", None))


def run_generator_to_model_forward(
    *,
    points: int = 4096,
    stage: str = "stage2",
    # model config knobs (keep defaults light)
    filter_count: int = 32,
    dilations=(1, 2, 4, 8, 16),
    blocks: int = 3,
    convolution_kernel=(4, 2),
    use_custom_padding: bool = True,
    separate_activation: bool = True,
    use_dropout: bool = False,
) -> None:
    """
    Pull one sample from the generator, reshape to model layout, run forward pass,
    and assert output is finite.
    """
    gen = generator(load_default_config())
    x, target = next(gen)

    x = _to_model_layout(x, points=points)
    target = tf.convert_to_tensor(target)

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

    y = model(x, training=False, stage=stage)

    print("x (model input):", x.shape, x.dtype)
    print("target:", target.shape, target.dtype)
    print("y:", y.shape, y.dtype)

    _assert_no_nan_inf(y, "model output")
    print("✅ generator → model forward pass OK")
