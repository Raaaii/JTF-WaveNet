#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import nmrglue as ng
from tensorflow import keras

from jtf_wavenet.model.builders import build_jtfwavenet
from jtf_wavenet.model.jtf_wavenet import JTFWaveNetConfig
from jtf_wavenet.training.callbacks import CustomLearningSchedule, get_checkpoint_objects


# ----------------------------
# IO
# ----------------------------
def load_json(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text())


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------
# FID helpers
# ----------------------------
def to_ri_points(x_complex: np.ndarray, points: int) -> np.ndarray:
    """
    Convert complex FID -> (points, 2) float32 [real, imag].
    Accepts (points,) complex, or (n, points) complex (takes first row).
    """
    x = np.asarray(x_complex)
    if x.ndim > 1:
        x = x[0]
    x = x[:points]
    ri = np.stack([np.real(x), np.imag(x)], axis=-1).astype(np.float32)  # (P,2)
    return ri


def read_pipe_fid(path: Path) -> tuple[dict, np.ndarray]:
    dic, data = ng.pipe.read(str(path))
    return dic, data


def make_model_input(hf_ri: np.ndarray, vib_ri: np.ndarray) -> np.ndarray:
    """
    Model expects (B, points, 2, 2):
      - last axis is [HF, Vib]
      - axis=2 is [real, imag]
    """
    if hf_ri.shape != vib_ri.shape:
        raise ValueError(f"HF shape {hf_ri.shape} != vib shape {vib_ri.shape}")
    if hf_ri.ndim != 2 or hf_ri.shape[1] != 2:
        raise ValueError(f"Expected (points,2) arrays, got {hf_ri.shape}")

    x = np.stack([hf_ri, vib_ri], axis=-1)        # (P,2,2)
    x = x[None, ...]                              # (1,P,2,2)
    return x.astype(np.float32)


# ----------------------------
# Model loading
# ----------------------------
def build_model_from_cfg(points: int, model_cfg: dict):

    dilations = list(np.geomspace(1, 100, num=17, dtype=int))

    from jtf_wavenet.model.jtf_wavenet import JTFWaveNetConfig
    from jtf_wavenet.model.builders import build_jtfwavenet

    cfg = JTFWaveNetConfig(
        points=points,
        filter_count=int(model_cfg.get("filter_count", 48)),
        dilations = dilations,
        blocks=int(model_cfg.get("blocks", 3)),
        convolution_kernel=tuple(model_cfg.get("convolution_kernel", (4, 2))),
        separate_activation=bool(model_cfg.get("separate_activation", True)),
        use_dropout=bool(model_cfg.get("use_dropout", False)),
        use_custom_padding=bool(model_cfg.get("use_custom_padding", True)),
        scale_factor_ft=float(model_cfg.get("scale_factor_ft", 1.0)),
        initial_factor=float(model_cfg.get("initial_factor", 0.0)),
    )

    model = build_jtfwavenet(cfg=cfg)
    model.build(input_shape=(None, points, 2, 2))
    return model


def restore_latest_checkpoint(model: keras.Model, ckpt_dir: Path) -> None:
    schedule = CustomLearningSchedule(warmup_steps=20000, d_model=10)
    opt = keras.optimizers.legacy.Adam(learning_rate=schedule)

    ckpt, mgr = get_checkpoint_objects(model, opt, str(ckpt_dir))
    if not mgr.latest_checkpoint:
        raise RuntimeError(f"No checkpoint found in: {ckpt_dir}")
    ckpt.restore(mgr.latest_checkpoint).expect_partial()
    print(f"Restored from: {mgr.latest_checkpoint}")

def resolve_paths(run_dir, cfg):
    data_dir = (run_dir / cfg["data_dir"]).resolve()
    hf_path = (data_dir / cfg.get("hf_file", "hf.fid")).resolve()
    fid_name = cfg.get("fid_name", "fid_phased.fid")
    vd_glob = cfg.get("vd_glob", "vd*")
    return data_dir, hf_path, fid_name, vd_glob


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_json("config.json")

    points = int(cfg["points"])
    stage = str(cfg.get("stage", "stage2"))

    run_dir = Path(__file__).resolve().parent
    out_root = ensure_dir(run_dir / cfg.get("out_dir", "outputs"))
    out_npy = ensure_dir(out_root / "npy")

    data_dir, hf_path, fid_name, vd_glob = resolve_paths(run_dir, cfg)

    ckpt_dir = (run_dir / cfg["checkpoint_dir"]).resolve()

    vd_glob = cfg.get("vd_glob", "vd*")
    vd_dirs = sorted([p for p in data_dir.glob(vd_glob) if p.is_dir()])

    if not vd_dirs:
        raise FileNotFoundError(f"No VD folders found in {data_dir} with glob '{vd_glob}'")

    print(f"[INFO] Found {len(vd_dirs)} VD folders using '{vd_glob}'")

    fid_name = str(cfg.get("fid_name", "fid_phased.fid"))
    delays = list(cfg.get("vd_delays_sec", []))
    save_sigma = bool(cfg.get("save_sigma", True))

    # ---- read & normalize HF once ----
    _, hf_data = read_pipe_fid(hf_path)
    hf_ri = to_ri_points(hf_data, points)

    norm_factor = np.max(np.abs(hf_ri))
    if not np.isfinite(norm_factor) or norm_factor <= 0:
        raise RuntimeError(f"Bad HF norm_factor: {norm_factor}")
    hf_ri = hf_ri / norm_factor

    # ---- model ----
    model_cfg = cfg.get("model", {})
    model = build_model_from_cfg(points, model_cfg)

    restore_latest_checkpoint(model, ckpt_dir)

    # ---- loop VD folders ----
    for vd_dir in vd_dirs:
        fid_path = vd_dir / cfg.get("fid_name", "fid_phased.fid")
        if not fid_path.exists():
            print(f"[SKIP] Missing {fid_path}")
            continue


        dic, vib_all = read_pipe_fid(fid_path)

        # vib_all may be (n_delays, points) complex
        vib_all = np.asarray(vib_all)
        if vib_all.ndim == 1:
            vib_all = vib_all[None, :]

        n_delays = vib_all.shape[0]
        vd_name = vd_dir.name              # e.g. "vd300"
        vd_num = int(vd_name.replace("vd", ""))
        print(f"[INFO] {vd_name} (#{vd_num}): {n_delays} delays")
        print(f"[INFO] {vd_name}: {n_delays} delays")

        for idx in range(n_delays):
            delay = delays[idx] if idx < len(delays) else float(idx)

            vib_ri = to_ri_points(vib_all[idx], points) / norm_factor

            x = make_model_input(hf_ri, vib_ri)  # (1,P,2,2)
            y = model(tf.convert_to_tensor(x), training=False, stage=stage)[0].numpy()  # (P,2,2)

            pred = y[..., 0]   # (P,2)
            sigma = y[..., 1]  # (P,2)

            tag = f"vd{vd_num:03d}_delay{int(delay*1000):04d}ms"
            np.save(out_npy / f"hf_{tag}.npy", hf_ri)
            np.save(out_npy / f"nf_{tag}.npy", vib_ri)
            np.save(out_npy / f"pred_{tag}.npy", pred)

            if save_sigma:
                np.save(out_npy / f"sigma_{tag}.npy", sigma)

            print(f"  [✓] {tag}")

    print(f"\nSaved .npy outputs to: {out_npy}")


if __name__ == "__main__":
    main()
