#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng


# ----------------------------
# IO
# ----------------------------
def load_json(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text())


def load_ppm_axis(fid_for_ppm: Path, points: int) -> np.ndarray:
    dic, data = ng.pipe.read(str(fid_for_ppm))
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[0]
    data = data[:points]
    uc = ng.pipe.make_uc(dic, data)
    ppm = uc.ppm_scale()
    # We plot length 2*points (because we zero-pad to 2P in FT)
    return np.linspace(ppm[0], ppm[-1], 2 * len(ppm))


# ----------------------------
# Fourier transforms (match your old style)
# ----------------------------
def _to_complex_time(x_ri: np.ndarray) -> np.ndarray:
    """(P,2) -> complex (P,)"""
    x_ri = np.asarray(x_ri)
    if x_ri.ndim != 2 or x_ri.shape[1] != 2:
        raise ValueError(f"Expected (points,2), got {x_ri.shape}")
    return x_ri[:, 0] + 1j * x_ri[:, 1]


def make_ft_time_to_freq(x_ri: np.ndarray, *, points: int, acq_time: float, decay: float) -> np.ndarray:
    """
    Time-domain RI -> window -> zero-pad to 2P -> fftshift(fft())
    Returns complex length 2P.
    """
    sig = _to_complex_time(x_ri)[:points]
    times = np.linspace(0.0, acq_time, points)
    win = np.exp(-decay * times)
    padded = np.pad(sig * win, (0, points), mode="constant")
    padded[0] /= 2.0
    return np.fft.fftshift(np.fft.fft(padded))


def make_ft_freqlike_to_freq(x_freqlike_ri: np.ndarray, *, points: int, acq_time: float, decay: float) -> np.ndarray:
    """
    Your predicted output is in 'freq-like' RI (P,2) (per your training target).
    Old behavior: convert to complex -> ifft -> window -> pad -> fft (no fftshift here in your code).
    We keep that behavior, then plot with ppm axis that already matches 2P.
    """
    spec = _to_complex_time(x_freqlike_ri)[:points]
    spec = np.fft.ifftshift(spec)  # keep your old default behavior
    time = np.fft.ifft(spec)

    times = np.linspace(0.0, acq_time, len(time))
    win = np.exp(-decay * times)
    padded = np.pad(time * win, (0, len(time)), mode="constant")
    return np.fft.fftshift(np.fft.fft(padded))


def sigma_to_freq_band(
    sigma: np.ndarray, *,
    points: int,
    acq_time: float,
    decay: float,
) -> np.ndarray | None:
    """
    Accept sigma as:
      - (P,2) time-domain RI  -> convert using make_ft_freqlike_to_freq (same as pred)
      - (P,)   time-domain real -> map into (P,2) with imag=0 then convert
      - (2P,) complex/real freq-domain already -> return as complex array
      - (2P,2) RI freq-domain -> convert to complex
    Returns band as real abs() array length 2P.
    """
    if sigma is None:
        return None

    s = np.asarray(sigma)

    # (2P,) already
    if s.ndim == 1 and s.shape[0] == 2 * points:
        # may be real or complex
        s_c = s.astype(np.complex64) if not np.iscomplexobj(s) else s
        return np.abs(np.real(s_c))

    # (2P,2) RI freq-domain
    if s.ndim == 2 and s.shape[0] == 2 * points and s.shape[1] == 2:
        s_c = s[:, 0] + 1j * s[:, 1]
        return np.abs(np.real(s_c))

    # (P,) time-domain
    if s.ndim == 1 and s.shape[0] == points:
        tmp = np.zeros((points, 2), dtype=float)
        tmp[:, 0] = s
        s_fd = make_ft_freqlike_to_freq(tmp, points=points, acq_time=acq_time, decay=decay)
        return np.abs(np.real(s_fd))

    # (P,2) time-domain
    if s.ndim == 2 and s.shape[0] == points and s.shape[1] == 2:
        s_fd = make_ft_freqlike_to_freq(s, points=points, acq_time=acq_time, decay=decay)
        return np.abs(np.real(s_fd))

    # fallback: try to interpret as (P,2) by truncation if possible
    try:
        if s.ndim == 2 and s.shape[1] == 2:
            s2 = s[:points, :]
            s_fd = make_ft_freqlike_to_freq(s2, points=points, acq_time=acq_time, decay=decay)
            return np.abs(np.real(s_fd))
    except Exception:
        pass

    return None


# ----------------------------
# Viewer
# ----------------------------
def list_tags(npy_dir: Path) -> list[str]:
    tags = []
    for p in npy_dir.glob("nf_*.npy"):
        tag = p.stem[len("nf_"):]
        tags.append(tag)
    return sorted(tags)


def maybe_load(npy_dir: Path, prefix: str, tag: str) -> np.ndarray | None:
    p = npy_dir / f"{prefix}_{tag}.npy"
    if not p.exists():
        return None
    return np.load(p)


def main():
    cfg = load_json("config.json")

    points = int(cfg["points"])
    out_dir = Path(cfg.get("out_dir", "outputs"))
    npy_dir = (Path(__file__).resolve().parent / out_dir / "npy").resolve()

    ppm_ref = Path(cfg["ppm_ref_fid"])
    ppm_ref = (Path(__file__).resolve().parent / ppm_ref).resolve()

    plot_cfg = cfg.get("plot", {})
    decay = float(plot_cfg.get("decay", 1.0))
    acq_time = float(plot_cfg.get("acq_time", 2.43712))
    figsize = plot_cfg.get("figsize", [6, 6])
    figsize = (float(figsize[0]), float(figsize[1]))

    if not npy_dir.exists():
        raise FileNotFoundError(f"NPY directory not found: {npy_dir}")
    if not ppm_ref.exists():
        raise FileNotFoundError(f"ppm_ref_fid not found: {ppm_ref}")

    x_ppm = load_ppm_axis(ppm_ref, points)

    tags = list_tags(npy_dir)
    if not tags:
        raise RuntimeError(f"No nf_*.npy found in {npy_dir}")

    idx = 0
    jump = ""

    print("Controls: [n]=next | [digits]+[Enter]=jump to vd### | [Esc]=quit")

    while 0 <= idx < len(tags):
        tag = tags[idx]

        nf = maybe_load(npy_dir, "nf", tag)
        hf = maybe_load(npy_dir, "hf", tag)
        pred = maybe_load(npy_dir, "pred", tag)
        sigma = maybe_load(npy_dir, "sigma", tag)  # optional

        if nf is None or hf is None or pred is None:
            print(f"[SKIP] Missing hf/nf/pred for {tag}")
            idx += 1
            continue

        # --- FFTs ---
        hf_ft = make_ft_time_to_freq(hf, points=points, acq_time=acq_time, decay=decay)
        nf_ft = make_ft_time_to_freq(nf, points=points, acq_time=acq_time, decay=decay)
        pred_ft = make_ft_freqlike_to_freq(pred, points=points, acq_time=acq_time, decay=decay)

        # --- Figure layout ---
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )

        ax_top.plot(x_ppm, np.real(nf_ft), alpha=0.55, lw=2.8, label="Shuttled (NF)")
        ax_top.plot(x_ppm, np.real(pred_ft), lw=1.6, label="Prediction")
        ax_top.plot(x_ppm, np.real(hf_ft), alpha=0.25, lw=1.0, color="red", label="High Field (HF)")
        ax_top.invert_xaxis()
        ax_top.set_ylabel("Intensity (a.u.)")
        ax_top.set_title(f"{tag}")
        ax_top.legend(frameon=False, fontsize=9)

        # --- Bottom: residual + ±sigma band ---
        ax_bot.set_xlabel("ppm")
        ax_bot.set_ylabel("Residuals")
        ax_bot.invert_xaxis()

        residual = np.real(hf_ft) - np.real(pred_ft)

        # align lengths (should all be 2P, but be safe)
        m = min(len(x_ppm), len(residual))
        xs = x_ppm[:m]
        rs = residual[:m]

        # fill_between expects increasing x
        order = np.argsort(xs)
        xs_s = xs[order]
        rs_s = rs[order]

        # ax_bot.plot(xs_s, rs_s, lw=1.0, color="black", alpha=0.9, label="HF − Pred")

        band = sigma_to_freq_band(sigma, points=points, acq_time=acq_time, decay=decay)
        if band is not None:
            band = band[:m][order]
            ax_bot.fill_between(xs_s, -band, band, alpha=0.25, color="purple", label="±σ band")

        ax_bot.legend(frameon=False, fontsize=8)

        # --- Navigation ---
        def on_key(event):
            nonlocal idx, jump
            if event.key == "n":
                plt.close(fig)
                idx += 1
            elif event.key and event.key.isdigit():
                jump += event.key
            elif event.key == "enter":
                if jump:
                    want = f"vd{jump}"
                    hit = None
                    for j, t in enumerate(tags):
                        if t.startswith(want):
                            hit = j
                            break
                    if hit is not None:
                        idx = hit
                        print(f"⏩ Jump -> {tags[idx]}")
                jump = ""
                plt.close(fig)
            elif event.key == "escape":
                print("👋 bye")
                plt.close("all")
                raise SystemExit(0)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()


if __name__ == "__main__":
    main()
