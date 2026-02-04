from __future__ import annotations

import argparse
from pathlib import Path

from jtf_wavenet.training.train import main as train_main


def cli() -> None:
    p = argparse.ArgumentParser(prog="jtfwavenet-train")
    p.add_argument("--stage", choices=["stage1", "stage2", "both"], default="stage1")
    p.add_argument("--ckpt", default="checkpoints", help="Checkpoint root directory")
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--keep", type=int, default=2, help="How many checkpoints to keep")
    args = p.parse_args()

    # optional: create root
    Path(args.ckpt).mkdir(parents=True, exist_ok=True)

    train_main(
        stage=args.stage,
        checkpoint_root=args.ckpt,
        batch_size=args.batch,
        steps_per_epoch=args.steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        rounds=args.rounds,
        check_point_count=args.keep,
    )
