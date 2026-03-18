import argparse
import numpy as np
import matplotlib.pyplot as plt

from jtf_wavenet.training.callbacks import CustomLearningSchedule
from jtf_wavenet.vis.style import set_science_style


def main():
    p = argparse.ArgumentParser("Plot CustomLearningSchedule")
    p.add_argument("--steps", type=int, default=600_00000)
    p.add_argument("--every", type=int, default=1000, help="sample every N steps")
    p.add_argument("--d-model", type=float, default=20 * 20 * 13)
    p.add_argument("--warmup-steps", type=int, default=200_000)
    p.add_argument("--baseline-lr", type=float, default=2e-8)
    p.add_argument("--warmup-exp", type=float, default=-1.5)
    p.add_argument("--value-scalar", type=float, default=100.0)
    p.add_argument("--max-lr", type=float, default=9e-5)
    p.add_argument("--out", type=str, default="", help="optional path to save png/pdf")
    args = p.parse_args()

    set_science_style()

    sched = CustomLearningSchedule(
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
        baseline_learning_rate=args.baseline_lr,
        warmup_exponent=args.warmup_exp,
        value_scalar=args.value_scalar,
        max_learning_rate=args.max_lr,
    )

    steps = np.arange(0, args.steps + 1, args.every, dtype=np.int64)
    lrs = np.array([float(sched(int(s)).numpy()) for s in steps], dtype=float)

    plt.figure(figsize=(9, 4))
    plt.plot(steps, lrs)
    plt.xlabel("optimizer step")
    plt.ylabel("learning rate")
    plt.title("CustomLearningSchedule")
    plt.yscale("log")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
