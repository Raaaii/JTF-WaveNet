import argparse
from jtf_wavenet.vis.model_checks import run_model_shape_smoke_test


def main() -> None:
    p = argparse.ArgumentParser(description="JTF-WaveNet model shape smoke test")
    p.add_argument("--points", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--stage", type=str, default="stage2", choices=["stage1", "stage2"])
    args = p.parse_args()

    run_model_shape_smoke_test(points=args.points, batch_size=args.batch_size, stage=args.stage)


if __name__ == "__main__":
    main()
