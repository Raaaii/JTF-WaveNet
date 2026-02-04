import argparse

from jtf_wavenet.vis.model_checks_with_generator import (
    run_generator_shapes_only,
    run_generator_to_model_forward,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Generator ↔ Model smoke checks")
    p.add_argument("--mode", choices=["shapes", "forward"], default="forward")
    p.add_argument("--points", type=int, default=4096)
    p.add_argument("--stage", choices=["stage1", "stage2"], default="stage2")
    args = p.parse_args()

    if args.mode == "shapes":
        run_generator_shapes_only()
    else:
        run_generator_to_model_forward(points=args.points, stage=args.stage)


if __name__ == "__main__":
    main()
