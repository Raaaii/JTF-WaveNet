import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_xy(path: str):
    arr = np.genfromtxt(path, delimiter=",")
    if arr.ndim == 1 and arr.size == 0:
        raise ValueError(f"Empty file: {path}")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] == 2:
        x, y = arr[:, 0], arr[:, 1]
    elif arr.shape[0] == 2:
        x, y = arr[0], arr[1]
    else:
        raise ValueError(f"Expected 2 columns or 2 rows in {path}, got {arr.shape}")
    return x, y


def plot(loss_x, loss_y, label="loss", color="purple"):
    plt.plot(loss_x, loss_y, color=color, label=label)
    plt.scatter(loss_x, loss_y, color=color)
    plt.xlabel("batches")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend()
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", help="path to loss.log")
    p.add_argument("--label", default="loss")
    p.add_argument("--color", default="purple")
    args = p.parse_args()

    x, y = load_xy(args.file)
    plot(x, y, label=args.label, color=args.color)


if __name__ == "__main__":
    main()
