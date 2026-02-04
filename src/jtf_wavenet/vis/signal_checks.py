import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from jtf_wavenet.vis.style import set_science_style
from jtf_wavenet.signal.signal_function import (
    signal,
    calculate_vibration,
    combine_signal_and_vibration,
)


def quick_signal_plot(example_kwargs, title="Signal sanity check"):
    set_science_style()

    s = signal(**example_kwargs).numpy()
    plt.figure(figsize=(12, 4))
    plt.plot(np.real(s), label="real")
    plt.plot(np.imag(s), label="imag", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
