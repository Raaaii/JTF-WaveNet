"""
Run all loss checks from one entrypoint.
"""

from __future__ import annotations

from jtf_wavenet.vis.loss_checks.check_sbece import run_check_sbece
from jtf_wavenet.vis.loss_checks.check_xi_momenta import run_check_xi_momenta


def main():
    # Keep these small-ish so it runs fast while you iterate.
    run_check_sbece(batch_size=4, points=4096, num_bins=10, temp=0.01, seed=0, show_plots=True)
    run_check_xi_momenta(batch_size=2, points=4096, seed=1, show_plots=True)


if __name__ == "__main__":
    main()
