from __future__ import annotations

import runpy
from pathlib import Path


def cli() -> None:
    """
    Console entrypoint for evaluation.

    This runs the repository script at scripts/eval.py while keeping
    the installed package clean (no heavy plotting code inside src/).
    """
    repo_root = (
        Path(__file__).resolve().parents[3]
    )  # .../src/jtf_wavenet/cli/eval_runner.py -> repo root
    script = repo_root / "scripts" / "eval.py"

    if not script.exists():
        raise FileNotFoundError(
            f"Could not find {script}. " "Are you running this command from a source checkout?"
        )

    # Executes scripts/eval.py as if: python scripts/eval.py <args>
    runpy.run_path(str(script), run_name="__main__")
