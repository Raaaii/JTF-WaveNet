from __future__ import annotations

import os
from pathlib import Path

from jtf_wavenet.data.parameter_sampling import load_config


def get_default_config_path() -> Path:
    """
    Return the default generator config JSON path.

    Precedence:
      1) env var JTFWAVENET_CONFIG
      2) <repo_root>/configs/default_generator.json (dev)
      3) <package_root>/../configs/default_generator.json (installed fallback)
    """
    env = os.getenv("JTFWAVENET_CONFIG")
    if env:
        return Path(env).expanduser().resolve()

    # Locate repo root by walking up until we find pyproject.toml
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists():
            return (p / "configs" / "default_generator.json").resolve()

    # Fallback: if installed, try alongside package (best effort)
    return (here.parents[1] / "configs" / "default_generator.json").resolve()


def load_default_config() -> dict:
    return load_config(get_default_config_path())
