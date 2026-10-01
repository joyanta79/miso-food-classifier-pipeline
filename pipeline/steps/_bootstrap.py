"""Runtime dependency bootstrap for SageMaker DLC containers.

The TensorFlow DLC image ships TensorFlow but not the lightweight ML libs the
processing steps need (scikit-learn, xgboost, cleanlab, pillow, pyyaml). Each
step calls ``ensure(...)`` at startup to pip-install any missing packages into
the running container before importing them. This keeps the pipeline self
contained without a custom image build.

Python 3.10 compatible.
"""

from __future__ import annotations

import importlib
import subprocess
import sys


def ensure(packages: dict[str, str]) -> None:
    """Ensure each importable module exists, pip-installing its dist if missing.

    ``packages`` maps an import name -> pip requirement string, e.g.
    {"PIL": "pillow", "sklearn": "scikit-learn"}.
    """
    missing = []
    for module_name, requirement in packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(requirement)
    if missing:
        print(f"[bootstrap] installing: {missing}", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-input", *missing]
        )
