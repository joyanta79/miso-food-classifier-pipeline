"""Runtime dependency bootstrap for SageMaker DLC containers.

The TensorFlow DLC image ships TensorFlow but not the lightweight ML libs the
processing steps need (scikit-learn, xgboost, cleanlab, pillow, pyyaml). Each
step calls ``ensure(...)`` at startup to pip-install any missing packages into
the running container before importing them. This keeps the pipeline self
contained without a custom image build.

Security contract (CWE-494: Download of Code Without Integrity Check):
    * The install is NOT allowed to fall back to bare package names resolved
      from whatever index the environment happens to reach. If an operator
      wants a runtime install, they MUST supply a fully-pinned, hash-locked
      requirements file via ``MISO_PIP_REQUIREMENTS_LOCK`` (default search
      path ``/opt/ml/processing/input/pkgcode/requirements.lock``) and a
      private index URL via ``MISO_PIP_INDEX_URL`` (or set
      ``MISO_PIP_NO_INDEX=1`` to disable network access altogether and rely
      on a wheel directory served via ``MISO_PIP_FIND_LINKS``).
    * ``--require-hashes --no-deps`` is always passed so no transitive
      distribution is pulled or executed without an operator-reviewed hash.
    * The preferred deployment is to bake every dependency into the container
      image and set ``MISO_PIP_DISABLE_BOOTSTRAP=1`` so ``ensure()`` becomes
      a no-op verification shim. In that mode missing modules raise instead
      of being silently installed.

Python 3.10 compatible.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

# Default locations when the pkgcode / code channel is mounted inside a job.
_DEFAULT_LOCK_SEARCH = (
    "/opt/ml/processing/input/pkgcode/requirements.lock",
    "/opt/ml/input/data/code/requirements.lock",
    "/opt/ml/code/requirements.lock",
)


def _truthy(value):
    return bool(value) and value.strip().lower() not in {"", "0", "false", "no", "off"}


def _find_lockfile():
    override = os.environ.get("MISO_PIP_REQUIREMENTS_LOCK")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    for candidate in _DEFAULT_LOCK_SEARCH:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def _missing_modules(packages):
    missing = []
    for module_name, requirement in packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(requirement)
    return missing


def ensure(packages):
    """Ensure each importable module exists.

    ``packages`` maps an import name -> pip requirement string, e.g.
    {"PIL": "pillow", "sklearn": "scikit-learn"}. The ``requirement`` string
    is retained only for diagnostic messaging; the actual install uses the
    pinned + hash-locked requirements file described in the module
    docstring. Bare names are never resolved against a remote index.
    """
    missing = _missing_modules(packages)
    if not missing:
        return

    if _truthy(os.environ.get("MISO_PIP_DISABLE_BOOTSTRAP")):
        raise RuntimeError(
            "[bootstrap] missing modules "
            + repr(missing)
            + " but MISO_PIP_DISABLE_BOOTSTRAP=1 forbids runtime installs; "
            "rebuild the container image with these dependencies baked in."
        )

    lockfile = _find_lockfile()
    if lockfile is None:
        raise RuntimeError(
            "[bootstrap] missing modules "
            + repr(missing)
            + " but no hash-locked requirements file was found. Set "
            "MISO_PIP_REQUIREMENTS_LOCK to a pip-compile-generated "
            "requirements.lock (with --generate-hashes) or bake the deps "
            "into the container image and set MISO_PIP_DISABLE_BOOTSTRAP=1."
        )

    cmd = _build_pip_argv(lockfile)
    print(
        "[bootstrap] installing from " + str(lockfile) + " (missing: " + repr(missing) + ")",
        flush=True,
    )
    subprocess.check_call(cmd)


def _build_pip_argv(lockfile):
    """Build pip argv that refuses to resolve bare names from the public index."""
    index_url = os.environ.get("MISO_PIP_INDEX_URL")
    no_index = _truthy(os.environ.get("MISO_PIP_NO_INDEX"))
    find_links = os.environ.get("MISO_PIP_FIND_LINKS")
    trusted_host = os.environ.get("MISO_PIP_TRUSTED_HOST")

    if not no_index and not index_url:
        raise RuntimeError(
            "[bootstrap] refusing to run pip install without a pinned "
            "index. Set MISO_PIP_INDEX_URL to a private, write-restricted "
            "mirror, or set MISO_PIP_NO_INDEX=1 together with "
            "MISO_PIP_FIND_LINKS pointing at a vetted wheel directory."
        )
    if no_index and not find_links:
        raise RuntimeError(
            "[bootstrap] MISO_PIP_NO_INDEX=1 requires MISO_PIP_FIND_LINKS to "
            "point at a vetted local wheel directory."
        )

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-input",
        "--require-hashes",
        "--no-deps",
        "--only-binary=:all:",
        "-r",
        str(lockfile),
    ]
    if no_index:
        cmd.append("--no-index")
    if index_url:
        cmd.extend(["--index-url", index_url])
    if find_links:
        cmd.extend(["--find-links", find_links])
    if trusted_host:
        cmd.extend(["--trusted-host", trusted_host])
    return cmd
