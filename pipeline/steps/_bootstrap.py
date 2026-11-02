"""Runtime dependency bootstrap for SageMaker DLC containers.

The TensorFlow DLC image ships TensorFlow but not the lightweight ML libs the
processing steps need (scikit-learn, xgboost, cleanlab, pillow, pyyaml). Each
step calls ensure(...) at startup to pip-install any missing packages into
the running container before importing them. This keeps the pipeline self
contained without a custom image build.

The import probe accepts an optional "module:attribute" form so the caller
can require a concrete public symbol (for example
"cleanlab.filter:find_label_issues") rather than only a top-level package
name. This closes the silent-degradation window where "pip install cleanlab"
resolves to a release whose public API no longer exposes the symbol the step
actually uses.

Python 3.10 compatible.
"""

from __future__ import annotations

import importlib
import subprocess
import sys


def _probe(spec: str) -> None:
    """Import module or verify module:attribute is accessible.

    Raises ImportError when either the module or the required attribute is
    missing. Callers treat the raise as "not installed" and pin-install the
    requirement.
    """

    module_name, _, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    if attribute and not hasattr(module, attribute):
        raise ImportError(
            f"module {module_name!r} does not expose required symbol {attribute!r}"
        )


def ensure(packages: dict[str, str]) -> None:
    """Ensure each importable module (or module symbol) exists.

    packages maps an import spec -> pip requirement string. The spec is
    either a bare module name ("sklearn") or "<module>:<attribute>" to
    require a concrete symbol ("cleanlab.filter:find_label_issues"). The
    requirement string SHOULD be pinned ("cleanlab==2.6.4") to match
    requirements.txt so the runtime container does not resolve to an
    API-incompatible release.
    """
    missing = []
    for spec, requirement in packages.items():
        try:
            _probe(spec)
        except ImportError:
            missing.append(requirement)
    if missing:
        print(f"[bootstrap] installing: {missing}", flush=True)
        # check=True (via check_call) is load-bearing: a failed pip install
        # must abort the step loudly rather than let a downstream import
        # swallow the ImportError and silently fall back to a weaker detector.
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-input", *missing]
        )
        # Re-verify every spec after install so a resolver that reports success
        # but does not actually provide the required symbol still fails closed.
        for spec in packages:
            _probe(spec)
