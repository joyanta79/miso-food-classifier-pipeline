"""Offline tests for the SDK deploy script's optional wiring (no AWS calls)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.deploy_pipeline import _bootstrap_env, _correction_rate_wiring

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _drift_config(**extra: object) -> dict:
    return {"drift": {"correction_rate_multiplier": 3, **extra}}


def test_bootstrap_env_absent_keeps_fail_closed_default() -> None:
    assert _bootstrap_env({}) == {}
    assert _bootstrap_env({"bootstrap": None}) == {}


def test_bootstrap_env_maps_configured_keys() -> None:
    env = _bootstrap_env(
        {"bootstrap": {"pip_index_url": "https://mirror.example/simple", "disable": False}}
    )
    assert env == {
        "MISO_PIP_INDEX_URL": "https://mirror.example/simple",
        "MISO_PIP_DISABLE_BOOTSTRAP": "0",
    }
    assert _bootstrap_env({"bootstrap": {"disable": True}}) == {
        "MISO_PIP_DISABLE_BOOTSTRAP": "1"
    }


def test_bootstrap_env_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown bootstrap config keys"):
        _bootstrap_env({"bootstrap": {"pip_index": "typo"}})


def test_correction_rate_absent_wires_nothing() -> None:
    assert _correction_rate_wiring(_drift_config()) == ({}, [])


def test_correction_rate_uri_wires_env_and_input() -> None:
    env, inputs = _correction_rate_wiring(
        _drift_config(correction_rate_source_uri="s3://bucket/prefix/rates.json")
    )
    assert env == {
        "CORRECTION_RATE_SOURCE": "/opt/ml/processing/input/correction_rate/rates.json"
    }
    assert len(inputs) == 1
    assert inputs[0].source == "s3://bucket/prefix/rates.json"
    assert inputs[0].destination == "/opt/ml/processing/input/correction_rate"
    assert inputs[0].input_name == "in_correction_rate"


@pytest.mark.parametrize(
    "uri", ["/local/rates.json", "https://bucket/rates.json", "s3://bucket/rates.csv"]
)
def test_correction_rate_uri_must_be_s3_json(uri: str) -> None:
    with pytest.raises(ValueError, match="correction_rate_source_uri"):
        _correction_rate_wiring(_drift_config(correction_rate_source_uri=uri))


def test_requirements_lock_is_pinned_and_hashed() -> None:
    text = (PROJECT_ROOT / "requirements.lock").read_text(encoding="utf-8")
    entries = re.findall(
        r"^([A-Za-z0-9_.-]+)==(\S+) \\\n((?:\s+--hash=sha256:[0-9a-f]{64}(?: \\)?\n)+)",
        text,
        flags=re.MULTILINE,
    )
    names = {name.lower() for name, _version, _hashes in entries}
    assert {"xgboost", "cleanlab"} <= names
    # The DLC's TensorFlow stack must never be replaced by the runtime install.
    assert not names & {"numpy", "scipy", "scikit-learn", "tensorflow"}
    requirement_lines = [
        line for line in text.splitlines() if line and not line.startswith((" ", "#"))
    ]
    assert len(requirement_lines) == len(entries)
