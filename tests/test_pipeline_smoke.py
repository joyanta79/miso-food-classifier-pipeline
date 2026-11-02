from __future__ import annotations

import json
from pathlib import Path

from pipeline.pipeline import (
    DEFAULT_CONFIG_PATH,
    build_dry_run_definition,
    load_config,
    render_dry_run_json,
    validate_dry_run_definition,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFRA_DIR = PROJECT_ROOT / "infra"


def _flatten_steps(steps: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for step in steps:
        flattened.append(step)
        if step["Type"] == "Condition":
            flattened.extend(_flatten_steps(step["Arguments"]["IfSteps"]))
            flattened.extend(_flatten_steps(step["Arguments"]["ElseSteps"]))
    return flattened


def test_local_dry_run_is_complete_for_every_declared_brand() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    for brand in config["brands"]:
        definition = build_dry_run_definition(DEFAULT_CONFIG_PATH, brand["id"])
        validate_dry_run_definition(definition, config=config)

        parameters = {item["Name"]: item["DefaultValue"] for item in definition["Parameters"]}
        register_step = next(
            step
            for step in _flatten_steps(definition["Steps"])
            if step["Name"] == "RegisterModelPendingManualApproval"
        )

        assert parameters["BrandId"] == brand["id"]
        assert parameters["BrandNamingPattern"] == brand["naming_pattern"]
        assert parameters["ModelPackageGroup"] == brand["model_package_group"]
        assert register_step["Arguments"]["ModelApprovalStatus"] == "PendingManualApproval"


def test_dry_run_parameters_match_all_configured_gates() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    definition = build_dry_run_definition(DEFAULT_CONFIG_PATH)
    parameters = {item["Name"]: item["DefaultValue"] for item in definition["Parameters"]}

    assert parameters["RecencyMonths"] == config["data_selection"]["recency_months"]
    assert parameters["CosineDedupThreshold"] == config["embedding"]["cosine_dedup_threshold"]
    assert parameters["KFoldCount"] == config["kfold"]["n_folds"]
    assert (
        parameters["CleanlabFlagThresholdFraction"] == config["cleanlab"]["flag_threshold_fraction"]
    )
    assert parameters["UseSpot"] is config["training"]["use_spot"]
    assert parameters["AccuracyThreshold"] == config["evaluation"]["accuracy_threshold"]
    assert parameters["CorrectionRateMultiplier"] == config["drift"]["correction_rate_multiplier"]
    assert parameters["ReviewTopicArn"] == config["notifications"]["review_topic_arn"]
    assert parameters["MockAws"] is config["local_mode"]["mock_aws"]


def test_each_checked_in_brand_definition_matches_its_fresh_dry_run() -> None:
    """Every per-brand definition that Terraform consumes must match a freshly
    rendered dry-run DAG for that brand. This catches the "single-brand
    baseline cloned as a multi-brand template" failure mode in which a brand's
    BrandNamingPattern leaks into another brand's committed definition.
    """
    config = load_config(DEFAULT_CONFIG_PATH)
    for brand in config["brands"]:
        path = INFRA_DIR / f"pipeline_definition.{brand['id']}.json"
        assert path.exists(), (
            f"Missing per-brand definition {path.name}; regenerate with "
            f"`python pipeline/export_definition.py`."
        )
        checked_in = json.loads(path.read_text(encoding="utf-8"))
        expected = json.loads(render_dry_run_json(DEFAULT_CONFIG_PATH, brand["id"]))
        assert checked_in == expected, (
            f"Checked-in {path.name} is out of date; regenerate with the exporter."
        )


def test_each_checked_in_brand_definition_passes_consistency_validation() -> None:
    """The BrandId/BrandNamingPattern/ModelPackageGroup triple in each committed
    definition must be internally consistent with config.yaml.
    """
    config = load_config(DEFAULT_CONFIG_PATH)
    for brand in config["brands"]:
        path = INFRA_DIR / f"pipeline_definition.{brand['id']}.json"
        definition = json.loads(path.read_text(encoding="utf-8"))
        validate_dry_run_definition(definition, config=config)
        parameters = {item["Name"]: item["DefaultValue"] for item in definition["Parameters"]}
        assert parameters["BrandId"] == brand["id"]
        assert parameters["BrandNamingPattern"] == brand["naming_pattern"]
        assert parameters["ModelPackageGroup"] == brand["model_package_group"]


def test_validate_dry_run_rejects_brand_pattern_mismatch() -> None:
    """Pairing one brand's id with another brand's naming pattern must fail.

    This is the exact failure mode of the previous Terraform template clone:
    the brand id was substituted but the naming pattern token was not.
    """
    import pytest

    config = load_config(DEFAULT_CONFIG_PATH)
    definition = build_dry_run_definition(DEFAULT_CONFIG_PATH, "brand-b")
    # Simulate the "template clone" bug: overwrite BrandNamingPattern with the
    # white-castle default while leaving BrandId=brand-b.
    for item in definition["Parameters"]:
        if item["Name"] == "BrandNamingPattern":
            item["DefaultValue"] = "wc_*"
            break
    with pytest.raises(ValueError, match="BrandNamingPattern"):
        validate_dry_run_definition(definition, config=config)
