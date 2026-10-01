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
CHECKED_IN_DEFINITION = PROJECT_ROOT / "infra" / "pipeline_definition.json"


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
        validate_dry_run_definition(definition)

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
    assert parameters["CleanlabFlagThresholdFraction"] == config["cleanlab"]["flag_threshold_fraction"]
    assert parameters["UseSpot"] is config["training"]["use_spot"]
    assert parameters["AccuracyThreshold"] == config["evaluation"]["accuracy_threshold"]
    assert parameters["CorrectionRateMultiplier"] == config["drift"]["correction_rate_multiplier"]
    assert parameters["ReviewTopicArn"] == config["notifications"]["review_topic_arn"]
    assert parameters["MockAws"] is config["local_mode"]["mock_aws"]


def test_checked_in_definition_matches_white_castle_dry_run() -> None:
    checked_in = json.loads(CHECKED_IN_DEFINITION.read_text(encoding="utf-8"))
    expected = json.loads(render_dry_run_json(DEFAULT_CONFIG_PATH, "white-castle"))

    assert checked_in == expected
