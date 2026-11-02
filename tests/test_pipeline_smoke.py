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
    assert parameters["UseSpot"] is config["training"]["use_spot"]
    assert parameters["ReviewTopicArn"] == config["notifications"]["review_topic_arn"]
    assert parameters["MockAws"] is config["local_mode"]["mock_aws"]

    # Safety-critical gate thresholds must not be exposed as execution parameters,
    # because SageMaker PipelineParameters are overridable per execution. They are
    # instead baked as literal RightValues into the Condition steps.
    forbidden = {
        "AccuracyThreshold",
        "FpRateEmergencyThreshold",
        "LatencyBudgetMs",
        "CleanlabFlagThresholdFraction",
        "Level1ConfidenceFloor",
        "CorrectionRateMultiplier",
        "TargetDatasetMin",
        "TargetDatasetMax",
        "MinImagesPerClass",
    }
    assert forbidden.isdisjoint(parameters), (
        "Safety-critical thresholds leaked back into pipeline parameters: "
        f"{sorted(forbidden & parameters.keys())}"
    )


def test_safety_gate_right_values_are_literal_and_match_config() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    definition = build_dry_run_definition(DEFAULT_CONFIG_PATH)
    gates = {
        step["Name"]: step
        for step in _flatten_steps(definition["Steps"])
        if step["Type"] == "Condition"
    }

    def _right_values(step_name: str) -> list:
        return [
            condition["RightValue"]
            for condition in gates[step_name]["Arguments"]["Conditions"]
        ]

    # None of the RightValues may be a parameter reference ({"Get": "Parameters.*"}).
    for name, step in gates.items():
        for condition in step["Arguments"]["Conditions"]:
            right = condition["RightValue"]
            assert not isinstance(right, dict), (
                f"Gate {name!r} RightValue must be a literal, not a reference: {right!r}"
            )
            assert isinstance(right, (int, float)) and not isinstance(right, bool), (
                f"Gate {name!r} RightValue must be numeric: {right!r}"
            )

    evaluation = config["evaluation"]
    drift = config["drift"]
    cleanlab = config["cleanlab"]
    selection = config["data_selection"]
    assert _right_values("EvaluateModelQualityGate") == [
        evaluation["accuracy_threshold"],
        evaluation["fp_rate_emergency_threshold"],
        evaluation["latency_budget_ms"],
    ]
    assert _right_values("DriftGate") == [
        drift["level1_confidence_floor"],
        drift["correction_rate_multiplier"],
    ]
    assert _right_values("CleanlabQualityGate") == [cleanlab["flag_threshold_fraction"]]
    assert _right_values("DatasetBoundsGate") == [
        selection["target_dataset_min"],
        selection["target_dataset_max"],
        selection["min_images_per_class"],
    ]


def test_checked_in_definition_matches_white_castle_dry_run() -> None:
    checked_in = json.loads(CHECKED_IN_DEFINITION.read_text(encoding="utf-8"))
    expected = json.loads(render_dry_run_json(DEFAULT_CONFIG_PATH, "white-castle"))

    assert checked_in == expected
