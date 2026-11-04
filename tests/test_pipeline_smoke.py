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
        steps = _flatten_steps(definition["Steps"])
        register_step = next(
            step
            for step in steps
            if step["Name"] == "RegisterModelPendingManualApproval"
        )
        select_step = next(
            step for step in steps if step["Name"] == "SelectRecentBrandData"
        )

        # Brand identity is baked into the definition as literals (not pipeline
        # parameters) so a StartPipelineExecution caller cannot redirect the
        # registered model into another brand's registry or forge the brand_id
        # metadata (CWE-639).
        assert "BrandId" not in parameters
        assert "BrandNamingPattern" not in parameters
        assert "ModelPackageGroup" not in parameters
        assert definition["Metadata"]["BrandId"] == brand["id"]
        assert select_step["Arguments"]["Environment"]["BRAND_ID"] == brand["id"]
        assert (
            select_step["Arguments"]["Environment"]["NAMING_PATTERN"]
            == brand["naming_pattern"]
        )
        assert (
            register_step["Arguments"]["ModelPackageGroupName"]
            == brand["model_package_group"]
        )
        assert (
            register_step["Arguments"]["CustomerMetadataProperties"]["brand_id"]
            == brand["id"]
        )
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

    # Safety-critical gate thresholds AND brand-identity values must not be
    # exposed as execution parameters, because SageMaker PipelineParameters are
    # overridable per execution. The identity triple is baked as literals into
    # the definition (RegisterModel.ModelPackageGroupName, Metadata.BrandId,
    # SelectRecentBrandData.Environment); the gate thresholds are baked as
    # literal RightValues into the Condition steps.
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
        "BrandId",
        "BrandNamingPattern",
        "ModelPackageGroup",
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
            if condition["Type"] == "ConditionEquals" and isinstance(right, str):
                continue  # literal string equality, e.g. detector == "cleanlab"
            assert isinstance(right, int | float) and not isinstance(right, bool), (
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
        evaluation["min_sample_count"],
        evaluation["min_per_class_count"],
        evaluation["min_per_class_count"],
        0,
    ]
    assert _right_values("DriftGate") == [
        drift["level1_confidence_floor"],
        drift["correction_rate_multiplier"],
    ]
    assert _right_values("CleanlabQualityGate") == [
        cleanlab["flag_threshold_fraction"],
        "cleanlab",
        selection["target_dataset_min"],
        selection["min_images_per_class"],
    ]
    assert _right_values("DatasetBoundsGate") == [
        selection["target_dataset_min"],
        selection["target_dataset_max"],
        selection["min_images_per_class"],
    ]


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
        steps = _flatten_steps(definition["Steps"])
        select_step = next(
            step for step in steps if step["Name"] == "SelectRecentBrandData"
        )
        register_step = next(
            step for step in steps if step["Name"] == "RegisterModelPendingManualApproval"
        )
        assert definition["Metadata"]["BrandId"] == brand["id"]
        assert select_step["Arguments"]["Environment"]["BRAND_ID"] == brand["id"]
        assert (
            select_step["Arguments"]["Environment"]["NAMING_PATTERN"]
            == brand["naming_pattern"]
        )
        assert (
            register_step["Arguments"]["ModelPackageGroupName"]
            == brand["model_package_group"]
        )


def test_validate_dry_run_rejects_brand_pattern_mismatch() -> None:
    """Pairing one brand's id with another brand's naming pattern must fail.

    This is the exact failure mode of the previous Terraform template clone:
    the brand id was substituted but the naming pattern token was not.
    """
    import pytest

    config = load_config(DEFAULT_CONFIG_PATH)
    definition = build_dry_run_definition(DEFAULT_CONFIG_PATH, "brand-b")
    # Simulate the "template clone" bug: overwrite the SelectRecentBrandData
    # NAMING_PATTERN literal with the white-castle default while leaving the
    # BrandId literal as brand-b.
    for step in _flatten_steps(definition["Steps"]):
        if step["Name"] == "SelectRecentBrandData":
            step["Arguments"]["Environment"]["NAMING_PATTERN"] = "wc_*"
            break
    with pytest.raises(ValueError, match="BrandNamingPattern"):
        validate_dry_run_definition(definition, config=config)
