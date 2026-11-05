import json
from pathlib import Path

import pytest

from processing.step6_evaluate import (
    EvaluationAbort,
    evaluate,
    main,
    parse_args,
)


def test_evaluate_calculates_done_when_not_done_fp_rate_and_fails_gate():
    predictions = [
        {"label": "done", "prediction": "done", "confidence": 0.99},
        {"label": "not_done", "prediction": "done", "confidence": 0.98},
        {"label": "not_done", "prediction": "not_done", "confidence": 0.80},
        {"label": "done", "prediction": "done", "confidence": 0.95},
    ]

    report = evaluate(
        predictions,
        {"evaluation": {"accuracy_threshold": 0.75, "fp_rate_emergency_threshold": 0.25}},
    )

    assert report["metrics"]["accuracy"] == 0.75
    assert report["metrics"]["done_when_not_done_fp_rate"] == 0.5
    assert report["quality_gate"] == "failed"
    assert (
        report["model_metrics"]["classification_metrics"]["done_when_not_done_fp_rate"]["value"]
        == 0.5
    )


def test_main_writes_sagemaker_evaluation_report(tmp_path: Path, monkeypatch):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"label": "done", "prediction": "done", "confidence": 0.99}),
                json.dumps({"label": "not_done", "prediction": "not_done", "confidence": 0.89}),
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "evaluation:\n  accuracy_threshold: 0.97\n  fp_rate_emergency_threshold: 0.05\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        "sys.argv",
        [
            "step6_evaluate.py",
            "--config",
            str(config),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
        ],
    )

    report = main(parse_args())

    assert report["quality_gate"] == "passed"
    written = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert written == report


def test_evaluate_fails_closed_when_hazardous_class_absent():
    # Dataset uses 'raw' as the hazardous label; without --not-done-label
    # propagation the primitive used to publish a null fp_rate instead of
    # failing closed.
    predictions = [
        {"label": "done", "prediction": "done", "confidence": 0.99},
        {"label": "raw", "prediction": "done", "confidence": 0.80},
    ]

    report = evaluate(predictions, {"evaluation": {}}, done_label="done")

    assert report["quality_gate"] == "failed"
    assert report["metrics"]["emergency_false_positive_rate"] == 1.0
    assert report["metrics"]["accuracy"] == 0.0
    assert "abort_reason" in report["metrics"]
    # Canonical gate-consumed fields are present.
    for key in (
        "accuracy",
        "emergency_false_positive_rate",
        "latency_ms",
        "sample_count",
        "not_done_sample_count",
        "done_sample_count",
        "missing_images",
    ):
        assert key in report["metrics"], key


def test_evaluate_honours_custom_not_done_label():
    predictions = [
        {"label": "done", "prediction": "done", "confidence": 0.99},
        {"label": "raw", "prediction": "done", "confidence": 0.80},
        {"label": "raw", "prediction": "raw", "confidence": 0.70},
    ]

    report = evaluate(
        predictions,
        {"evaluation": {"accuracy_threshold": 0.5, "fp_rate_emergency_threshold": 0.9}},
        done_label="done",
        not_done_label="raw",
    )

    assert report["metrics"]["not_done_sample_count"] == 2
    assert report["metrics"]["done_sample_count"] == 1
    assert report["metrics"]["emergency_false_positive_rate"] == 0.5
    assert report["metrics"]["done_when_not_done_fp_rate"] == 0.5


def test_main_predictions_cli_fails_closed_when_hazardous_class_absent(
    tmp_path: Path, monkeypatch
):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"label": "done", "prediction": "done", "confidence": 0.99}),
                json.dumps({"label": "raw", "prediction": "done", "confidence": 0.80}),
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "evaluation:\n  accuracy_threshold: 0.5\n  fp_rate_emergency_threshold: 0.5\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(
        "sys.argv",
        [
            "step6_evaluate.py",
            "--config",
            str(config),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
            # Default not_done_label='not_done' which is absent from the data.
        ],
    )

    with pytest.raises(EvaluationAbort):
        main(parse_args())

    # The sentinel report must still be written so operators can inspect it.
    written = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert written["emergency_false_positive_rate"] == 1.0
    assert written["accuracy"] == 0.0
    assert "abort_reason" in written
