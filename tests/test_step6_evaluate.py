import json
from pathlib import Path

from processing.step6_evaluate import evaluate, main, parse_args


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
    assert report["model_metrics"]["classification_metrics"]["done_when_not_done_fp_rate"]["value"] == 0.5


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
