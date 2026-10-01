# Miso Robotics ML Pipeline Architecture

## Scope and boundaries

This repository defines the SageMaker orchestration, its infrastructure, and its
interface contracts. It deliberately does **not** contain implementations for the
processing or training workloads. At execution time, the SageMaker containers read
these files from `s3://<pipeline-artifacts-bucket>/code/`:

- `data_selection.py`
- `embedding_dedup.py`
- `kfold_cleanlab.py`
- `drift_detection.py`
- `train.py`
- `evaluation.py`

Those files are owned and released by their respective implementation tracks. The
pipeline definition makes their inputs, outputs, environments, and metrics
contracts explicit without duplicating them.

## Per-brand pipeline isolation

`config.yaml` is the source of truth for the configured brands. Terraform creates
one SageMaker pipeline and one Model Package Group per brand. Each pipeline is
compiled from the same baseline definition but replaces the baseline brand and
registry group with the declared values for that brand.

This ensures that:

- raw data is selected only through the configured brand prefix;
- training and pipeline artifact prefixes are brand-specific;
- model packages never share a registry group across brands; and
- a caller cannot select an undeclared brand through the exporter.

The checked-in definition is the `white-castle` baseline. The local exporter can
render the dry-run DAG for any configured brand:

```bash
python pipeline/export_definition.py --brand brand-b --output /tmp/brand-b-pipeline.json
```

## Pipeline flow

```text
SelectRecentBrandData
  └─ DatasetBoundsGate
       ├─ fail: FailDatasetBounds
       └─ EmbedAndDeduplicate
            └─ KFoldCleanlabQuality
                 └─ CleanlabQualityGate
                      ├─ fail: FailCleanlabQuality
                      └─ DetectDrift
                           └─ DriftGate
                                ├─ fail: FailDriftGate
                                └─ TrainBrandClassifier
                                     └─ EvaluateModel
                                          └─ EvaluateModelQualityGate
                                               ├─ fail: FailEvaluationGate
                                               └─ RegisterModelPendingManualApproval
```

### Data and quality controls

| Stage | Configuration controls | Required output metric |
| --- | --- | --- |
| `SelectRecentBrandData` | brand naming pattern, six-month recency window, per-class minimum, dataset min/max | `dataset_size`, `minimum_class_count` |
| `EmbedAndDeduplicate` | backbone provenance/path, 1280-dim embeddings, cosine threshold | deduplicated dataset |
| `KFoldCleanlabQuality` | five folds, XGBoost classifier, label-quality flag threshold | `flagged_fraction` |
| `DetectDrift` | confidence baseline, Level 1 floor, Level 2 threshold, correction multiplier | `confidence`, `correction_rate_multiplier` |
| `TrainBrandClassifier` | instance type, managed spot setting, epochs, batch size, learning rate, checkpoint interval, image size | model artifacts and checkpoints |
| `EvaluateModel` | accuracy, emergency false-positive, and latency thresholds | `accuracy`, `emergency_false_positive_rate`, `latency_ms` |

The four gates fail closed. A model can reach the registry only after dataset,
Cleanlab, drift, and evaluation gates succeed.

## Approval and notifications

The final registry step sets `ModelApprovalStatus` to `PendingManualApproval`.
This is intentionally a registry state, not an automatic deploy signal. A human
reviewer must inspect the package and explicitly promote it using the normal
Model Registry approval workflow. The review, ML lead, and VP Engineering SNS ARN
parameters are carried through the model metadata contract for the notification
implementation; this track does not create a Lambda or notification handler.

## Local dry run and SDK mode

`pipeline/pipeline.py` always supports a complete local, JSON-serialisable DAG.
It validates required steps, branch uniqueness, required parameters, and the
manual approval safeguard without importing the SageMaker SDK or using AWS
credentials.

When the optional SageMaker SDK is installed, `build_sagemaker_pipeline()` builds
a native SDK graph against the same config and remote code contracts. Use
`python pipeline/export_definition.py --sdk` only after replacing the placeholder
account, bucket, backbone, and role values in `config.yaml`.

## Terraform controls

Terraform creates four private, SSE-S3 encrypted buckets:

- training
- model-artifacts
- pipeline-artifacts
- raw-archive

All are S3 Standard: no lifecycle transition or non-Standard storage rule exists.
Only `model-artifacts` enables S3 versioning. Terraform also creates per-brand
Model Package Groups and SageMaker pipeline resources. It expects an existing
least-privilege SageMaker execution role rather than creating an over-privileged
role.
