# miso-food-classifier-pipeline

> ## ⚠️ Disclaimer — Template / Reference Code Only
>
> **This repository is TEMPLATE code and is NOT intended to be deployed to
> production under any circumstances.** It exists solely to help the Miso
> Robotics team get started with an MLOps pipeline and model-training workflow —
> as a reference and learning scaffold, not a production-ready system.
>
> Before any real use, the Miso team must independently review, test, harden,
> and adapt this code to their own requirements, data, security standards, and
> operational practices. In particular:
>
> - **Not production-hardened** — no guarantees of correctness, security,
>   reliability, performance, or fitness for any purpose.
> - **Smoke-scale only** — any accuracy or quality numbers produced by the
>   included smoke configuration are **not meaningful** (tiny, non-representative
>   sample data). They demonstrate that the pipeline *executes*, not that the
>   model *works*.
> - **Review before deploying** — IAM policies, data handling, model-quality
>   gates, and infrastructure settings are illustrative and must be validated
>   against Miso's own standards.
>
> Provided "as is", without warranty of any kind. Use at your own discretion.

---

## Overview

An MLOps pipeline that preprocesses image data and trains a per-brand food
image classifier on Amazon SageMaker. It models the full workflow — from raw
data selection through data-quality checks, drift detection, training,
evaluation, and model registration — as a single SageMaker Pipeline DAG with
fail-closed quality gates, so a model can only be registered (for manual
approval) after every gate passes.

The project is **config-driven**: `config.yaml` is the source of truth for the
brands, data thresholds, and quality gates. Terraform provisions one SageMaker
pipeline and one Model Package Group per configured brand.

## Pipeline flow

```text
SelectRecentBrandData
  └─ DatasetBoundsGate            (fail → FailDatasetBounds)
       └─ EmbedAndDeduplicate
            └─ KFoldCleanlabQuality
                 └─ CleanlabQualityGate   (fail → FailCleanlabQuality)
                      └─ DetectDrift
                           └─ DriftGate    (fail → FailDriftGate)
                                └─ TrainBrandClassifier
                                     └─ EvaluateModel
                                          └─ EvaluateModelQualityGate  (fail → FailEvaluationGate)
                                               └─ RegisterModelPendingManualApproval
```

The final step registers the model with `ModelApprovalStatus =
PendingManualApproval` — a registry state, **not** an automatic deploy. A human
must review and promote it through the normal Model Registry approval workflow.

## Repository structure

| Path | Contents |
| --- | --- |
| `pipeline/` | Pipeline DAG builder (`pipeline.py`), SDK deploy script (`deploy_pipeline.py`), definition exporter, and per-step container entrypoints under `steps/` |
| `processing/` | Data-cleaning step implementations (selection, embeddings, k-fold + CleanLab, evaluate, register) |
| `training/` | Model training entry point (`train.py`) |
| `infra/` | Terraform for S3 buckets, IAM, SageMaker pipelines and model package groups |
| `lambdas/` | Trigger/handler Lambda stubs |
| `monitoring/` | Model-monitor / drift setup |
| `migration/` | DataSync migration helper |
| `tests/` | Offline unit tests (no AWS required) |
| `config.yaml` | Production configuration (placeholders — fill in before use) |
| `config.smoke.yaml` | Relaxed thresholds for an end-to-end smoke run (never for real training) |

## Configuration

`config.yaml` ships with **placeholder** values (`<replace-with-...>`) for the
AWS account ID, role ARN, bucket names, backbone path, and SNS topics. Replace
them with your own before running anything against AWS. The production floors
(`min_images_per_class: 500`, `target_dataset_min: 6000`,
`accuracy_threshold: 0.97`) reflect real data-scale and quality expectations.

`config.smoke.yaml` is a **separate overlay** with those floors deliberately
relaxed so a tiny sample dataset can traverse the whole DAG. It is for proving
the pipeline *executes* end to end — never for a real training run, and its
results are not meaningful.

## Running locally (no AWS)

```bash
pip install -r requirements.txt

# Validate the DAG as JSON (no SageMaker SDK, no AWS credentials)
python pipeline/export_definition.py --brand white-castle --output /tmp/pipeline.json

# Run the offline unit tests
pytest -q
```

## Deploying the pipeline (requires AWS)

> Only after replacing the placeholders in `config.yaml` with real values, and
> after your own security review.

```bash
# Provision infrastructure (per-brand pipelines, buckets, model groups)
cd infra
terraform init
terraform plan
terraform apply

# Or build/upsert the pipeline via the SageMaker SDK
python pipeline/deploy_pipeline.py --config config.yaml --upsert
```

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request:

- **ruff** lint (`ruff check .`)
- **pytest** (offline, no AWS)
- **terraform** `fmt -check` + `validate`

## License / warranty

See the disclaimer at the top. Provided "as is", without warranty of any kind.
