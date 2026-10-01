"""Step 2: inexpensive K-fold out-of-sample probabilities from embeddings."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def _make_classifier(kind: str, num_classes: int, seed: int):
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.15,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob" if num_classes > 2 else "binary:logistic",
                eval_metric="mlogloss",
                n_jobs=1,
                random_state=seed,
            )
        except ImportError:
            pass
    # This fallback is intentionally non-logistic and requires only scikit-learn.
    return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=1)


def kfold_predict(
    embeddings: np.ndarray,
    labels: Iterable[str | int],
    *,
    n_folds: int = 5,
    classifier: str = "xgboost",
    seed: int = 2026,
) -> tuple[np.ndarray, list[str]]:
    """Generate an N x classes matrix where every row is out-of-sample."""
    features = np.asarray(embeddings, dtype=np.float32)
    raw_labels = np.asarray(list(labels))
    if features.ndim != 2 or len(features) != len(raw_labels):
        raise ValueError("embeddings and labels must have matching row counts")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(raw_labels)
    counts = np.bincount(encoded)
    folds = min(n_folds, int(counts.min()))
    if folds < 2:
        raise ValueError("Each class needs at least two records for K-fold prediction")
    probabilities = np.zeros((len(features), len(encoder.classes_)), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_indices, validation_indices in splitter.split(features, encoded):
        model = _make_classifier(classifier, len(encoder.classes_), seed)
        model.fit(features[train_indices], encoded[train_indices])
        fold_probabilities = model.predict_proba(features[validation_indices])
        # A fold's training split may omit a class, so map each predicted column
        # back to its global class index instead of assuming all classes appear.
        for column, class_index in enumerate(model.classes_.astype(int)):
            probabilities[validation_indices, class_index] = fold_probabilities[:, column]
    return probabilities.astype(np.float32), encoder.classes_.astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cheap embedding-space K-fold prediction")
    parser.add_argument("--embeddings", default="/opt/ml/processing/input/embeddings.npy")
    parser.add_argument("--mapping", default="/opt/ml/processing/input/embedding_mapping.json")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--classifier", default="xgboost", choices=("xgboost", "fallback"))
    args = parser.parse_args()
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    probabilities, classes = kfold_predict(
        np.load(args.embeddings),
        [record["label"] for record in mapping],
        n_folds=args.n_folds,
        classifier=args.classifier,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "pred_probs.npy", probabilities)
    (output_dir / "classes.json").write_text(json.dumps(classes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
