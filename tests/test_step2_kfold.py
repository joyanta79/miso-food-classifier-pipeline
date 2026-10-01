import numpy as np

from processing.step2_kfold_predict import kfold_predict
from tests.synthetic_data.generator import embedding_fixture


def test_kfold_predictions_have_expected_shape_and_normalized_rows() -> None:
    embeddings, labels = embedding_fixture(classes=3, samples_per_class=8)

    probabilities, classes = kfold_predict(
        embeddings, labels, n_folds=4, classifier="fallback", seed=2026
    )

    assert probabilities.shape == (len(labels), 3)
    assert classes == ["class-0", "class-1", "class-2"]
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probabilities >= 0)
