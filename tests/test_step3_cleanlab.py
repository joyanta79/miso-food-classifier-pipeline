from processing.step3_cleanlab_filter import filter_label_issues
from tests.synthetic_data.generator import mislabeled_probabilities


def test_cleanlab_filter_recalls_at_least_eighty_percent_of_injected_mislabels() -> None:
    probabilities, mapping, classes, injected_ids = mislabeled_probabilities(
        injected_mislabels=10
    )

    flagged, cleaned, report = filter_label_issues(
        probabilities, mapping, classes, flag_threshold_fraction=0.25
    )

    flagged_ids = {record["id"] for record in flagged}
    recall = len(flagged_ids & injected_ids) / len(injected_ids)
    assert recall >= 0.80
    assert report["flagged_fraction"] == len(flagged) / len(mapping)
    assert len(cleaned) + len(flagged) == len(mapping)
    assert all(record["reason"] in {"mislabel", "ambiguous"} for record in flagged)
