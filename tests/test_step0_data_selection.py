from datetime import UTC, datetime

from processing.step0_data_selection import select_data
from tests.synthetic_data.generator import inventory_for_selection


def test_select_data_filters_brand_recency_and_underrepresented_classes() -> None:
    now = datetime(2026, 9, 16, tzinfo=UTC)
    result = select_data(
        inventory_for_selection(now),
        naming_pattern="wc_*",
        recency_months=6,
        min_images_per_class=3,
        target_dataset_min=6,
        target_dataset_max=6,
        now=now,
    )

    assert len(result) == 6
    assert {item["label"] for item in result} == {"burger", "fries"}
    assert all(item["key"].split("/")[-1].startswith("wc_") for item in result)


def test_select_data_rejects_dataset_below_target_minimum() -> None:
    now = datetime(2026, 9, 16, tzinfo=UTC)
    try:
        select_data(
            inventory_for_selection(now),
            naming_pattern="wc_*",
            recency_months=6,
            min_images_per_class=3,
            target_dataset_min=7,
            target_dataset_max=10,
            now=now,
        )
    except ValueError as error:
        assert "below required minimum" in str(error)
    else:
        raise AssertionError("Expected under-sized curated dataset to be rejected")
