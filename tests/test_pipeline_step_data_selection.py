"""Regression tests for pipeline/steps/data_selection.py.

Pins the SelectRecentBrandData entrypoint's enforcement of RECENCY_MONTHS:
files older than the window are excluded, the real file modification time
is recorded on each manifest entry, and the step fails closed when the
environment variable is missing or non-numeric.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline.steps.data_selection import main, select

UTC = timezone.utc


def _write_image(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))


def test_select_excludes_back_dated_files(tmp_path):
    now = datetime(2026, 11, 5, tzinfo=UTC)
    fresh_ts = (now - timedelta(days=10)).timestamp()
    stale_ts = (now - timedelta(days=365 * 3)).timestamp()

    _write_image(tmp_path / "wc_done_0.91.jpg", stale_ts)
    _write_image(tmp_path / "wc_done_0.92.jpg", fresh_ts)
    _write_image(tmp_path / "wc_not_done_0.90.jpg", fresh_ts)

    manifest, metrics = select(
        tmp_path,
        "wc_*",
        recency_months=6,
        min_images_per_class=1,
        target_dataset_min=1,
        target_dataset_max=10,
        now=now,
    )

    keys = {record["key"] for record in manifest}
    assert "wc_done_0.91.jpg" not in keys
    assert keys == {"wc_done_0.92.jpg", "wc_not_done_0.90.jpg"}
    assert metrics["dataset_size"] == 2


def test_select_records_real_file_mtime(tmp_path):
    now = datetime(2026, 11, 5, tzinfo=UTC)
    file_mtime = (now - timedelta(days=15)).timestamp()
    _write_image(tmp_path / "wc_done_0.91.jpg", file_mtime)
    _write_image(tmp_path / "wc_not_done_0.90.jpg", file_mtime)

    manifest, _ = select(
        tmp_path,
        "wc_*",
        recency_months=6,
        min_images_per_class=1,
        target_dataset_min=1,
        target_dataset_max=10,
        now=now,
    )

    expected = datetime.fromtimestamp(file_mtime, tz=UTC).isoformat()
    for record in manifest:
        assert record["last_modified"] == expected


def test_select_rejects_zero_recency(tmp_path):
    with pytest.raises(ValueError, match="RECENCY_MONTHS"):
        select(
            tmp_path,
            "wc_*",
            recency_months=0,
            min_images_per_class=1,
            target_dataset_min=1,
            target_dataset_max=10,
        )


def test_select_rejects_non_integer_recency(tmp_path):
    with pytest.raises(ValueError, match="RECENCY_MONTHS"):
        select(
            tmp_path,
            "wc_*",
            recency_months="six",
            min_images_per_class=1,
            target_dataset_min=1,
            target_dataset_max=10,
        )


def test_main_fails_closed_when_recency_env_missing(tmp_path, monkeypatch, capsys):
    images = tmp_path / "images"
    images.mkdir()
    monkeypatch.setenv("NAMING_PATTERN", "wc_*")
    monkeypatch.setenv("BRAND_ID", "white-castle")
    monkeypatch.delenv("RECENCY_MONTHS", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "data_selection.py",
            "--images",
            str(images),
            "--dataset-out",
            str(tmp_path / "dataset"),
            "--metrics-out",
            str(tmp_path / "metrics"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "RECENCY_MONTHS" in captured.err


def test_main_fails_closed_when_recency_env_non_numeric(tmp_path, monkeypatch, capsys):
    images = tmp_path / "images"
    images.mkdir()
    monkeypatch.setenv("NAMING_PATTERN", "wc_*")
    monkeypatch.setenv("BRAND_ID", "white-castle")
    monkeypatch.setenv("RECENCY_MONTHS", "not-a-number")
    monkeypatch.setattr(
        "sys.argv",
        [
            "data_selection.py",
            "--images",
            str(images),
            "--dataset-out",
            str(tmp_path / "dataset"),
            "--metrics-out",
            str(tmp_path / "metrics"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "RECENCY_MONTHS" in captured.err
