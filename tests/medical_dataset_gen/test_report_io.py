from __future__ import annotations

import pytest

from experiments.medical_dataset_gen.reports.report_io import report_io_workers


def test_report_io_uses_one_worker_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('REPORT_IO_WORKERS', raising=False)
    assert report_io_workers(656) == 1


def test_report_io_worker_override_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('REPORT_IO_WORKERS', '8')
    assert report_io_workers(3) == 3
