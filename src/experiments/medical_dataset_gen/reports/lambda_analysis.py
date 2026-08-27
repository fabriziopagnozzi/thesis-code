"""Validation-grid lambda diagnostics shared by report generation and plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.reports.analysis_scope import INTERACTION_FAMILY_ID
from experiments.medical_dataset_gen.reports.helpers import ordered_embedding_models
from experiments.medical_dataset_gen.reports.models import ExperimentRecord


def select_reference_record(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> ExperimentRecord | None:
    """Select one stable, non-interaction reference cell for an illustrative plot."""
    candidates = [
        record
        for record in records
        if 'reference' in record.tags
        and record.family_id != INTERACTION_FAMILY_ID
        and _validation_grid_path(record).is_file()
    ]
    if not candidates:
        warnings.append('lambda reference plot skipped: no completed reference grid was found')
        return None

    model_order = ordered_embedding_models(record.embedding_model for record in candidates)
    model_rank = {model: index for index, model in enumerate(model_order)}
    return min(
        candidates,
        key=lambda record: (model_rank.get(record.embedding_model, len(model_order)), record.name),
    )


def load_reference_validation_grid(
    record: ExperimentRecord,
    *,
    warnings: list[str],
) -> pl.DataFrame | None:
    path = _validation_grid_path(record)
    try:
        stats = pl.read_parquet(path)
    except Exception as exc:
        warnings.append(f'{record.name}: could not read reference lambda grid ({exc})')
        return None
    if stats.is_empty():
        warnings.append(f'{record.name}: reference lambda grid is empty')
        return None
    return stats


def _validation_grid_path(record: ExperimentRecord) -> Path:
    return record.paths.table_path('evaluation_selection_stats')
