"""Shared coverage-stress audit over frozen query-geometry artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

FAIL_MISSING_FACET = 'fail_missing_facet'
FAIL_WEAK_FIRST_AXIS = 'fail_weak_primary_axis_dominance'
FAIL_EXCESS_EARLY_COVERAGE = 'fail_excess_stress_horizon_facet_coverage'
GOLD_NEAR_MISS_MARGIN = 'gold_minus_near_miss_similarity_margin'
GOLD_BACKGROUND_MARGIN = 'gold_minus_background_outlier_similarity_margin'
LEGACY_PASS = 'passes_filter'
COVERAGE_STRESS_DEFINITION = f'not {FAIL_EXCESS_EARLY_COVERAGE}'


@dataclass(frozen=True)
class GeometryCoverageSummary:
    query_count: int
    coverage_stress_pass_count: int
    coverage_stress_rate: float
    gold_near_miss_margin: float | None
    gold_background_margin: float | None


def geometry_coverage_columns() -> tuple[str, ...]:
    return (
        LEGACY_PASS,
        FAIL_MISSING_FACET,
        FAIL_WEAK_FIRST_AXIS,
        FAIL_EXCESS_EARLY_COVERAGE,
        GOLD_NEAR_MISS_MARGIN,
        GOLD_BACKGROUND_MARGIN,
    )


def representation_audit_manifest_metadata(input_suites: Sequence[str]) -> dict[str, object]:
    """Describe the report-facing audit without coupling it to report assembly."""
    return {
        'metric': 'coverage_stress',
        'definition': COVERAGE_STRESS_DEFINITION,
        'descriptive_only': True,
        'filters_evaluation_queries': False,
        'input_suites': list(input_suites),
        'legacy_passes_filter': {
            'validated': True,
            'used_for_report_metric': False,
        },
    }


def summarize_geometry_coverage(
    frame: pl.DataFrame,
    *,
    source: str,
) -> GeometryCoverageSummary:
    """Compute coverage stress while verifying the untouched legacy audit."""
    if frame.is_empty():
        raise ValueError(f'{source}: empty geometry table')
    missing = set(geometry_coverage_columns()).difference(frame.columns)
    if missing:
        raise ValueError(f'{source}: missing geometry columns {sorted(missing)}')

    scored = frame.with_columns(
        (~pl.col(FAIL_EXCESS_EARLY_COVERAGE).fill_null(True)).alias('coverage_stress_pass'),
        (
            ~pl.col(FAIL_MISSING_FACET).fill_null(True)
            & ~pl.col(FAIL_WEAK_FIRST_AXIS).fill_null(True)
            & ~pl.col(FAIL_EXCESS_EARLY_COVERAGE).fill_null(True)
        ).alias('legacy_pass_from_components'),
    )
    mismatch_count = scored.filter(
        pl.col(LEGACY_PASS).fill_null(False) != pl.col('legacy_pass_from_components')
    ).height
    if mismatch_count:
        raise ValueError(
            f'{source}: {mismatch_count} legacy pass rows disagree with their three stored '
            'component flags'
        )

    query_count = scored.height
    pass_count = int(scored['coverage_stress_pass'].sum())
    return GeometryCoverageSummary(
        query_count=query_count,
        coverage_stress_pass_count=pass_count,
        coverage_stress_rate=pass_count / query_count,
        gold_near_miss_margin=_mean_or_none(scored, GOLD_NEAR_MISS_MARGIN),
        gold_background_margin=_mean_or_none(scored, GOLD_BACKGROUND_MARGIN),
    )


def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
    value = frame.select(pl.col(column).mean()).item()
    return float(value) if isinstance(value, int | float) else None
