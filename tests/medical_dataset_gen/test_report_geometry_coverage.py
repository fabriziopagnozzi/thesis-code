from __future__ import annotations

import polars as pl
import pytest

from experiments.medical_dataset_gen.reports.geometry_coverage import (
    COVERAGE_STRESS_DEFINITION,
    representation_audit_manifest_metadata,
    summarize_geometry_coverage,
)


def _geometry_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'passes_filter': [True, False, False],
            'fail_missing_facet': [False, False, True],
            'fail_weak_primary_axis_dominance': [False, False, False],
            'fail_excess_stress_horizon_facet_coverage': [False, True, False],
            'gold_minus_near_miss_similarity_margin': [0.2, 0.1, 0.3],
            'gold_minus_background_outlier_similarity_margin': [0.5, 0.4, 0.6],
        }
    )


def test_coverage_stress_uses_only_the_early_facet_coverage_component() -> None:
    summary = summarize_geometry_coverage(_geometry_frame(), source='fixture')

    assert summary.query_count == 3
    assert summary.coverage_stress_pass_count == 2
    assert summary.coverage_stress_rate == pytest.approx(2 / 3)
    assert summary.gold_near_miss_margin == pytest.approx(0.2)
    assert summary.gold_background_margin == pytest.approx(0.5)


def test_legacy_pass_is_validated_but_not_used_as_coverage_stress() -> None:
    inconsistent = _geometry_frame().with_columns(pl.lit(True).alias('passes_filter'))

    with pytest.raises(ValueError, match='legacy pass rows disagree'):
        summarize_geometry_coverage(inconsistent, source='fixture')


def test_manifest_metadata_records_the_report_facing_definition() -> None:
    metadata = representation_audit_manifest_metadata(('thesis_v5', 'extension'))

    assert metadata['definition'] == COVERAGE_STRESS_DEFINITION
    assert metadata['input_suites'] == ['thesis_v5', 'extension']
    assert metadata['filters_evaluation_queries'] is False
    assert metadata['legacy_passes_filter'] == {
        'validated': True,
        'used_for_report_metric': False,
    }
