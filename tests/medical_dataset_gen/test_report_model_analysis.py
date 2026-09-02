from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

from experiments.medical_dataset_gen.reports.model_analysis import (
    embedding_geometry_summary_rows,
    embedding_metric_range_rows,
    embedding_metric_summary_rows,
    lambda_curve_by_embedding_model_rows,
    model_grid_coverage_rows,
)
from experiments.medical_dataset_gen.reports.plot_models import write_embedding_model_figures

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _metric_row(*, model: str, family: str, delta: float) -> dict[str, object]:
    return {
        'EmbeddingModel': model,
        'ExperimentFamilyLabel': family,
        'BudgetCategory': 'low_budget',
        'TopK_FCP': 0.4,
        'MMR_FCP': 0.5,
        'FacLoc_FCP': 0.5 + delta,
        'Delta_FacLoc_MMR_FCP': delta,
        'Delta_FacLoc_TopK_FCP': 0.1 + delta,
        'Delta_MMR_TopK_FCP': 0.1,
    }


def test_model_first_summary_and_ranges_do_not_reweight_dense_families() -> None:
    rows = [
        _metric_row(model='model-a', family='family-a', delta=0.0),
        _metric_row(model='model-a', family='family-a', delta=0.0),
        _metric_row(model='model-a', family='family-b', delta=1.0),
        _metric_row(model='model-b', family='family-a', delta=3.0),
        _metric_row(model='model-b', family='family-b', delta=3.0),
    ]

    summaries = embedding_metric_summary_rows(rows)
    overall = {
        row['EmbeddingModel']: row
        for row in summaries
        if row['MetricLabel'] == 'FCP' and row['Scope'] == 'overall'
    }
    assert overall['model-a']['MeanDeltaFacLocMMR'] == pytest.approx(0.5)
    assert overall['model-b']['MeanDeltaFacLocMMR'] == pytest.approx(3.0)

    ranges = embedding_metric_range_rows(summaries, complete_crossing=True)
    fcp_overall = next(
        row for row in ranges if row['MetricLabel'] == 'FCP' and row['Scope'] == 'overall'
    )
    assert fcp_overall['MeanDeltaFacLocMMR'] == pytest.approx(1.75)
    assert fcp_overall['MeanDeltaFacLocMMRMinModel'] == pytest.approx(0.5)
    assert fcp_overall['MeanDeltaFacLocMMRMaxModel'] == pytest.approx(3.0)


def test_model_grid_coverage_requires_the_same_distribution_wording_crossing() -> None:
    rows = [
        {'EmbeddingModel': model, 'Distribution': distribution, 'WordingConfig': 'wording'}
        for model in ('model-a', 'model-b')
        for distribution in ('one', 'two')
    ]
    complete, missing = model_grid_coverage_rows(
        rows,
        embedding_models=('model-a', 'model-b'),
    )
    assert all(row['Complete'] is True for row in complete)
    assert missing == []

    incomplete, missing = model_grid_coverage_rows(
        rows[:-1],
        embedding_models=('model-a', 'model-b'),
    )
    assert {row['EmbeddingModel']: row['Complete'] for row in incomplete} == {
        'model-a': True,
        'model-b': False,
    }
    assert missing == [
        {'EmbeddingModel': 'model-b', 'Distribution': 'two', 'WordingConfig': 'wording'}
    ]


def test_model_figures_write_png_and_pdf_outputs(tmp_path: Path) -> None:
    model = 'Qwen/Qwen3-Embedding-0.6B'
    metric_rows = embedding_metric_summary_rows(
        [_metric_row(model=model, family='Balanced gold support', delta=0.2)]
    )
    geometry_rows, geometry_family_rows = embedding_geometry_summary_rows(
        [
            {
                'EmbeddingModel': model,
                'ExperimentFamilyLabel': 'Balanced gold support',
                'CoverageStressRate': 0.9,
                'GoldNearMissMargin': 0.1,
                'GoldBackgroundMargin': 0.3,
            }
        ]
    )
    lambda_rows = lambda_curve_by_embedding_model_rows(
        [
            {
                'EmbeddingModel': model,
                'ExperimentFamily': 'balanced_clean',
                'strategy': strategy,
                'lambda_norm': 0.0,
                'DeltaStrategyTopK_FCP': delta,
                'DeltaStrategyTopK_DistractorRate': -delta,
            }
            for strategy, delta in (('mmr', -0.1), ('fac_loc', 0.1))
        ]
    )

    written = write_embedding_model_figures(
        plt=plt,
        output_dir=tmp_path,
        metric_rows=metric_rows,
        geometry_rows=geometry_rows,
        geometry_family_rows=geometry_family_rows,
        lambda_rows=lambda_rows,
    )

    assert len(written) == 8
    assert {path.suffix for path in written} == {'.png', '.pdf'}
    assert all(path.is_file() for path in written)
