from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.reports.analysis_constants import ExperimentFamilyId
from experiments.medical_dataset_gen.reports.lambda_analysis import select_reference_record
from experiments.medical_dataset_gen.reports.latex_macros import (
    _lambda_curve_result_macros,
    _lambda_robustness_result_macros,
)
from experiments.medical_dataset_gen.reports.models import ExperimentRecord
from experiments.medical_dataset_gen.reports.plot_diagnostics import _save_plot_with_vector_copy
from experiments.medical_dataset_gen.reports.summaries import (
    lambda_curve_summary_rows,
    lambda_robustness_summary_rows,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


class _FakeFigure:
    def savefig(self, path: Path, **_kwargs: object) -> None:
        path.touch()


def test_lambda_plots_write_png_and_vector_pdf(tmp_path: Path) -> None:
    written = _save_plot_with_vector_copy(
        figure=_FakeFigure(),
        output_dir=tmp_path,
        stem='lambda_summary',
        plot_format='png',
    )

    assert [path.name for path in written] == ['lambda_summary.png', 'lambda_summary.pdf']
    assert all(path.is_file() for path in written)


def _grid_row(
    *,
    strategy: str,
    lambda_norm: float,
    lam: float,
    model: str,
    family: str,
    fcp_delta: float,
    distractor_delta: float,
) -> dict[str, object]:
    return {
        'strategy': strategy,
        'lambda_norm': lambda_norm,
        'lam': lam,
        'EmbeddingModel': model,
        'ExperimentFamily': family,
        'DeltaStrategyTopK_FCP': fcp_delta,
        'DeltaStrategyTopK_DistractorRate': distractor_delta,
    }


def test_lambda_curve_summary_balances_model_family_strata() -> None:
    rows = [
        _grid_row(
            strategy='fac_loc',
            lambda_norm=0.0,
            lam=0.01,
            model='model-a',
            family='family-a',
            fcp_delta=1.0,
            distractor_delta=-0.2,
        ),
        _grid_row(
            strategy='fac_loc',
            lambda_norm=0.0,
            lam=0.01,
            model='model-a',
            family='family-a',
            fcp_delta=3.0,
            distractor_delta=0.0,
        ),
        _grid_row(
            strategy='fac_loc',
            lambda_norm=0.0,
            lam=0.01,
            model='model-a',
            family='family-b',
            fcp_delta=5.0,
            distractor_delta=0.2,
        ),
        _grid_row(
            strategy='fac_loc',
            lambda_norm=0.0,
            lam=0.01,
            model='model-b',
            family='family-a',
            fcp_delta=-1.0,
            distractor_delta=0.4,
        ),
        _grid_row(
            strategy='fac_loc',
            lambda_norm=0.0,
            lam=0.01,
            model='model-b',
            family='family-b',
            fcp_delta=1.0,
            distractor_delta=-0.4,
        ),
    ]

    summary = lambda_curve_summary_rows(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row['MeanDeltaStrategyTopK_FCP'] == 1.75
    assert row['CellMeanDeltaStrategyTopK_FCP'] == 1.8
    assert row['BalancedSafeLambdaFraction'] == 0.75
    assert row['CellSafeLambdaFraction'] == 0.8
    assert row['MeanDeltaStrategyTopK_DistractorRate'] == 0.025


def _safety_row(
    *,
    strategy: str,
    model: str,
    family: str,
    k: int,
    safe: float,
    worst: float,
    span: float,
) -> dict[str, object]:
    return {
        'strategy': strategy,
        'EmbeddingModel': model,
        'ExperimentFamily': family,
        'k': k,
        'SafeLambdaFraction': safe,
        'WorstDeltaStrategyTopK_FCP': worst,
        'DeltaStrategyTopK_FCPRange': span,
        'LambdaCount': 120,
    }


def test_lambda_robustness_summary_reports_overall_and_scoped_rows() -> None:
    rows = [
        _safety_row(
            strategy='fac_loc',
            model='model-a',
            family='family-a',
            k=4,
            safe=1.0,
            worst=0.1,
            span=0.2,
        ),
        _safety_row(
            strategy='fac_loc',
            model='model-a',
            family='family-b',
            k=4,
            safe=0.5,
            worst=-0.2,
            span=0.4,
        ),
        _safety_row(
            strategy='fac_loc',
            model='model-b',
            family='family-a',
            k=4,
            safe=1.0,
            worst=0.05,
            span=0.1,
        ),
        _safety_row(
            strategy='fac_loc',
            model='model-b',
            family='family-b',
            k=4,
            safe=1.0,
            worst=0.03,
            span=0.08,
        ),
    ]

    summary = lambda_robustness_summary_rows(rows)
    overall = next(row for row in summary if row['Scope'] == 'overall')

    assert overall['MeanSafeLambdaFraction'] == 0.875
    assert overall['MedianSafeLambdaFraction'] == 1.0
    assert overall['AllGridSafeRows'] == 3
    assert overall['AllGridSafeRate'] == 0.75
    assert overall['MedianWorstDeltaStrategyTopK_FCP'] == 0.04
    assert overall['MinWorstDeltaStrategyTopK_FCP'] == -0.2


def test_lambda_result_macros_expose_grid_and_robustness_claims() -> None:
    curve_rows = [
        {
            'strategy': 'fac_loc',
            'MeanDeltaStrategyTopK_FCP': 0.1,
            'lambda_norm': 0.0,
        },
        {
            'strategy': 'fac_loc',
            'MeanDeltaStrategyTopK_FCP': -0.02,
            'lambda_norm': 1.0,
        },
        {
            'strategy': 'mmr',
            'MeanDeltaStrategyTopK_FCP': -0.2,
            'lambda_norm': 0.0,
        },
        {
            'strategy': 'mmr',
            'MeanDeltaStrategyTopK_FCP': 0.01,
            'lambda_norm': 1.0,
        },
    ]
    robustness_rows = [
        {
            'Scope': 'overall',
            'strategy': 'fac_loc',
            'AllGridSafeRows': 3,
            'AllGridSafeRate': 0.75,
            'MeanSafeLambdaFraction': 0.875,
            'MedianSafeLambdaFraction': 1.0,
            'MedianWorstDeltaStrategyTopK_FCP': 0.04,
            'MinWorstDeltaStrategyTopK_FCP': -0.2,
        },
        {
            'Scope': 'overall',
            'strategy': 'mmr',
            'AllGridSafeRows': 1,
            'AllGridSafeRate': 0.25,
            'MeanSafeLambdaFraction': 0.5,
            'MedianSafeLambdaFraction': 0.5,
            'MedianWorstDeltaStrategyTopK_FCP': -0.3,
            'MinWorstDeltaStrategyTopK_FCP': -0.7,
        },
    ]

    curve_macros = _lambda_curve_result_macros(curve_rows)
    robustness_macros = _lambda_robustness_result_macros(robustness_rows)

    assert curve_macros['ResultLambdaGridPoints'] == '2'
    assert curve_macros['ResultFacLocAggregateBelowTopKLambdaCount'] == '1'
    assert curve_macros['ResultFacLocAggregateMinMeanFcpDelta'] == '-0.0200'
    assert curve_macros['ResultMmrAggregateBelowTopKLambdaCount'] == '1'
    assert robustness_macros['ResultFacLocAllGridSafeRate'] == r'75\%'
    assert robustness_macros['ResultMmrMedianWorstLambdaFcpDelta'] == '-0.3000'


def _reference_record(
    tmp_path: Path,
    *,
    name: str,
    model: str,
    tags: tuple[str, ...],
    family: str = 'balanced_clean',
) -> ExperimentRecord:
    root = tmp_path / name
    (root / 'evaluation_selection_stats.parquet').parent.mkdir(parents=True)
    pl.DataFrame({'strategy': ['top_k']}).write_parquet(
        root / 'evaluation_selection_stats.parquet'
    )
    return ExperimentRecord(
        name=name,
        experiment_dir=root,
        distribution_id='balanced_reference',
        run_label=name,
        is_subexperiment=True,
        cfg=None,
        paths=MedicalDatasetGenPaths(name, artifact_root=root),
        config_error=None,
        family_id=cast(ExperimentFamilyId, family),
        family_label='Balanced',
        tags=tags,
    )


def test_reference_record_selection_is_stable_and_excludes_interactions(tmp_path: Path) -> None:
    records = [
        _reference_record(
            tmp_path,
            name='reference-z',
            model='model-b',
            tags=('reference',),
        ),
        _reference_record(
            tmp_path,
            name='reference-a',
            model='model-a',
            tags=('reference',),
        ),
        _reference_record(
            tmp_path,
            name='interaction',
            model='model-a',
            tags=('reference',),
            family='interaction',
        ),
    ]

    warnings: list[str] = []
    selected = select_reference_record(records, warnings=warnings)

    assert selected is not None
    assert selected.name == 'reference-a'
    assert warnings == []
