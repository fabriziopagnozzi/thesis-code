from experiments.medical_dataset_gen.reports.latex_macros import (
    _embedding_analysis_result_macros,
)
from experiments.medical_dataset_gen.reports.model_analysis import (
    embedding_geometry_summary_rows,
    embedding_metric_range_rows,
    embedding_metric_summary_rows,
    lambda_curve_by_embedding_model_rows,
    model_grid_coverage_rows,
)
from experiments.medical_dataset_gen.reports.suite_analysis import (
    suite_distribution_and_family_rows,
)
from experiments.medical_dataset_gen.suites.core import SuiteManifest, SuiteManifestCell


def _budget_row(*, model: str, family: str, budget: str, delta: float) -> dict[str, object]:
    return {
        'EmbeddingModel': model,
        'ExperimentFamilyLabel': family,
        'BudgetCategory': budget,
        'BudgetCategoryLabel': budget,
        'TopK_FCP': 0.4,
        'MMR_FCP': 0.6,
        'FacLoc_FCP': 0.6 + delta,
        'Delta_FacLoc_MMR_FCP': delta,
        'Delta_FacLoc_TopK_FCP': 0.2 + delta,
        'Delta_MMR_TopK_FCP': 0.2,
    }


def test_model_range_uses_equal_weight_model_means() -> None:
    rows = [
        _budget_row(model='model-a', family='A', budget='low_budget', delta=0.1),
        _budget_row(model='model-a', family='A', budget='low_budget', delta=0.3),
        _budget_row(model='model-b', family='A', budget='low_budget', delta=-0.2),
    ]
    model_rows = embedding_metric_summary_rows(rows)
    ranges = embedding_metric_range_rows(
        model_rows,
        embedding_models=('model-a', 'model-b'),
    )
    fcp_low = next(
        row
        for row in ranges
        if row['MetricLabel'] == 'FCP'
        and row['Scope'] == 'budget'
        and row['BudgetCategory'] == 'low_budget'
    )
    assert fcp_low['MeanDeltaFacLocMMR'] == 0.0
    assert fcp_low['MeanDeltaFacLocMMRMinModel'] == -0.2
    assert fcp_low['MeanDeltaFacLocMMRMaxModel'] == 0.2
    assert fcp_low['PositiveModelCount'] == 1


def test_model_range_suppresses_incomplete_crossing() -> None:
    model_rows = embedding_metric_summary_rows(
        [_budget_row(model='model-a', family='A', budget='low_budget', delta=0.1)]
    )
    ranges = embedding_metric_range_rows(
        model_rows,
        embedding_models=('model-a', 'model-b'),
    )
    fcp_low = next(
        row
        for row in ranges
        if row['MetricLabel'] == 'FCP'
        and row['Scope'] == 'budget'
        and row['BudgetCategory'] == 'low_budget'
    )
    assert fcp_low['CompleteModelCrossing'] is False
    assert fcp_low['MeanDeltaFacLocMMR'] is None


def test_geometry_summary_is_family_balanced_within_model() -> None:
    rows = [
        {
            'EmbeddingModel': 'model-a',
            'ExperimentFamily': 'a',
            'ExperimentFamilyLabel': 'A',
            'IncludeInFamilySummary': True,
            'GeometryPassRate': 1.0,
            'FailMissingFacetRate': 0.0,
            'FailWeakPrimaryAxisDominanceRate': 0.0,
            'FailExcessStressHorizonFacetCoverageRate': 0.0,
        },
        {
            'EmbeddingModel': 'model-a',
            'ExperimentFamily': 'a',
            'ExperimentFamilyLabel': 'A',
            'IncludeInFamilySummary': True,
            'GeometryPassRate': 1.0,
            'FailMissingFacetRate': 0.0,
            'FailWeakPrimaryAxisDominanceRate': 0.0,
            'FailExcessStressHorizonFacetCoverageRate': 0.0,
        },
        {
            'EmbeddingModel': 'model-a',
            'ExperimentFamily': 'b',
            'ExperimentFamilyLabel': 'B',
            'IncludeInFamilySummary': True,
            'GeometryPassRate': 0.0,
            'FailMissingFacetRate': 0.0,
            'FailWeakPrimaryAxisDominanceRate': 1.0,
            'FailExcessStressHorizonFacetCoverageRate': 0.0,
        },
    ]
    model_rows, family_rows = embedding_geometry_summary_rows(rows)
    assert model_rows[0]['GeometryPassRate'] == 0.5
    assert model_rows[0]['PrimaryAxisStressPassRate'] == 0.5
    assert len(family_rows) == 2


def test_lambda_curve_is_family_balanced_within_model() -> None:
    rows = [
        {
            'EmbeddingModel': 'model-a',
            'ExperimentFamily': 'a',
            'strategy': 'mmr',
            'lambda_norm': 0.0,
            'lam': 0.01,
            'DeltaStrategyTopK_FCP': 1.0,
        },
        {
            'EmbeddingModel': 'model-a',
            'ExperimentFamily': 'a',
            'strategy': 'mmr',
            'lambda_norm': 0.0,
            'lam': 0.01,
            'DeltaStrategyTopK_FCP': 1.0,
        },
        {
            'EmbeddingModel': 'model-a',
            'ExperimentFamily': 'b',
            'strategy': 'mmr',
            'lambda_norm': 0.0,
            'lam': 0.01,
            'DeltaStrategyTopK_FCP': -1.0,
        },
    ]
    summary = lambda_curve_by_embedding_model_rows(rows)
    assert summary[0]['MeanDeltaStrategyTopK_FCP'] == 0.0
    assert summary[0]['CellSafeLambdaFraction'] == 2 / 3


def test_model_grid_coverage_uses_declared_cells() -> None:
    cell_a = SuiteManifestCell.model_construct(
        cell_id='a',
        name='suite/a',
        distribution_id='dist',
        run_profile_id='profile-a',
        run_profile_factors={'embedding': 'model-a'},
    )
    cell_b = SuiteManifestCell.model_construct(
        cell_id='b',
        name='suite/b',
        distribution_id='dist',
        run_profile_id='profile-b',
        run_profile_factors={'embedding': 'model-b'},
    )
    manifest = SuiteManifest.model_construct(cells=[cell_a, cell_b])
    coverage, missing, complete = model_grid_coverage_rows(
        manifest=manifest,
        completed_experiments={'suite/a'},
        embedding_models=('model-a', 'model-b'),
    )
    assert [row['CompletedCells'] for row in coverage] == [1, 0]
    assert missing[0]['Experiment'] == 'suite/b'
    assert complete is False


def test_suite_family_summary_retains_model_and_metrics() -> None:
    distribution_rows, family_rows = suite_distribution_and_family_rows(
        [
            {
                'IncludeInCausalSummaries': True,
                'IncludeInFamilySummary': True,
                'ExperimentFamily': 'balanced_clean',
                'ExperimentFamilyLabel': 'Balanced clean',
                'AnalysisBlocks': 'core',
                'Distribution': 'balanced',
                'EmbeddingModel': 'model-a',
                'ArtifactOrigin': 'native',
                'k': 6,
                'Delta_FacLoc_MMR_FCP': 0.1,
            }
        ]
    )
    assert distribution_rows[0]['EmbeddingModel'] == 'model-a'
    assert family_rows[0]['EmbeddingModel'] == 'model-a'
    assert family_rows[0]['Delta_FacLoc_MMR_FCP_mean'] == 0.1


def test_embedding_analysis_macros_include_model_means_and_ranges() -> None:
    model_rows = embedding_metric_summary_rows(
        [
            _budget_row(
                model='Qwen/Qwen3-Embedding-0.6B',
                family='A',
                budget='low_budget',
                delta=0.1,
            ),
            _budget_row(
                model='Qwen/Qwen3-Embedding-4B',
                family='A',
                budget='low_budget',
                delta=0.2,
            ),
        ]
    )
    ranges = embedding_metric_range_rows(
        model_rows,
        embedding_models=(
            'Qwen/Qwen3-Embedding-0.6B',
            'Qwen/Qwen3-Embedding-4B',
        ),
    )
    macros = _embedding_analysis_result_macros(
        model_rows=model_rows,
        range_rows=ranges,
        geometry_rows=[],
        lambda_robustness_rows=[],
    )
    assert macros['ResultEmbeddingQwenSmallLowFcpFacLocMmrMeanDelta'] == '+0.100'
    assert macros['ResultFcpLowFacLocMmrMinModelDelta'] == '+0.100'
    assert macros['ResultFcpLowFacLocMmrMaxModelDelta'] == '+0.200'
