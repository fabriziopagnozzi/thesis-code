from collections.abc import Sequence

from experiments.medical_dataset_gen.reports.report_config import LOW_BUDGET_K, REPORT_METRIC_SPECS
from experiments.medical_dataset_gen.reports.wording_result_macros import (
    WORDING_EXPERIMENT_FAMILIES,
    render_wording_result_macros,
)

_MODEL = 'Qwen/Qwen3-Embedding-0.6B'


def _budget_row(
    *,
    experiment: str,
    family: str,
    distribution: str,
    chunk_text_mode: str,
    k: int,
    query_mode: str = 'unbiased',
) -> dict[str, object]:
    row: dict[str, object] = {
        'Experiment': experiment,
        'ExperimentFamily': family,
        'Distribution': distribution,
        'EmbeddingModel': _MODEL,
        'QueryMode': query_mode,
        'FocusMode': 'natural',
        'ChunkTextMode': chunk_text_mode,
        'BudgetCategory': 'low_budget',
        'k': k,
        'TopK_n_queries': 50,
    }
    for spec in REPORT_METRIC_SPECS:
        metric = spec.metric_label
        row[f'TopK_{metric}'] = 0.5
        row[f'MMR_{metric}'] = 0.6
        row[f'FacLoc_{metric}'] = 0.7
        row[f'Delta_FacLoc_MMR_{metric}'] = 0.1
        row[f'Delta_FacLoc_TopK_{metric}'] = 0.2
    return row


def _complete_grid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_mode in ('biased', 'unbiased'):
        for chunk_text_mode in ('simple', 'hardened'):
            for family in WORDING_EXPERIMENT_FAMILIES:
                distribution = f'{family}_reference'
                rows.append(
                    _budget_row(
                        experiment=f'{distribution}_{query_mode}_{chunk_text_mode}',
                        family=family,
                        distribution=distribution,
                        chunk_text_mode=chunk_text_mode,
                        k=LOW_BUDGET_K,
                        query_mode=query_mode,
                    )
                )
    return rows


def _geometry_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    geometry_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        pass_rate = {
            ('biased', 'simple'): 0.9,
            ('biased', 'hardened'): 0.8,
            ('unbiased', 'simple'): 0.7,
            ('unbiased', 'hardened'): 0.6,
        }[(str(row['QueryMode']), str(row['ChunkTextMode']))]
        geometry_rows.append(
            {
                'Experiment': row['Experiment'],
                'ExperimentFamily': row['ExperimentFamily'],
                'EmbeddingModel': row['EmbeddingModel'],
                'QueryMode': row['QueryMode'],
                'FocusMode': row['FocusMode'],
                'ChunkTextMode': row['ChunkTextMode'],
                'GeometryQueries': 50,
                'GeometryPassRate': pass_rate,
                'FailMissingFacetRate': 0.1,
                'FailWeakPrimaryAxisDominanceRate': 0.2,
                'FailExcessStressHorizonFacetCoverageRate': 0.3,
                'QueryToGoldMeanMean': 0.7,
                'QueryToNearMissMeanMean': None if index == 0 else 0.5,
                'QueryToBackgroundOutlierMeanMean': 0.2,
            }
        )
    return geometry_rows


def test_wording_macros_select_the_global_low_budget_grid() -> None:
    complete_rows = _complete_grid_rows()
    extra_row = _budget_row(
        experiment='balanced_clean_reference_simple_k12',
        family='balanced_clean',
        distribution='balanced_clean_reference',
        chunk_text_mode='simple',
        k=LOW_BUDGET_K + 6,
    )
    warnings: list[str] = []

    macros = render_wording_result_macros(
        budget_rows=[*complete_rows, extra_row],
        geometry_rows=_geometry_rows(complete_rows),
        embedding_models=(_MODEL,),
        require_complete_grid=True,
        warnings=warnings,
    )

    assert macros['ResultWordingConfigurations'] == '4'
    assert macros['ResultWordingLowBudgetK'] == str(LOW_BUDGET_K)
    assert macros['ResultWordingCoreCells'] == str(len(complete_rows))
    assert macros['ResultWordingGeometryQwenPassRateMean'] == '0.750'
    assert macros['ResultWordingGeometryQwenPassRateMin'] == '0.600'
    assert macros['ResultWordingGeometryQwenPassRateMax'] == '0.900'
    assert macros['ResultWordingGeometryQwenFacetCompletenessRateMean'] == '0.900'
    assert macros['ResultWordingGeometryQwenPrimaryAxisStressPassRateMean'] == '0.800'
    assert macros['ResultWordingGeometryQwenEarlyFacetCoverageStressPassRateMean'] == '0.700'
    assert macros['ResultWordingGeometryQwenQueryToGoldSimilarityMean'] == '0.700'
    assert macros['ResultWordingGeometryQwenQueryToNearMissSimilarityMean'] == '0.500'
    assert macros['ResultWordingGeometryQwenQueryToBackgroundSimilarityMean'] == '0.200'
    assert macros['ResultWordingGeometryQwenConfigBiasedNaturalSimplePassRateMean'] == '0.900'
    assert macros['ResultWordingGeometryQwenConfigBiasedNaturalHardenedPassRateMean'] == '0.800'
    assert macros['ResultWordingGeometryQwenConfigUnbiasedNaturalSimplePassRateMean'] == '0.700'
    assert macros['ResultWordingGeometryQwenConfigUnbiasedNaturalHardenedPassRateMean'] == '0.600'
    assert macros['ResultWordingLowFamilyBalancedCleanFCPMmrMean'] == '0.600'
    assert macros['ResultWordingLowFamilyBalancedCleanFCPFacLocMmrMeanDelta'] == '+0.100'
    assert macros['ResultWordingLowFamilyBalancedCleanFCPFacLocMmrWinRate'] == r'100\%'
    assert macros['ResultWordingLowFamilyBalancedCleanFCPTopKMean'] == '0.500'
    assert macros['ResultWordingLowFamilyBalancedCleanFCPFacLocTopKMeanDelta'] == '+0.200'
    assert macros['ResultWordingLowFamilyBalancedCleanFCPFacLocTopKWinRate'] == r'100\%'
    assert warnings == [
        f'Wording result macros use the global k={LOW_BUDGET_K} low-budget grid and exclude 1 row(s) '
        'from alternative k values.'
    ]
