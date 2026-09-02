from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.reports.analysis_constants import (
    DeltaMetricLabel,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import (
    family_balanced_mean,
    ordered_embedding_models,
)
from experiments.medical_dataset_gen.reports.latex_tables import (
    _METRIC_RESULT_TOKENS,
    _budget_result_token,
    _budget_rows,
    _embedding_model_result_token,
    _experiment_embedding_budget_key,
    _family_counts,
    _family_label,
    _fixed,
    _float,
    _integer,
    _label_token,
    _median,
    _metric_result_token,
    _negative_family_summary,
    _outcome_count,
    _signed,
    _tex_percent,
    _values,
    thesis_result_macros_path,
)
from experiments.medical_dataset_gen.reports.report_config import LOW_BUDGET_K
from experiments.medical_dataset_gen.reports.wording_result_macros import (
    render_wording_result_macros,
)

type ReportRow = dict[str, object]


def _mean(rows: Sequence[Mapping[str, object]], column: str) -> float | None:
    values = _values(rows, column)
    return sum(values) / len(values) if values else None


def _geometry_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    pass_rates = _values(rows, 'GeometryPassRate')
    qwen_rates = [
        rate
        for row, rate in zip(
            rows, (_float(row.get('GeometryPassRate')) for row in rows), strict=True
        )
        if rate is not None and str(row.get('EmbeddingModel') or '').startswith('Qwen/')
    ]
    return {
        'ResultGeometryPassMean': _fixed(family_balanced_mean(rows, 'GeometryPassRate'), digits=3),
        'ResultGeometryPassMedian': _fixed(_median(pass_rates), digits=3),
        'ResultGeometryBgeMean': _fixed(
            family_balanced_mean(
                [row for row in rows if row.get('EmbeddingModel') == 'BAAI/bge-m3'],
                'GeometryPassRate',
            ),
            digits=3,
        ),
        'ResultGeometryQwenMin': _fixed(min(qwen_rates) if qwen_rates else None, digits=3),
        'ResultGeometryQwenMax': _fixed(max(qwen_rates) if qwen_rates else None, digits=3),
    }


def _synthetic_artifact_result_macros(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Expose the synthetic-text validity diagnostics cited in the thesis."""
    duplicate_rates = _values(rows, 'ExactDuplicateChunkRate')
    return {
        'ResultArtifactDiagnosticDistributions': _integer(len(rows)),
        'ResultArtifactDuplicateChunkRateMin': _tex_percent(
            min(duplicate_rates) if duplicate_rates else None
        ),
        'ResultArtifactDuplicateChunkRateMax': _tex_percent(
            max(duplicate_rates) if duplicate_rates else None
        ),
        'ResultArtifactWithinMinusBetweenJaccard': _fixed(
            _mean(rows, 'WithinMinusBetweenJaccard'), digits=3
        ),
    }


def _comparison_result_macros(
    comparison_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    fcp_rows = [
        row for row in comparison_rows if _float(row.get('Delta_FacLoc_MMR_FCP')) is not None
    ]
    low_rows = _budget_rows(budget_rows, 'low_budget')
    medium_rows = _budget_rows(budget_rows, 'medium_budget')
    high_rows = _budget_rows(budget_rows, 'high_budget')
    negative_rows = [
        row for row in fcp_rows if (_float(row.get('Delta_FacLoc_MMR_FCP')) or 0.0) < 0.0
    ]
    negative_family_counts = _family_counts(negative_rows)
    return {
        'ResultFcpRows': _integer(len(fcp_rows)),
        'ResultMmrTopKBetterRows': _integer(
            sum((_float(row.get('Delta_MMR_TopK_FCP')) or 0.0) > 0.0 for row in fcp_rows)
        ),
        'ResultFacLocTopKBetterRows': _integer(
            sum((_float(row.get('Delta_FacLoc_TopK_FCP')) or 0.0) > 0.0 for row in fcp_rows)
        ),
        'ResultMmrTopKMeanFcpDelta': _signed(family_balanced_mean(fcp_rows, 'Delta_MMR_TopK_FCP')),
        'ResultMmrTopKMedianFcpDelta': _signed(_median(_values(fcp_rows, 'Delta_MMR_TopK_FCP'))),
        'ResultFacLocTopKMeanFcpDelta': _signed(
            family_balanced_mean(fcp_rows, 'Delta_FacLoc_TopK_FCP')
        ),
        'ResultFacLocTopKMedianFcpDelta': _signed(
            _median(_values(fcp_rows, 'Delta_FacLoc_TopK_FCP'))
        ),
        'ResultFacLocMmrBetterRows': _integer(_outcome_count(fcp_rows, 'facloc_better')),
        'ResultFacLocMmrTiedRows': _integer(_outcome_count(fcp_rows, 'tied')),
        'ResultFacLocMmrWorseRows': _integer(_outcome_count(fcp_rows, 'facloc_worse')),
        'ResultLowBudgetRows': _integer(len(low_rows)),
        'ResultLowBudgetFacLocMmrBetterRows': _integer(_outcome_count(low_rows, 'facloc_better')),
        'ResultLowBudgetFacLocMmrTiedRows': _integer(_outcome_count(low_rows, 'tied')),
        'ResultLowBudgetFacLocMmrWorseRows': _integer(_outcome_count(low_rows, 'facloc_worse')),
        'ResultMediumBudgetRows': _integer(len(medium_rows)),
        'ResultMediumBudgetFacLocMmrBetterRows': _integer(
            _outcome_count(medium_rows, 'facloc_better')
        ),
        'ResultHighBudgetRows': _integer(len(high_rows)),
        'ResultHighBudgetFacLocMmrBetterRows': _integer(_outcome_count(high_rows, 'facloc_better')),
        'ResultNegativeFacLocMmrRows': _integer(len(negative_rows)),
        'ResultNegativeFacLocMmrFamilySummary': _negative_family_summary(negative_family_counts),
    }


def _metric_budget_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for row in rows:
        metric_token = _metric_result_token(row.get('MetricLabel'))
        budget_token = _budget_result_token(row.get('BudgetCategory'))
        if metric_token is None or budget_token is None:
            continue
        prefix = f'Result{metric_token}{budget_token}'
        macros.update(
            {
                f'{prefix}Rows': _integer(row.get('Rows')),
                f'{prefix}FacLocMmrBetterRows': _integer(row.get('FacLocBetterRows')),
                f'{prefix}FacLocMmrTiedRows': _integer(row.get('FacLocTiedRows')),
                f'{prefix}FacLocMmrWorseRows': _integer(row.get('FacLocWorseRows')),
                f'{prefix}FacLocMmrBetterPct': _tex_percent(row.get('FacLocBetterPct')),
                f'{prefix}FacLocMmrTiedPct': _tex_percent(row.get('FacLocTiedPct')),
                f'{prefix}FacLocMmrWorsePct': _tex_percent(row.get('FacLocWorsePct')),
                f'{prefix}FacLocMmrMeanDelta': _signed(row.get('MeanDeltaFacLocMMR'), digits=3),
                f'{prefix}FacLocMmrMedianDelta': _signed(row.get('MedianDeltaFacLocMMR'), digits=3),
                f'{prefix}FacLocTopKMeanDelta': _signed(row.get('MeanDeltaFacLocTopK'), digits=3),
                f'{prefix}MmrTopKMeanDelta': _signed(row.get('MeanDeltaMMRTopK'), digits=3),
            }
        )
    return macros


def _metric_family_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for row in rows:
        metric_token = _metric_result_token(row.get('MetricLabel'))
        family_token = _label_token(_family_label(row.get('ExperimentFamilyLabel')))
        if metric_token is None or not family_token:
            continue
        prefix = f'Result{metric_token}Family{family_token}'
        macros.update(
            {
                f'{prefix}Rows': _integer(row.get('Rows')),
                f'{prefix}FacLocMmrBetterPct': _tex_percent(row.get('FacLocBetterPct')),
                f'{prefix}FacLocMmrTiedPct': _tex_percent(row.get('FacLocTiedPct')),
                f'{prefix}FacLocMmrWorsePct': _tex_percent(row.get('FacLocWorsePct')),
                f'{prefix}FacLocMmrMeanDelta': _signed(row.get('MeanDeltaFacLocMMR')),
                f'{prefix}FacLocTopKMeanDelta': _signed(row.get('MeanDeltaFacLocTopK')),
            }
        )
    return macros


def _metric_family_budget_result_macros(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Expose descriptive family means at each thesis retrieval budget."""
    macros: dict[str, str] = {}
    for row in rows:
        metric_token = _metric_result_token(row.get('MetricLabel'))
        family_token = _label_token(_family_label(row.get('ExperimentFamilyLabel')))
        budget_token = _budget_result_token(row.get('BudgetCategory'))
        if metric_token is None or not family_token or budget_token is None:
            continue
        prefix = f'Result{metric_token}Family{family_token}{budget_token}'
        macros.update(
            {
                f'{prefix}Rows': _integer(row.get('Rows')),
                f'{prefix}FacLocMmrMeanDelta': _signed(row.get('MeanDeltaFacLocMMR'), digits=3),
                f'{prefix}FacLocTopKMeanDelta': _signed(row.get('MeanDeltaFacLocTopK'), digits=3),
            }
        )
    return macros


def _paired_suite_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for row in rows:
        if row.get('MetricLabel') != 'FCP':
            continue
        budget_token = _budget_result_token(row.get('BudgetCategory'))
        scope = str(row.get('Scope') or '')
        scope_token = 'Core' if scope == 'Core suite' else _label_token(_family_label(scope))
        if budget_token is None or not scope_token:
            continue
        prefix = f'ResultPaired{scope_token}{budget_token}Fcp'
        macros.update(
            {
                f'{prefix}Distributions': _integer(row.get('Distributions')),
                f'{prefix}Runs': _integer(row.get('Runs')),
                f'{prefix}MeanDelta': _signed(row.get('MeanDeltaFacLocMMR'), digits=3),
                f'{prefix}CiLow': _signed(row.get('CI95Low'), digits=3),
                f'{prefix}CiHigh': _signed(row.get('CI95High'), digits=3),
            }
        )
    return macros


def _background_topology_endpoint_result_macros(
    comparison_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Generate reproducible endpoint values for the fixed-mass topology table."""
    grouped_rows: dict[tuple[int, str], list[Mapping[str, object]]] = {}
    for row in comparison_rows:
        if (
            row.get('ExperimentFamily') != 'background_variant'
            or row.get('AnalysisBlocks') != 'background_topology'
            or _float(row.get('k')) != LOW_BUDGET_K
        ):
            continue
        distribution = str(row.get('Distribution') or '')
        topology = distribution.rsplit('_', maxsplit=1)[-1]
        clusters, separator, chunks = topology.partition('x')
        if not separator or not clusters.isdecimal() or not chunks.isdecimal():
            continue
        grouped_rows.setdefault((int(clusters), distribution), []).append(row)

    if len(grouped_rows) < 2:
        return {}

    endpoints = {
        'Dispersed': max(grouped_rows),
        'Compact': min(grouped_rows),
    }
    macros: dict[str, str] = {}
    for endpoint, key in endpoints.items():
        rows = grouped_rows[key]
        prefix = f'ResultBackgroundTopology{endpoint}'
        macros.update(
            {
                f'{prefix}MmrBackgroundOutlierRate': _tex_percent(
                    _mean(rows, 'MMR_BackgroundOutlierRate')
                ),
                f'{prefix}FacLocBackgroundOutlierRate': _tex_percent(
                    _mean(rows, 'FacLoc_BackgroundOutlierRate')
                ),
                f'{prefix}FacLocMmrFcpMeanDelta': _signed(
                    _mean(rows, 'Delta_FacLoc_MMR_FCP'), digits=3
                ),
            }
        )
    return macros


def _distribution_budget_result_macros(
    comparison_rows: Sequence[Mapping[str, object]],
    *,
    embedding_models: Sequence[str] = (),
) -> dict[str, str]:
    """Expose model-specific and equally model-weighted distribution results."""
    grouped_rows: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    pooled_rows: dict[tuple[str, int], dict[str, list[Mapping[str, object]]]] = {}
    for row in comparison_rows:
        embedding_model = str(row.get('EmbeddingModel') or '')
        model_token = _embedding_model_result_token(row.get('EmbeddingModel'))
        distribution = str(row.get('Distribution') or '')
        k_value = _float(row.get('k'))
        if not embedding_model or model_token is None or not distribution or k_value is None:
            continue
        integer_k = int(k_value)
        grouped_rows.setdefault((model_token, distribution, integer_k), []).append(row)
        pooled_rows.setdefault((distribution, integer_k), {}).setdefault(
            embedding_model, []
        ).append(row)

    macros: dict[str, str] = {}
    for (model_token, distribution, k_value), rows in grouped_rows.items():
        prefix = f'Result{model_token}Distribution{_label_token(distribution)}K{k_value}'
        mmr_near_miss_rate = _mean(rows, 'MMR_NearMissDistractorRate')
        facloc_near_miss_rate = _mean(rows, 'FacLoc_NearMissDistractorRate')
        macros.update(
            {
                f'{prefix}FacLocMmrFcpMeanDelta': _signed(
                    _mean(rows, 'Delta_FacLoc_MMR_FCP'), digits=3
                ),
                f'{prefix}MmrNearMissDistractorRate': _tex_percent(mmr_near_miss_rate),
                f'{prefix}FacLocNearMissDistractorRate': _tex_percent(facloc_near_miss_rate),
                f'{prefix}NearMissReductionPp': _fixed(
                    None
                    if mmr_near_miss_rate is None or facloc_near_miss_rate is None
                    else (mmr_near_miss_rate - facloc_near_miss_rate) * 100,
                    digits=1,
                ),
            }
        )

    effective_models = tuple(
        embedding_models
        or ordered_embedding_models(
            model for model_rows in pooled_rows.values() for model in model_rows
        )
    )
    expected_models = set(effective_models)
    macros['ResultDistributionPooledModels'] = _integer(len(effective_models))
    for (distribution, k_value), rows_by_model in pooled_rows.items():
        # Omit partially crossed cells: every pooled value must represent the same
        # embedding-model scope declared by this report.
        if set(rows_by_model) != expected_models:
            continue
        prefix = f'ResultDistribution{_label_token(distribution)}K{k_value}'
        fcp_deltas = _model_means(rows_by_model, 'Delta_FacLoc_MMR_FCP')
        mmr_near_miss_rates = _model_means(rows_by_model, 'MMR_NearMissDistractorRate')
        facloc_near_miss_rates = _model_means(rows_by_model, 'FacLoc_NearMissDistractorRate')
        mean_mmr_near_miss_rate = _mean_values(mmr_near_miss_rates)
        mean_facloc_near_miss_rate = _mean_values(facloc_near_miss_rates)
        macros.update(
            {
                f'{prefix}ModelCount': _integer(len(rows_by_model)),
                f'{prefix}FacLocMmrFcpMeanDelta': _signed(_mean_values(fcp_deltas), digits=3),
                f'{prefix}FacLocMmrFcpMinModelDelta': _signed(
                    min(fcp_deltas) if fcp_deltas else None, digits=3
                ),
                f'{prefix}FacLocMmrFcpMaxModelDelta': _signed(
                    max(fcp_deltas) if fcp_deltas else None, digits=3
                ),
                f'{prefix}FacLocMmrFcpPositiveModelCount': _integer(
                    sum(delta > 0.0 for delta in fcp_deltas)
                ),
                f'{prefix}MmrNearMissDistractorRate': _tex_percent(mean_mmr_near_miss_rate),
                f'{prefix}FacLocNearMissDistractorRate': _tex_percent(mean_facloc_near_miss_rate),
                f'{prefix}NearMissReductionPp': _fixed(
                    None
                    if mean_mmr_near_miss_rate is None or mean_facloc_near_miss_rate is None
                    else (mean_mmr_near_miss_rate - mean_facloc_near_miss_rate) * 100,
                    digits=1,
                ),
            }
        )
    return macros


def _model_means(
    rows_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    column: str,
) -> list[float]:
    """Average configurations within models before averaging across models."""
    return [
        model_mean
        for model in sorted(rows_by_model)
        if (model_mean := _mean(rows_by_model[model], column)) is not None
    ]


def _mean_values(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _embedding_model_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for row in rows:
        token = _embedding_model_result_token(row.get('EmbeddingModel'))
        if token is None:
            continue
        prefix = f'ResultEmbedding{token}'
        macros.update(
            {
                f'{prefix}Runs': _integer(row.get('Runs')),
                f'{prefix}GeometryPassMean': _fixed(
                    _float(row.get('GeometryPassRate_mean')),
                    digits=3,
                ),
                f'{prefix}GeometryPassMedian': _fixed(
                    _float(row.get('GeometryPassRate_median')),
                    digits=3,
                ),
                f'{prefix}FacLocFcpMean': _fixed(_float(row.get('FacLoc_FCP_mean')), digits=3),
                f'{prefix}FacLocMmrFcpMeanDelta': _signed(row.get('Delta_FacLoc_MMR_FCP_mean')),
                f'{prefix}FacLocTopKFcpMeanDelta': _signed(row.get('Delta_FacLoc_TopK_FCP_mean')),
            }
        )
    return macros


def _embedding_metric_result_macros(
    rows: Sequence[Mapping[str, object]],
    range_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    macros: dict[str, str] = {}
    for row in rows:
        if row.get('MetricLabel') != 'FCP' or row.get('Scope') != 'overall':
            continue
        token = _embedding_model_result_token(row.get('EmbeddingModel'))
        if token is None:
            continue
        prefix = f'ResultEmbedding{token}OverallFcp'
        macros.update(
            {
                f'{prefix}TopKMean': _fixed(_float(row.get('MeanTopK')), digits=3),
                f'{prefix}MmrMean': _fixed(_float(row.get('MeanMMR')), digits=3),
                f'{prefix}FacLocMean': _fixed(_float(row.get('MeanFacLoc')), digits=3),
                f'{prefix}FacLocMmrMeanDelta': _signed(row.get('MeanDeltaFacLocMMR'), digits=3),
            }
        )
    for row in range_rows:
        if row.get('MetricLabel') != 'FCP':
            continue
        scope = str(row.get('Scope') or '')
        if scope == 'overall':
            prefix = 'ResultFcpAll'
        elif scope == 'budget':
            budget_token = _budget_result_token(row.get('ScopeValue'))
            if budget_token is None:
                continue
            prefix = f'ResultFcp{budget_token}'
        elif scope == 'experiment_family':
            family_token = _label_token(_family_label(row.get('ScopeValue')))
            if not family_token:
                continue
            prefix = f'ResultFcpFamily{family_token}'
        elif scope == 'family_budget':
            family, separator, budget = str(row.get('ScopeValue') or '').partition('|')
            family_token = _label_token(_family_label(family))
            budget_token = _budget_result_token(budget)
            if not separator or not family_token or budget_token is None:
                continue
            prefix = f'ResultFcpFamily{family_token}{budget_token}'
        else:
            continue
        macros.update(
            {
                f'{prefix}FacLocMmrMeanDelta': _signed(row.get('MeanDeltaFacLocMMR'), digits=3),
                f'{prefix}FacLocMmrMinModelDelta': _signed(
                    row.get('MeanDeltaFacLocMMRMinModel'), digits=3
                ),
                f'{prefix}FacLocMmrMaxModelDelta': _signed(
                    row.get('MeanDeltaFacLocMMRMaxModel'), digits=3
                ),
            }
        )
    return macros


def _embedding_geometry_result_macros(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    macros: dict[str, str] = {}
    stress_values: list[float] = []
    for row in rows:
        token = _embedding_model_result_token(row.get('EmbeddingModel'))
        if token is None:
            continue
        prefix = f'ResultEmbedding{token}Geometry'
        stress = _float(row.get('CoverageStressRate'))
        if stress is not None:
            stress_values.append(stress)
        macros.update(
            {
                f'{prefix}CoverageStressMean': _fixed(stress, digits=3),
                f'{prefix}GoldNearMissMarginMean': _fixed(
                    _float(row.get('GoldNearMissMargin')), digits=3
                ),
                f'{prefix}GoldBackgroundMarginMean': _fixed(
                    _float(row.get('GoldBackgroundMargin')), digits=3
                ),
            }
        )
    if stress_values:
        macros.update(
            {
                'ResultGeometryCoverageStressMean': _fixed(
                    sum(stress_values) / len(stress_values), digits=3
                ),
                'ResultGeometryCoverageStressMedian': _fixed(_median(stress_values), digits=3),
                'ResultGeometryCoverageStressMinModel': _fixed(min(stress_values), digits=3),
                'ResultGeometryCoverageStressMaxModel': _fixed(max(stress_values), digits=3),
            }
        )
    return macros


def _embedding_lambda_result_macros(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    macros: dict[str, str] = {}
    for row in rows:
        if row.get('Scope') != 'embedding_model':
            continue
        token = _embedding_model_result_token(row.get('ScopeValue'))
        strategy = str(row.get('strategy') or '')
        strategy_token = {'fac_loc': 'FacLoc', 'mmr': 'Mmr'}.get(strategy)
        if token is None or strategy_token is None:
            continue
        prefix = f'ResultEmbedding{token}{strategy_token}Lambda'
        macros[f'{prefix}MeanSafeFraction'] = _fixed(
            _float(row.get('MeanSafeLambdaFraction')), digits=3
        )
        macros[f'{prefix}AllGridSafeRate'] = _tex_percent(row.get('AllGridSafeRate'))
        macros[f'{prefix}MinWorstFcpDelta'] = _signed(row.get('MinWorstDeltaStrategyTopK_FCP'))
    return macros


def _embedding_low_budget_result_macros(
    comparison_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    low_budget_keys = {
        key
        for row in budget_rows
        if row.get('BudgetCategory') == 'low_budget'
        and (key := _experiment_embedding_budget_key(row)) is not None
    }
    grouped_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in comparison_rows:
        if _experiment_embedding_budget_key(row) not in low_budget_keys:
            continue
        token = _embedding_model_result_token(row.get('EmbeddingModel'))
        if token is None:
            continue
        grouped_rows.setdefault(token, []).append(row)

    macros: dict[str, str] = {}
    for model_token, rows in grouped_rows.items():
        for metric, metric_token in _METRIC_RESULT_TOKENS.items():
            deltas = _values(rows, f'Delta_FacLoc_MMR_{metric}')
            threshold = practical_effect_threshold(cast(DeltaMetricLabel, metric))
            prefix = f'ResultEmbedding{model_token}Low{metric_token}FacLocMmr'
            macros.update(
                {
                    f'{prefix}Rows': _integer(len(deltas)),
                    f'{prefix}BetterRows': _integer(sum(delta > threshold for delta in deltas)),
                    f'{prefix}TiedRows': _integer(sum(abs(delta) <= threshold for delta in deltas)),
                    f'{prefix}WorseRows': _integer(sum(delta < -threshold for delta in deltas)),
                    f'{prefix}MeanDelta': _signed(
                        family_balanced_mean(rows, f'Delta_FacLoc_MMR_{metric}')
                    ),
                    f'{prefix}MedianDelta': _signed(_median(deltas)),
                }
            )
    return macros


def _embedding_edge_case_result_macros(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Expose the paired DOM_M03 low-budget reversal used in the thesis claims."""
    target_distribution = 'DOM_M03_dominance_high'
    matching_budgets = sorted(
        {
            int(k)
            for row in rows
            if row.get('Distribution') == target_distribution
            and (k := _float(row.get('k'))) is not None
        }
    )
    target_budget = matching_budgets[0] if matching_budgets else None
    model_prefixes = {
        'Bge': 'BAAI/bge-m3',
        'Qwen': 'Qwen/Qwen3-Embedding-0.6B',
    }
    macros: dict[str, str] = {}
    for label, model_name in model_prefixes.items():
        match = next(
            (
                row
                for row in rows
                if row.get('Distribution') == target_distribution
                and row.get('EmbeddingModel') == model_name
                and target_budget is not None
                and _float(row.get('k')) == target_budget
            ),
            None,
        )
        macros[f'ResultDom{label}LowFacetCoverageDelta'] = _signed(
            _float(None if match is None else match.get('Delta_FacLoc_MMR_FacetCoverage'))
        )
        macros[f'ResultDom{label}LowFcpDelta'] = _signed(
            _float(None if match is None else match.get('Delta_FacLoc_MMR_FCP'))
        )
    return macros


def _lambda_safety_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    facloc_rows = [row for row in rows if row.get('strategy') == 'fac_loc']
    mmr_rows = [row for row in rows if row.get('strategy') == 'mmr']
    return {
        'ResultLambdaSafetyRows': _integer(len(facloc_rows)),
        'ResultFacLocSafeLambdaMean': _fixed(
            family_balanced_mean(facloc_rows, 'SafeLambdaFraction'), digits=3
        ),
        'ResultFacLocSafeLambdaMedian': _fixed(
            _median(_values(facloc_rows, 'SafeLambdaFraction')), digits=3
        ),
        'ResultFacLocSafeLambdaMin': _fixed(
            min(_values(facloc_rows, 'SafeLambdaFraction') or [0.0]), digits=3
        ),
        'ResultMmrSafeLambdaMean': _fixed(
            family_balanced_mean(mmr_rows, 'SafeLambdaFraction'), digits=3
        ),
        'ResultMmrSafeLambdaMedian': _fixed(
            _median(_values(mmr_rows, 'SafeLambdaFraction')), digits=3
        ),
        'ResultFacLocWorstLambdaMedianDelta': _signed(
            _median(_values(facloc_rows, 'WorstDeltaStrategyTopK_FCP'))
        ),
        'ResultMmrWorstLambdaMedianDelta': _signed(
            _median(_values(mmr_rows, 'WorstDeltaStrategyTopK_FCP'))
        ),
        'ResultMmrWorstLambdaMinDelta': _signed(
            min(_values(mmr_rows, 'WorstDeltaStrategyTopK_FCP') or [0.0])
        ),
    }


def _lambda_curve_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for strategy, label in (('fac_loc', 'FacLoc'), ('mmr', 'Mmr')):
        strategy_rows = [row for row in rows if row.get('strategy') == strategy]
        mean_deltas = _values(strategy_rows, 'MeanDeltaStrategyTopK_FCP')
        macros[f'Result{label}AggregateBelowTopKLambdaCount'] = _integer(
            sum(delta < 0.0 for delta in mean_deltas)
        )
        macros[f'Result{label}AggregateMinMeanFcpDelta'] = _signed(
            min(mean_deltas) if mean_deltas else None
        )
        macros[f'Result{label}AggregateLambdaPointCount'] = _integer(len(mean_deltas))
    grid_counts = [
        len([row for row in rows if row.get('strategy') == strategy])
        for strategy in ('fac_loc', 'mmr')
    ]
    macros['ResultLambdaGridPoints'] = _integer(max(grid_counts, default=0))
    return macros


def _lambda_robustness_result_macros(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    overall = [row for row in rows if row.get('Scope') == 'overall']
    macros: dict[str, str] = {}
    for strategy, label in (('fac_loc', 'FacLoc'), ('mmr', 'Mmr')):
        match = next((row for row in overall if row.get('strategy') == strategy), None)
        macros[f'Result{label}AllGridSafeRows'] = _integer(
            None if match is None else match.get('AllGridSafeRows')
        )
        macros[f'Result{label}AllGridSafeRate'] = _tex_percent(
            None if match is None else match.get('AllGridSafeRate')
        )
        macros[f'Result{label}MeanSafeLambdaFraction'] = _fixed(
            _float(None if match is None else match.get('MeanSafeLambdaFraction')), digits=3
        )
        macros[f'Result{label}MedianSafeLambdaFraction'] = _fixed(
            _float(None if match is None else match.get('MedianSafeLambdaFraction')), digits=3
        )
        macros[f'Result{label}MedianWorstLambdaFcpDelta'] = _signed(
            None if match is None else match.get('MedianWorstDeltaStrategyTopK_FCP')
        )
        macros[f'Result{label}MinWorstLambdaFcpDelta'] = _signed(
            None if match is None else match.get('MinWorstDeltaStrategyTopK_FCP')
        )
    return macros


def _alpha_ndcg_result_macros(
    comparison_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    rows = [
        row for row in comparison_rows if _float(row.get('Delta_FacLoc_MMR_alpha_nDCG')) is not None
    ]
    low_rows = _budget_rows(budget_rows, 'low_budget')
    return {
        'ResultAlphaNdcgRows': _integer(len(rows)),
        'ResultAlphaNdcgFacLocMmrBetterRows': _integer(
            sum((_float(row.get('Delta_FacLoc_MMR_alpha_nDCG')) or 0.0) > 0.0 for row in rows)
        ),
        'ResultAlphaNdcgFacLocTopKBetterRows': _integer(
            sum((_float(row.get('Delta_FacLoc_TopK_alpha_nDCG')) or 0.0) > 0.0 for row in rows)
        ),
        'ResultLowBudgetAlphaNdcgFacLocMmrMeanDelta': _signed(
            family_balanced_mean(low_rows, 'Delta_FacLoc_MMR_alpha_nDCG')
        ),
    }


def render_thesis_result_macros(
    *,
    geometry_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
    lambda_safety_rows: Sequence[Mapping[str, object]],
    synthetic_artifact_rows: Sequence[Mapping[str, object]] = (),
    metric_summary_rows: Sequence[Mapping[str, object]] = (),
    metric_family_summary_rows: Sequence[Mapping[str, object]] = (),
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]] = (),
    paired_suite_rows: Sequence[Mapping[str, object]] = (),
    embedding_summary_rows: Sequence[Mapping[str, object]] = (),
    embedding_models: Sequence[str] = (),
    require_complete_wording_grid: bool = False,
    warnings: list[str] | None = None,
    lambda_curve_rows: Sequence[Mapping[str, object]] = (),
    lambda_robustness_rows: Sequence[Mapping[str, object]] = (),
    embedding_metric_rows: Sequence[Mapping[str, object]] = (),
    embedding_metric_range_rows: Sequence[Mapping[str, object]] = (),
    embedding_geometry_rows: Sequence[Mapping[str, object]] = (),
) -> str:
    """Render scalar result macros imported by the thesis text."""
    macros = {
        **_geometry_result_macros(geometry_rows),
        **_synthetic_artifact_result_macros(synthetic_artifact_rows),
        **_comparison_result_macros(comparison_rows, budget_rows),
        **_metric_budget_result_macros(metric_summary_rows),
        **_metric_family_result_macros(metric_family_summary_rows),
        **_metric_family_budget_result_macros(metric_family_budget_summary_rows),
        **_paired_suite_result_macros(paired_suite_rows),
        **_background_topology_endpoint_result_macros(comparison_rows),
        **_distribution_budget_result_macros(
            comparison_rows,
            embedding_models=embedding_models,
        ),
        **_embedding_model_result_macros(embedding_summary_rows),
        **_embedding_metric_result_macros(embedding_metric_rows, embedding_metric_range_rows),
        **_embedding_geometry_result_macros(embedding_geometry_rows),
        **_embedding_lambda_result_macros(lambda_robustness_rows),
        **_embedding_low_budget_result_macros(comparison_rows, budget_rows),
        **_embedding_edge_case_result_macros(comparison_rows),
        **_lambda_safety_result_macros(lambda_safety_rows),
        **_lambda_curve_result_macros(lambda_curve_rows),
        **_lambda_robustness_result_macros(lambda_robustness_rows),
        **_alpha_ndcg_result_macros(comparison_rows, budget_rows),
        **render_wording_result_macros(
            budget_rows=budget_rows,
            geometry_rows=geometry_rows,
            embedding_models=embedding_models,
            require_complete_grid=require_complete_wording_grid,
            warnings=warnings,
        ),
    }
    lines = [
        '% Auto-generated by experiments.medical_dataset_gen.reports.',
        '% Do not edit this file directly; rerun the report instead.',
        '',
    ]
    for name in sorted(macros):
        if any(character.isdigit() for character in name):
            # TeX control-word names end at digits. Defining numeric names through
            # \csname keeps generated identifiers such as ``...K10...`` addressable.
            lines.append(rf'\expandafter\def\csname {name}\endcsname{{{macros[name]}}}')
        else:
            lines.append(rf'\newcommand{{\{name}}}{{{macros[name]}}}')
    return '\n'.join(lines) + '\n'


def generate_exp_results_macros(
    *,
    report_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Regenerate thesis result macros from an existing report's CSV artifacts."""
    report_dir = report_dir.expanduser().resolve()
    data_dir = report_dir / 'data'
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Report data directory not found: {data_dir}')

    output_path = (
        thesis_result_macros_path(report_dir)
        if output_path is None
        else output_path.expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_thesis_result_macros(
            geometry_rows=_read_rows(data_dir / 'geometry_filter_summary.csv'),
            comparison_rows=_read_rows(data_dir / 'comparison_by_k.csv'),
            budget_rows=_read_rows(data_dir / 'budget_strategy_summary.csv'),
            lambda_safety_rows=_read_rows(data_dir / 'lambda_safety_summary.csv', required=False),
            synthetic_artifact_rows=_read_rows(data_dir / 'synthetic_artifact_diagnostics.csv'),
            metric_summary_rows=_read_rows(data_dir / 'metric_aggregate_summary.csv'),
            metric_family_summary_rows=_read_rows(data_dir / 'metric_family_summary.csv'),
            metric_family_budget_summary_rows=_read_rows(
                data_dir / 'metric_family_budget_summary.csv'
            ),
            paired_suite_rows=_read_rows(
                data_dir / 'paired_suite_effect_summary.csv', required=False
            ),
            embedding_summary_rows=_read_rows(data_dir / 'embedding_model_summary.csv'),
            embedding_metric_rows=_read_rows(
                data_dir / 'embedding_model_metric_summary.csv', required=False
            ),
            embedding_metric_range_rows=_read_rows(
                data_dir / 'embedding_model_metric_ranges.csv', required=False
            ),
            embedding_geometry_rows=_read_rows(
                data_dir / 'embedding_geometry_summary.csv', required=False
            ),
            embedding_models=tuple(
                str(row.get('EmbeddingModel') or '')
                for row in _read_rows(
                    data_dir / 'embedding_model_grid_coverage.csv', required=False
                )
                if row.get('EmbeddingModel')
            ),
            require_complete_wording_grid=True,
            lambda_curve_rows=_read_rows(data_dir / 'lambda_curve_summary.csv', required=False),
            lambda_robustness_rows=_read_rows(
                data_dir / 'lambda_robustness_summary.csv', required=False
            ),
        )
    )
    return output_path


def _read_rows(path: Path, *, required: bool = True) -> list[ReportRow]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'Required report artifact not found: {path}')
        return []
    if path.stat().st_size == 0:
        if required:
            raise ValueError(f'Required report artifact is empty: {path}')
        return []
    return cast(list[ReportRow], pl.read_csv(path).to_dicts())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Regenerate exp_results_macros.tex from an existing experiment report.'
    )
    parser.add_argument(
        '--report-dir',
        type=Path,
        required=True,
        help='Existing experiment_comparison report containing data/*.csv.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output TeX file. Defaults to <report-dir>/latex/exp_results_macros.tex.',
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_path = generate_exp_results_macros(
        report_dir=cast(Path, args.report_dir),
        output_path=cast(Path | None, args.output),
    )
    print(f'wrote result macros to {output_path}')


if __name__ == '__main__':
    main()
