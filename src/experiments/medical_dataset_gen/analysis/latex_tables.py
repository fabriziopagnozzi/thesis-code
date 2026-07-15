from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from tabulate import tabulate

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    DeltaMetricLabel,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.analysis.report_config import REPORT_METRIC_LABELS

THESIS_AGGREGATE_TABLES_PATH = Path(
    '/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/docs/thesis/exp_results_tables.tex'
)
THESIS_RESULT_MACROS_PATH = Path(
    '/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/docs/thesis/exp_results_macros.tex'
)

type TableKind = Literal['metric_budget', 'metric_family', 'metric_family_budget']


@dataclass(frozen=True)
class ThesisTableSpec:
    kind: TableKind
    subsection_title: str
    caption: str
    label: str
    column_spec: str
    tabcolsep: str
    headers: tuple[str, ...]
    group_macro: str
    metric_order: tuple[str, ...]


THESIS_TABLE_SPECS: tuple[ThesisTableSpec, ...] = (
    ThesisTableSpec(
        kind='metric_budget',
        subsection_title='By Evaluation Metric and Retrieval Budget',
        caption='Aggregate comparison by evaluation metric and retrieval budget.',
        label='tab:aggregate',
        column_spec='@{}llrrrrrr@{}',
        tabcolsep='8pt',
        headers=(
            r'\textbf{Budget}',
            r'\textbf{\#Exp.}',
            r'\textbf{$F>M$}',
            r'\textbf{Tie}',
            r'\textbf{$F<M$}',
            r'\textbf{Mean $F-M$}',
            r'\textbf{Mean $F-$}\TopK{}',
        ),
        group_macro='MetricSevenGroupRow',
        metric_order=REPORT_METRIC_LABELS,
    ),
    ThesisTableSpec(
        kind='metric_family',
        subsection_title='By Evaluation Metric and Experiment Family',
        caption='Aggregate comparison by evaluation metric and experiment family.',
        label='tab:metric-family-aggregate',
        column_spec='@{}llrrrrr@{}',
        tabcolsep='8pt',
        headers=(
            r'\textbf{Family}',
            r'\textbf{\#Exp.}',
            r'\textbf{$F>M$}',
            r'\textbf{Tie}',
            r'\textbf{$F<M$}',
            r'\textbf{Mean $F-M$}',
            r'\textbf{Mean $F-$}\TopK{}',
        ),
        group_macro='MetricSevenGroupRow',
        metric_order=REPORT_METRIC_LABELS,
    ),
    ThesisTableSpec(
        kind='metric_family_budget',
        subsection_title='By Evaluation Metric, Experiment Family, and Retrieval Budget',
        caption='Aggregate comparison by evaluation metric, experiment family, and retrieval budget.',
        label='tab:metric-family-budget-aggregate',
        column_spec='@{}lllrrrrrr@{}',
        tabcolsep='6pt',
        headers=(
            r'\textbf{Family}',
            r'\textbf{Budget}',
            r'\textbf{\#Exp.}',
            r'\textbf{$F>M$}',
            r'\textbf{Tie}',
            r'\textbf{$F<M$}',
            r'\textbf{Mean $F-M$}',
            r'\textbf{Mean $F-$}\TopK{}',
        ),
        group_macro='MetricNineGroupRow',
        metric_order=REPORT_METRIC_LABELS,
    ),
)

_METRIC_MACROS = {
    'FCP': r'\FCP{}',
    'FacetCoverage': r'\FacetCoverage{}',
    'AllFacetCoverageRate': r'\AllFacetCoverageRate{}',
    'FacetWeightedRecall': r'\FacetWeightedRecall{}',
    'Precision': r'\PrecisionK{}',
    'alpha_nDCG': r'\AlphanDCG{}',
    'AllFacetCleanRate': r'\ACR{}',
}
_FAMILY_LABELS = {
    'Balanced clean distributions': 'Balanced clean',
    'Near-miss-heavy distributions': 'Near-miss-heavy',
    'Dominance distributions': 'Dominance',
    'Background variants': 'Background',
    'Sparse-niche distributions': 'Sparse niche',
}
_BUDGET_LABELS = {
    'all_k': 'All budgets',
    'low_budget': 'Low',
    'medium_budget': 'Medium',
    'high_budget': 'High',
}
_BUDGET_RESULT_TOKENS = {
    'all_k': 'All',
    'low_budget': 'Low',
    'medium_budget': 'Medium',
    'high_budget': 'High',
}
_METRIC_RESULT_TOKENS = {
    'FCP': 'Fcp',
    'FacetCoverage': 'FacetCoverage',
    'AllFacetCoverageRate': 'AllFacetCoverageRate',
    'AllFacetCleanRate': 'AllFacetCleanRate',
    'FacetWeightedRecall': 'FacetWeightedRecall',
    'Precision': 'Precision',
    'alpha_nDCG': 'AlphaNdcg',
}
_EMBEDDING_MODEL_RESULT_TOKENS = {
    'BAAI/bge-m3': 'Bge',
    'Qwen/Qwen3-Embedding-0.6B': 'QwenSmall',
    'Qwen/Qwen3-Embedding-4B': 'QwenFourB',
    'Qwen/Qwen3-Embedding-8B': 'QwenEightB',
    'multi-qa-mpnet-base-cos-v1': 'MultiMpnet',
    'jinaai/jina-embeddings-v5-text-small': 'Jina',
    'abhinand/MedEmbed-large-v0.1': 'MedEmbed',
}


def render_thesis_aggregate_tables(
    *,
    metric_summary_rows: Sequence[Mapping[str, object]],
    metric_family_summary_rows: Sequence[Mapping[str, object]],
    metric_family_budget_summary_rows: Sequence[Mapping[str, object]],
) -> str:
    """Render the aggregate tables imported directly by the thesis document."""
    rows_by_kind: dict[TableKind, Sequence[Mapping[str, object]]] = {
        'metric_budget': metric_summary_rows,
        'metric_family': metric_family_summary_rows,
        'metric_family_budget': metric_family_budget_summary_rows,
    }
    rendered_tables = [
        _render_table(spec=spec, rows=rows_by_kind[spec.kind]) for spec in THESIS_TABLE_SPECS
    ]
    return (
        '% Auto-generated by experiments.medical_dataset_gen.analysis.\n'
        '% Do not edit this file directly; rerun the report instead.\n\n'
        + '\n\n'.join(rendered_tables)
        + '\n'
    )


def render_thesis_result_macros(
    *,
    geometry_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    budget_rows: Sequence[Mapping[str, object]],
    lambda_safety_rows: Sequence[Mapping[str, object]],
    metric_summary_rows: Sequence[Mapping[str, object]] = (),
    metric_family_summary_rows: Sequence[Mapping[str, object]] = (),
    paired_suite_rows: Sequence[Mapping[str, object]] = (),
    embedding_summary_rows: Sequence[Mapping[str, object]] = (),
) -> str:
    """Render scalar result macros imported by the thesis text."""
    macros = {
        **_geometry_result_macros(geometry_rows),
        **_comparison_result_macros(comparison_rows, budget_rows),
        **_metric_budget_result_macros(metric_summary_rows),
        **_metric_family_result_macros(metric_family_summary_rows),
        **_paired_suite_result_macros(paired_suite_rows),
        **_embedding_model_result_macros(embedding_summary_rows),
        **_embedding_low_budget_result_macros(comparison_rows, budget_rows),
        **_embedding_edge_case_result_macros(comparison_rows),
        **_lambda_safety_result_macros(lambda_safety_rows),
        **_alpha_ndcg_result_macros(comparison_rows, budget_rows),
    }
    lines = [
        '% Auto-generated by experiments.medical_dataset_gen.analysis.',
        '% Do not edit this file directly; rerun the report instead.',
        '',
    ]
    for name in sorted(macros):
        lines.append(rf'\newcommand{{\{name}}}{{{macros[name]}}}')
    return '\n'.join(lines) + '\n'


def _geometry_result_macros(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    pass_rates = _values(rows, 'GeometryPassRate')
    bge_rates = [
        rate
        for row, rate in zip(
            rows, (_float(row.get('GeometryPassRate')) for row in rows), strict=True
        )
        if rate is not None and row.get('EmbeddingModel') == 'BAAI/bge-m3'
    ]
    qwen_rates = [
        rate
        for row, rate in zip(
            rows, (_float(row.get('GeometryPassRate')) for row in rows), strict=True
        )
        if rate is not None and str(row.get('EmbeddingModel') or '').startswith('Qwen/')
    ]
    return {
        'ResultGeometryPassMean': _fixed(_mean(pass_rates), digits=3),
        'ResultGeometryPassMedian': _fixed(_median(pass_rates), digits=3),
        'ResultGeometryBgeMean': _fixed(_mean(bge_rates), digits=3),
        'ResultGeometryQwenMin': _fixed(min(qwen_rates) if qwen_rates else None, digits=3),
        'ResultGeometryQwenMax': _fixed(max(qwen_rates) if qwen_rates else None, digits=3),
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
        'ResultMmrTopKMeanFcpDelta': _signed(_mean(_values(fcp_rows, 'Delta_MMR_TopK_FCP'))),
        'ResultMmrTopKMedianFcpDelta': _signed(_median(_values(fcp_rows, 'Delta_MMR_TopK_FCP'))),
        'ResultFacLocTopKMeanFcpDelta': _signed(_mean(_values(fcp_rows, 'Delta_FacLoc_TopK_FCP'))),
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
                f'{prefix}FacLocMmrMeanDelta': _signed(row.get('MeanDeltaFacLocMMR')),
                f'{prefix}FacLocMmrMedianDelta': _signed(row.get('MedianDeltaFacLocMMR')),
                f'{prefix}FacLocTopKMeanDelta': _signed(row.get('MeanDeltaFacLocTopK')),
                f'{prefix}MmrTopKMeanDelta': _signed(row.get('MeanDeltaMMRTopK')),
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
                    f'{prefix}TiedRows': _integer(
                        sum(abs(delta) <= threshold for delta in deltas)
                    ),
                    f'{prefix}WorseRows': _integer(sum(delta < -threshold for delta in deltas)),
                    f'{prefix}MeanDelta': _signed(_mean(deltas)),
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
            _mean(_values(facloc_rows, 'SafeLambdaFraction')), digits=3
        ),
        'ResultFacLocSafeLambdaMedian': _fixed(
            _median(_values(facloc_rows, 'SafeLambdaFraction')), digits=3
        ),
        'ResultFacLocSafeLambdaMin': _fixed(
            min(_values(facloc_rows, 'SafeLambdaFraction') or [0.0]), digits=3
        ),
        'ResultMmrSafeLambdaMean': _fixed(_mean(_values(mmr_rows, 'SafeLambdaFraction')), digits=3),
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
            _mean(_values(low_rows, 'Delta_FacLoc_MMR_alpha_nDCG'))
        ),
    }


def _render_table(*, spec: ThesisTableSpec, rows: Sequence[Mapping[str, object]]) -> str:
    grouped_rows = _group_rows(rows=rows, metric_order=spec.metric_order)
    header = ' & '.join(spec.headers) + r' \\'
    lines = [
        rf'\subsubsection{{{spec.subsection_title}}}',
        r'\begingroup',
        r'\small',
        rf'\setlength{{\tabcolsep}}{{{spec.tabcolsep}}}',
        r'\renewcommand{\arraystretch}{1.10}',
        rf'\begin{{longtable}}{{{spec.column_spec}}}',
        rf'\caption{{{spec.caption}}}',
        rf'\label{{{spec.label}}} \\',
        '',
        r'\toprule',
        header,
        r'\midrule',
        r'\endfirsthead',
        '',
        rf'\caption[]{{{spec.caption.removesuffix(".")} continued.}} \\',
        r'\toprule',
        header,
        r'\midrule',
        r'\endhead',
        '',
        r'\midrule',
        rf'\multicolumn{{{len(spec.headers)}}}{{r}}{{Continued on next page}} \\',
        r'\endfoot',
        '',
        r'\bottomrule',
        r'\endlastfoot',
        '',
    ]
    for index, (metric, metric_rows) in enumerate(grouped_rows):
        color = 'metricgroupwhite' if index % 2 == 0 else 'metricgroupgray'
        metric_macro = _METRIC_MACROS.get(metric, metric)
        lines.append(rf'\{spec.group_macro}{{{color}}}{{{metric_macro}}}')
        lines.extend(rf'\rowcolor{{{color}}}{row}' for row in _render_tabulate_rows(metric_rows))
        if index < len(grouped_rows) - 1:
            lines.extend([r'\midrule', ''])
    lines.extend([r'\end{longtable}', r'\endgroup'])
    return '\n'.join(lines)


def _group_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    metric_order: Sequence[str],
) -> list[tuple[str, list[Mapping[str, object]]]]:
    by_metric: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        metric = str(row.get('MetricLabel') or '')
        if metric:
            by_metric.setdefault(metric, []).append(row)
    ordered = [(metric, by_metric.pop(metric)) for metric in metric_order if metric in by_metric]
    return [*ordered, *sorted(by_metric.items())]


def _render_tabulate_rows(rows: Sequence[Mapping[str, object]]) -> list[str]:
    row_cells = _table_cells(rows)
    tabulate_output = tabulate(
        row_cells,
        headers=tuple(f'Column {index}' for index in range(len(row_cells[0]))),
        tablefmt='latex_longtable',
        disable_numparse=True,
    )
    return _extract_longtable_body(tabulate_output)


def _table_cells(rows: Sequence[Mapping[str, object]]) -> list[list[str]]:
    first_row = rows[0]
    is_family_budget = 'BudgetCategoryLabel' in first_row
    is_metric_budget = 'BudgetView' in first_row
    cells: list[list[str]] = []
    previous_family: str | None = None
    for row in rows:
        family = _family_label(row.get('ExperimentFamilyLabel'))
        if is_metric_budget:
            cells.append(_aggregate_cells(row))
        elif is_family_budget:
            displayed_family = '' if family == previous_family else family
            cells.append(_family_budget_cells(row, displayed_family))
            previous_family = family
        else:
            cells.append(_family_cells(row, family))
    return cells


def _aggregate_cells(row: Mapping[str, object]) -> list[str]:
    return [
        _budget_label(row.get('BudgetCategory')),
        _integer(row.get('Rows')),
        _percent(row.get('FacLocBetterPct')),
        _percent(row.get('FacLocTiedPct')),
        _percent(row.get('FacLocWorsePct')),
        _delta(row.get('MeanDeltaFacLocMMR')),
        _delta(row.get('MeanDeltaFacLocTopK')),
    ]


def _family_cells(row: Mapping[str, object], family: str) -> list[str]:
    return [
        family,
        _integer(row.get('Rows')),
        _percent(row.get('FacLocBetterPct')),
        _percent(row.get('FacLocTiedPct')),
        _percent(row.get('FacLocWorsePct')),
        _delta(row.get('MeanDeltaFacLocMMR')),
        _delta(row.get('MeanDeltaFacLocTopK')),
    ]


def _family_budget_cells(row: Mapping[str, object], family: str) -> list[str]:
    return [
        family,
        _budget_label(row.get('BudgetCategory')),
        _integer(row.get('Rows')),
        _percent(row.get('FacLocBetterPct')),
        _percent(row.get('FacLocTiedPct')),
        _percent(row.get('FacLocWorsePct')),
        _delta(row.get('MeanDeltaFacLocMMR')),
        _delta(row.get('MeanDeltaFacLocTopK')),
    ]


def _extract_longtable_body(tabulate_output: str) -> list[str]:
    lines = tabulate_output.splitlines()
    try:
        body_start = lines.index(r'\endhead') + 1
        body_end = len(lines) - 2
    except ValueError as exc:
        raise ValueError('Unexpected tabulate latex_longtable output.') from exc
    if body_end <= body_start or lines[body_end] != r'\hline':
        raise ValueError('Unexpected tabulate latex_longtable body boundaries.')
    return [line.strip() for line in lines[body_start:body_end]]


def _family_label(value: object) -> str:
    label = str(value or 'Unknown')
    return _FAMILY_LABELS.get(label, label.removesuffix(' distributions'))


def _budget_label(value: object) -> str:
    return _BUDGET_LABELS.get(str(value or ''), str(value or 'Unknown'))


def _budget_result_token(value: object) -> str | None:
    return _BUDGET_RESULT_TOKENS.get(str(value or ''))


def _metric_result_token(value: object) -> str | None:
    return _METRIC_RESULT_TOKENS.get(str(value or ''))


def _embedding_model_result_token(value: object) -> str | None:
    return _EMBEDDING_MODEL_RESULT_TOKENS.get(str(value or ''))


def _label_token(value: object) -> str:
    alphanumeric_text = ''.join(ch if ch.isalnum() else ' ' for ch in str(value))
    return ''.join(part for part in alphanumeric_text.title().split())


def _experiment_embedding_budget_key(row: Mapping[str, object]) -> tuple[str, str, int] | None:
    k_value = _float(row.get('k'))
    if k_value is None:
        return None
    return (str(row.get('Experiment') or ''), str(row.get('EmbeddingModel') or ''), int(k_value))


def _integer(value: object) -> str:
    return f'{int(value)}' if isinstance(value, int | float) else ''


def _percent(value: object) -> str:
    if not isinstance(value, int | float):
        return ''
    return f'{value * 100:.1f}'.rstrip('0').rstrip('.') + '%'


def _tex_percent(value: object) -> str:
    if not isinstance(value, int | float):
        return ''
    return f'{value * 100:.1f}'.rstrip('0').rstrip('.') + r'\%'


def _delta(value: object) -> str:
    return f'{value:+.4f}' if isinstance(value, int | float) else ''


def _values(rows: Sequence[Mapping[str, object]], column: str) -> list[float]:
    return [value for value in (_float(row.get(column)) for row in rows) if value is not None]


def _float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _fixed(value: float | None, *, digits: int) -> str:
    return f'{value:.{digits}f}' if value is not None else 'n/a'


def _signed(value: object, *, digits: int = 4) -> str:
    numeric = _float(value)
    return f'{numeric:+.{digits}f}' if numeric is not None else 'n/a'


def _outcome_count(rows: Sequence[Mapping[str, object]], outcome: str) -> int:
    return sum(row.get('FacLocVsMMR_FCPOutcome') == outcome for row in rows)


def _budget_rows(
    rows: Sequence[Mapping[str, object]],
    budget_category: str,
) -> list[Mapping[str, object]]:
    return [row for row in rows if row.get('BudgetCategory') == budget_category]


def _family_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        family = _family_label(row.get('ExperimentFamilyLabel'))
        counts[family] = counts.get(family, 0) + 1
    return counts


def _negative_family_summary(counts: Mapping[str, int]) -> str:
    if not counts:
        return 'none'
    parts = []
    for family, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        family_text = family.lower().replace(' ', '-')
        noun = 'configuration' if count == 1 else 'configurations'
        parts.append(f'{family_text} {noun} ({count})')
    if len(parts) == 1:
        return parts[0]
    return ', '.join(parts[:-1]) + f' and {parts[-1]}'
