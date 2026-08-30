from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tabulate import tabulate

from experiments.medical_dataset_gen.reports.report_config import (
    REPORT_METRIC_LABELS,
    embedding_model_macro_token,
)

LATEX_OUTPUT_DIRNAME = 'latex'
THESIS_AGGREGATE_TABLES_FILENAME = 'exp_results_tables.tex'
THESIS_RESULT_MACROS_FILENAME = 'exp_results_macros.tex'
THESIS_STATISTICAL_TABLES_FILENAME = 'paired_statistical_tables.tex'


def thesis_latex_dir(report_dir: Path) -> Path:
    return report_dir / LATEX_OUTPUT_DIRNAME


def thesis_aggregate_tables_path(report_dir: Path) -> Path:
    return thesis_latex_dir(report_dir) / THESIS_AGGREGATE_TABLES_FILENAME


def thesis_result_macros_path(report_dir: Path) -> Path:
    return thesis_latex_dir(report_dir) / THESIS_RESULT_MACROS_FILENAME


def thesis_statistical_tables_path(report_dir: Path) -> Path:
    return thesis_latex_dir(report_dir) / THESIS_STATISTICAL_TABLES_FILENAME


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
    'NearMissDistractorRate': 'Near-miss rate',
    'BackgroundOutlierRate': 'Background-outlier rate',
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
        '% Auto-generated by experiments.medical_dataset_gen.reports.\n'
        '% Do not edit this file directly; rerun the report instead.\n\n'
        + '\n\n'.join(rendered_tables)
        + '\n'
    )


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
    return embedding_model_macro_token(str(value or ''))


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
