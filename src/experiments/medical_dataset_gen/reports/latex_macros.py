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
from experiments.medical_dataset_gen.reports.helpers import family_balanced_mean
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
from experiments.medical_dataset_gen.reports.wording_result_macros import (
    render_wording_result_macros,
)

type ReportRow = dict[str, object]


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
    metric_summary_rows: Sequence[Mapping[str, object]] = (),
    metric_family_summary_rows: Sequence[Mapping[str, object]] = (),
    paired_suite_rows: Sequence[Mapping[str, object]] = (),
    embedding_summary_rows: Sequence[Mapping[str, object]] = (),
    embedding_models: Sequence[str] = (),
    require_complete_wording_grid: bool = False,
    warnings: list[str] | None = None,
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
            lambda_safety_rows=_read_rows(data_dir / 'lambda_safety_summary.csv'),
            metric_summary_rows=_read_rows(data_dir / 'metric_aggregate_summary.csv'),
            metric_family_summary_rows=_read_rows(data_dir / 'metric_family_summary.csv'),
            paired_suite_rows=_read_rows(
                data_dir / 'paired_suite_effect_summary.csv', required=False
            ),
            embedding_summary_rows=_read_rows(data_dir / 'embedding_model_summary.csv'),
            require_complete_wording_grid=True,
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
