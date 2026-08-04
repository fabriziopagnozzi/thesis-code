from __future__ import annotations

import polars as pl

from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
    select_best_lambda_row,
)
from experiments.medical_dataset_gen.retrieval.reranker import DENSE_RERANKER_STRATEGY
from experiments.medical_dataset_gen.utils.global_schemas import EvaluationMode, ExperimentCfg

_SELECTION_SPLIT = 'validation'
_REPORT_SPLIT = 'test'


def stats_sliced_results_df(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame()
    slice_columns = [
        'condition_id',
        'cohort_dimension_id',
        'cohort_contrast_family',
        'cohort_contrast_id',
        'primary_axis',
        'secondary_axis',
        'template_id',
        'passes_geometry_filter',
        'n_topk_retrieved_facets',
    ]
    slice_columns = [column for column in slice_columns if column in results.columns]
    agg_exprs: list[pl.Expr] = [
        pl.col('query_id').n_unique().alias('n_queries'),
        pl.col('facet_coverage').mean().alias('FacetCoverage@k'),
        _all_facet_coverage_expr(results),
        pl.col('weighted_facet_coverage').mean().alias('FacetWeightedRecall@k'),
        pl.col('facet_coverage_purity').mean().alias('FacetCoveragePurity@k'),
        pl.col('all_facet_clean').mean().alias('AllFacetCleanRate@k'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('gold_recall').mean().alias('Recall@k'),
        pl.col('gold_f1').mean().alias('F1@k'),
        pl.col('primary_axis_rate').mean().alias('PrimaryAxisRate'),
        pl.col('dominant_facet_rate').mean().alias('DominantFacetRate'),
        pl.col('redundant_gold_rate').mean().alias('RedundantGoldRate'),
    ]
    optional_rouge_exprs = [
        ('answer_rouge1_recall', 'AnswerROUGE1Recall@k'),
        ('answer_rouge1_precision', 'AnswerROUGE1Precision@k'),
        ('answer_rouge1_f1', 'AnswerROUGE1F1@k'),
        ('answer_rouge2_recall', 'AnswerROUGE2Recall@k'),
    ]
    agg_exprs.extend(
        pl.col(source_col).mean().alias(target_col)
        for source_col, target_col in optional_rouge_exprs
        if source_col in results.columns
    )
    return (
        results.group_by(*slice_columns, 'strategy', 'lam', 'k')
        .agg(agg_exprs)
        .sort(*slice_columns, 'k', 'strategy', 'lam')
    )


def stats_aggregated_results_df(results: pl.DataFrame) -> pl.DataFrame:
    if len(results) == 0:
        return pl.DataFrame()

    agg_polars_exprs: list[pl.Expr] = [
        pl.col('query_id').n_unique().alias('n_queries'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('gold_recall').mean().alias('Recall@k'),
        pl.col('gold_f1').mean().alias('F1@k'),
        pl.col('average_precision_at_k').mean().alias('MAP@k'),
        pl.col('facet_coverage').mean().alias('FacetCoverage@k'),
        _all_facet_coverage_expr(results),
        pl.col('weighted_facet_coverage').mean().alias('FacetWeightedRecall@k'),
        pl.col('facet_coverage_purity').mean().alias('FacetCoveragePurity@k'),
        pl.col('all_facet_clean').mean().alias('AllFacetCleanRate@k'),
        pl.col('facet_mrr_at_k').mean().alias('FacetMRR@k'),
        pl.col('alpha_ndcg').mean().alias('alpha-nDCG@k'),
        pl.col('distractor_rate').mean().alias('DistractorRate'),
        pl.col('near_miss_distractor_rate').mean().alias('NearMissDistractorRate'),
        pl.col('background_outlier_rate').mean().alias('BackgroundOutlierRate'),
        pl.col('primary_axis_rate').mean().alias('PrimaryAxisRate'),
        pl.col('dominant_facet_rate').mean().alias('DominantFacetRate'),
        pl.col('redundant_gold_rate').mean().alias('RedundantGoldRate'),
        pl.col('fac_cov_score').mean().alias('fac'),
        pl.col('avg_cos').mean().alias('avg_cos'),
        pl.col('jaccard_vs_topk').mean().alias('jac'),
    ]
    optional_rouge_exprs = [
        ('answer_rouge1_recall', 'AnswerROUGE1Recall@k'),
        ('answer_rouge1_precision', 'AnswerROUGE1Precision@k'),
        ('answer_rouge1_f1', 'AnswerROUGE1F1@k'),
        ('answer_rouge2_recall', 'AnswerROUGE2Recall@k'),
    ]
    agg_polars_exprs.extend(
        pl.col(source_col).mean().alias(target_col)
        for source_col, target_col in optional_rouge_exprs
        if source_col in results.columns
    )
    stats = (
        results.group_by('strategy', 'lam', 'k').agg(agg_polars_exprs).sort('k', 'strategy', 'lam')
    )
    ordered_columns = [
        'strategy',
        'lam',
        'k',
        'n_queries',
        'Precision@k',
        'Recall@k',
        'F1@k',
        'MAP@k',
        'FacetCoverage@k',
        'AllFacetCoverageRate@k',
        'FacetWeightedRecall@k',
        'FacetCoveragePurity@k',
        'AllFacetCleanRate@k',
        'FacetMRR@k',
        'alpha-nDCG@k',
        'AnswerROUGE1Recall@k',
        'AnswerROUGE1Precision@k',
        'AnswerROUGE1F1@k',
        'AnswerROUGE2Recall@k',
        'DistractorRate',
        'NearMissDistractorRate',
        'BackgroundOutlierRate',
        'PrimaryAxisRate',
        'DominantFacetRate',
        'RedundantGoldRate',
        'fac',
        'avg_cos',
        'jac',
    ]
    return stats.select([col for col in ordered_columns if col in stats.columns])


def stats_for_evaluation_mode(
    results: pl.DataFrame,
    *,
    mode: EvaluationMode,
    cfg: ExperimentCfg,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if mode == 'exploring':
        all_stats = stats_aggregated_results_df(results)
        return all_stats, pl.DataFrame(), pl.DataFrame()

    selection_results = results_for_split(results, _SELECTION_SPLIT)
    report_results = results_for_split(results, _REPORT_SPLIT)
    selection_stats = stats_aggregated_results_df(selection_results)
    report_grid_stats = stats_aggregated_results_df(report_results)
    report_stats = _heldout_report_stats(
        selection_stats=selection_stats,
        report_grid_stats=report_grid_stats,
        cfg=cfg,
    )
    return report_stats, selection_stats, report_grid_stats


def results_for_split(results: pl.DataFrame, split: str) -> pl.DataFrame:
    if results.is_empty() or 'split' not in results.columns:
        return pl.DataFrame()
    return results.filter(pl.col('split') == split)


def _all_facet_coverage_expr(results: pl.DataFrame) -> pl.Expr:
    if 'all_facet_coverage' in results.columns:
        return pl.col('all_facet_coverage').mean().alias('AllFacetCoverageRate@k')
    return (pl.col('facet_coverage') == 1.0).cast(pl.Float64).mean().alias('AllFacetCoverageRate@k')


def _heldout_report_stats(
    *,
    selection_stats: pl.DataFrame,
    report_grid_stats: pl.DataFrame,
    cfg: ExperimentCfg,
) -> pl.DataFrame:
    if selection_stats.is_empty() or report_grid_stats.is_empty():
        return pl.DataFrame()

    rows: list[pl.DataFrame] = []
    k_values = sorted(set(int(k) for k in cfg.retrieval.k_values))
    for k in k_values:
        topk_row = report_grid_stats.filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k))
        if not topk_row.is_empty():
            rows.append(_annotate_heldout_row(topk_row.head(1), selected_on_metric_value=None))

        for strategy in _heldout_non_topk_strategies(cfg, report_grid_stats):
            if _has_lambda_grid(selection_stats, strategy, k):
                selected = select_best_lambda_row(
                    selection_stats,
                    strategy=strategy,
                    k=k,
                    cfg=cfg.evaluation.lambda_selection,
                )
                if selected is None:
                    continue
                selected_lam = selected.get('lam', None)
                if selected_lam is None:
                    continue

                report_row = report_grid_stats.filter(
                    (pl.col('strategy') == strategy)
                    & (pl.col('k') == k)
                    & (pl.col('lam') == selected_lam)
                )
                selected_metric_value = selected.get(LAMBDA_SELECTION_MAXIMIZING_METRIC)
            else:
                report_row = report_grid_stats.filter(
                    (pl.col('strategy') == strategy)
                    & (pl.col('k') == k)
                    & (pl.col('lam').is_null())
                )
                selected_metric_value = None

            if report_row.is_empty():
                continue

            rows.append(
                _annotate_heldout_row(
                    report_row.head(1),
                    selected_on_metric_value=selected_metric_value,
                )
            )

    return pl.concat(rows).sort('k', 'strategy', 'lam') if rows else pl.DataFrame()


def _heldout_non_topk_strategies(cfg: ExperimentCfg, report_grid_stats: pl.DataFrame) -> list[str]:
    configured = {str(strategy) for strategy in cfg.retrieval.strategies if strategy != 'top_k'}
    if cfg.evaluation.use_reranker:
        configured.add(DENSE_RERANKER_STRATEGY)

    present = set(str(strategy) for strategy in report_grid_stats['strategy'].unique().to_list())
    return sorted(configured & present, key=_evaluation_strategy_sort_key)


def _evaluation_strategy_sort_key(strategy: str) -> tuple[int, str]:
    preferred_order = {
        'top_k': 0,
        'fac_loc': 1,
        'mmr': 2,
        DENSE_RERANKER_STRATEGY: 3,
    }
    return preferred_order.get(strategy, len(preferred_order)), strategy


def _has_lambda_grid(stats_df: pl.DataFrame, strategy: str, k: int) -> bool:
    return (
        stats_df.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
        .select(pl.col('lam').drop_nulls().len())
        .item()
        != 0
    )


def _annotate_heldout_row(
    row: pl.DataFrame,
    *,
    selected_on_metric_value: float | None,
) -> pl.DataFrame:
    return row.with_columns(
        pl.lit(_SELECTION_SPLIT).alias('lambda_selection_split'),
        pl.lit(_REPORT_SPLIT).alias('report_split'),
        pl.lit(LAMBDA_SELECTION_MAXIMIZING_METRIC).alias('lambda_selection_metric'),
        pl.lit(selected_on_metric_value, dtype=pl.Float64).alias('lambda_selection_metric_value'),
    )
