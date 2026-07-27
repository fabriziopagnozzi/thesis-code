from __future__ import annotations

import polars as pl


def adapt_results_for_synthetic_plots(results_df: pl.DataFrame) -> pl.DataFrame:
    if results_df.is_empty():
        return results_df

    exprs: list[pl.Expr] = []
    if 'aspect_recall' in results_df.columns and 'facet_coverage' not in results_df.columns:
        exprs.append(pl.col('aspect_recall').alias('facet_coverage'))
    if (
        'weighted_aspect_recall' in results_df.columns
        and 'weighted_facet_coverage' not in results_df.columns
    ):
        exprs.append(pl.col('weighted_aspect_recall').alias('weighted_facet_coverage'))
    if 'gold_precision' in results_df.columns and 'distractor_rate' not in results_df.columns:
        exprs.append((1.0 - pl.col('gold_precision')).alias('distractor_rate'))
    if exprs:
        results_df = results_df.with_columns(exprs)
    return results_df


def adapt_stats_for_synthetic_plots(stats_df: pl.DataFrame) -> pl.DataFrame:
    if stats_df.is_empty():
        return stats_df

    exprs: list[pl.Expr] = []
    if 'AR' in stats_df.columns and 'FacetCoverage@k' not in stats_df.columns:
        exprs.append(pl.col('AR').alias('FacetCoverage@k'))
    if 'GP' in stats_df.columns and 'Precision@k' not in stats_df.columns:
        exprs.append(pl.col('GP').alias('Precision@k'))
    if 'GR' in stats_df.columns and 'Recall@k' not in stats_df.columns:
        exprs.append(pl.col('GR').alias('Recall@k'))
    if 'WAR' in stats_df.columns and 'MeanFacetRecall@k' not in stats_df.columns:
        exprs.append(pl.col('WAR').alias('MeanFacetRecall@k'))
    if 'GP' in stats_df.columns and 'DistractorRate' not in stats_df.columns:
        exprs.append((1.0 - pl.col('GP')).alias('DistractorRate'))
    if 'cos' in stats_df.columns and 'avg_cos' not in stats_df.columns:
        exprs.append(pl.col('cos').alias('avg_cos'))
    if 'ans_rouge1_rec' in stats_df.columns and 'AnswerROUGE1Recall@k' not in stats_df.columns:
        exprs.append(pl.col('ans_rouge1_rec').alias('AnswerROUGE1Recall@k'))
    if 'ans_rouge1_prec' in stats_df.columns and 'AnswerROUGE1Precision@k' not in stats_df.columns:
        exprs.append(pl.col('ans_rouge1_prec').alias('AnswerROUGE1Precision@k'))
    if exprs:
        stats_df = stats_df.with_columns(exprs)
    return stats_df
