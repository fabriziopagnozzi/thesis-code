import polars as pl

from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    pair_kernel_polars_expr,
    sigmoid_polars_expr,
)
from experiments.medical_dataset_gen.global_config import (
    MethodsComparisonKernelMetricCfg,
    MethodsComparisonKernelsCfg,
)
from experiments.medical_dataset_gen.schemas.metrics_schemas import METRIC_NAME_TO_FIELD

_LAMBDA_PAIR_AGREEMENT_EVAL_METRICS = [
    'Precision@k',
    'Recall@k',
    'F1@k',
    'MAP@k',
    'MeanFacetHitRate@k',
    'MeanFacetRecall@k',
    'FacetMRR@k',
    'alpha-nDCG@k',
    'AnswerROUGE1Recall@k',
    'AnswerROUGE1Precision@k',
    'AnswerROUGE2Recall@k',
    'MacroFacetAnswerROUGE1Recall@k',
]

_LAMBDA_PAIR_AGREEMENT_DIAGNOSTIC_METRICS = [
    'DistractorRate',
    'NearMissDistractorRate',
    'BackgroundOutlierRate',
    'PrimaryAxisRate',
    'fac',
    'avg_cos',
    'jac',
]

_LAMBDA_PAIR_AGREEMENT_METRICS = (
    _LAMBDA_PAIR_AGREEMENT_EVAL_METRICS + _LAMBDA_PAIR_AGREEMENT_DIAGNOSTIC_METRICS
)


def build_lambda_pair_agreement(
    stats_df: pl.DataFrame,
    *,
    results_df: pl.DataFrame | None = None,
    kernel_cfg: MethodsComparisonKernelsCfg | None = None,
) -> pl.DataFrame:
    metric_cols = lambda_pair_agreement_metric_cols(stats_df)
    eval_metric_cols = lambda_pair_agreement_eval_metric_cols(stats_df)
    active_kernel_metrics = _active_lambda_pair_kernel_metrics(
        kernel_cfg or MethodsComparisonKernelsCfg(),
        stats_df,
        results_df,
    )
    schema = {
        'k': pl.Int64,
        'fac_loc_lam': pl.Float64,
        'mmr_lam': pl.Float64,
        **{f'abs_diff__{metric}': pl.Float64 for metric in metric_cols},
        **{
            f'{strategy}_gain_vs_topk__{metric_cfg.summary_metric}': pl.Float64
            for metric_cfg in active_kernel_metrics
            for strategy in ['fac_loc', 'mmr']
        },
        **{
            f'{strategy}_lower95_gain_vs_topk__{metric_cfg.summary_metric}': pl.Float64
            for metric_cfg in active_kernel_metrics
            for strategy in ['fac_loc', 'mmr']
        },
        **{
            f'{strategy}_kernel__{metric_cfg.summary_metric}': pl.Float64
            for metric_cfg in active_kernel_metrics
            for strategy in ['fac_loc', 'mmr']
        },
        'fac_loc_kernel_score': pl.Float64,
        'mmr_kernel_score': pl.Float64,
        'pair_quality_kernel': pl.Float64,
        'weighted_mean_abs_diff': pl.Float64,
        'mean_abs_diff': pl.Float64,
        'rank_within_k': pl.UInt32,
        'weighted_rank_within_k': pl.UInt32,
    }
    if stats_df.is_empty() or not metric_cols or not eval_metric_cols:
        return pl.DataFrame(schema=schema)

    cfg = kernel_cfg or MethodsComparisonKernelsCfg()
    lambda_max = float(cfg.lambda_max)
    fac_df = stats_df.filter((pl.col('strategy') == 'fac_loc') & (pl.col('lam') <= lambda_max))
    mmr_df = stats_df.filter((pl.col('strategy') == 'mmr') & (pl.col('lam') <= lambda_max))
    if fac_df.is_empty() or mmr_df.is_empty():
        return pl.DataFrame(schema=schema)

    fac_df = _attach_strategy_kernel_columns(
        fac_df,
        strategy='fac_loc',
        results_df=results_df,
        metric_cfgs=active_kernel_metrics,
    )
    mmr_df = _attach_strategy_kernel_columns(
        mmr_df,
        strategy='mmr',
        results_df=results_df,
        metric_cfgs=active_kernel_metrics,
    )

    fac_select = [
        pl.col('k'),
        pl.col('lam').cast(pl.Float64).alias('fac_loc_lam'),
        *[pl.col(metric).alias(f'fac__{metric}') for metric in metric_cols],
        *[
            pl.col(f'gain_vs_topk__{metric_cfg.summary_metric}').alias(
                f'fac_loc_gain_vs_topk__{metric_cfg.summary_metric}'
            )
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'lower95_gain_vs_topk__{metric_cfg.summary_metric}').alias(
                f'fac_loc_lower95_gain_vs_topk__{metric_cfg.summary_metric}'
            )
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'kernel__{metric_cfg.summary_metric}').alias(
                f'fac_loc_kernel__{metric_cfg.summary_metric}'
            )
            for metric_cfg in active_kernel_metrics
        ],
        pl.col('kernel_score').alias('fac_loc_kernel_score'),
    ]
    mmr_select = [
        pl.col('k'),
        pl.col('lam').cast(pl.Float64).alias('mmr_lam'),
        *[pl.col(metric).alias(f'mmr__{metric}') for metric in metric_cols],
        *[
            pl.col(f'gain_vs_topk__{metric_cfg.summary_metric}').alias(
                f'mmr_gain_vs_topk__{metric_cfg.summary_metric}'
            )
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'lower95_gain_vs_topk__{metric_cfg.summary_metric}').alias(
                f'mmr_lower95_gain_vs_topk__{metric_cfg.summary_metric}'
            )
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'kernel__{metric_cfg.summary_metric}').alias(
                f'mmr_kernel__{metric_cfg.summary_metric}'
            )
            for metric_cfg in active_kernel_metrics
        ],
        pl.col('kernel_score').alias('mmr_kernel_score'),
    ]
    joined = fac_df.select(fac_select).join(mmr_df.select(mmr_select), on='k', how='inner')
    if joined.is_empty():
        return pl.DataFrame(schema=schema)

    diff_cols = [f'abs_diff__{metric}' for metric in metric_cols]
    eval_diff_cols = [f'abs_diff__{metric}' for metric in eval_metric_cols]
    agreement = joined.select(
        'k',
        'fac_loc_lam',
        'mmr_lam',
        *[
            (pl.col(f'fac__{metric}') - pl.col(f'mmr__{metric}')).abs().alias(f'abs_diff__{metric}')
            for metric in metric_cols
        ],
        *[
            pl.col(f'fac_loc_gain_vs_topk__{metric_cfg.summary_metric}')
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'fac_loc_lower95_gain_vs_topk__{metric_cfg.summary_metric}')
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'fac_loc_kernel__{metric_cfg.summary_metric}')
            for metric_cfg in active_kernel_metrics
        ],
        pl.col('fac_loc_kernel_score'),
        *[
            pl.col(f'mmr_gain_vs_topk__{metric_cfg.summary_metric}')
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'mmr_lower95_gain_vs_topk__{metric_cfg.summary_metric}')
            for metric_cfg in active_kernel_metrics
        ],
        *[
            pl.col(f'mmr_kernel__{metric_cfg.summary_metric}')
            for metric_cfg in active_kernel_metrics
        ],
        pl.col('mmr_kernel_score'),
    ).with_columns(pl.mean_horizontal(eval_diff_cols).alias('mean_abs_diff'))
    agreement = agreement.with_columns(
        pair_kernel_polars_expr(cfg).alias('pair_quality_kernel'),
    ).with_columns(
        (
            pl.col('mean_abs_diff')
            / pl.max_horizontal(
                pl.col('pair_quality_kernel'),
                pl.lit(float(cfg.kernel_floor)),
            ).pow(float(cfg.agreement_alpha))
        ).alias('weighted_mean_abs_diff')
    )
    agreement = agreement.sort(
        ['k', 'mean_abs_diff', 'fac_loc_lam', 'mmr_lam'],
        descending=[False, False, False, False],
    ).with_row_index('_row_idx')
    agreement = agreement.with_columns(
        (pl.col('_row_idx') - pl.col('_row_idx').min().over('k') + 1)
        .cast(pl.UInt32)
        .alias('rank_within_k')
    )
    weighted_ranked = agreement.sort(
        ['k', 'weighted_mean_abs_diff', 'fac_loc_lam', 'mmr_lam'],
        descending=[False, False, False, False],
    ).with_row_index('_weighted_row_idx')
    agreement = weighted_ranked.with_columns(
        (pl.col('_weighted_row_idx') - pl.col('_weighted_row_idx').min().over('k') + 1)
        .cast(pl.UInt32)
        .alias('weighted_rank_within_k')
    ).drop('_row_idx', '_weighted_row_idx')
    output_cols = [
        'k',
        'fac_loc_lam',
        'mmr_lam',
        *diff_cols,
        *[
            f'fac_loc_gain_vs_topk__{metric_cfg.summary_metric}'
            for metric_cfg in active_kernel_metrics
        ],
        *[
            f'fac_loc_lower95_gain_vs_topk__{metric_cfg.summary_metric}'
            for metric_cfg in active_kernel_metrics
        ],
        *[f'fac_loc_kernel__{metric_cfg.summary_metric}' for metric_cfg in active_kernel_metrics],
        'fac_loc_kernel_score',
        *[f'mmr_gain_vs_topk__{metric_cfg.summary_metric}' for metric_cfg in active_kernel_metrics],
        *[
            f'mmr_lower95_gain_vs_topk__{metric_cfg.summary_metric}'
            for metric_cfg in active_kernel_metrics
        ],
        *[f'mmr_kernel__{metric_cfg.summary_metric}' for metric_cfg in active_kernel_metrics],
        'mmr_kernel_score',
        'pair_quality_kernel',
        'weighted_mean_abs_diff',
        'mean_abs_diff',
        'rank_within_k',
        'weighted_rank_within_k',
    ]
    return agreement.select(output_cols)


def lambda_pair_agreement_metric_cols(stats_df: pl.DataFrame) -> list[str]:
    return [metric for metric in _LAMBDA_PAIR_AGREEMENT_METRICS if metric in stats_df.columns]


def lambda_pair_agreement_eval_metrics() -> list[str]:
    return list(_LAMBDA_PAIR_AGREEMENT_EVAL_METRICS)


def lambda_pair_agreement_eval_metric_cols(stats_df: pl.DataFrame) -> list[str]:
    return [metric for metric in _LAMBDA_PAIR_AGREEMENT_EVAL_METRICS if metric in stats_df.columns]


def _active_lambda_pair_kernel_metrics(
    kernel_cfg: MethodsComparisonKernelsCfg,
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame | None,
) -> list[MethodsComparisonKernelMetricCfg]:
    if results_df is None or results_df.is_empty():
        return []
    active: list[MethodsComparisonKernelMetricCfg] = []
    for metric_cfg in kernel_cfg.metrics:
        if not metric_cfg.enabled:
            continue
        metric_spec = METRIC_NAME_TO_FIELD.get(metric_cfg.summary_metric)
        if metric_spec is None:
            continue
        if (
            metric_cfg.summary_metric not in stats_df.columns
            or metric_spec.result_col not in results_df.columns
        ):
            continue
        active.append(metric_cfg)
    return active


def _attach_strategy_kernel_columns(
    stats_df: pl.DataFrame,
    *,
    strategy: str,
    results_df: pl.DataFrame | None,
    metric_cfgs: list[MethodsComparisonKernelMetricCfg],
) -> pl.DataFrame:
    if results_df is None or results_df.is_empty() or not metric_cfgs or stats_df.is_empty():
        return stats_df.with_columns(pl.lit(0.0).alias('kernel_score'))
    metric_meta = [
        (
            metric_cfg,
            METRIC_NAME_TO_FIELD[metric_cfg.summary_metric].result_col,
            METRIC_NAME_TO_FIELD[metric_cfg.summary_metric].higher_is_better,
        )
        for metric_cfg in metric_cfgs
    ]
    result_cols = list(dict.fromkeys(result_col for _, result_col, _ in metric_meta))
    topk_df = results_df.filter(pl.col('strategy') == 'top_k').select(
        'query_id',
        'k',
        *[pl.col(col).alias(f'topk__{col}') for col in result_cols],
    )
    strat_df = results_df.filter(pl.col('strategy') == strategy).select(
        'query_id',
        'k',
        'lam',
        *[pl.col(col).alias(f'strategy__{col}') for col in result_cols],
    )
    paired_df = topk_df.join(strat_df, on=['query_id', 'k'], how='inner')
    if paired_df.is_empty():
        return stats_df.with_columns(pl.lit(0.0).alias('kernel_score'))

    summary = paired_df.group_by('k', 'lam').agg(*[
        expr
        for result_col in result_cols
        for expr in (
            (pl.col(f'strategy__{result_col}') - pl.col(f'topk__{result_col}'))
            .mean()
            .alias(f'mean_delta__{result_col}'),
            (pl.col(f'strategy__{result_col}') - pl.col(f'topk__{result_col}'))
            .std(ddof=1)
            .alias(f'std_delta__{result_col}'),
            (pl.col(f'strategy__{result_col}') - pl.col(f'topk__{result_col}'))
            .count()
            .alias(f'n_delta__{result_col}'),
        )
    ])

    kernel_exprs: list[pl.Expr] = []
    weighted_kernel_sum: pl.Expr | None = None
    total_weight = 0.0
    for metric_cfg, result_col, higher_is_better in metric_meta:
        direction = 1.0 if higher_is_better else -1.0
        gain_expr = pl.col(f'mean_delta__{result_col}') * direction
        count_expr = pl.col(f'n_delta__{result_col}').cast(pl.Float64)
        ci_half_width_expr = (
            pl
            .when(count_expr >= 2.0)
            .then(1.96 * pl.col(f'std_delta__{result_col}') / count_expr.sqrt())
            .otherwise(pl.lit(float('nan')))
        )
        lower95_expr = gain_expr - ci_half_width_expr
        kernel_expr = sigmoid_polars_expr(
            (gain_expr - float(metric_cfg.target_gain_vs_topk)) / float(metric_cfg.gain_bandwidth)
        ) * sigmoid_polars_expr(
            (lower95_expr - float(metric_cfg.target_lower_bound_vs_topk))
            / float(metric_cfg.lower_bound_bandwidth)
        )
        kernel_exprs.extend([
            gain_expr.alias(f'gain_vs_topk__{metric_cfg.summary_metric}'),
            lower95_expr.alias(f'lower95_gain_vs_topk__{metric_cfg.summary_metric}'),
            kernel_expr.alias(f'kernel__{metric_cfg.summary_metric}'),
        ])
        weight = float(metric_cfg.weight)
        weighted_component = kernel_expr * weight
        weighted_kernel_sum = (
            weighted_component
            if weighted_kernel_sum is None
            else weighted_kernel_sum + weighted_component
        )
        total_weight += weight

    kernel_df = summary.select(
        'k',
        'lam',
        *kernel_exprs,
        (
            weighted_kernel_sum / total_weight if weighted_kernel_sum is not None else pl.lit(0.0)
        ).alias('kernel_score'),
    )
    return stats_df.join(kernel_df, on=['k', 'lam'], how='left').with_columns(
        pl.col('kernel_score').fill_null(0.0)
    )
