from __future__ import annotations

from collections.abc import Iterator
from typing import NotRequired, TypedDict, cast

import polars as pl

from experiments.medical_dataset_gen.global_config import LambdaSelectionCfg

LAMBDA_SELECTION_MAXIMIZING_METRIC = 'FacetCoveragePurity@k'

LambdaSelectionMetricRow = TypedDict(
    'LambdaSelectionMetricRow',
    {
        'strategy': NotRequired[str],
        'lam': NotRequired[float],
        'k': NotRequired[int],
        'n_queries': NotRequired[int],
        'Precision@k': NotRequired[float],
        'Recall@k': NotRequired[float],
        'F1@k': NotRequired[float],
        'MAP@k': NotRequired[float],
        'FacetCoverage@k': NotRequired[float],
        'MeanFacetRecall@k': NotRequired[float],
        'FacetCoveragePurity@k': NotRequired[float],
        'AllFacetCleanRate@k': NotRequired[float],
        'FacetMRR@k': NotRequired[float],
        'alpha-nDCG@k': NotRequired[float],
        'AnswerROUGE1Recall@k': NotRequired[float],
        'AnswerROUGE1Precision@k': NotRequired[float],
        'AnswerROUGE1F1@k': NotRequired[float],
        'AnswerROUGE2Recall@k': NotRequired[float],
        'DistractorRate': NotRequired[float],
        'NearMissDistractorRate': NotRequired[float],
        'BackgroundOutlierRate': NotRequired[float],
        'PrimaryAxisRate': NotRequired[float],
        'CalibratedFacetRate': NotRequired[float],
        'RedundantGoldRate': NotRequired[float],
        'fac': NotRequired[float],
        'avg_cos': NotRequired[float],
        'jac': NotRequired[float],
    },
    total=False,
)


def select_best_lambda_rows(
    stats_df: pl.DataFrame,
    *,
    strategy: str,
    k_values: list[int],
    cfg: LambdaSelectionCfg,
) -> pl.DataFrame:
    strategy_df = stats_df.filter(pl.col('strategy') == strategy)
    rows: list[pl.DataFrame] = []
    for k in k_values:
        k_df = strategy_df.filter(pl.col('k') == k)
        if k_df.is_empty():
            continue
        selected = select_best_lambda_row_frame(k_df, cfg=cfg)
        if selected is not None:
            rows.append(selected)
    return pl.concat(rows).sort('k') if rows else pl.DataFrame()


def select_best_lambda_row(
    stats_df: pl.DataFrame,
    *,
    strategy: str,
    k: int,
    cfg: LambdaSelectionCfg,
) -> LambdaSelectionMetricRow | None:
    sub = stats_df.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
    selected = select_best_lambda_row_frame(sub, cfg=cfg)
    if selected is None or selected.height == 0:
        return None
    return cast(LambdaSelectionMetricRow, selected.row(0, named=True))


def select_best_lambda_row_frame(
    lambda_df: pl.DataFrame,
    *,
    cfg: LambdaSelectionCfg,
) -> pl.DataFrame | None:
    if lambda_df.is_empty():
        return None

    rows = list(_iter_metric_rows(lambda_df))
    best_idx = _select_best_by_primary_metric(rows, cfg=cfg)
    return lambda_df.slice(best_idx, 1) if best_idx is not None else None


def lambda_selection_policy_note(cfg: LambdaSelectionCfg) -> str:
    tie_break = 'lower lambda' if cfg.tie_break == 'lower_lambda' else 'higher lambda'
    return (
        f'lambda*: maximize {LAMBDA_SELECTION_MAXIMIZING_METRIC} within strategy x k '
        f'(ties -> {tie_break})'
    )


def lambda_selection_short_label(cfg: LambdaSelectionCfg) -> str:
    tie_break = 'lower lambda' if cfg.tie_break == 'lower_lambda' else 'higher lambda'
    return f'lambda* = argmax {LAMBDA_SELECTION_MAXIMIZING_METRIC} (ties -> {tie_break})'


def _select_best_by_primary_metric(
    rows: list[LambdaSelectionMetricRow],
    *,
    cfg: LambdaSelectionCfg,
) -> int | None:
    best: tuple[float, float, int] | None = None
    best_idx: int | None = None
    for idx, row in enumerate(rows):
        primary_value = row.get(LAMBDA_SELECTION_MAXIMIZING_METRIC)
        if primary_value is None:
            raise ValueError(
                f'Lambda-selection metric is unavailable: {LAMBDA_SELECTION_MAXIMIZING_METRIC}'
            )
        lambda_rank = _lambda_tie_break_rank(row, cfg)
        candidate = (
            float(primary_value),
            lambda_rank,
            -idx,
        )
        if best is None or candidate > best:
            best = candidate
            best_idx = idx

    return best_idx


def _iter_metric_rows(lambda_df: pl.DataFrame) -> Iterator[LambdaSelectionMetricRow]:
    for row in lambda_df.iter_rows(named=True):
        yield cast(LambdaSelectionMetricRow, row)


def _lambda_tie_break_rank(
    row: LambdaSelectionMetricRow,
    cfg: LambdaSelectionCfg,
) -> float:
    lam = row.get('lam') or 0.0
    return -lam if cfg.tie_break == 'lower_lambda' else lam
