from __future__ import annotations

from collections.abc import Iterator
from typing import NotRequired, TypedDict, cast

import polars as pl

from experiments.medical_dataset_gen.global_config import LambdaSelectionCfg
from experiments.medical_dataset_gen.schemas.metrics_schemas import METRIC_NAME_TO_FIELD

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
        'MeanFacetHitRate@k': NotRequired[float],
        'MeanFacetRecall@k': NotRequired[float],
        'FacetMRR@k': NotRequired[float],
        'alpha-nDCG@k': NotRequired[float],
        'AnswerROUGE1Recall@k': NotRequired[float],
        'AnswerROUGE1Precision@k': NotRequired[float],
        'AnswerROUGE1F1@k': NotRequired[float],
        'AnswerROUGE2Recall@k': NotRequired[float],
        'DistractorRate': NotRequired[float],
        'NearMissDistractorRate': NotRequired[float],
        'BackgroundOutlierRate': NotRequired[float],
        'SameConditionWrongAxisRate': NotRequired[float],
        'PrimaryAxisRate': NotRequired[float],
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
    primary_metric = cfg.primary_metric
    if primary_metric is None:
        best_idx = _select_best_without_primary(rows, cfg=cfg)
    else:
        best_idx = _select_best_with_primary(
            rows,
            cfg=cfg,
            primary_metric=primary_metric,
        )

    return lambda_df.slice(best_idx, 1) if best_idx is not None else None


def lambda_selection_policy_note(cfg: LambdaSelectionCfg) -> str:
    weights = ', '.join(_metric_weight_labels(cfg))
    if cfg.primary_metric is None:
        return f'lambda*: weighted composite within strategy x k (weights: {weights})'
    return (
        f'lambda*: weighted composite with primary guard within strategy x k '
        f'(primary={cfg.primary_metric}, eps={float(cfg.primary_tolerance):g}; weights: {weights})'
    )


def lambda_selection_short_label(cfg: LambdaSelectionCfg) -> str:
    if cfg.primary_metric is None:
        return 'lambda* = weighted composite within strategy x k'
    return 'lambda* = weighted composite with primary guard within strategy x k'


def _metric_weight_labels(cfg: LambdaSelectionCfg) -> list[str]:
    labels: list[str] = []
    for metric in cfg.metrics:
        if not metric.enabled:
            continue
        direction = (
            '' if _higher_is_better(metric.metric, override=metric.higher_is_better) else '-low'
        )
        labels.append(f'{metric.metric}{direction}={float(metric.weight):g}')
    return labels


def _composite_score(row: LambdaSelectionMetricRow, cfg: LambdaSelectionCfg) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for metric in cfg.metrics:
        if not metric.enabled:
            continue
        value = row.get(metric.metric)
        if value is None:
            _handle_missing_metric(cfg, metric.metric)
            continue
        higher_is_better = _higher_is_better(metric.metric, override=metric.higher_is_better)
        oriented_value = value if higher_is_better else 1.0 - value
        weight = float(metric.weight)
        weighted_sum += weight * oriented_value
        total_weight += weight

    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def _select_best_without_primary(
    rows: list[LambdaSelectionMetricRow],
    *,
    cfg: LambdaSelectionCfg,
) -> int | None:
    best: tuple[float, float, float, int] | None = None
    best_idx: int | None = None
    primary_rank = _rank_value(None, higher_is_better=True)

    for idx, row in enumerate(rows):
        score = _composite_score(row, cfg)
        lambda_rank = _lambda_tie_break_rank(row, cfg)
        candidate = (
            score if score is not None else primary_rank,
            primary_rank,
            lambda_rank,
            -idx,
        )
        if best is None or candidate > best:
            best = candidate
            best_idx = idx

    return best_idx


def _select_best_with_primary(
    rows: list[LambdaSelectionMetricRow],
    *,
    cfg: LambdaSelectionCfg,
    primary_metric: str,
) -> int | None:
    primary_higher_is_better = _higher_is_better(
        primary_metric,
        override=cfg.primary_higher_is_better,
    )
    primary_values = [value for row in rows if (value := row.get(primary_metric)) is not None]
    if not primary_values:
        _handle_missing_metric(cfg, primary_metric)

    primary_best = (
        (max(primary_values) if primary_higher_is_better else min(primary_values))
        if primary_values
        else None
    )

    best: tuple[float, float, float, int] | None = None
    best_idx: int | None = None
    for idx, row in enumerate(rows):
        primary_value = row.get(primary_metric)
        if primary_best is not None:
            if primary_value is None:
                continue
            if not _passes_primary_guard(
                value=primary_value,
                best=primary_best,
                tolerance=float(cfg.primary_tolerance),
                higher_is_better=primary_higher_is_better,
            ):
                continue

        score = _composite_score(row, cfg)
        primary_rank = _rank_value(primary_value, higher_is_better=primary_higher_is_better)
        lambda_rank = _lambda_tie_break_rank(row, cfg)
        candidate = (
            score if score is not None else primary_rank,
            primary_rank,
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


def _higher_is_better(metric: str, *, override: bool | None) -> bool:
    if override is not None:
        return override
    spec = METRIC_NAME_TO_FIELD.get(metric)
    return True if spec is None else spec.higher_is_better


def _passes_primary_guard(
    *,
    value: float,
    best: float,
    tolerance: float,
    higher_is_better: bool,
) -> bool:
    if higher_is_better:
        return value >= best - tolerance
    return value <= best + tolerance


def _rank_value(value: float | None, *, higher_is_better: bool) -> float:
    if value is None:
        return float('-inf')
    return value if higher_is_better else -value


def _lambda_tie_break_rank(
    row: LambdaSelectionMetricRow,
    cfg: LambdaSelectionCfg,
) -> float:
    lam = row.get('lam') or 0.0
    return -lam if cfg.tie_break == 'lower_lambda' else lam


def _handle_missing_metric(cfg: LambdaSelectionCfg, metric: str) -> None:
    if cfg.missing_metric_policy == 'error':
        raise ValueError(f'Lambda-selection metric is unavailable: {metric}')
