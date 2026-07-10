from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import polars as pl
from tabulate import tabulate

from experiments.medical_dataset_gen.analysis.analysis_constants import (
    DEFAULT_TABLE_COL_WIDTH,
    INTEGER_TABLE_COLUMNS,
    TABLE_COL_WIDTHS,
    TABLE_HEADERS,
    StrategyName,
)
from experiments.medical_dataset_gen.analysis.models import ExperimentRecord, ScalarItem


def _base_experiment_row(record: ExperimentRecord) -> dict[str, object]:
    metadata = embedding_metadata(record)
    return {
        'Experiment': record.name,
        'ShortExperiment': short_experiment_id(record.name),
        'Distribution': record.distribution_id,
        'ShortDistribution': short_token(record.distribution_id),
        'ExperimentFamily': record.family_id,
        'ExperimentFamilyLabel': record.family_label,
        'RunLabel': record.run_label,
        'IsSubexperiment': record.is_subexperiment,
        'EmbeddingModel': metadata.get('model_name') or record.embedding_model,
        'EmbeddingDimension': metadata.get('dimension'),
        'OnlyPassGeometry': record.only_pass_geometry,
        'QueryScope': _query_scope_label(record.only_pass_geometry),
    }


def embedding_metadata(record: ExperimentRecord) -> dict[str, object]:
    path = record.paths.embeddings_paths('metadata')
    fallback_path = record.experiment_dir / 'embeddings_metadata.json'
    metadata_path = path if path.is_file() else fallback_path
    if not metadata_path.is_file():
        return {}
    try:
        raw = json.loads(metadata_path.read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if _is_jsonish_scalar(value)}


def _mean_min_max_for_columns(df: pl.DataFrame, *, columns: Sequence[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = [
            value
            for value in (_float_or_none(value) for value in df[column].to_list())
            if value is not None
        ]
        if not values:
            continue
        out[f'{column}Mean'] = statistics.fmean(values)
        out[f'{column}Min'] = min(values)
        out[f'{column}Max'] = max(values)
    return out


def _primary_dominance_ratio(summary: Mapping[str, object]) -> float | None:
    dominant = _float_or_none(summary.get('DominantPrimaryGoldCountMean'))
    other_primary = _float_or_none(summary.get('OtherPrimaryGoldCountMean'))
    secondary = _float_or_none(summary.get('SecondaryGoldCountMean'))
    reference_values = [value for value in (other_primary, secondary) if value and value > 0.0]
    if dominant is None or not reference_values:
        return None
    return dominant / statistics.fmean(reference_values)


def _distribution_category(summary: Mapping[str, object]) -> str | None:
    pool_size = _float_or_none(summary.get('PoolSizeMean'))
    if pool_size is None:
        return None
    size_label = 'small' if pool_size <= 80 else 'medium' if pool_size <= 160 else 'large'
    dominance = _primary_dominance_ratio(summary)
    if dominance is None:
        dominance_label = 'unknown-dominance'
    elif dominance < 1.25:
        dominance_label = 'balanced-primary'
    elif dominance < 1.75:
        dominance_label = 'mild-primary-skew'
    else:
        dominance_label = 'strong-primary-skew'
    background = _float_or_none(summary.get('BackgroundOutlierCountMean')) or 0.0
    background_label = 'no-bg' if background == 0.0 else 'bg-outliers'
    niche = _float_or_none(summary.get('NicheGoldCountMean')) or 0.0
    niche_label = 'niche' if niche > 0.0 else 'no-niche'
    near_miss = _float_or_none(summary.get('NearMissDistractorCountMean')) or 0.0
    distractor_label = 'low-distractors' if near_miss <= 12 else 'hard-distractors'
    return '/'.join([size_label, dominance_label, niche_label, background_label, distractor_label])


def _lambda_norm(
    record: ExperimentRecord,
    strategy: StrategyName,
    lam: float | None,
    stats: pl.DataFrame,
) -> float | None:
    if strategy == 'top_k' or lam is None:
        return None
    start: float | None = None
    stop: float | None = None
    if record.cfg is not None:
        grid = (
            record.cfg.retrieval.lambdas_mmr
            if strategy == 'mmr'
            else record.cfg.retrieval.lambdas_fac_loc
        )
        start = float(grid.start)
        stop = float(grid.stop)
    else:
        sub = stats.filter(pl.col('strategy') == strategy).drop_nulls(subset=['lam'])
        if not sub.is_empty():
            start = _float_or_none(sub['lam'].min())
            stop = _float_or_none(sub['lam'].max())
    if start is None or stop is None:
        return None
    denominator = stop - start
    if denominator == 0:
        return 0.0
    return (lam - start) / denominator


def _winner_for_metric(row: Mapping[str, object], metric_label: str) -> str | None:
    values = {
        'top_k': _float_or_none(row.get(f'TopK_{metric_label}')),
        'mmr': _float_or_none(row.get(f'MMR_{metric_label}')),
        'fac_loc': _float_or_none(row.get(f'FacLoc_{metric_label}')),
    }
    present = {key: value for key, value in values.items() if value is not None}
    if not present:
        return None
    return max(present.items(), key=lambda item: item[1])[0]


def _numeric_stats(values: Sequence[float], prefix: str) -> dict[str, object]:
    if not values:
        return {
            f'{prefix}_mean': None,
            f'{prefix}_std': None,
            f'{prefix}_min': None,
            f'{prefix}_max': None,
            f'{prefix}_median': None,
            f'{prefix}_iqr': None,
        }
    sorted_values = sorted(values)
    return {
        f'{prefix}_mean': statistics.fmean(values),
        f'{prefix}_std': statistics.stdev(values) if len(values) > 1 else 0.0,
        f'{prefix}_min': min(values),
        f'{prefix}_max': max(values),
        f'{prefix}_median': statistics.median(values),
        f'{prefix}_iqr': _quantile(sorted_values, 0.75) - _quantile(sorted_values, 0.25),
    }


def _numeric_values(rows: Sequence[Mapping[str, object]], column: str) -> list[float]:
    return [
        value for value in (_float_or_none(row.get(column)) for row in rows) if value is not None
    ]


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _boundary_rate(values: Sequence[float]) -> float | None:
    if not values:
        return None
    boundary_count = sum(value <= 0.02 or value >= 0.98 for value in values)
    return boundary_count / len(values)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _delta_outcome(delta: float | None, *, epsilon: float) -> str | None:
    if delta is None:
        return None
    if delta < -epsilon:
        return 'facloc_worse'
    if abs(delta) <= epsilon:
        return 'tied'
    return 'facloc_better'


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, ScalarItem):
        return _int_or_none(value.item())
    return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        numeric = float(value)
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return None
        return None if math.isnan(numeric) or math.isinf(numeric) else numeric
    if isinstance(value, ScalarItem):
        return _float_or_none(value.item())
    return None


def _json_scalar(value: object) -> object:
    if _is_jsonish_scalar(value):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, ScalarItem):
        item = value.item()
        return item if _is_jsonish_scalar(item) else str(item)
    return str(value)


def _series_mean(series: pl.Series) -> float | None:
    values = [_float_or_none(value) for value in series.to_list()]
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _is_jsonish_scalar(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _strategy_label(strategy: StrategyName) -> str:
    return {'top_k': 'TopK', 'mmr': 'MMR', 'fac_loc': 'FacLoc'}[strategy]


def _query_scope_label(only_pass_geometry: bool | None) -> str:
    if only_pass_geometry is True:
        return 'pass-only'
    if only_pass_geometry is False:
        return 'all-query'
    return 'unknown'


def _title_token(value: str) -> str:
    return ''.join(part.capitalize() for part in value.split('_') if part)


def _short_experiment_label(value: str) -> str:
    if len(value) <= 42:
        return value
    if '/' in value:
        parent, child = value.split('/', 1)
        parent = parent[:24]
        child = child[:17]
        return f'{parent}/{child}'
    return value[:39] + '...'


def short_experiment_id(value: str) -> str:
    return '/'.join(short_token(part) for part in Path(value).parts)


def short_token(value: str) -> str:
    canonical_match = re.match(r'^(?P<token>[A-Z]+_[SML]\d{2})(?:_|$)', value)
    if canonical_match is not None:
        return canonical_match.group('token')
    return value.split('_', 1)[0]


def _short_model_label(value: str) -> str:
    return value.rsplit('/', 1)[-1].replace('Embedding-', '').replace('-cos-v1', '')


def _sorted_rows(
    rows: Sequence[Mapping[str, object]],
    column: str,
    *,
    descending: bool = True,
) -> list[Mapping[str, object]]:
    with_values = [row for row in rows if _float_or_none(row.get(column)) is not None]
    without_values = [row for row in rows if _float_or_none(row.get(column)) is None]
    return (
        sorted(
            with_values,
            key=lambda row: cast(float, _float_or_none(row.get(column))),
            reverse=descending,
        )
        + without_values
    )


def _select_extreme_rows(
    rows: Sequence[Mapping[str, object]],
    column: str,
    max_rows: int,
) -> list[Mapping[str, object]]:
    if len(rows) <= max_rows:
        return _sorted_rows(rows, column, descending=True)
    half = max(1, max_rows // 2)
    best = _sorted_rows(rows, column, descending=True)[:half]
    worst = _sorted_rows(rows, column, descending=False)[: max_rows - half]
    selected: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for row in [*best, *worst]:
        experiment = str(row.get('Experiment'))
        if experiment in seen:
            continue
        seen.add(experiment)
        selected.append(row)
    return _sorted_rows(selected, column, descending=True)


def _section_with_table(
    title: str,
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
    tablefmt: str,
    max_rows: int,
) -> list[str]:
    lines = [f'## {title}', '']
    if not rows:
        lines.extend(['No rows available.', ''])
        return lines
    shown = rows[:max_rows]
    table_rows = [
        [_format_table_value(row.get(column), column=column) for column in columns] for row in shown
    ]
    lines.append(
        tabulate(
            table_rows,
            headers=[_table_header(column) for column in columns],
            tablefmt=tablefmt,
            maxcolwidths=[_table_col_width(column) for column in columns],
            disable_numparse=True,
        )
    )
    if len(rows) > max_rows:
        lines.append('')
        lines.append(f'Showing {max_rows}/{len(rows)} rows. Full data is in the CSV outputs.')
    lines.append('')
    return lines


def _format_table_value(value: object, *, column: str) -> object:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 70 else value[:67] + '...'
    numeric = _float_or_none(value)
    if numeric is not None:
        if column in INTEGER_TABLE_COLUMNS:
            return f'{numeric:.0f}'
        if column.endswith('Pct'):
            return f'{numeric:.1%}'
        if abs(numeric) >= 1000:
            return f'{numeric:.0f}'
        return f'{numeric:.4f}'
    if value is None:
        return ''
    text = str(value)
    return text if len(text) <= 70 else text[:67] + '...'


def _table_header(column: str) -> str:
    return TABLE_HEADERS.get(column, column)


def _table_col_width(column: str) -> int:
    return TABLE_COL_WIDTHS.get(column, DEFAULT_TABLE_COL_WIDTH)


def _bullets(items: Iterable[str]) -> list[str]:
    return [f'- {item}' for item in items]
