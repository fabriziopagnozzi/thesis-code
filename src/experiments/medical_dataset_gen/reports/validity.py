from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, cast

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
    select_best_lambda_row,
)
from experiments.medical_dataset_gen.evaluation.statistics import stats_aggregated_results_df
from experiments.medical_dataset_gen.reports.analysis_constants import (
    DIVERSIFYING_STRATEGIES,
    EVALUATION_METRICS,
    StrategyName,
)
from experiments.medical_dataset_gen.reports.helpers import (
    base_experiment_row,
    float_or_none,
    int_or_none,
    json_scalar,
    lambda_norm,
)
from experiments.medical_dataset_gen.reports.models import ExperimentRecord
from experiments.medical_dataset_gen.utils.global_schemas import LambdaSelectionCfg

type LambdaTransferPolicy = Literal['global', 'leave_one_distribution_out']
type GeometryPopulation = Literal['all_generated', 'geometry_eligible', 'geometry_ineligible']

_QUERY_ID = 'query_id'
_SPLIT = 'split'
_STRATEGY = 'strategy'
_LAMBDA = 'lam'
_K = 'k'
_N_QUERIES = 'n_queries'
_VALIDATION_SPLIT = 'validation'
_TEST_SPLIT = 'test'
_WORD_RE = re.compile(r'[A-Za-z0-9]+')
_TEXT_SAMPLE_SIZE = 5_000
_JACCARD_QUERY_LIMIT = 128
_MAX_CHUNKS_PER_FACET_FOR_JACCARD = 2


@dataclass(frozen=True)
class _StatsBundle:
    record: ExperimentRecord
    selection_stats: pl.DataFrame
    report_grid_stats: pl.DataFrame


@dataclass(frozen=True)
class _LambdaChoice:
    lam: float
    selection_metric_value: float
    training_records: int
    training_queries: int


def global_lambda_strategy_rows(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> list[dict[str, object]]:
    """Evaluate each run with one suite-wide lambda per strategy and budget."""
    bundles = _load_stats_bundles(records, warnings=warnings)
    return _transfer_lambda_strategy_rows(
        bundles=bundles,
        policy='global',
        warnings=warnings,
    )


def lodo_lambda_strategy_rows(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> list[dict[str, object]]:
    """Evaluate each run with lambdas selected after holding out its distribution."""
    bundles = _load_stats_bundles(records, warnings=warnings)
    return _transfer_lambda_strategy_rows(
        bundles=bundles,
        policy='leave_one_distribution_out',
        warnings=warnings,
    )


def geometry_population_strategy_rows(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> list[dict[str, object]]:
    """Report all-query, eligible-query, and ineligible-query performance.

    Current evaluation writes every query to evaluation_results.parquet and
    stores the geometry pass flag on each row. Population views are therefore
    derived from each experiment's own result table rather than from separate
    all-query sibling branches.
    """
    rows: list[dict[str, object]] = []
    for record in records:
        results_path = record.paths.table_path('evaluation_results')
        if not results_path.is_file():
            warnings.append(f'{record.name}: evaluation_results missing at {results_path}')
            continue

        try:
            results = _load_population_results(
                results_path,
                geometry_path=record.paths.table_path('geometry_stats'),
            )
        except Exception as exc:
            warnings.append(f'{record.name}: could not load population results ({exc})')
            continue

        for population in _GEOMETRY_POPULATIONS:
            population_results = _filter_geometry_population(results, population)
            if population_results.is_empty():
                warnings.append(f'{record.name}: no rows for geometry population {population}')
                continue
            rows.extend(
                _population_selected_strategy_rows(
                    record=record,
                    population=population,
                    results=population_results,
                    warnings=warnings,
                )
            )
    return rows


def synthetic_artifact_diagnostic_rows(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> list[dict[str, object]]:
    """Compute lightweight diagnostics for templatic or lexical artifacts."""
    representatives = _representative_distribution_records(records)
    rows: list[dict[str, object]] = []
    for record in representatives:
        chunk_path = record.paths.table_path('chunk_documents')
        qrels_path = record.paths.table_path('qrels')
        if not chunk_path.is_file() or not qrels_path.is_file():
            warnings.append(f'{record.name}: missing chunk_documents or qrels for diagnostics')
            continue
        try:
            chunks = pl.read_parquet(chunk_path, columns=['chunk_id', 'text'])
            qrels = pl.read_parquet(
                qrels_path,
                columns=['query_id', 'chunk_id', 'facet_id', 'cluster_role', 'axis', 'is_gold'],
            )
        except Exception as exc:
            warnings.append(f'{record.name}: could not read synthetic diagnostics inputs ({exc})')
            continue

        if chunks.is_empty() or qrels.is_empty():
            warnings.append(f'{record.name}: empty synthetic diagnostics inputs')
            continue

        out = base_experiment_row(record)
        out.update(_duplicate_text_stats(chunks))
        out.update(_lexical_jaccard_stats(chunks=chunks, qrels=qrels))
        out.update(_lexical_classifier_stats(chunks=chunks, qrels=qrels))
        rows.append(out)
    return rows


_GEOMETRY_POPULATIONS: tuple[GeometryPopulation, ...] = (
    'all_generated',
    'geometry_eligible',
    'geometry_ineligible',
)
_POPULATION_LABELS: dict[GeometryPopulation, str] = {
    'all_generated': 'All generated queries',
    'geometry_eligible': 'Geometry-eligible queries',
    'geometry_ineligible': 'Geometry-ineligible queries',
}
_POPULATION_QUERY_SCOPE_LABELS: dict[GeometryPopulation, str] = {
    'all_generated': 'all-query',
    'geometry_eligible': 'geometry-eligible',
    'geometry_ineligible': 'geometry-ineligible',
}
_POPULATION_PASS_FILTER_VALUE: dict[GeometryPopulation, bool | None] = {
    'all_generated': None,
    'geometry_eligible': True,
    'geometry_ineligible': False,
}


def _load_stats_bundles(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> list[_StatsBundle]:
    bundles: list[_StatsBundle] = []
    for record in records:
        selection_path = record.paths.table_path('evaluation_selection_stats')
        report_path = record.paths.table_path('evaluation_report_grid_stats')
        if not selection_path.is_file() or not report_path.is_file():
            warnings.append(
                f'{record.name}: missing validation/test lambda grids for transfer analysis'
            )
            continue
        try:
            selection_stats = pl.read_parquet(selection_path)
            report_grid_stats = pl.read_parquet(report_path)
        except Exception as exc:
            warnings.append(f'{record.name}: could not read lambda grids ({exc})')
            continue
        if selection_stats.is_empty() or report_grid_stats.is_empty():
            warnings.append(f'{record.name}: empty lambda grid stats for transfer analysis')
            continue
        bundles.append(
            _StatsBundle(
                record=record,
                selection_stats=selection_stats,
                report_grid_stats=report_grid_stats,
            )
        )
    return bundles


def _transfer_lambda_strategy_rows(
    *,
    bundles: Sequence[_StatsBundle],
    policy: LambdaTransferPolicy,
    warnings: list[str],
) -> list[dict[str, object]]:
    compatible_bundles = _transfer_compatible_bundles(bundles, warnings=warnings)
    rows: list[dict[str, object]] = []
    global_choices = (
        _lambda_choices_by_strategy_k(compatible_bundles, warnings=warnings)
        if policy == 'global'
        else {}
    )
    for bundle in compatible_bundles:
        choices = (
            global_choices
            if policy == 'global'
            else _lambda_choices_by_strategy_k(
                [
                    other
                    for other in compatible_bundles
                    if other.record.distribution_id != bundle.record.distribution_id
                ],
                warnings=warnings,
            )
        )
        rows.extend(
            _report_rows_with_transfer_choices(
                bundle=bundle,
                choices=choices,
                policy=policy,
                warnings=warnings,
            )
        )
    return rows


def _transfer_compatible_bundles(
    bundles: Sequence[_StatsBundle],
    *,
    warnings: list[str],
) -> list[_StatsBundle]:
    if not bundles:
        return []

    signatures = [_lambda_grid_signature(bundle.report_grid_stats) for bundle in bundles]
    signature_counts = Counter(signatures)
    modal_signature, _count = signature_counts.most_common(1)[0]
    compatible: list[_StatsBundle] = []
    for bundle, signature in zip(bundles, signatures, strict=True):
        if signature == modal_signature:
            compatible.append(bundle)
        else:
            warnings.append(
                f'{bundle.record.name}: excluded from lambda-transfer summaries because '
                'its lambda grid differs from the modal suite grid'
            )
    return compatible


def _lambda_grid_signature(
    stats: pl.DataFrame,
) -> tuple[tuple[StrategyName, tuple[float, ...]], ...]:
    signature: list[tuple[StrategyName, tuple[float, ...]]] = []
    for strategy in DIVERSIFYING_STRATEGIES:
        lambdas = tuple(
            sorted(
                {
                    round(float(value), 6)
                    for value in stats.filter(pl.col(_STRATEGY) == strategy)[_LAMBDA]
                    .drop_nulls()
                    .to_list()
                }
            )
        )
        signature.append((strategy, lambdas))
    return tuple(signature)


def _lambda_choices_by_strategy_k(
    bundles: Sequence[_StatsBundle],
    *,
    warnings: list[str],
) -> dict[tuple[StrategyName, int], _LambdaChoice]:
    choices: dict[tuple[StrategyName, int], _LambdaChoice] = {}
    if not bundles:
        return choices

    k_values = sorted(
        {
            int(k)
            for bundle in bundles
            for k in bundle.selection_stats[_K].drop_nulls().unique().to_list()
        }
    )
    for strategy in DIVERSIFYING_STRATEGIES:
        for k in k_values:
            choice = _select_weighted_lambda_choice(
                bundles=bundles,
                strategy=strategy,
                k=k,
                warnings=warnings,
            )
            if choice is not None:
                choices[(strategy, k)] = choice
    return choices


def _select_weighted_lambda_choice(
    *,
    bundles: Sequence[_StatsBundle],
    strategy: StrategyName,
    k: int,
    warnings: list[str],
) -> _LambdaChoice | None:
    frames: list[pl.DataFrame] = []
    for bundle in bundles:
        stats = bundle.selection_stats
        required = {_STRATEGY, _K, _LAMBDA, _N_QUERIES, LAMBDA_SELECTION_MAXIMIZING_METRIC}
        if not required <= set(stats.columns):
            warnings.append(f'{bundle.record.name}: incomplete lambda grid for transfer selection')
            continue
        sub = stats.filter((pl.col(_STRATEGY) == strategy) & (pl.col(_K) == k)).select(
            pl.col(_LAMBDA),
            pl.col(_N_QUERIES).cast(pl.Float64),
            pl.col(LAMBDA_SELECTION_MAXIMIZING_METRIC).cast(pl.Float64),
        )
        if not sub.is_empty():
            frames.append(sub.with_columns(pl.lit(bundle.record.name).alias('record_name')))
    if not frames:
        return None

    combined = pl.concat(frames)
    grouped = (
        combined.group_by(_LAMBDA)
        .agg(
            (
                (pl.col(LAMBDA_SELECTION_MAXIMIZING_METRIC) * pl.col(_N_QUERIES)).sum()
                / pl.col(_N_QUERIES).sum()
            ).alias('weighted_metric'),
            pl.col(_N_QUERIES).sum().alias('training_queries'),
            pl.col('record_name').n_unique().alias('training_records'),
        )
        .drop_nulls(subset=[_LAMBDA, 'weighted_metric'])
    )
    if grouped.is_empty():
        return None

    tie_break = (
        bundles[0].record.cfg.evaluation.lambda_selection.tie_break
        if bundles[0].record.cfg is not None
        else 'lower_lambda'
    )
    descending = [True, tie_break == 'higher_lambda']
    selected = grouped.sort(['weighted_metric', _LAMBDA], descending=descending).row(0, named=True)
    return _LambdaChoice(
        lam=float(selected[_LAMBDA]),
        selection_metric_value=float(selected['weighted_metric']),
        training_records=int(selected['training_records']),
        training_queries=int(selected['training_queries']),
    )


def _report_rows_with_transfer_choices(
    *,
    bundle: _StatsBundle,
    choices: Mapping[tuple[StrategyName, int], _LambdaChoice],
    policy: LambdaTransferPolicy,
    warnings: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stats = bundle.report_grid_stats
    k_values = sorted(int(k) for k in stats[_K].drop_nulls().unique().to_list())
    for k in k_values:
        topk_row = stats.filter((pl.col(_STRATEGY) == 'top_k') & (pl.col(_K) == k)).head(1)
        if not topk_row.is_empty():
            rows.append(
                _strategy_row_from_stats(
                    record=bundle.record,
                    stats=stats,
                    row=topk_row.row(0, named=True),
                    selection_source=f'{policy}_lambda',
                    lambda_policy=policy,
                    lambda_choice=None,
                )
            )

        for strategy in DIVERSIFYING_STRATEGIES:
            choice = choices.get((strategy, k))
            if choice is None:
                warnings.append(
                    f'{bundle.record.name}: no {policy} lambda choice for {strategy}@{k}'
                )
                continue
            report_row = _report_row_for_lambda(stats, strategy=strategy, k=k, lam=choice.lam)
            if report_row is None:
                warnings.append(
                    f'{bundle.record.name}: {strategy}@{k} lacks selected lambda {choice.lam}'
                )
                continue
            rows.append(
                _strategy_row_from_stats(
                    record=bundle.record,
                    stats=stats,
                    row=report_row,
                    selection_source=f'{policy}_lambda',
                    lambda_policy=policy,
                    lambda_choice=choice,
                )
            )
    return rows


def _report_row_for_lambda(
    stats: pl.DataFrame,
    *,
    strategy: StrategyName,
    k: int,
    lam: float,
) -> dict[str, object] | None:
    sub = stats.filter(
        (pl.col(_STRATEGY) == strategy)
        & (pl.col(_K) == k)
        & ((pl.col(_LAMBDA) - lam).abs() <= 1e-9)
    )
    if sub.is_empty():
        return None
    return cast(dict[str, object], sub.row(0, named=True))


def _strategy_row_from_stats(
    *,
    record: ExperimentRecord,
    stats: pl.DataFrame,
    row: Mapping[str, object],
    selection_source: str,
    lambda_policy: str,
    lambda_choice: _LambdaChoice | None,
) -> dict[str, object]:
    strategy = cast(StrategyName, row.get(_STRATEGY))
    lam = float_or_none(row.get(_LAMBDA))
    out = base_experiment_row(record)
    out.update(
        {
            'SelectionSource': selection_source,
            'LambdaPolicy': lambda_policy,
            'strategy': strategy,
            'k': int_or_none(row.get(_K)),
            'lam': lam,
            'lambda_norm': lambda_norm(record, strategy, lam, stats),
            'LambdaSelectionMetric': LAMBDA_SELECTION_MAXIMIZING_METRIC
            if lambda_choice is not None
            else None,
            'LambdaSelectionValidationValue': lambda_choice.selection_metric_value
            if lambda_choice is not None
            else None,
            'LambdaSelectionTrainingRecords': lambda_choice.training_records
            if lambda_choice is not None
            else None,
            'LambdaSelectionTrainingQueries': lambda_choice.training_queries
            if lambda_choice is not None
            else None,
        }
    )
    _copy_metric_values(out, row)
    return out


def _load_population_results(results_path: Path, *, geometry_path: Path) -> pl.DataFrame:
    needed_columns = [
        _QUERY_ID,
        _SPLIT,
        _STRATEGY,
        _K,
        _LAMBDA,
        'gold_precision',
        'gold_recall',
        'gold_f1',
        'average_precision_at_k',
        'facet_coverage',
        'weighted_facet_coverage',
        'facet_coverage_purity',
        'all_facet_coverage',
        'all_facet_clean',
        'facet_mrr_at_k',
        'alpha_ndcg',
        'distractor_rate',
        'near_miss_distractor_rate',
        'background_outlier_rate',
        'primary_axis_rate',
        'dominant_facet_rate',
        'redundant_gold_rate',
        'fac_cov_score',
        'avg_cos',
        'jaccard_vs_topk',
    ]
    available = set(pl.scan_parquet(results_path).collect_schema().names())
    pass_column = _population_pass_column(available)
    selected_columns = [column for column in needed_columns if column in available]
    if pass_column is not None:
        selected_columns.append(pass_column)
    results = pl.scan_parquet(results_path).select(selected_columns)
    if pass_column is None:
        collected = results.collect()
        if geometry_path.is_file() and _QUERY_ID in collected.columns:
            geometry = pl.read_parquet(geometry_path, columns=[_QUERY_ID, 'passes_filter'])
            return collected.join(
                geometry.select(
                    _QUERY_ID,
                    pl.col('passes_filter').fill_null(False).alias('passes_geometry_filter'),
                ),
                on=_QUERY_ID,
                how='left',
            ).with_columns(pl.col('passes_geometry_filter').fill_null(False))
        return collected.with_columns(pl.lit(True).alias('passes_geometry_filter'))
    if pass_column == 'passes_geometry_filter':
        return results.with_columns(pl.col(pass_column).fill_null(False)).collect()
    return (
        results.rename({pass_column: 'passes_geometry_filter'})
        .with_columns(pl.col('passes_geometry_filter').fill_null(False))
        .collect()
    )


def _population_pass_column(columns: set[str]) -> str | None:
    for column in ('passes_geometry_filter', 'passes_filter'):
        if column in columns:
            return column
    return None


def _filter_geometry_population(
    results: pl.DataFrame,
    population: GeometryPopulation,
) -> pl.DataFrame:
    pass_value = _POPULATION_PASS_FILTER_VALUE[population]
    if pass_value is None:
        return results
    return results.filter(pl.col('passes_geometry_filter') == pass_value)


def _population_selected_strategy_rows(
    *,
    record: ExperimentRecord,
    population: GeometryPopulation,
    results: pl.DataFrame,
    warnings: list[str],
) -> list[dict[str, object]]:
    selection_results = results.filter(pl.col(_SPLIT) == _VALIDATION_SPLIT)
    report_results = results.filter(pl.col(_SPLIT) == _TEST_SPLIT)
    if selection_results.is_empty() or report_results.is_empty():
        warnings.append(f'{record.name}: missing split rows for population {population}')
        return []

    selection_stats = stats_aggregated_results_df(selection_results)
    report_grid_stats = stats_aggregated_results_df(report_results)
    if selection_stats.is_empty() or report_grid_stats.is_empty():
        warnings.append(f'{record.name}: empty stats for population {population}')
        return []

    rows: list[dict[str, object]] = []
    k_values = sorted(int(k) for k in report_grid_stats[_K].drop_nulls().unique().to_list())
    for k in k_values:
        topk = report_grid_stats.filter((pl.col(_STRATEGY) == 'top_k') & (pl.col(_K) == k)).head(1)
        if not topk.is_empty():
            rows.append(
                _population_row_from_stats(
                    record=record,
                    population=population,
                    stats=report_grid_stats,
                    row=topk.row(0, named=True),
                    selected_validation_value=None,
                )
            )
        for strategy in DIVERSIFYING_STRATEGIES:
            selected = select_best_lambda_row(
                selection_stats,
                strategy=strategy,
                k=k,
                cfg=record.cfg.evaluation.lambda_selection
                if record.cfg is not None
                else LambdaSelectionCfg(),
            )
            if selected is None:
                warnings.append(f'{record.name}: no selected {strategy}@{k} for {population}')
                continue
            lam = selected.get(_LAMBDA)
            if lam is None:
                continue
            report_row = _report_row_for_lambda(
                report_grid_stats,
                strategy=strategy,
                k=k,
                lam=float(lam),
            )
            if report_row is None:
                warnings.append(f'{record.name}: no test {strategy}@{k} lambda {lam}')
                continue
            rows.append(
                _population_row_from_stats(
                    record=record,
                    population=population,
                    stats=report_grid_stats,
                    row=report_row,
                    selected_validation_value=float_or_none(
                        selected.get(LAMBDA_SELECTION_MAXIMIZING_METRIC)
                    ),
                )
            )
    return rows


def _population_row_from_stats(
    *,
    record: ExperimentRecord,
    population: GeometryPopulation,
    stats: pl.DataFrame,
    row: Mapping[str, object],
    selected_validation_value: float | None,
) -> dict[str, object]:
    strategy = cast(StrategyName, row.get(_STRATEGY))
    lam = float_or_none(row.get(_LAMBDA))
    out = base_experiment_row(record)
    out.update(
        {
            'OnlyPassGeometry': population == 'geometry_eligible',
            'QueryScope': _POPULATION_QUERY_SCOPE_LABELS[population],
            'GeometrySourceExperiment': record.name,
            'GeometryPopulation': population,
            'GeometryPopulationLabel': _POPULATION_LABELS[population],
            'PopulationPassFilterValue': _POPULATION_PASS_FILTER_VALUE[population],
            'SelectionSource': 'population_validation',
            'strategy': strategy,
            'k': int_or_none(row.get(_K)),
            'lam': lam,
            'lambda_norm': lambda_norm(record, strategy, lam, stats),
            'LambdaSelectionMetric': LAMBDA_SELECTION_MAXIMIZING_METRIC
            if strategy != 'top_k'
            else None,
            'LambdaSelectionValidationValue': selected_validation_value,
        }
    )
    _copy_metric_values(out, row)
    return out


def _copy_metric_values(out: dict[str, object], row: Mapping[str, object]) -> None:
    for metric in EVALUATION_METRICS:
        if metric in row:
            out[metric] = json_scalar(row[metric])


def _representative_distribution_records(
    records: Sequence[ExperimentRecord],
) -> list[ExperimentRecord]:
    candidates = [
        record
        for record in records
        if record.paths.table_path('chunk_documents').is_file()
        and record.paths.table_path('qrels').is_file()
    ]
    preferred_run_order = {
        'qwen3_06': 0,
        'bge_m3': 1,
    }
    by_distribution: dict[str, ExperimentRecord] = {}
    for record in sorted(
        candidates,
        key=lambda item: (
            item.distribution_id,
            preferred_run_order.get(item.run_label, 99),
            item.run_label,
        ),
    ):
        by_distribution.setdefault(record.distribution_id, record)
    return [by_distribution[key] for key in sorted(by_distribution)]


def _duplicate_text_stats(chunks: pl.DataFrame) -> dict[str, object]:
    normalized = chunks.select(
        pl.col('chunk_id'),
        pl.col('text')
        .fill_null('')
        .str.to_lowercase()
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
        .alias('normalized_text'),
        pl.col('text').fill_null('').str.split(by=' ').list.len().alias('word_count'),
    ).filter(pl.col('normalized_text') != '')
    n_chunks = normalized.height
    n_unique_texts = normalized['normalized_text'].n_unique() if n_chunks else 0
    duplicate_chunks = n_chunks - n_unique_texts
    word_counts = [
        value
        for value in (int_or_none(value) for value in normalized['word_count'].to_list())
        if value is not None
    ]
    return {
        'DiagnosticChunks': n_chunks,
        'UniqueNormalizedTexts': n_unique_texts,
        'ExactDuplicateChunks': duplicate_chunks,
        'ExactDuplicateChunkRate': duplicate_chunks / n_chunks if n_chunks else None,
        'ChunkWordCountMean': statistics.fmean(word_counts) if word_counts else None,
        'ChunkWordCountMedian': statistics.median(word_counts) if word_counts else None,
    }


def _lexical_jaccard_stats(*, chunks: pl.DataFrame, qrels: pl.DataFrame) -> dict[str, object]:
    gold = qrels.filter(pl.col('is_gold').fill_null(False))
    query_ids = sorted(str(value) for value in gold['query_id'].drop_nulls().unique().to_list())
    query_ids = query_ids[:_JACCARD_QUERY_LIMIT]
    if not query_ids:
        return _empty_jaccard_stats()

    sampled_gold = gold.filter(pl.col('query_id').is_in(query_ids))
    needed_chunk_ids = set(str(value) for value in sampled_gold['chunk_id'].drop_nulls().to_list())
    docs_by_id = {
        str(chunk_id): _word_set(str(text or ''))
        for chunk_id, text in chunks.filter(pl.col('chunk_id').is_in(needed_chunk_ids))
        .select('chunk_id', 'text')
        .iter_rows(named=False)
    }
    by_query_facet: dict[str, dict[str, list[set[str]]]] = {}
    for row in sampled_gold.iter_rows(named=True):
        query_id = str(row['query_id'])
        facet_id = str(row['facet_id'])
        chunk_words = docs_by_id.get(str(row['chunk_id']))
        if not chunk_words:
            continue
        facet_sets = by_query_facet.setdefault(query_id, {}).setdefault(facet_id, [])
        if len(facet_sets) < _MAX_CHUNKS_PER_FACET_FOR_JACCARD:
            facet_sets.append(chunk_words)

    within: list[float] = []
    between: list[float] = []
    for facets in by_query_facet.values():
        for chunk_sets in facets.values():
            within.extend(_jaccard(left, right) for left, right in combinations(chunk_sets, 2))
        facet_items = list(facets.items())
        for (left_facet, left_sets), (right_facet, right_sets) in combinations(facet_items, 2):
            if left_facet == right_facet:
                continue
            for left_set in left_sets:
                for right_set in right_sets:
                    between.append(_jaccard(left_set, right_set))

    within_mean = statistics.fmean(within) if within else None
    between_mean = statistics.fmean(between) if between else None
    return {
        'LexicalJaccardQueries': len(by_query_facet),
        'WithinFacetJaccardMean': within_mean,
        'BetweenFacetJaccardMean': between_mean,
        'WithinMinusBetweenJaccard': (
            within_mean - between_mean
            if within_mean is not None and between_mean is not None
            else None
        ),
    }


def _empty_jaccard_stats() -> dict[str, object]:
    return {
        'LexicalJaccardQueries': 0,
        'WithinFacetJaccardMean': None,
        'BetweenFacetJaccardMean': None,
        'WithinMinusBetweenJaccard': None,
    }


def _lexical_classifier_stats(*, chunks: pl.DataFrame, qrels: pl.DataFrame) -> dict[str, object]:
    gold_labels = (
        qrels.filter(pl.col('is_gold').fill_null(False))
        .select('chunk_id', 'axis', 'cluster_role')
        .drop_nulls(subset=['chunk_id', 'axis', 'cluster_role'])
        .unique(subset=['chunk_id'])
    )
    labeled = gold_labels.join(chunks.select('chunk_id', 'text'), on='chunk_id', how='inner')
    if labeled.height > _TEXT_SAMPLE_SIZE:
        labeled = labeled.sample(n=_TEXT_SAMPLE_SIZE, seed=42, shuffle=True)

    texts = [str(text or '') for text in labeled['text'].to_list()]
    axis_labels = [str(label) for label in labeled['axis'].to_list()]
    role_labels = [str(label) for label in labeled['cluster_role'].to_list()]
    return {
        'LexicalClassifierRows': labeled.height,
        'AxisLabelCount': len(set(axis_labels)),
        'ClusterRoleLabelCount': len(set(role_labels)),
        'AxisBowAccuracy': _bag_of_words_accuracy(texts, axis_labels),
        'ClusterRoleBowAccuracy': _bag_of_words_accuracy(texts, role_labels),
    }


def _bag_of_words_accuracy(texts: Sequence[str], labels: Sequence[str]) -> float | None:
    if len(texts) < 100 or len(set(labels)) < 2:
        return None
    label_counts = Counter(labels)
    stratify = labels if min(label_counts.values()) >= 2 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            list(texts),
            list(labels),
            test_size=0.30,
            random_state=42,
            stratify=stratify,
        )
        model = make_pipeline(
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=20_000,
            ),
            MultinomialNB(),
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
    except Exception:
        return None
    return float(accuracy_score(y_test, predictions))


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
