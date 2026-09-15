from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from itertools import combinations

import polars as pl

from experiments.medical_dataset_gen.reports.helpers import base_experiment_row, int_or_none
from experiments.medical_dataset_gen.reports.models import ExperimentRecord

_WORD_RE = re.compile(r'[A-Za-z0-9]+')
_JACCARD_QUERY_LIMIT = 128
_MAX_CHUNKS_PER_FACET_FOR_JACCARD = 2


def synthetic_artifact_diagnostic_rows(
    records: Sequence[ExperimentRecord],
    *,
    warnings: list[str],
) -> list[dict[str, object]]:
    """Compute the synthetic-text diagnostics cited in the thesis conclusions."""
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
                columns=['query_id', 'chunk_id', 'facet_id', 'is_gold'],
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
        rows.append(out)
    return rows


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
        for (_, left_sets), (_, right_sets) in combinations(facets.items(), 2):
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


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
