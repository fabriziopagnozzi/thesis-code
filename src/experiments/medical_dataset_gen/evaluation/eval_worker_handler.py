import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast

import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    build_query_to_facet_gold_map,
    get_qrels_by_query_chunk,
    load_embedding_arrays_mmap_ids,
)
from experiments.medical_dataset_gen.schemas.evaluation_schemas import (
    AnswerReferenceTexts,
    EmbeddingIdArray,
    EvaluationIndexMaps,
    EvaluationWorkerState,
    LightweightChunkRecord,
    LightweightQueryRecord,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    SyntheticMedicalDatasetTableName,
)
from experiments.medical_dataset_gen.utils.io_utils import json_loads

_QUERY_COLUMNS = [
    'query_id',
    'evidence_profile_id',
    'pool_id',
    'query_type',
    'template_id',
    'condition_id',
    'cohort_dimension_id',
    'cohort_contrast_id',
    'primary_axis',
    'secondary_axis',
    'calibrated_primary_facet_id',
    'split',
    'query_text',
]
_CHUNK_COLUMNS_NO_ROUGE = ['chunk_id', 'admission_id']
_CHUNK_COLUMNS_WITH_ROUGE = [*_CHUNK_COLUMNS_NO_ROUGE, 'text']
_MEMBERSHIP_COLUMNS = ['query_id', 'chunk_id']
_QREL_COLUMNS = [
    'query_id',
    'chunk_id',
    'facet_id',
    'cluster_id',
    'cluster_role',
    'axis',
    'is_gold',
]
_GEOMETRY_WORKER_COLUMNS = ['query_id', 'passes_filter']
_GOLD_ANSWER_COLUMNS = [
    'query_id',
    'answer_text',
    'facet_summaries_json',
    'answer_facts_json',
]

type MMapEmbeddingIdArray = NDArray[Any]

_eval_worker_state: EvaluationWorkerState | None = None


def set_evaluation_worker_state(target: EvaluationWorkerState | None) -> None:
    global _eval_worker_state
    _eval_worker_state = target


def get_evaluation_worker_state() -> EvaluationWorkerState | None:
    return _eval_worker_state


def init_evaluation_worker(cfg: ExperimentCfg, exp_name: str) -> None:
    paths = MedicalDatasetGenPaths(exp_name, result_dir_overrides=cfg.global_.result_dir_overrides)

    compute_answer_rouge = cfg.retrieval.compute_answer_rouge
    chunk_columns = _CHUNK_COLUMNS_WITH_ROUGE if compute_answer_rouge else _CHUNK_COLUMNS_NO_ROUGE
    chunk_documents = load_selected_parquet_columns(
        paths,
        'chunk_documents',
        chunk_columns,
        optional_columns=['admission_id'],
    )
    chunk_memberships = load_selected_parquet_columns(
        paths, 'chunk_memberships', _MEMBERSHIP_COLUMNS
    )
    queries = load_selected_parquet_columns(paths, 'queries', _QUERY_COLUMNS)
    qrels = load_selected_parquet_columns(
        paths, 'qrels', _QREL_COLUMNS, optional_columns=['cluster_id']
    )
    geometry = load_selected_parquet_columns(paths, 'geometry_stats', _GEOMETRY_WORKER_COLUMNS)

    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays_mmap_ids(paths)
    maps = _build_evaluation_index_maps(
        chunk_documents=chunk_documents,
        chunk_memberships=chunk_memberships,
        chunk_ids=cast(MMapEmbeddingIdArray, chunk_ids),
        query_ids=cast(MMapEmbeddingIdArray, query_ids),
    )

    facet_gold = build_query_to_facet_gold_map(qrels)
    answer_refs_by_query = (
        _answer_refs_by_query(
            load_selected_parquet_columns(paths, 'gold_answers', _GOLD_ANSWER_COLUMNS)
        )
        if compute_answer_rouge
        else {}
    )
    query_id_to_gold_chunks = {
        qid: {chunk_id for ids in facet_map.values() for chunk_id in ids}
        for qid, facet_map in facet_gold.items()
    }
    pass_map = {
        str(query_id): bool(passes_filter)
        for query_id, passes_filter in zip(
            geometry['query_id'].to_list(),
            geometry['passes_filter'].to_list(),
            strict=True,
        )
    }

    worker_state: EvaluationWorkerState = {
        'cfg': cfg,
        'queries_by_id': build_lightweight_query_map(queries),
        'chunk_vectors': chunk_vectors,
        'query_vectors': query_vectors,
        'chunk_ids': cast(EmbeddingIdArray, chunk_ids),
        'maps': maps,
        'facet_gold': facet_gold,
        'gold_by_query': query_id_to_gold_chunks,
        'qrels_by_query_chunk': get_qrels_by_query_chunk(qrels),
        'answer_refs_by_query': answer_refs_by_query,
        'pass_map': pass_map,
        'k_values': sorted(set(int(k) for k in cfg.retrieval.k_values)),
    }
    set_evaluation_worker_state(worker_state)


def get_evaluation_worker_count(cfg: ExperimentCfg, n_queries: int) -> int:
    requested = cfg.evaluation.workers

    if requested is None:
        raw_env = os.getenv('EVALUATION_WORKERS')
        if raw_env is not None:
            try:
                requested = int(raw_env)
            except ValueError as exc:
                raise ValueError('EVALUATION_WORKERS must be an integer') from exc

    workers = requested if requested is not None else (os.cpu_count() or 1)
    return max(1, min(n_queries, workers))


def get_evaluation_chunksize(n_queries: int, worker_count: int) -> int:
    raw_env = os.getenv('EVALUATION_CHUNKSIZE')
    if raw_env is not None:
        try:
            chunksize = int(raw_env)
        except ValueError as exc:
            raise ValueError('EVALUATION_CHUNKSIZE must be an integer') from exc
        return max(1, chunksize)
    return max(1, min(16, n_queries // max(worker_count * 4, 1)))


def _answer_refs_by_query(gold_answers: pl.DataFrame) -> dict[str, AnswerReferenceTexts]:
    refs: dict[str, AnswerReferenceTexts] = {}

    for row in gold_answers.iter_rows(named=True):
        query_id = str(row['query_id'])
        answer_text = str(row.get('answer_text') or '')
        facet_summaries = cast(
            dict[str, str],
            json_loads(row.get('facet_summaries_json'), default={}),
        )
        answer_facts = cast(
            list[dict[str, object]],
            json_loads(row.get('answer_facts_json'), default=[]),
        )
        facet_references = (
            list(facet_summaries.values())
            if facet_summaries
            else [str(fact.get('summary') or '') for fact in answer_facts]
        )
        refs[query_id] = {
            'answer_text': answer_text,
            'facet_references': [text for text in facet_references if text.strip()],
        }

    return refs


def load_selected_parquet_columns(
    paths: MedicalDatasetGenPaths,
    table: SyntheticMedicalDatasetTableName,
    columns: Sequence[str],
    optional_columns: Sequence[str] = (),
) -> pl.DataFrame:
    scan = pl.scan_parquet(paths.table_path(table))
    schema = scan.collect_schema()
    optional_column_set = set(optional_columns)
    select_exprs: list[pl.Expr] = []
    missing_required_columns: list[str] = []

    for column in columns:
        if column in schema:
            select_exprs.append(pl.col(column))
        elif column in optional_column_set:
            select_exprs.append(pl.lit(None).alias(column))
        else:
            missing_required_columns.append(column)

    if missing_required_columns:
        missing = ', '.join(missing_required_columns)
        raise ValueError(f'{table} is missing required column(s): {missing}')

    return scan.select(select_exprs).collect()


def _build_evaluation_index_maps(
    *,
    chunk_documents: pl.DataFrame,
    chunk_memberships: pl.DataFrame,
    chunk_ids: MMapEmbeddingIdArray,
    query_ids: MMapEmbeddingIdArray,
) -> EvaluationIndexMaps:
    chunk_id_to_idx = {str(chunk_id): idx for idx, chunk_id in enumerate(chunk_ids)}
    query_id_to_idx = {str(query_id): idx for idx, query_id in enumerate(query_ids)}

    chunks_by_source_query: dict[str, list[int]] = defaultdict(list)
    seen_by_query: dict[str, set[int]] = defaultdict(set)
    for query_id, chunk_id in chunk_memberships.iter_rows(named=False):
        chunk_idx = chunk_id_to_idx.get(str(chunk_id))
        if chunk_idx is None:
            continue
        query_id_str = str(query_id)
        if chunk_idx not in seen_by_query[query_id_str]:
            chunks_by_source_query[query_id_str].append(chunk_idx)
            seen_by_query[query_id_str].add(chunk_idx)

    return {
        'query_id_to_idx': query_id_to_idx,
        'chunk_by_id': build_lightweight_chunk_map(chunk_documents),
        'chunks_by_source_query': chunks_by_source_query,
    }


def build_lightweight_query_map(queries: pl.DataFrame) -> dict[str, LightweightQueryRecord]:
    result: dict[str, LightweightQueryRecord] = {}
    for row in queries.iter_rows(named=True):
        query = LightweightQueryRecord(
            query_id=str(row['query_id']),
            evidence_profile_id=str(row['evidence_profile_id']),
            pool_id=str(row['pool_id']),
            query_type=str(row['query_type']),
            template_id=str(row['template_id']),
            condition_id=None if row['condition_id'] is None else str(row['condition_id']),
            cohort_dimension_id=str(row['cohort_dimension_id']),
            cohort_contrast_id=str(row['cohort_contrast_id']),
            primary_axis=str(row['primary_axis']),
            secondary_axis=str(row['secondary_axis']),
            calibrated_primary_facet_id=str(row['calibrated_primary_facet_id']),
            split=str(row['split']),
            query_text=str(row.get('query_text') or ''),
        )
        result[query.query_id] = query
    return result


def build_lightweight_chunk_map(chunk_documents: pl.DataFrame) -> dict[str, LightweightChunkRecord]:
    include_text = 'text' in chunk_documents.columns
    result: dict[str, LightweightChunkRecord] = {}
    for row in chunk_documents.iter_rows(named=True):
        chunk_id = str(row['chunk_id'])
        result[chunk_id] = LightweightChunkRecord(
            admission_id=row.get('admission_id'),
            text=str(row.get('text') or '') if include_text else '',
        )
    return result
