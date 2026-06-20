import os

import polars as pl

from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    assert_pool_scope_match,
    build_index_maps,
    build_query_to_facet_gold_map,
    get_qrels_by_query_chunk,
    load_embedding_arrays,
)
from experiments.medical_dataset_gen.schemas.evaluation_schemas import (
    AnswerReferenceTexts,
    EvaluationIndexMaps,
    EvaluationWorkerState,
    GoldAnswerRecord,
    QueryRecord,
)
from experiments.medical_dataset_gen.schemas.retrieval_schemas import (
    RetrievalIndexMaps as RawRetrievalIndexMaps,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet

EVALUATION_WORKER_STATE: EvaluationWorkerState | None = None


def set_evaluation_worker_state(target: EvaluationWorkerState | None) -> None:
    global EVALUATION_WORKER_STATE
    EVALUATION_WORKER_STATE = target


def get_evaluation_worker_state() -> EvaluationWorkerState | None:
    return EVALUATION_WORKER_STATE


def init_evaluation_worker(cfg_dump: dict[str, object], exp_name: str) -> None:
    cfg = ExperimentCfg.model_validate(cfg_dump)
    paths = MedicalDatasetGenPaths(exp_name, result_dir_overrides=cfg.global_.result_dir_overrides)

    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    gold_answers = read_parquet(paths, 'gold_answers')
    qrels = read_parquet(paths, 'qrels')
    geometry = read_parquet(paths, 'geometry_stats')

    assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    raw_maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)
    maps = _build_evaluation_index_maps(raw_maps)

    facet_gold = build_query_to_facet_gold_map(qrels)
    answer_refs_by_query = _answer_refs_by_query(gold_answers)
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

    set_evaluation_worker_state({
        'cfg': cfg,
        'queries_by_id': {
            query_record.query_id: query_record
            for query_record in (
                QueryRecord.model_validate(row) for row in queries.iter_rows(named=True)
            )
        },
        'chunk_vectors': chunk_vectors,
        'query_vectors': query_vectors,
        'chunk_ids': chunk_ids,
        'maps': maps,
        'facet_gold': facet_gold,
        'gold_by_query': query_id_to_gold_chunks,
        'qrels_by_query_chunk': get_qrels_by_query_chunk(qrels),
        'answer_refs_by_query': answer_refs_by_query,
        'pass_map': pass_map,
        'k_values': sorted(set(int(k) for k in cfg.retrieval.k_values)),
    })


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


def _build_evaluation_index_maps(raw_maps: RawRetrievalIndexMaps) -> EvaluationIndexMaps:
    return {
        'query_id_to_idx': raw_maps['query_id_to_idx'],
        'chunk_by_id': raw_maps['chunk_by_id'],
        'chunks_by_source_query': raw_maps['chunks_by_source_query'],
        'chunks_by_condition': raw_maps['chunks_by_condition'],
    }


def _answer_refs_by_query(gold_answers: pl.DataFrame) -> dict[str, AnswerReferenceTexts]:
    refs: dict[str, AnswerReferenceTexts] = {}

    for row in gold_answers.iter_rows(named=True):
        answer = GoldAnswerRecord.model_validate(row)
        facet_references = (
            list(answer.facet_summaries_json.values())
            if answer.facet_summaries_json
            else [fact.summary for fact in answer.answer_facts_json or []]
        )
        refs[answer.query_id] = {
            'answer_text': answer.answer_text,
            'facet_references': [text for text in facet_references if text.strip()],
        }

    return refs
