from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.evaluation.eval_worker_handler import (
    get_evaluation_chunksize,
    get_evaluation_worker_count,
    get_evaluation_worker_state,
    init_evaluation_worker,
)
from experiments.medical_dataset_gen.evaluation.schemas import (
    EvaluationResultRow,
    LightweightQueryRecord,
)
from experiments.medical_dataset_gen.retrieval.metrics_answer import (
    empty_answer_reference_texts,
    prepare_answer_rouge_scorer,
)
from experiments.medical_dataset_gen.retrieval.metrics_retrieval import compute_retrieval_metrics
from experiments.medical_dataset_gen.retrieval.reranker import DENSE_RERANKER_STRATEGY
from experiments.medical_dataset_gen.retrieval.retrieval_utils import (
    compute_retrieval_diagnostics,
    get_candidate_pool_indices,
    run_topn_cosine_retrieval,
    select_indices,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths


def evaluate_queries(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    query_ids: list[str],
) -> list[EvaluationResultRow]:
    if not query_ids:
        return []

    validate_reranker_evaluation_cfg(cfg)
    worker_count = get_evaluation_worker_count(cfg, len(query_ids))
    chunksize = get_evaluation_chunksize(len(query_ids), worker_count)

    if worker_count == 1:
        init_evaluation_worker(cfg, paths.exp_name)
        iterator = map(evaluate_query, query_ids)
    else:
        print(f'[evaluate] scoring {len(query_ids):,} queries with {worker_count} workers')
        worker_context = mp.get_context('spawn')
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=worker_context,
            initializer=init_evaluation_worker,
            initargs=(cfg, paths.exp_name),
        )
        iterator = executor.map(evaluate_query, query_ids, chunksize=chunksize)

    rows: list[EvaluationResultRow] = []
    try:
        for batch_rows in tqdm(
            iterator, total=len(query_ids), desc='Evaluating', dynamic_ncols=True
        ):
            rows.extend(batch_rows)
    finally:
        if worker_count != 1:
            executor.shutdown()  # type: ignore[possibly-undefined]
    return rows


def evaluate_query(qid: str) -> list[EvaluationResultRow]:
    worker_state = get_evaluation_worker_state()
    if worker_state is None:
        raise RuntimeError('evaluation worker was not initialized')

    cfg: ExperimentCfg = worker_state['cfg']
    query: LightweightQueryRecord | None = worker_state['queries_by_id'].get(qid)
    if query is None:
        return []
    query_facet_gold = worker_state['facet_gold'].get(qid)
    query_all_gold = worker_state['gold_by_query'].get(qid)
    if not query_facet_gold or not query_all_gold:
        return []

    maps = worker_state['maps']
    qidx = maps['query_id_to_idx'].get(qid)
    if qidx is None:
        return []

    chunk_ids = worker_state['chunk_ids']
    chunk_vectors = worker_state['chunk_vectors']
    query_vectors = worker_state['query_vectors']

    candidate_idx = get_candidate_pool_indices(
        query_id=qid,
        chunks_by_source_query=maps['chunks_by_source_query'],
    )
    topn_global, topn_sims = run_topn_cosine_retrieval(
        candidate_indices=candidate_idx,
        chunk_vectors=chunk_vectors,
        query_vector=query_vectors[qidx],
        n=cfg.retrieval.candidate_pool_n,
    )
    if len(topn_global) == 0:
        return []

    candidate_vectors = chunk_vectors[topn_global]
    sim_matrix = candidate_vectors @ candidate_vectors.T
    sim_to_query = topn_sims.astype(np.float32)
    candidate_chunk_ids = [str(chunk_ids[int(i)]) for i in topn_global]

    valid_k_values = [k for k in worker_state['k_values'] if k <= len(candidate_chunk_ids)]
    if not valid_k_values:
        return []
    max_k = valid_k_values[-1]
    topk_full = np.arange(max_k, dtype=np.intp)

    answer_rouge_scorer = None
    if cfg.retrieval.compute_answer_rouge:
        answer_rouge_scorer = prepare_answer_rouge_scorer(
            query_text=query.query_text,
            candidate_chunk_ids=candidate_chunk_ids,
            chunk_by_id=maps['chunk_by_id'],
            answer_refs=worker_state['answer_refs_by_query'].get(
                qid, empty_answer_reference_texts()
            ),
        )

    eval_result_rows: list[EvaluationResultRow] = []

    def append_rows_for_selection(
        *,
        strategy: str,
        lam: float | None,
        selected_indices_list_max_k: np.ndarray,
    ) -> None:
        for k in valid_k_values:
            selected_local = selected_indices_list_max_k[:k]
            if len(selected_local) < k:
                continue
            selected_chunk_ids = [candidate_chunk_ids[int(i)] for i in selected_local]

            eval_result_rows.append(
                {
                    'query_id': qid,
                    'evidence_profile_id': query.evidence_profile_id,
                    'pool_id': query.pool_id,
                    'query_type': query.query_type,
                    'template_id': query.template_id,
                    'condition_id': query.condition_id,
                    'cohort_dimension_id': query.cohort_dimension_id,
                    'cohort_contrast_family': query.cohort_contrast_family,
                    'cohort_contrast_id': query.cohort_contrast_id,
                    'primary_axis': query.primary_axis,
                    'secondary_axis': query.secondary_axis,
                    'dominant_primary_facet_id': query.dominant_primary_facet_id,
                    'split': query.split,
                    'strategy': strategy,
                    'k': k,
                    'lam': lam,
                    'reranker_model_name': (
                        cfg.evaluation.reranker.model_name
                        if strategy == DENSE_RERANKER_STRATEGY
                        else None
                    ),
                    'pool_scope': cfg.retrieval.pool_scope,
                    'pool_size': len(candidate_chunk_ids),
                    **compute_retrieval_metrics(
                        selected_chunk_ids=selected_chunk_ids,
                        chunk_by_id=maps['chunk_by_id'],
                        query_qrels=worker_state['qrels_by_query_chunk'].get(qid, {}),
                        facet_to_gold=query_facet_gold,
                        all_gold_ids=query_all_gold,
                        primary_axis=query.primary_axis,
                        dominant_primary_facet_id=query.dominant_primary_facet_id,
                        all_clean_rate_precision_threshold=(
                            cfg.evaluation.all_clean_rate_precision_threshold
                        ),
                    ),
                    **(
                        answer_rouge_scorer.score(selected_chunk_ids)
                        if answer_rouge_scorer is not None
                        else {}
                    ),
                    **compute_retrieval_diagnostics(
                        selected_local,
                        sim_to_query,
                        sim_matrix,
                        topk_local_indices=(topk_full[:k] if strategy != 'top_k' else None),
                    ),
                }
            )

    for strategy in cfg.retrieval.strategies:
        for lam in cfg.retrieval.lambda_values_for_strategy(strategy):
            selected_indices_list_max_k = (
                topk_full
                if strategy == 'top_k'
                else select_indices(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=max_k,
                    lam=lam,
                    mmr_window=cfg.retrieval.mmr_window,
                )
            )
            append_rows_for_selection(
                strategy=strategy,
                lam=lam,
                selected_indices_list_max_k=selected_indices_list_max_k,
            )

    if cfg.evaluation.use_reranker:
        reranker = worker_state['reranker']
        if reranker is None:
            raise RuntimeError('evaluation.use_reranker is true, but no reranker was initialized')

        reranker_pool_n = min(
            len(candidate_chunk_ids),
            int(cfg.evaluation.reranker.candidate_pool_n or len(candidate_chunk_ids)),
        )
        selected_indices_list_max_k = reranker.rank_indices(
            query_text=query.query_text,
            candidate_chunk_ids=candidate_chunk_ids[:reranker_pool_n],
            chunk_by_id=maps['chunk_by_id'],
            top_k=max_k,
        )
        append_rows_for_selection(
            strategy=DENSE_RERANKER_STRATEGY,
            lam=None,
            selected_indices_list_max_k=selected_indices_list_max_k,
        )

    return eval_result_rows


def validate_reranker_evaluation_cfg(cfg: ExperimentCfg) -> None:
    if not cfg.evaluation.use_reranker or cfg.evaluation.reranker.candidate_pool_n is None:
        return

    max_k = max(int(k) for k in cfg.retrieval.k_values)
    if cfg.evaluation.reranker.candidate_pool_n < max_k:
        raise ValueError(
            'evaluation.reranker.candidate_pool_n must be at least the largest retrieval k '
            f'({max_k}) when evaluation.use_reranker is true'
        )


def get_query_ids_to_evaluate(
    *,
    queries: pl.DataFrame,
    facet_gold: dict[str, dict[str, list[str]]],
    gold_by_query: dict[str, set[str]],
) -> list[str]:
    query_ids: list[str] = []
    for query in queries.iter_rows(named=True):
        qid = str(query['query_id'])
        if not facet_gold.get(qid) or not gold_by_query.get(qid):
            continue
        query_ids.append(qid)
    return query_ids
