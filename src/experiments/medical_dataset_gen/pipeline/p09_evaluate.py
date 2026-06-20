"""Evaluate retrieval strategies on the synthetic medical benchmark.

This module exists to score top-k, MMR, and facility-location against the gold
facets and distractors generated earlier in the pipeline. It uses shared
candidate-pool logic, per-query metric aggregation, and redundancy-aware
ranking metrics so the benchmark can expose coverage differences rather than
just nearest-neighbor accuracy.
"""

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
from experiments.medical_dataset_gen.evaluation.lambda_agreement import build_lambda_pair_agreement
from experiments.medical_dataset_gen.evaluation.metrics_answer import (
    empty_answer_reference_texts,
    prepare_answer_rouge_scorer,
)
from experiments.medical_dataset_gen.evaluation.metrics_retrieval import compute_retrieval_metrics
from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    assert_pool_scope_match,
    build_query_to_facet_gold_map,
    compute_retrieval_diagnostics,
    get_candidate_pool_indices,
    run_topn_cosine_retrieval,
    select_indices,
)
from experiments.medical_dataset_gen.schemas.evaluation_schemas import (
    EvaluationResultRow,
    QueryRecord,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


def run_evaluate(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    geometry = read_parquet(paths, 'geometry_stats')
    assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')

    facet_gold = build_query_to_facet_gold_map(qrels)
    gold_by_query = {
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

    # Three results: per-query eval, aggregated eval, and lambda_agreement stats between fac_loc and MMR
    eval_results_df = pl.DataFrame(
        _evaluate_queries(
            cfg,
            paths,
            _get_query_ids_to_evaluate(
                queries=queries,
                facet_gold=facet_gold,
                gold_by_query=gold_by_query,
                pass_map=pass_map,
                only_pass_geometry=cfg.retrieval.only_pass_geometry,
            ),
        )
    )
    aggregated_eval_stats_df = stats_aggregated_results_df(eval_results_df)
    lambda_pair_agreement_df = build_lambda_pair_agreement(
        aggregated_eval_stats_df,
        results_df=eval_results_df,
        kernel_cfg=cfg.evaluation.fac_loc_mmr_comparison_kernels,
    )

    write_parquet(paths, 'evaluation_results', eval_results_df)
    write_parquet(paths, 'evaluation_stats', aggregated_eval_stats_df)
    write_parquet(paths, 'lambda_pair_agreement', lambda_pair_agreement_df)

    print(aggregated_eval_stats_df)
    return eval_results_df


def _evaluate_queries(
    cfg: ExperimentCfg, paths: MedicalDatasetGenPaths, query_ids: list[str]
) -> list[EvaluationResultRow]:
    if not query_ids:
        return []

    worker_count = get_evaluation_worker_count(cfg, len(query_ids))
    chunksize = get_evaluation_chunksize(len(query_ids), worker_count)

    if worker_count == 1:
        init_evaluation_worker(cfg.model_dump(mode='python'), paths.exp_name)
        iterator = map(_evaluate_query, query_ids)
    else:
        print(f'[evaluate] scoring {len(query_ids):,} queries with {worker_count} workers')
        worker_context = mp.get_context('spawn')

        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=worker_context,
            initializer=init_evaluation_worker,
            initargs=(cfg.model_dump(mode='python'), paths.exp_name),
        )
        iterator = executor.map(_evaluate_query, query_ids, chunksize=chunksize)

    rows: list[EvaluationResultRow] = []
    try:
        for batch_rows in tqdm(
            iterator, total=len(query_ids), desc='Evaluating', dynamic_ncols=True
        ):
            rows.extend(batch_rows)
    finally:
        if worker_count != 1:
            executor.shutdown()
    return rows


def _evaluate_query(qid: str) -> list[EvaluationResultRow]:
    worker_state = get_evaluation_worker_state()
    if worker_state is None:
        raise RuntimeError('evaluation worker was not initialized')

    cfg: ExperimentCfg = worker_state['cfg']
    query: QueryRecord | None = worker_state['queries_by_id'].get(qid)
    if query is None:
        return []
    if cfg.retrieval.only_pass_geometry and not bool(worker_state['pass_map'].get(qid, False)):
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
        pool_scope=cfg.retrieval.pool_scope,
        n_chunks=len(chunk_ids),
        chunks_by_source_query=maps['chunks_by_source_query'],
        chunks_by_condition=maps['chunks_by_condition'],
        query_condition_id=query.condition_id,
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

    # Validate k values and select max_k to run the algorithms only once and re-use
    # results for lower values of k, saving some compute
    valid_k_values = [k for k in worker_state['k_values'] if k <= len(candidate_chunk_ids)]
    if not valid_k_values:
        return []
    max_k = valid_k_values[-1]
    topk_full = np.arange(max_k, dtype=np.intp)

    # Optionally prepare the ROUGE scorer
    compute_answer_rouge = cfg.retrieval.compute_answer_rouge
    answer_rouge_scorer = None
    if compute_answer_rouge:
        answer_rouge_scorer = prepare_answer_rouge_scorer(
            query_text=query.query_text,
            candidate_chunk_ids=candidate_chunk_ids,
            chunk_by_id=maps['chunk_by_id'],
            answer_refs=worker_state['answer_refs_by_query'].get(
                qid, empty_answer_reference_texts()
            ),
        )

    eval_result_rows: list[EvaluationResultRow] = []
    for strategy in cfg.retrieval.strategies:
        lam_values: list[None] | list[float] = (
            [None] if strategy == 'top_k' else cfg.retrieval.lambda_values
        )

        for lam in lam_values:
            if strategy == 'top_k':
                selected_indices_list_max_k = topk_full
            else:
                selected_indices_list_max_k = select_indices(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=max_k,
                    lam=lam,
                    mmr_window=cfg.retrieval.mmr_window,
                )

            for k in valid_k_values:
                selected_local = selected_indices_list_max_k[:k]
                selected_chunk_ids = [candidate_chunk_ids[int(i)] for i in selected_local]

                eval_result_rows.append({
                    'query_id': qid,
                    'query_type': query.query_type,
                    'condition_id': query.condition_id,
                    'split': query.split,
                    'strategy': strategy,
                    'k': k,
                    'lam': lam,
                    'pool_scope': cfg.retrieval.pool_scope,
                    'pool_size': len(candidate_chunk_ids),
                    **compute_retrieval_metrics(
                        selected_chunk_ids=selected_chunk_ids,
                        chunk_by_id=maps['chunk_by_id'],
                        query_qrels=worker_state['qrels_by_query_chunk'].get(qid, {}),
                        facet_to_gold=query_facet_gold,
                        all_gold_ids=query_all_gold,
                        dominant_facet_id=query.dominant_facet_id,
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
                        topk_local_indices=topk_full[:k] if strategy != 'top_k' else None,
                    ),
                })
            # end for k in valid_k_values
        # end for lam in lam_values
    # end for strategy in strategies

    return eval_result_rows


def _get_query_ids_to_evaluate(
    *,
    queries: pl.DataFrame,
    facet_gold: dict[str, dict[str, list[str]]],
    gold_by_query: dict[str, set[str]],
    pass_map: dict[str, bool],
    only_pass_geometry: bool,
) -> list[str]:
    query_ids: list[str] = []

    for query in queries.iter_rows(named=True):
        query_row = QueryRecord.model_validate(query)
        qid = query_row.query_id

        if only_pass_geometry and not bool(pass_map.get(qid, False)):
            continue
        if not facet_gold.get(qid) or not gold_by_query.get(qid):
            continue
        query_ids.append(qid)

    return query_ids


def stats_aggregated_results_df(results: pl.DataFrame) -> pl.DataFrame:
    if len(results) == 0:
        return pl.DataFrame()

    agg_polars_exprs: list[pl.Expr] = [
        pl.col('query_id').n_unique().alias('n_queries'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('gold_recall').mean().alias('Recall@k'),
        pl.col('gold_f1').mean().alias('F1@k'),
        pl.col('average_precision_at_k').mean().alias('MAP@k'),
        pl.col('facet_coverage').mean().alias('MeanFacetHitRate@k'),
        pl.col('weighted_facet_coverage').mean().alias('MeanFacetRecall@k'),
        pl.col('facet_mrr_at_k').mean().alias('FacetMRR@k'),
        pl.col('alpha_ndcg').mean().alias('alpha-nDCG@k'),
        pl.col('distractor_rate').mean().alias('DistractorRate'),
        pl.col('near_miss_distractor_rate').mean().alias('NearMissDistractorRate'),
        pl.col('background_outlier_rate').mean().alias('BackgroundOutlierRate'),
        pl.col('dominant_facet_rate').mean().alias('DominantFacetRate'),
        pl.col('redundant_gold_rate').mean().alias('RedundantGoldRate'),
        pl.col('fac_cov_score').mean().alias('fac'),
        pl.col('avg_cos').mean().alias('avg_cos'),
        pl.col('jaccard_vs_topk').mean().alias('jac'),
    ]
    optional_rouge_exprs = [
        ('answer_rouge1_recall', 'AnswerROUGE1Recall@k'),
        ('answer_rouge1_precision', 'AnswerROUGE1Precision@k'),
        ('answer_rouge2_recall', 'AnswerROUGE2Recall@k'),
        ('macro_facet_answer_rouge1_recall', 'MacroFacetAnswerROUGE1Recall@k'),
    ]
    agg_polars_exprs.extend(
        pl.col(source_col).mean().alias(target_col)
        for source_col, target_col in optional_rouge_exprs
        if source_col in results.columns
    )
    stats = (
        results.group_by('strategy', 'lam', 'k').agg(agg_polars_exprs).sort('k', 'strategy', 'lam')
    )

    STATS_DF_ORDERED_COLS = [
        'strategy',
        'lam',
        'k',
        'n_queries',
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
        'DistractorRate',
        'NearMissDistractorRate',
        'BackgroundOutlierRate',
        'DominantFacetRate',
        'RedundantGoldRate',
        'fac',
        'avg_cos',
        'jac',
    ]

    return stats.select([col for col in STATS_DF_ORDERED_COLS if col in stats.columns])


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_evaluate(cfg, paths)
