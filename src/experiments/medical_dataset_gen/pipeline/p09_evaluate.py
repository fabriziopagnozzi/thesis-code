"""Evaluate retrieval strategies on the synthetic medical benchmark.

This module exists to score top-k, MMR, and facility-location against the gold
facets and distractors generated earlier in the pipeline. It uses shared
candidate-pool logic, per-query metric aggregation, and redundancy-aware
ranking metrics so the benchmark can expose coverage differences rather than
just nearest-neighbor accuracy.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, cast, get_args

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
from experiments.medical_dataset_gen.global_config import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
    setup_logging,
)
from experiments.medical_dataset_gen.schemas.evaluation_schemas import (
    EvaluationResultRow,
    QueryRecord,
)
from experiments.medical_dataset_gen.utils.io_utils import (
    read_parquet,
    read_parquet_if_exists_else_empty_df,
    write_parquet,
)

type EvaluationStep = Literal[
    'evaluation_results',
    'evaluation_stats',
    'evaluation_slice_stats',
    'lambda_agreement',
]
EVALUATION_STEP_NAMES = set[EvaluationStep](get_args(EvaluationStep.__value__))
EVALUATION_STEP_ALIASES: dict[str, EvaluationStep] = {
    'lambda_pair_agreement': 'lambda_agreement',
}


def run_evaluate(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    selected_steps: set[EvaluationStep] | None = None,
) -> pl.DataFrame:
    requested_steps = (
        selected_steps if selected_steps is not None else set[EvaluationStep](EVALUATION_STEP_NAMES)
    )
    eval_results_df: pl.DataFrame | None = None
    aggregated_eval_stats_df: pl.DataFrame | None = None

    if 'evaluation_results' in requested_steps:
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
        if not eval_results_df.is_empty():
            geometry_dimensions = geometry.select(
                'query_id',
                'calibration_warning',
                'n_topk_retrieved_facets',
            )
            eval_results_df = eval_results_df.join(
                geometry_dimensions,
                on='query_id',
                how='left',
                validate='m:1',
            )
        write_parquet(paths, 'evaluation_results', eval_results_df)

    if 'evaluation_stats' in requested_steps:
        eval_results_df = _ensure_eval_results_loaded(
            cfg=cfg,
            paths=paths,
            eval_results_df=eval_results_df,
            requesting_step='evaluation_stats',
        )
        aggregated_eval_stats_df = stats_aggregated_results_df(eval_results_df)
        write_parquet(paths, 'evaluation_stats', aggregated_eval_stats_df)

    if 'evaluation_slice_stats' in requested_steps:
        eval_results_df = _ensure_eval_results_loaded(
            cfg=cfg,
            paths=paths,
            eval_results_df=eval_results_df,
            requesting_step='evaluation_slice_stats',
        )
        sliced_eval_stats_df = stats_sliced_results_df(eval_results_df)
        write_parquet(paths, 'evaluation_slice_stats', sliced_eval_stats_df)

    if 'lambda_agreement' in requested_steps:
        eval_results_df = _ensure_eval_results_loaded(
            cfg=cfg,
            paths=paths,
            eval_results_df=eval_results_df,
            requesting_step='lambda_agreement',
        )
        if aggregated_eval_stats_df is None:
            aggregated_eval_stats_df = _load_eval_stats_or_compute_from_results(
                cfg=cfg,
                paths=paths,
                eval_results_df=eval_results_df,
            )
        lambda_pair_agreement_df = build_lambda_pair_agreement(
            aggregated_eval_stats_df,
            results_df=eval_results_df,
            kernel_cfg=cfg.evaluation.fac_loc_mmr_comparison_kernels,
        )
        write_parquet(paths, 'lambda_pair_agreement', lambda_pair_agreement_df)

    if aggregated_eval_stats_df is not None:
        print(aggregated_eval_stats_df)

    return eval_results_df if eval_results_df is not None else pl.DataFrame()


def stats_sliced_results_df(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame()
    slice_columns = [
        'condition_id',
        'cohort_dimension_id',
        'cohort_contrast_id',
        'primary_axis',
        'secondary_axis',
        'template_id',
        'calibration_warning',
        'n_topk_retrieved_facets',
    ]
    agg_exprs: list[pl.Expr] = [
        pl.col('query_id').n_unique().alias('n_queries'),
        pl.col('facet_coverage').mean().alias('MeanFacetHitRate@k'),
        pl.col('weighted_facet_coverage').mean().alias('MeanFacetRecall@k'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('gold_recall').mean().alias('Recall@k'),
        pl.col('gold_f1').mean().alias('F1@k'),
        pl.col('same_condition_wrong_axis_rate').mean().alias('SameConditionWrongAxisRate'),
        pl.col('primary_axis_rate').mean().alias('PrimaryAxisRate'),
    ]
    optional_rouge_exprs = [
        ('answer_rouge1_recall', 'AnswerROUGE1Recall@k'),
        ('answer_rouge1_precision', 'AnswerROUGE1Precision@k'),
        ('answer_rouge1_f1', 'AnswerROUGE1F1@k'),
        ('answer_rouge2_recall', 'AnswerROUGE2Recall@k'),
    ]
    agg_exprs.extend(
        pl.col(source_col).mean().alias(target_col)
        for source_col, target_col in optional_rouge_exprs
        if source_col in results.columns
    )
    return (
        results
        .group_by(*slice_columns, 'strategy', 'lam', 'k')
        .agg(agg_exprs)
        .sort(*slice_columns, 'k', 'strategy', 'lam')
    )


def _ensure_eval_results_loaded(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    eval_results_df: pl.DataFrame | None,
    requesting_step: EvaluationStep,
) -> pl.DataFrame:
    if eval_results_df is not None:
        return eval_results_df

    loaded_df = _read_required_table(paths, 'evaluation_results', requesting_step)
    assert_pool_scope_match(loaded_df, cfg.retrieval.pool_scope, table_name='evaluation_results')
    return loaded_df


def _load_eval_stats_or_compute_from_results(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    eval_results_df: pl.DataFrame,
) -> pl.DataFrame:
    stats_df = read_parquet_if_exists_else_empty_df(paths, 'evaluation_stats')
    if stats_df.is_empty():
        return stats_aggregated_results_df(eval_results_df)

    assert_pool_scope_match(stats_df, cfg.retrieval.pool_scope, table_name='evaluation_stats')
    return stats_df


def _read_required_table(
    paths: MedicalDatasetGenPaths,
    table_name: Literal['evaluation_results'],
    requesting_step: EvaluationStep,
) -> pl.DataFrame:
    table_path = paths.table_path(table_name)
    if not table_path.exists():
        raise FileNotFoundError(
            f'Step "{requesting_step}" requires {table_name}. Run p09_evaluate.py with '
            f'--steps evaluation_results first, or omit --steps to run the full stage.'
        )
    return read_parquet(paths, table_name)


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
            executor.shutdown()  # type: ignore
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
                    'evidence_profile_id': query.evidence_profile_id,
                    'pool_id': query.pool_id,
                    'query_type': query.query_type,
                    'template_id': query.template_id,
                    'condition_id': query.condition_id,
                    'cohort_dimension_id': query.cohort_dimension_id,
                    'cohort_contrast_id': query.cohort_contrast_id,
                    'primary_axis': query.primary_axis,
                    'secondary_axis': query.secondary_axis,
                    'calibrated_primary_facet_id': query.calibrated_primary_facet_id,
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
                        primary_axis=query.primary_axis,
                        calibrated_primary_facet_id=query.calibrated_primary_facet_id,
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
        pl.col('same_condition_wrong_axis_rate').mean().alias('SameConditionWrongAxisRate'),
        pl.col('primary_axis_rate').mean().alias('PrimaryAxisRate'),
        pl.col('fac_cov_score').mean().alias('fac'),
        pl.col('avg_cos').mean().alias('avg_cos'),
        pl.col('jaccard_vs_topk').mean().alias('jac'),
    ]
    optional_rouge_exprs = [
        ('answer_rouge1_recall', 'AnswerROUGE1Recall@k'),
        ('answer_rouge1_precision', 'AnswerROUGE1Precision@k'),
        ('answer_rouge1_f1', 'AnswerROUGE1F1@k'),
        ('answer_rouge2_recall', 'AnswerROUGE2Recall@k'),
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
        'AnswerROUGE1F1@k',
        'AnswerROUGE2Recall@k',
        'DistractorRate',
        'NearMissDistractorRate',
        'BackgroundOutlierRate',
        'SameConditionWrongAxisRate',
        'PrimaryAxisRate',
        'fac',
        'avg_cos',
        'jac',
    ]

    return stats.select([col for col in STATS_DF_ORDERED_COLS if col in stats.columns])


def parse_evaluation_steps(raw_value: str | None) -> set[EvaluationStep] | None:
    if raw_value is None:
        return None

    raw_steps = {part.strip() for part in raw_value.split(',') if part.strip()}
    if not raw_steps:
        raise ValueError('--steps was provided but no evaluation step names were specified')

    normalized_steps = {
        EVALUATION_STEP_ALIASES.get(step_name, cast(EvaluationStep | str, step_name))
        for step_name in raw_steps
    }
    unknown_steps = sorted(step for step in normalized_steps if step not in EVALUATION_STEP_NAMES)
    if unknown_steps:
        available = ', '.join(sorted(EVALUATION_STEP_NAMES))
        unknown = ', '.join(unknown_steps)
        raise ValueError(f'Unknown evaluation step name(s): {unknown}. Available steps: {available}')

    return cast(set[EvaluationStep], normalized_steps)


def parse_evaluate_cli_args(argv: list[str]) -> tuple[ExperimentCfg, set[EvaluationStep] | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--steps',
        type=str,
        help='Comma-separated evaluation step names to recompute selectively.',
    )
    args, remaining_argv = parser.parse_known_args(argv)

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining_argv]
        cfg = load_config_from_cli()
    finally:
        sys.argv = original_argv
    return cfg, parse_evaluation_steps(args.steps)


if __name__ == '__main__':
    cfg, selected_steps = parse_evaluate_cli_args(sys.argv[1:])
    paths = paths_for(cfg)
    setup_logging(paths)
    run_evaluate(cfg, paths, selected_steps=selected_steps)
