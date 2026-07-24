"""Evaluate retrieval strategies on the synthetic medical benchmark.

This module exists to score top-k, MMR, and facility-location against the gold
facets and distractors generated earlier in the pipeline. It uses shared
candidate-pool logic, per-query metric aggregation, and redundancy-aware
ranking metrics so the benchmark can expose coverage differences rather than
just nearest-neighbor accuracy.
"""

from __future__ import annotations

import argparse
import gc
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
    load_selected_parquet_columns,
)
from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    LAMBDA_SELECTION_MAXIMIZING_METRIC,
    select_best_lambda_row,
)
from experiments.medical_dataset_gen.evaluation.metrics_answer import (
    empty_answer_reference_texts,
    prepare_answer_rouge_scorer,
)
from experiments.medical_dataset_gen.evaluation.metrics_retrieval import compute_retrieval_metrics
from experiments.medical_dataset_gen.evaluation.reranker import DENSE_RERANKER_STRATEGY
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
    LightweightQueryRecord,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    EvaluationMode,
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)
from experiments.medical_dataset_gen.utils.io_utils import (
    read_parquet,
    write_parquet,
)
from experiments.medical_dataset_gen.utils.logging_utils import setup_logging

type EvaluationStep = Literal[
    'evaluation_results',
    'evaluation_stats',
    'evaluation_slice_stats',
]
EVALUATION_STEP_NAMES = set[EvaluationStep](get_args(EvaluationStep.__value__))
_PARENT_QUERY_COLUMNS = ['query_id']
_PARENT_QREL_COLUMNS = ['query_id', 'chunk_id', 'facet_id', 'is_gold']
_PARENT_GEOMETRY_COLUMNS = [
    'query_id',
    'passes_filter',
    'pool_scope',
    'n_topk_retrieved_facets',
]
_SELECTION_SPLIT = 'validation'
_REPORT_SPLIT = 'test'


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
        queries = load_selected_parquet_columns(paths, 'queries', _PARENT_QUERY_COLUMNS)
        qrels = load_selected_parquet_columns(paths, 'qrels', _PARENT_QREL_COLUMNS)
        geometry = load_selected_parquet_columns(paths, 'geometry_stats', _PARENT_GEOMETRY_COLUMNS)
        assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')

        facet_gold = build_query_to_facet_gold_map(qrels)
        gold_by_query = {
            qid: {chunk_id for ids in facet_map.values() for chunk_id in ids}
            for qid, facet_map in facet_gold.items()
        }
        geometry_dimensions = geometry.select(
            'query_id',
            pl.col('passes_filter').fill_null(False).alias('passes_geometry_filter'),
            'n_topk_retrieved_facets',
        )
        query_ids_to_evaluate = _get_query_ids_to_evaluate(
            queries=queries,
            facet_gold=facet_gold,
            gold_by_query=gold_by_query,
        )
        del queries, qrels, geometry, facet_gold, gold_by_query
        gc.collect()

        eval_results_df = pl.DataFrame(
            _evaluate_queries(
                cfg,
                paths,
                query_ids_to_evaluate,
            ),
            infer_schema_length=None,
            schema_overrides={'reranker_model_name': pl.String},
        )
        if not eval_results_df.is_empty():
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
        (
            aggregated_eval_stats_df,
            selection_stats_df,
            report_grid_stats_df,
        ) = stats_for_evaluation_mode(eval_results_df, mode=cfg.evaluation.mode, cfg=cfg)
        write_parquet(paths, 'evaluation_stats', aggregated_eval_stats_df)
        if cfg.evaluation.mode == 'testing':
            write_parquet(paths, 'evaluation_selection_stats', selection_stats_df)
            write_parquet(paths, 'evaluation_report_grid_stats', report_grid_stats_df)

    if 'evaluation_slice_stats' in requested_steps:
        eval_results_df = _ensure_eval_results_loaded(
            cfg=cfg,
            paths=paths,
            eval_results_df=eval_results_df,
            requesting_step='evaluation_slice_stats',
        )
        slice_results_df = (
            _results_for_split(eval_results_df, _REPORT_SPLIT)
            if cfg.evaluation.mode == 'testing'
            else eval_results_df
        )
        sliced_eval_stats_df = stats_sliced_results_df(slice_results_df)
        write_parquet(paths, 'evaluation_slice_stats', sliced_eval_stats_df)

    if aggregated_eval_stats_df is not None:
        print(aggregated_eval_stats_df)

    return eval_results_df if eval_results_df is not None else pl.DataFrame()


def stats_sliced_results_df(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame()
    slice_columns = [
        'condition_id',
        'cohort_dimension_id',
        'cohort_contrast_family',
        'cohort_contrast_id',
        'primary_axis',
        'secondary_axis',
        'template_id',
        'passes_geometry_filter',
        'n_topk_retrieved_facets',
    ]
    slice_columns = [column for column in slice_columns if column in results.columns]
    agg_exprs: list[pl.Expr] = [
        pl.col('query_id').n_unique().alias('n_queries'),
        pl.col('facet_coverage').mean().alias('FacetCoverage@k'),
        _all_facet_coverage_expr(results),
        pl.col('weighted_facet_coverage').mean().alias('FacetWeightedRecall@k'),
        pl.col('facet_coverage_purity').mean().alias('FacetCoveragePurity@k'),
        pl.col('all_facet_clean').mean().alias('AllFacetCleanRate@k'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('gold_recall').mean().alias('Recall@k'),
        pl.col('gold_f1').mean().alias('F1@k'),
        pl.col('primary_axis_rate').mean().alias('PrimaryAxisRate'),
        pl.col('dominant_facet_rate').mean().alias('DominantFacetRate'),
        pl.col('redundant_gold_rate').mean().alias('RedundantGoldRate'),
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

    _validate_reranker_evaluation_cfg(cfg)
    worker_count = get_evaluation_worker_count(cfg, len(query_ids))
    chunksize = get_evaluation_chunksize(len(query_ids), worker_count)

    if worker_count == 1:
        init_evaluation_worker(cfg, paths.exp_name)
        iterator = map(_evaluate_query, query_ids)
    else:
        print(f'[evaluate] scoring {len(query_ids):,} queries with {worker_count} workers')
        worker_context = mp.get_context('spawn')

        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=worker_context,
            initializer=init_evaluation_worker,
            initargs=(cfg, paths.exp_name),
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

            eval_result_rows.append({
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
            })

    for strategy in cfg.retrieval.strategies:
        lam_values = cfg.retrieval.lambda_values_for_strategy(strategy)

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

            append_rows_for_selection(
                strategy=strategy,
                lam=lam,
                selected_indices_list_max_k=selected_indices_list_max_k,
            )
            # end for k in valid_k_values
        # end for lam in lam_values
    # end for strategy in strategies

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


def _validate_reranker_evaluation_cfg(cfg: ExperimentCfg) -> None:
    if not cfg.evaluation.use_reranker or cfg.evaluation.reranker.candidate_pool_n is None:
        return

    max_k = max(int(k) for k in cfg.retrieval.k_values)
    if cfg.evaluation.reranker.candidate_pool_n < max_k:
        raise ValueError(
            'evaluation.reranker.candidate_pool_n must be at least the largest retrieval k '
            f'({max_k}) when evaluation.use_reranker is true'
        )


def _get_query_ids_to_evaluate(
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


def stats_aggregated_results_df(results: pl.DataFrame) -> pl.DataFrame:
    if len(results) == 0:
        return pl.DataFrame()

    agg_polars_exprs: list[pl.Expr] = [
        pl.col('query_id').n_unique().alias('n_queries'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('gold_recall').mean().alias('Recall@k'),
        pl.col('gold_f1').mean().alias('F1@k'),
        pl.col('average_precision_at_k').mean().alias('MAP@k'),
        pl.col('facet_coverage').mean().alias('FacetCoverage@k'),
        _all_facet_coverage_expr(results),
        pl.col('weighted_facet_coverage').mean().alias('FacetWeightedRecall@k'),
        pl.col('facet_coverage_purity').mean().alias('FacetCoveragePurity@k'),
        pl.col('all_facet_clean').mean().alias('AllFacetCleanRate@k'),
        pl.col('facet_mrr_at_k').mean().alias('FacetMRR@k'),
        pl.col('alpha_ndcg').mean().alias('alpha-nDCG@k'),
        pl.col('distractor_rate').mean().alias('DistractorRate'),
        pl.col('near_miss_distractor_rate').mean().alias('NearMissDistractorRate'),
        pl.col('background_outlier_rate').mean().alias('BackgroundOutlierRate'),
        pl.col('primary_axis_rate').mean().alias('PrimaryAxisRate'),
        pl.col('dominant_facet_rate').mean().alias('DominantFacetRate'),
        pl.col('redundant_gold_rate').mean().alias('RedundantGoldRate'),
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
        'FacetCoverage@k',
        'AllFacetCoverageRate@k',
        'FacetWeightedRecall@k',
        'FacetCoveragePurity@k',
        'AllFacetCleanRate@k',
        'FacetMRR@k',
        'alpha-nDCG@k',
        'AnswerROUGE1Recall@k',
        'AnswerROUGE1Precision@k',
        'AnswerROUGE1F1@k',
        'AnswerROUGE2Recall@k',
        'DistractorRate',
        'NearMissDistractorRate',
        'BackgroundOutlierRate',
        'PrimaryAxisRate',
        'DominantFacetRate',
        'RedundantGoldRate',
        'fac',
        'avg_cos',
        'jac',
    ]

    return stats.select([col for col in STATS_DF_ORDERED_COLS if col in stats.columns])


def _all_facet_coverage_expr(results: pl.DataFrame) -> pl.Expr:
    if 'all_facet_coverage' in results.columns:
        return pl.col('all_facet_coverage').mean().alias('AllFacetCoverageRate@k')
    return (pl.col('facet_coverage') == 1.0).cast(pl.Float64).mean().alias('AllFacetCoverageRate@k')


def stats_for_evaluation_mode(
    results: pl.DataFrame,
    *,
    mode: EvaluationMode,
    cfg: ExperimentCfg,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if mode == 'exploring':
        all_stats = stats_aggregated_results_df(results)
        return all_stats, pl.DataFrame(), pl.DataFrame()

    selection_results = _results_for_split(results, _SELECTION_SPLIT)
    report_results = _results_for_split(results, _REPORT_SPLIT)
    selection_stats = stats_aggregated_results_df(selection_results)
    report_grid_stats = stats_aggregated_results_df(report_results)
    report_stats = _heldout_report_stats(
        selection_stats=selection_stats,
        report_grid_stats=report_grid_stats,
        cfg=cfg,
    )
    return report_stats, selection_stats, report_grid_stats


def _results_for_split(results: pl.DataFrame, split: str) -> pl.DataFrame:
    if results.is_empty() or 'split' not in results.columns:
        return pl.DataFrame()
    return results.filter(pl.col('split') == split)


def _heldout_report_stats(
    *,
    selection_stats: pl.DataFrame,
    report_grid_stats: pl.DataFrame,
    cfg: ExperimentCfg,
) -> pl.DataFrame:
    if selection_stats.is_empty() or report_grid_stats.is_empty():
        return pl.DataFrame()

    rows: list[pl.DataFrame] = []
    k_values = sorted(set(int(k) for k in cfg.retrieval.k_values))
    for k in k_values:
        topk_row = report_grid_stats.filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k))
        if not topk_row.is_empty():
            rows.append(_annotate_heldout_row(topk_row.head(1), selected_on_metric_value=None))

        for strategy in _heldout_non_topk_strategies(cfg, report_grid_stats):
            if _has_lambda_grid(selection_stats, strategy, k):
                selected = select_best_lambda_row(
                    selection_stats,
                    strategy=strategy,
                    k=k,
                    cfg=cfg.evaluation.lambda_selection,
                )
                if selected is None:
                    continue
                selected_lam = selected.get('lam', None)
                if selected_lam is None:
                    continue

                report_row = report_grid_stats.filter(
                    (pl.col('strategy') == strategy)
                    & (pl.col('k') == k)
                    & (pl.col('lam') == selected_lam)
                )
                selected_metric_value = selected.get(LAMBDA_SELECTION_MAXIMIZING_METRIC)
            else:
                report_row = report_grid_stats.filter(
                    (pl.col('strategy') == strategy)
                    & (pl.col('k') == k)
                    & (pl.col('lam').is_null())
                )
                selected_metric_value = None

            if report_row.is_empty():
                continue

            rows.append(
                _annotate_heldout_row(
                    report_row.head(1), selected_on_metric_value=selected_metric_value
                )
            )

    return pl.concat(rows).sort('k', 'strategy', 'lam') if rows else pl.DataFrame()


def _heldout_non_topk_strategies(cfg: ExperimentCfg, report_grid_stats: pl.DataFrame) -> list[str]:
    configured = {str(strategy) for strategy in cfg.retrieval.strategies if strategy != 'top_k'}
    if cfg.evaluation.use_reranker:
        configured.add(DENSE_RERANKER_STRATEGY)

    present = set(str(strategy) for strategy in report_grid_stats['strategy'].unique().to_list())
    return sorted(configured & present, key=_evaluation_strategy_sort_key)


def _evaluation_strategy_sort_key(strategy: str) -> tuple[int, str]:
    preferred_order = {
        'top_k': 0,
        'fac_loc': 1,
        'mmr': 2,
        DENSE_RERANKER_STRATEGY: 3,
    }
    return preferred_order.get(strategy, len(preferred_order)), strategy


def _has_lambda_grid(stats_df: pl.DataFrame, strategy: str, k: int) -> bool:
    return (
        stats_df
        .filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
        .select(pl.col('lam').drop_nulls().len())
        .item()
        != 0
    )


def _annotate_heldout_row(
    row: pl.DataFrame,
    *,
    selected_on_metric_value: float | None,
) -> pl.DataFrame:
    return row.with_columns(
        pl.lit(_SELECTION_SPLIT).alias('lambda_selection_split'),
        pl.lit(_REPORT_SPLIT).alias('report_split'),
        pl.lit(LAMBDA_SELECTION_MAXIMIZING_METRIC).alias('lambda_selection_metric'),
        pl.lit(selected_on_metric_value, dtype=pl.Float64).alias('lambda_selection_metric_value'),
    )


def parse_evaluation_steps(raw_value: str | None) -> set[EvaluationStep] | None:
    if raw_value is None:
        return None

    raw_steps = {part.strip() for part in raw_value.split(',') if part.strip()}
    if not raw_steps:
        raise ValueError('--steps was provided but no evaluation step names were specified')

    normalized_steps = {cast(EvaluationStep | str, step_name) for step_name in raw_steps}
    unknown_steps = sorted(step for step in normalized_steps if step not in EVALUATION_STEP_NAMES)
    if unknown_steps:
        available = ', '.join(sorted(EVALUATION_STEP_NAMES))
        unknown = ', '.join(unknown_steps)
        raise ValueError(
            f'Unknown evaluation step name(s): {unknown}. Available steps: {available}'
        )

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
