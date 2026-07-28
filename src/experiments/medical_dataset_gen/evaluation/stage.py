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
from typing import Literal, cast, get_args

import polars as pl

from experiments.medical_dataset_gen.evaluation.eval_worker_handler import (
    load_selected_parquet_columns,
)
from experiments.medical_dataset_gen.evaluation.scoring import (
    evaluate_queries,
    get_query_ids_to_evaluate,
)
from experiments.medical_dataset_gen.evaluation.statistics import (
    results_for_split,
    stats_for_evaluation_mode,
    stats_sliced_results_df,
)
from experiments.medical_dataset_gen.retrieval.retrieval_utils import (
    assert_pool_scope_match,
    build_query_to_facet_gold_map,
)
from experiments.medical_dataset_gen.utils.cli_parsing import parse_comma_separated_names
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import (
    read_parquet,
    write_parquet,
)

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
        query_ids_to_evaluate = get_query_ids_to_evaluate(
            queries=queries,
            facet_gold=facet_gold,
            gold_by_query=gold_by_query,
        )
        del queries, qrels, geometry, facet_gold, gold_by_query
        gc.collect()

        eval_results_df = pl.DataFrame(
            evaluate_queries(
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
            results_for_split(eval_results_df, _REPORT_SPLIT)
            if cfg.evaluation.mode == 'testing'
            else eval_results_df
        )
        sliced_eval_stats_df = stats_sliced_results_df(slice_results_df)
        write_parquet(paths, 'evaluation_slice_stats', sliced_eval_stats_df)

    if aggregated_eval_stats_df is not None:
        print(aggregated_eval_stats_df)

    return eval_results_df if eval_results_df is not None else pl.DataFrame()


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
            f'Step "{requesting_step}" requires {table_name}. Run the orchestrator with '
            f'--run "eval --steps evaluation_results" first, or omit --steps to run the full stage.'
        )
    return read_parquet(paths, table_name)


def parse_evaluation_steps(raw_value: str | None) -> set[EvaluationStep] | None:
    parsed = parse_comma_separated_names(
        raw_value=raw_value,
        valid_names=EVALUATION_STEP_NAMES,
        option_name='--steps',
    )
    if parsed is None:
        return None
    return cast(set[EvaluationStep], set(parsed))


def parse_evaluate_cli_args(argv: list[str]) -> set[EvaluationStep] | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--steps',
        type=str,
        help='Comma-separated evaluation step names to recompute selectively.',
    )
    args, remaining_argv = parser.parse_known_args(argv)
    if remaining_argv:
        parser.error(f'unknown evaluation argument(s): {" ".join(remaining_argv)}')
    return parse_evaluation_steps(args.steps)
