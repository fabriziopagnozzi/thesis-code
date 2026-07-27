import os
from collections import defaultdict
from collections.abc import Container, Mapping
from pathlib import Path
from typing import Any, cast

import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.eval_worker_handler import (
    load_selected_parquet_columns,
)
from experiments.medical_dataset_gen.retrieval.retrieval_utils import (
    load_embedding_arrays_mmap_ids,
)
from experiments.medical_dataset_gen.query_geometry.artifacts import build_best_lambda_maps
from experiments.medical_dataset_gen.query_geometry.geom_plots_configs import GeomPlotName
from experiments.medical_dataset_gen.utils.global_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.query_geometry.schemas import (
    EmbeddingGeometryWorkerState,
    GeometryChunkRecord,
    GeometryEmbeddingIdArray,
    GeometryIndexMaps,
    GeometryQrelRecord,
    GeometryQueryRecord,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    SyntheticMedicalDatasetTableName,
    paths_for,
)

type MMapEmbeddingIdArray = NDArray[Any]

_QUERY_COLUMNS = [
    'query_id',
    'query_type',
    'condition_id',
    'condition_display',
    'primary_axis',
    'secondary_axis',
    'facets_json',
    'query_text',
]
_CHUNK_COLUMNS = [
    'chunk_id',
    'condition_id',
    'condition_display',
    'subgroup_id',
    'subgroup_label',
    'axis',
]
_MEMBERSHIP_COLUMNS = ['query_id', 'chunk_id']
_QREL_COLUMNS = [
    'query_id',
    'chunk_id',
    'facet_id',
    'target_facet_id',
    'cluster_id',
    'cluster_role',
    'is_gold',
    'distractor_type',
]
_EVAL_RESULTS_COLUMNS = [
    'query_id',
    'strategy',
    'k',
    'lam',
    'facet_coverage_purity',
    'all_facet_clean',
    'all_facet_coverage',
    'facet_coverage',
    'gold_precision',
    'distractor_rate',
    'weighted_facet_coverage',
    'alpha_ndcg',
    'gold_recall',
]
_EVAL_STATS_COLUMNS = [
    'strategy',
    'k',
    'lam',
    'FacetCoveragePurity@k',
    'AllFacetCleanRate@k',
    'AllFacetCoverageRate@k',
    'FacetCoverage@k',
    'Precision@k',
    'DistractorRate',
    'FacetWeightedRecall@k',
    'alpha-nDCG@k',
]

_query_geometry_worker_state: EmbeddingGeometryWorkerState | None = None


def set_geom_worker_state(target: EmbeddingGeometryWorkerState | None) -> None:
    global _query_geometry_worker_state
    _query_geometry_worker_state = target


def get_geom_worker_state() -> EmbeddingGeometryWorkerState | None:
    return _query_geometry_worker_state


def query_geometry_worker_count(n_queries: int) -> int:
    requested = os.getenv('QUERY_GEOMETRY_WORKERS')
    if requested is not None:
        try:
            workers = int(requested)
        except ValueError as exc:
            raise ValueError('QUERY_GEOMETRY_WORKERS must be an integer') from exc
    else:
        workers = os.cpu_count() or 1
    return max(1, min(n_queries, workers))


def init_query_geometry_worker(
    cfg: ExperimentCfg,
    exp_name: str,
    out_dir: str,
    query_group_by_id: dict[str, str],
    query_dir_name_by_id: dict[str, str],
    selected_plot_names: Container[GeomPlotName] | None,
) -> None:
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if cfg.global_.output_experiment != exp_name:
        cfg = cfg.model_copy(deep=True)
        cfg.global_.output_experiment = exp_name
    paths = paths_for(cfg)

    chunk_documents = load_selected_parquet_columns(paths, 'chunk_documents', _CHUNK_COLUMNS)
    chunk_memberships = load_selected_parquet_columns(
        paths,
        'chunk_memberships',
        _MEMBERSHIP_COLUMNS,
    )
    queries = load_selected_parquet_columns(paths, 'queries', _QUERY_COLUMNS)
    qrels = load_selected_parquet_columns(paths, 'qrels', _QREL_COLUMNS)
    eval_stats = load_selected_parquet_columns_if_exists(
        paths,
        'evaluation_stats',
        required_columns=['strategy', 'k', 'lam'],
        optional_columns=[
            col for col in _EVAL_STATS_COLUMNS if col not in {'strategy', 'k', 'lam'}
        ],
    )
    eval_results = load_selected_parquet_columns_if_exists(
        paths,
        'evaluation_results',
        required_columns=['query_id', 'strategy', 'k', 'lam'],
        optional_columns=[
            col for col in _EVAL_RESULTS_COLUMNS if col not in {'query_id', 'strategy', 'k', 'lam'}
        ],
    )
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays_mmap_ids(paths)
    maps = _build_geometry_index_maps(
        chunk_documents=chunk_documents,
        chunk_memberships=chunk_memberships,
        chunk_ids=cast(MMapEmbeddingIdArray, chunk_ids),
        query_ids=cast(MMapEmbeddingIdArray, query_ids),
    )
    query_best_lambdas, global_best_lambdas = build_best_lambda_maps(eval_stats, eval_results)

    worker_state: EmbeddingGeometryWorkerState = {
        'cfg': cfg,
        'queries_by_id': build_lightweight_geometry_query_map(queries),
        'qrels_by_query_chunk': build_lightweight_geometry_qrels_by_query_chunk(qrels),
        'chunk_vectors': chunk_vectors,
        'query_vectors': query_vectors,
        'chunk_ids': cast(GeometryEmbeddingIdArray, chunk_ids),
        'maps': maps,
        'query_best_lambdas': query_best_lambdas,
        'global_best_lambdas': global_best_lambdas,
        'out_dir': Path(out_dir),
        'query_group_by_id': query_group_by_id,
        'query_dir_name_by_id': query_dir_name_by_id,
        'k_values': list(dict.fromkeys(cfg.retrieval.k_values)),
        'selected_plot_names': selected_plot_names,
    }
    set_geom_worker_state(worker_state)


def load_selected_parquet_columns_if_exists(
    paths: MedicalDatasetGenPaths,
    table: SyntheticMedicalDatasetTableName,
    *,
    required_columns: list[str],
    optional_columns: list[str],
) -> pl.DataFrame:
    if not paths.table_path(table).exists():
        return pl.DataFrame()
    return load_selected_parquet_columns(
        paths,
        table,
        [*required_columns, *optional_columns],
        optional_columns=optional_columns,
    )


def _build_geometry_index_maps(
    *,
    chunk_documents: pl.DataFrame,
    chunk_memberships: pl.DataFrame,
    chunk_ids: MMapEmbeddingIdArray,
    query_ids: MMapEmbeddingIdArray,
) -> GeometryIndexMaps:
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
        'chunk_by_id': build_lightweight_geometry_chunk_map(chunk_documents),
        'chunks_by_source_query': chunks_by_source_query,
    }


def build_lightweight_geometry_query_map(queries: pl.DataFrame) -> dict[str, GeometryQueryRecord]:
    result: dict[str, GeometryQueryRecord] = {}
    for row in queries.iter_rows(named=True):
        query = GeometryQueryRecord(
            query_id=str(row['query_id']),
            query_type=str(row['query_type']),
            condition_id=None if row['condition_id'] is None else str(row['condition_id']),
            condition_display=(
                None if row['condition_display'] is None else str(row['condition_display'])
            ),
            primary_axis=str(row['primary_axis']),
            secondary_axis=str(row['secondary_axis']),
            facets_json=None if row['facets_json'] is None else str(row['facets_json']),
            query_text=str(row.get('query_text') or ''),
        )
        result[query.query_id] = query
    return result


def build_lightweight_geometry_chunk_map(
    chunk_documents: pl.DataFrame,
) -> Mapping[str, GeometryChunkRecord]:
    result: dict[str, GeometryChunkRecord] = {}
    for row in chunk_documents.iter_rows(named=True):
        result[str(row['chunk_id'])] = GeometryChunkRecord(
            condition_id=None if row['condition_id'] is None else str(row['condition_id']),
            condition_display=(
                None if row['condition_display'] is None else str(row['condition_display'])
            ),
            subgroup_id=None if row['subgroup_id'] is None else str(row['subgroup_id']),
            subgroup_label=None if row['subgroup_label'] is None else str(row['subgroup_label']),
            axis=None if row['axis'] is None else str(row['axis']),
        )
    return result


def build_lightweight_geometry_qrels_by_query_chunk(
    qrels: pl.DataFrame,
) -> Mapping[str, dict[str, GeometryQrelRecord]]:
    result: dict[str, dict[str, GeometryQrelRecord]] = defaultdict(dict)
    for row in qrels.iter_rows(named=True):
        result[str(row['query_id'])][str(row['chunk_id'])] = GeometryQrelRecord(
            facet_id=None if row['facet_id'] is None else str(row['facet_id']),
            target_facet_id=(
                None if row['target_facet_id'] is None else str(row['target_facet_id'])
            ),
            cluster_id=None if row['cluster_id'] is None else str(row['cluster_id']),
            cluster_role=None if row['cluster_role'] is None else str(row['cluster_role']),
            is_gold=bool(row['is_gold']),
            distractor_type=None if row['distractor_type'] is None else str(row['distractor_type']),
        )
    return result
