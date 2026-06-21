import os
from collections.abc import Container
from pathlib import Path
from typing import cast

from experiments.medical_dataset_gen.evaluation.retrieval_utils import (
    build_index_maps,
    load_embedding_arrays,
)
from experiments.medical_dataset_gen.query_geometry.geom_plots_configs import GeomPlotFileName
from experiments.medical_dataset_gen.schemas.query_geometry_schemas import (
    EmbeddingGeometryWorkerState,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import (
    read_parquet,
    read_parquet_if_exists_else_empty_df,
)

QUERY_GEOMETRY_WORKER_STATE: EmbeddingGeometryWorkerState | None = None


def set_geom_worker_state(target: EmbeddingGeometryWorkerState | None) -> None:
    global QUERY_GEOMETRY_WORKER_STATE
    QUERY_GEOMETRY_WORKER_STATE = target


def get_geom_worker_state() -> EmbeddingGeometryWorkerState | None:
    return QUERY_GEOMETRY_WORKER_STATE


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
    cfg_dump: dict[str, object],
    exp_name: str,
    out_dir: str,
    query_group_by_id: dict[str, str],
    selected_plot_names: Container[GeomPlotFileName] | None,
) -> None:
    os.environ.setdefault('MPLBACKEND', 'Agg')
    cfg = ExperimentCfg.model_validate(cfg_dump)
    paths = MedicalDatasetGenPaths(exp_name, result_dir_overrides=cfg.global_.result_dir_overrides)

    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    qrels = read_parquet(paths, 'qrels')
    eval_stats = read_parquet_if_exists_else_empty_df(paths, 'evaluation_stats')
    eval_results = read_parquet_if_exists_else_empty_df(paths, 'evaluation_results')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)

    worker_state: EmbeddingGeometryWorkerState = {
        'cfg': cfg,
        'queries': queries,
        'qrels': qrels,
        'chunk_vectors': chunk_vectors,
        'query_vectors': query_vectors,
        'chunk_ids': chunk_ids,
        'maps': maps,
        'eval_stats': eval_stats,
        'eval_results': eval_results,
        'out_dir': Path(out_dir),
        'query_group_by_id': query_group_by_id,
        'k_values': cast(list[int], list(dict.fromkeys(cfg.retrieval.k_values))),
        'selected_plot_names': selected_plot_names,
    }
    set_geom_worker_state(worker_state)
