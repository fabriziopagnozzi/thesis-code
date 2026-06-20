from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import ConfigDict

from experiments.medical_dataset_gen.schemas.generation_schemas import BenchmarkModel
from experiments.medical_dataset_gen.schemas.retrieval_schemas import (
    ChunkDocumentRecord,
    QrelRecord,
    QueryRecord,
    RetrievalIndexMaps,
    RetrievalStrategy,
)
from experiments.medical_dataset_gen.utils.global_configs import ExperimentCfg


class GeometrySelection(BenchmarkModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_indices: NDArray[np.intp]
    lam: float | None


class GeometryArtifact(BenchmarkModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_id: str
    query: QueryRecord
    pool_scope: str
    candidate_chunk_ids: list[str]
    candidate_vectors: NDArray[np.float32]
    query_vector: NDArray[np.float32]
    sim_to_query: NDArray[np.float32]
    sim_matrix: NDArray[np.float32]
    coords: NDArray[np.float32]
    query_coord: NDArray[np.float32]
    reduction_method: str
    labels: list[str]
    label_ids: list[str]
    roles: list[str]
    is_gold: list[bool]
    facets_by_id: dict[str, str]
    cluster_labels: NDArray[np.int32]
    selections: dict[RetrievalStrategy, GeometrySelection]
    selection_variants: dict[RetrievalStrategy, list[GeometrySelection]]
    lambda_values: list[float]
    mmr_window: int | None
    k: int
    chunk_by_id: dict[str, ChunkDocumentRecord]
    qrel_by_chunk_id: dict[str, QrelRecord]
    selection_group: str | None = None


class EmbeddingGeometry2DPoint(TypedDict):
    query_id: str
    selection_group: str | None
    point_kind: str
    chunk_id: str | None
    rank: int
    x: float
    y: float
    reduction_method: str
    sim_to_query: float
    plot_label: str
    label_id: str
    cluster_role: str
    is_gold: bool
    facet_id: str | None
    target_facet_id: str | None
    distractor_type: str | None
    hdbscan_label: int | None
    selected_top_k: bool
    selected_mmr: bool
    selected_fac_loc: bool


class EmbeddingGeometryQueryStats(TypedDict, total=False):
    query_id: str
    selection_group: str | None
    query_type: str
    condition_id: str | None
    pool_scope: str
    pool_size: int
    plot_k: int
    reduction_method: str
    n_hidden_labels: int
    n_gold_points: int
    n_distractor_points: int
    gold_silhouette_cosine: float | None
    mean_in_facet_similarity: float
    mean_cross_facet_similarity: float
    in_minus_cross_similarity: float
    query_to_gold_mean: float | None
    query_to_distractor_mean: float | None
    hdbscan_n_clusters: int
    hdbscan_noise_rate: float
    hdbscan_ari_hidden: float | None
    hdbscan_nmi_hidden: float | None


class RenderedGeometryResult(TypedDict):
    point_rows: list[EmbeddingGeometry2DPoint]
    query_stats: EmbeddingGeometryQueryStats


class EmbeddingGeometryWorkerState(TypedDict):
    cfg: ExperimentCfg
    queries: pl.DataFrame
    qrels: pl.DataFrame
    chunk_vectors: NDArray[np.float32]
    query_vectors: NDArray[np.float32]
    chunk_ids: list[str]
    maps: RetrievalIndexMaps
    eval_stats: pl.DataFrame
    eval_results: pl.DataFrame
    out_dir: Path
    query_group_by_id: dict[str, str]
    k_values: list[int]
