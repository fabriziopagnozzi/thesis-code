from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

from experiments.medical_dataset_gen.query_geometry.geom_plots_configs import GeomPlotFileName
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    BenchmarkModel,
    ChunkPoolScope,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg
from experiments.medical_dataset_gen.schemas.retrieval_schemas import (
    ChunkDocumentRecord,
    QrelRecord,
    QueryRecord,
    RetrievalStrategy,
)

type GeometryGlobalLambdaKey = tuple[RetrievalStrategy, int]
type GeometryQueryLambdaKey = tuple[str, RetrievalStrategy, int]
type GeometryEmbeddingIdArray = Sequence[object]


@dataclass(frozen=True, slots=True)
class GeometryQueryRecord:
    query_id: str
    query_type: str
    condition_id: str | None
    condition_display: str | None
    primary_axis: str
    secondary_axis: str
    facets_json: str | None = None


@dataclass(frozen=True, slots=True)
class GeometryChunkRecord:
    condition_id: str | None
    condition_display: str | None
    subgroup_id: str | None
    subgroup_label: str | None
    axis: str | None


@dataclass(frozen=True, slots=True)
class GeometryQrelRecord:
    facet_id: str | None
    target_facet_id: str | None
    cluster_id: str | None
    cluster_role: str | None
    is_gold: bool
    distractor_type: str | None


type GeometryQueryLike = QueryRecord | GeometryQueryRecord
type GeometryChunkLike = ChunkDocumentRecord | GeometryChunkRecord
type GeometryQrelLike = QrelRecord | GeometryQrelRecord


class GeometrySelection(BenchmarkModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_indices: NDArray[np.intp]
    lam: float | None


class GeometryArtifact(BenchmarkModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_id: str
    query: GeometryQueryLike
    pool_scope: ChunkPoolScope
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
    cluster_labels: NDArray[np.int32] | None
    selections: dict[RetrievalStrategy, GeometrySelection]
    selection_variants: dict[RetrievalStrategy, list[GeometrySelection]]
    lambda_values_by_strategy: dict[RetrievalStrategy, list[float]]
    mmr_window: int | None
    k: int
    chunk_by_id: Mapping[str, GeometryChunkLike]
    qrel_by_chunk_id: Mapping[str, GeometryQrelLike]
    selection_group: str | None = None


class GeometryFilterStatsRow(BenchmarkModel):
    # The filter stage adds dynamic topk_{k}_* diagnostic columns, so the
    # stable schema is typed here and the flattened per-k fields are allowed as
    # validated extras.
    model_config = ConfigDict(extra='allow')

    query_id: str
    evidence_profile_id: str
    pool_id: str
    pool_scope: ChunkPoolScope
    pool_size: int
    topk_k: int
    n_facets: int
    n_facets_present: int
    all_facets_present: bool
    topk_dominant_count: int
    calibrated_primary_facet_id: str
    calibrated_primary_topk_count: int
    calibrated_primary_topk_fraction: float
    primary_axis: str
    secondary_axis: str
    primary_axis_topk_count: int
    primary_axis_topk_fraction: float
    n_topk_retrieved_facets: int
    max_topk_retrieved_facets: int | None
    rank_where_all_facets_first_covered: int | None
    all_facets_covered_before_primary_k: bool
    n_distractors_in_pool: int
    n_near_miss_distractors_in_pool: int
    mean_in_facet_similarity: float
    mean_cross_facet_similarity: float
    in_minus_cross_similarity: float
    mean_same_axis_different_cohort_similarity: float
    mean_same_cohort_different_axis_similarity: float
    mean_different_axis_cohort_similarity: float
    same_axis_cohort_gap: float
    same_cohort_axis_gap: float
    passes_filter: bool
    fail_missing_facet: bool
    fail_weak_primary_axis_dominance: bool
    fail_too_many_topk_facets: bool
    fail_weak_facet_separation: bool
    fail_weak_same_axis_cohort_separation: bool
    fail_weak_same_cohort_axis_separation: bool
    fail_too_few_near_miss_distractors: bool
    fail_missing_or_malformed_background_outlier: bool
    facets_present_json: str
    topk_retrieved_facets_json: str
    n_background_outliers_in_pool: int
    n_background_outlier_clusters_in_pool: int
    background_outlier_complete: bool
    background_outlier_mean_in_cluster_similarity: float | None
    query_to_background_outlier_mean: float | None
    query_to_gold_mean: float | None
    gold_minus_background_outlier_similarity_margin: float | None
    background_outlier_first_rank: int | None
    background_outlier_median_rank: float | None
    fac_topk: float
    fac_facloc: float
    avg_cos_topk: float
    avg_cos_facloc: float
    jaccard_topk_facloc: float


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
    pool_scope: ChunkPoolScope
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


class GeometryIndexMaps(TypedDict):
    query_id_to_idx: dict[str, int]
    chunk_by_id: Mapping[str, GeometryChunkLike]
    chunks_by_source_query: dict[str, list[int]]


class EmbeddingGeometryWorkerState(TypedDict):
    cfg: ExperimentCfg
    queries_by_id: Mapping[str, GeometryQueryLike]
    qrels_by_query_chunk: Mapping[str, Mapping[str, GeometryQrelLike]]
    chunk_vectors: NDArray[np.float32]
    query_vectors: NDArray[np.float32]
    chunk_ids: GeometryEmbeddingIdArray
    maps: GeometryIndexMaps
    query_best_lambdas: dict[GeometryQueryLambdaKey, float]
    global_best_lambdas: dict[GeometryGlobalLambdaKey, float]
    out_dir: Path
    query_group_by_id: dict[str, str]
    query_dir_name_by_id: dict[str, str]
    k_values: list[int]
    selected_plot_names: Container[GeomPlotFileName] | None
