from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import ConfigDict

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    BenchmarkModel,
    ClinicalAxis,
    ClusterRole,
    DataSplit,
    QueryType,
    SubgroupAxis,
)

type RetrievalStrategy = Literal['top_k', 'mmr', 'fac_loc']
type ChunkSupport = Literal[
    'positive', 'background_outlier', 'same_condition_wrong_axis', 'hard_negative'
]

type QueryIdToQrels = dict[str, QrelRecord]
type QrelsByQueryChunk = dict[str, QueryIdToQrels]

type FacetIdToGoldChunks = dict[str, list[str]]
type QueryIdToFacetMap = dict[str, FacetIdToGoldChunks]

type TopKDiagnosticsByK = dict[int, TopKDiagnostics]


class QrelRecord(BenchmarkModel):
    query_id: str
    evidence_profile_id: str
    pool_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    calibrated_primary_facet_id: str
    chunk_id: str
    fact_id: str | None = None
    facet_id: str | None = None
    target_facet_id: str | None = None
    cluster_id: str | None = None
    cluster_role: ClusterRole | None = None
    axis: ClinicalAxis | None = None
    facet_priority: Literal['primary', 'secondary'] | None = None
    is_gold: bool = False
    distractor_type: str | None = None
    relevance_grade: int | None = None
    support_type: ChunkSupport | None = None


class QueryRecord(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    query_id: str
    evidence_profile_id: str
    pool_id: str
    query_type: QueryType
    template_id: str
    condition_id: str | None
    condition_display: str | None = None
    split: DataSplit
    cohort_contrast_id: str
    cohort_dimension_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    calibrated_primary_facet_id: str
    facets_json: str | None = None
    query_text: str = ''


class ChunkDocumentRecord(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    chunk_id: str
    text: str = ''
    condition_id: str | None = None
    condition_display: str | None = None
    subgroup_id: str | None = None
    subgroup_label: str | None = None
    subgroup_axis: SubgroupAxis | None = None
    subgroup_field: str | None = None
    subgroup_value: str | None = None
    axis: ClinicalAxis | None = None
    value_bin: str | None = None
    admission_id: str | int | None = None


class ChunkMembershipRecord(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    chunk_id: str
    query_id: str
    pool_id: str
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis
    calibrated_primary_facet_id: str


class TopKDiagnostics(TypedDict):
    dominant_count: int
    dominant_fraction: float
    primary_axis_count: int
    primary_axis_fraction: float
    calibrated_primary_count: int
    calibrated_primary_fraction: float
    n_retrieved_facets: int
    facet_coverage: float
    all_facets_covered: bool
    retrieved_facets: list[str]


class BackgroundOutlierDiagnostics(TypedDict):
    n_background_outliers_in_pool: int
    n_background_outlier_clusters_in_pool: int
    background_outlier_complete: bool
    background_outlier_mean_in_cluster_similarity: float | None
    query_to_background_outlier_mean: float | None
    query_to_gold_mean: float | None
    gold_minus_background_outlier_similarity_margin: float | None
    background_outlier_first_rank: int | None
    background_outlier_median_rank: float | None


class RetrievalIndexMaps(TypedDict):
    chunk_id_to_idx: dict[str, int]
    query_id_to_idx: dict[str, int]
    chunk_by_id: dict[str, ChunkDocumentRecord]
    membership_by_query_chunk: dict[tuple[str, str], ChunkMembershipRecord]
    query_by_id: dict[str, QueryRecord]
    chunks_by_source_query: dict[str, list[int]]
