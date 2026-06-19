from __future__ import annotations

from typing import Literal, TypedDict, get_args

from pydantic import ConfigDict

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    BenchmarkModel,
    ClusterRole,
    QueryType,
    Split,
)

type RetrievalStrategy = Literal['top_k', 'mmr', 'fac_loc']
type ChunkSupportType = Literal['positive', 'background_outlier', 'hard_negative']
CHUNK_SUPPORT_TYPE_LIST: list[ChunkSupportType] = list(get_args(ChunkSupportType.__value__))

type QueryQrels = dict[str, QrelRecord]
type QrelsByQueryChunk = dict[str, QueryQrels]

type FacetGoldChunks = dict[str, list[str]]
type FacetGoldByQuery = dict[str, FacetGoldChunks]

type TopKDiagnosticsByK = dict[int, TopKDiagnostics]


class QrelRecord(BenchmarkModel):
    query_id: str
    chunk_id: str
    fact_id: str | None = None
    facet_id: str | None = None
    target_facet_id: str | None = None
    cluster_id: str | None = None
    cluster_role: ClusterRole | None = None
    is_gold: bool = False
    distractor_type: str | None = None
    relevance_grade: int | None = None
    support_type: ChunkSupportType | None = None


class QueryRecord(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    query_id: str
    query_type: QueryType
    condition_id: str | None
    split: Split
    dominant_facet_id: str
    query_text: str = ''


class ChunkDocumentRecord(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    chunk_id: str
    text: str = ''
    condition_id: str | None = None
    admission_id: str | int | None = None


class ChunkMembershipRecord(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    chunk_id: str
    source_query_id: str


class TopKDiagnostics(TypedDict):
    dominant_count: int
    dominant_fraction: float
    planned_dominant_count: int
    planned_dominant_fraction: float
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
    chunks_by_condition: dict[str, list[int]]
