from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Json

from experiments.medical_dataset_gen.dataset_generation.schemas import (
    AnswerFact,
    BenchmarkPydanticModel,
)
from experiments.medical_dataset_gen.retrieval.schemas import (
    ChunkDocumentRecord,
    QrelRecord,
    QueryRecord,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg

if TYPE_CHECKING:
    from experiments.medical_dataset_gen.retrieval.reranker import DenseReranker

__all__ = [
    'AnswerReferenceTexts',
    'ChunkDocumentRecord',
    'EvaluationIndexMaps',
    'EvaluationResultRow',
    'EvaluationResultScalar',
    'EvaluationWorkerState',
    'GoldAnswerRecord',
    'LightweightChunkRecord',
    'LightweightQrelRecord',
    'LightweightQueryRecord',
    'QrelRecord',
    'QueryRecord',
]

type EvaluationResultScalar = str | int | float | bool | None
type EvaluationResultRow = dict[str, EvaluationResultScalar]
type ChunkEmbeddingMatrix = NDArray[np.float32]
type EmbeddingIdArray = Sequence[object]


class GoldAnswerRecord(BenchmarkPydanticModel):
    model_config = ConfigDict(extra='ignore')

    query_id: str
    answer_text: str = ''
    facet_summaries_json: Json[dict[str, str]] | None = None
    answer_facts_json: Json[list[AnswerFact]] | None = None


@dataclass(frozen=True, slots=True)
class LightweightChunkRecord:
    admission_id: str | int | None = None
    text: str = ''


@dataclass(frozen=True, slots=True)
class LightweightQrelRecord:
    facet_id: str | None
    cluster_id: str | None
    cluster_role: str | None
    axis: str | None
    distractor_type: str | None
    is_gold: bool


@dataclass(frozen=True, slots=True)
class LightweightQueryRecord:
    query_id: str
    evidence_profile_id: str
    pool_id: str
    query_type: str
    template_id: str
    condition_id: str | None
    cohort_dimension_id: str
    cohort_contrast_id: str
    cohort_contrast_family: str
    primary_axis: str
    secondary_axis: str
    dominant_primary_facet_id: str
    split: str
    query_text: str = ''


class EvaluationIndexMaps(TypedDict):
    query_id_to_idx: dict[str, int]
    chunk_by_id: dict[str, LightweightChunkRecord]
    chunks_by_source_query: dict[str, list[int]]


class AnswerReferenceTexts(TypedDict):
    answer_text: str
    facet_references: list[str]


class EvaluationWorkerState(TypedDict):
    cfg: ExperimentCfg
    queries_by_id: dict[str, LightweightQueryRecord]
    chunk_vectors: ChunkEmbeddingMatrix
    query_vectors: ChunkEmbeddingMatrix
    chunk_ids: EmbeddingIdArray
    maps: EvaluationIndexMaps
    facet_gold: dict[str, dict[str, list[str]]]
    gold_by_query: dict[str, set[str]]
    qrels_by_query_chunk: dict[str, dict[str, LightweightQrelRecord]]
    answer_refs_by_query: dict[str, AnswerReferenceTexts]
    reranker: DenseReranker | None
    k_values: list[int]
