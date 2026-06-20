from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Json

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AnswerFact,
    BenchmarkModel,
)
from experiments.medical_dataset_gen.schemas.retrieval_schemas import (
    ChunkDocumentRecord,
    QrelRecord,
    QueryRecord,
)
from experiments.medical_dataset_gen.utils.global_configs import ExperimentCfg

type EvaluationResultScalar = str | int | float | None
type EvaluationResultRow = dict[str, EvaluationResultScalar]
type ChunkEmbeddingMatrix = NDArray[np.float32]


class GoldAnswerRecord(BenchmarkModel):
    model_config = ConfigDict(extra='ignore')

    query_id: str
    answer_text: str = ''
    facet_summaries_json: Json[dict[str, str]] | None = None
    answer_facts_json: Json[list[AnswerFact]] | None = None


class EvaluationIndexMaps(TypedDict):
    query_id_to_idx: dict[str, int]
    chunk_by_id: dict[str, ChunkDocumentRecord]
    chunks_by_source_query: dict[str, list[int]]
    chunks_by_condition: dict[str, list[int]]


class AnswerReferenceTexts(TypedDict):
    answer_text: str
    facet_references: list[str]


class EvaluationWorkerState(TypedDict):
    cfg: ExperimentCfg
    queries_by_id: dict[str, QueryRecord]
    chunk_vectors: ChunkEmbeddingMatrix
    query_vectors: ChunkEmbeddingMatrix
    chunk_ids: Sequence[str]
    maps: EvaluationIndexMaps
    facet_gold: dict[str, dict[str, list[str]]]
    gold_by_query: dict[str, set[str]]
    qrels_by_query_chunk: dict[str, dict[str, QrelRecord]]
    answer_refs_by_query: dict[str, AnswerReferenceTexts]
    pass_map: dict[str, bool]
    k_values: list[int]
