from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from experiments.medical_dataset_gen.global_configs import ExperimentCfg

type EvaluationResultScalar = str | int | float | None
type EvaluationResultRow = dict[str, EvaluationResultScalar]
type ChunkEmbeddingMatrix = NDArray[np.float32]


class QueryRecord(TypedDict):
    query_id: str
    query_type: str
    condition_id: str | None
    split: str
    dominant_facet_id: str
    query_text: str


class ChunkDocumentRecord(TypedDict, total=False):
    chunk_id: str
    text: str
    condition_id: str | None
    admission_id: str | int | None


class QrelRecord(TypedDict, total=False):
    query_id: str
    chunk_id: str
    facet_id: str
    cluster_role: str
    is_gold: bool


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


def coerce_query_record(row: Mapping[str, object]) -> QueryRecord:
    return {
        'query_id': str(row['query_id']),
        'query_type': str(row['query_type']),
        'condition_id': str(row['condition_id']) if row.get('condition_id') is not None else None,
        'split': str(row['split']),
        'dominant_facet_id': str(row['dominant_facet_id']),
        'query_text': str(row.get('query_text') or ''),
    }


def coerce_chunk_document_record(row: Mapping[str, object]) -> ChunkDocumentRecord:
    return {
        'chunk_id': str(row['chunk_id']),
        'text': str(row.get('text') or ''),
        'condition_id': str(row['condition_id']) if row.get('condition_id') is not None else None,
        'admission_id': _coerce_admission_id(row.get('admission_id')),
    }


def coerce_qrel_record(row: Mapping[str, object]) -> QrelRecord:
    record: QrelRecord = {
        'query_id': str(row['query_id']),
        'chunk_id': str(row['chunk_id']),
        'is_gold': bool(row.get('is_gold', False)),
    }
    if row.get('facet_id') is not None:
        record['facet_id'] = str(row['facet_id'])
    if row.get('cluster_role') is not None:
        record['cluster_role'] = str(row['cluster_role'])
    return record


def _coerce_admission_id(value: object) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return str(value)
