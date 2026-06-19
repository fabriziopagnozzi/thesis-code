from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MetricFieldSpec:
    result_col: str
    higher_is_better: bool


METRIC_NAME_TO_FIELD: dict[str, MetricFieldSpec] = {
    'Precision@k': MetricFieldSpec('gold_precision', higher_is_better=True),
    'Recall@k': MetricFieldSpec('gold_recall', higher_is_better=True),
    'F1@k': MetricFieldSpec('gold_f1', higher_is_better=True),
    'MAP@k': MetricFieldSpec('average_precision_at_k', higher_is_better=True),
    'MeanFacetHitRate@k': MetricFieldSpec('facet_coverage', higher_is_better=True),
    'MeanFacetRecall@k': MetricFieldSpec('weighted_facet_coverage', higher_is_better=True),
    'FacetMRR@k': MetricFieldSpec('facet_mrr_at_k', higher_is_better=True),
    'alpha-nDCG@k': MetricFieldSpec('alpha_ndcg', higher_is_better=True),
    'DistractorRate': MetricFieldSpec('distractor_rate', higher_is_better=False),
    'NearMissDistractorRate': MetricFieldSpec('near_miss_distractor_rate', higher_is_better=False),
    'BackgroundOutlierRate': MetricFieldSpec('background_outlier_rate', higher_is_better=False),
    'DominantFacetRate': MetricFieldSpec('dominant_facet_rate', higher_is_better=False),
    'RedundantGoldRate': MetricFieldSpec('redundant_gold_rate', higher_is_better=False),
    'fac': MetricFieldSpec('fac_cov_score', higher_is_better=True),
    'avg_cos': MetricFieldSpec('avg_cos', higher_is_better=True),
    'jac': MetricFieldSpec('jaccard_vs_topk', higher_is_better=True),
    'AnswerROUGE1Recall@k': MetricFieldSpec('answer_rouge1_recall', higher_is_better=True),
    'AnswerROUGE1Precision@k': MetricFieldSpec('answer_rouge1_precision', higher_is_better=True),
    'AnswerROUGE2Recall@k': MetricFieldSpec('answer_rouge2_recall', higher_is_better=True),
    'MacroFacetAnswerROUGE1Recall@k': MetricFieldSpec(
        'macro_facet_answer_rouge1_recall', higher_is_better=True
    ),
}

# Answer metrics types
type Ngram = tuple[str, ...]
type NgramCounter = Counter[Ngram]


class RougeNgramBundle(TypedDict):
    rouge1: NgramCounter
    rouge2: NgramCounter


class PreparedAnswerRougeRefs(TypedDict):
    answer_ngrams: RougeNgramBundle
    facet_rouge1_ngrams: list[NgramCounter]


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
