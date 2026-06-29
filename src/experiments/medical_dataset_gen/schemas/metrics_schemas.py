from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TypedDict

type Ngram = tuple[str, ...]
type NgramCounter = Counter[Ngram]


class RougeNgramBundle(TypedDict):
    rouge1: NgramCounter
    rouge2: NgramCounter


class PreparedAnswerRougeRefs(TypedDict):
    answer_ngrams: RougeNgramBundle
    facet_rouge1_ngrams: list[NgramCounter]


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
    'SameConditionWrongAxisRate': MetricFieldSpec(
        'same_condition_wrong_axis_rate', higher_is_better=False
    ),
    'PrimaryAxisRate': MetricFieldSpec('primary_axis_rate', higher_is_better=False),
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
