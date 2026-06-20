from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rouge_score import rouge_scorer

from experiments.medical_dataset_gen.constants import GENERIC_CLINICAL_STOPWORDS
from experiments.medical_dataset_gen.schemas.evaluation_schemas import (
    AnswerReferenceTexts,
    ChunkDocumentRecord,
)
from experiments.medical_dataset_gen.schemas.metrics_schemas import (
    NgramCounter,
    PreparedAnswerRougeRefs,
    RougeNgramBundle,
)

_ANSWER_ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
_TOKEN_RE = re.compile(r'[a-z0-9]+')
_MIN_ANSWER_TOKEN_LEN = 3


@dataclass
class AnswerRougeScorer:
    candidate_rouge_text_by_id: dict[str, str]
    reference_ngrams: RougeNgramBundle
    facet_reference_rouge1_ngrams: list[NgramCounter]
    _cache: dict[tuple[str, ...], dict[str, float]] = field(default_factory=dict)

    def score(self, selected_chunk_ids: list[str]) -> dict[str, float]:
        score_key = tuple(sorted(str(chunk_id) for chunk_id in selected_chunk_ids))
        cached_scores = self._cache.get(score_key)

        if cached_scores is None:
            cached_scores = compute_answer_rouge_metrics(
                selected_chunk_ids=selected_chunk_ids,
                candidate_rouge_text_by_id=self.candidate_rouge_text_by_id,
                reference_ngrams=self.reference_ngrams,
                facet_reference_rouge1_ngrams=self.facet_reference_rouge1_ngrams,
            )
            self._cache[score_key] = cached_scores

        return cached_scores


def compute_answer_rouge_metrics(
    selected_chunk_ids: list[str],
    candidate_rouge_text_by_id: dict[str, str],
    reference_ngrams: RougeNgramBundle,
    facet_reference_rouge1_ngrams: list[NgramCounter],
) -> dict[str, float]:
    selected_text = ' '.join(
        candidate_rouge_text_by_id.get(str(chunk_id), '') for chunk_id in selected_chunk_ids
    )
    candidate_ngrams = _rouge_ngram_bundle(selected_text)
    scores = _score_answer_rouge_ngrams(reference_ngrams, candidate_ngrams)

    facet_scores = []
    for facet_rouge1_ngrams in facet_reference_rouge1_ngrams:
        facet_scores.append(
            float(
                rouge_scorer._score_ngrams(
                    facet_rouge1_ngrams,
                    candidate_ngrams['rouge1'],
                ).recall
            )
        )

    return {
        'answer_rouge1_recall': scores['rouge1_recall'],
        'answer_rouge1_precision': scores['rouge1_precision'],
        'answer_rouge2_recall': scores['rouge2_recall'],
        'macro_facet_answer_rouge1_recall': float(np.mean(facet_scores)) if facet_scores else 0.0,
    }


def prepare_answer_rouge_scorer(
    query_text: str,
    candidate_chunk_ids: list[str],
    chunk_by_id: dict[str, ChunkDocumentRecord],
    answer_refs: AnswerReferenceTexts,
) -> AnswerRougeScorer:
    query_terms = _get_answer_terms(query_text)
    candidate_rouge_text_by_id = _preprocess_candidate_chunk_texts(
        candidate_chunk_ids=candidate_chunk_ids,
        chunk_by_id=chunk_by_id,
        query_terms=query_terms,
    )
    prepared_refs = _prepare_answer_rouge_refs(
        answer_refs=answer_refs,
        query_terms=query_terms,
    )

    return AnswerRougeScorer(
        candidate_rouge_text_by_id=candidate_rouge_text_by_id,
        reference_ngrams=prepared_refs['answer_ngrams'],
        facet_reference_rouge1_ngrams=prepared_refs['facet_rouge1_ngrams'],
    )


def empty_answer_reference_texts() -> AnswerReferenceTexts:
    return {
        'answer_text': '',
        'facet_references': [],
    }


def _get_answer_terms(query_text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(query_text.lower())
        if _is_answer_valid_token(token) and token not in GENERIC_CLINICAL_STOPWORDS
    }


def _is_answer_valid_token(token: str) -> bool:
    return token.isdigit() or len(token) >= _MIN_ANSWER_TOKEN_LEN


def _preprocess_candidate_chunk_texts(
    candidate_chunk_ids: list[str],
    chunk_by_id: dict[str, ChunkDocumentRecord],
    query_terms: set[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk_id in candidate_chunk_ids:
        chunk = chunk_by_id.get(chunk_id)
        result[str(chunk_id)] = _preprocess_answer_metric_text(
            chunk.text if chunk is not None else '',
            query_terms=query_terms,
        )
    return result


def _preprocess_answer_metric_text(text: str, query_terms: set[str]) -> str:
    tokens = []

    for token in _TOKEN_RE.findall(text.lower()):
        if not _is_answer_valid_token(token):
            continue
        if token in query_terms or token in GENERIC_CLINICAL_STOPWORDS:
            continue
        tokens.append(token)
    return ' '.join(tokens)


def _prepare_answer_rouge_refs(
    answer_refs: AnswerReferenceTexts,
    query_terms: set[str],
) -> PreparedAnswerRougeRefs:
    answer_text = _preprocess_answer_metric_text(
        answer_refs['answer_text'],
        query_terms=query_terms,
    )
    facet_references = [
        _preprocess_answer_metric_text(facet_reference, query_terms=query_terms)
        for facet_reference in answer_refs['facet_references']
    ]

    return {
        'answer_ngrams': _rouge_ngram_bundle(answer_text),
        'facet_rouge1_ngrams': [_rouge_ngrams(it, n=1) for it in facet_references if it],
    }


def _score_answer_rouge_ngrams(
    reference_ngrams: RougeNgramBundle,
    candidate_ngrams: RougeNgramBundle,
) -> dict[str, float]:
    rouge1 = rouge_scorer._score_ngrams(reference_ngrams['rouge1'], candidate_ngrams['rouge1'])
    rouge2 = rouge_scorer._score_ngrams(reference_ngrams['rouge2'], candidate_ngrams['rouge2'])

    return {
        'rouge1_recall': float(rouge1.recall),
        'rouge1_precision': float(rouge1.precision),
        'rouge2_recall': float(rouge2.recall),
    }


def _rouge_ngram_bundle(text: str) -> RougeNgramBundle:
    tokens = _ANSWER_ROUGE_SCORER._tokenizer.tokenize(text)
    return {
        'rouge1': rouge_scorer._create_ngrams(tokens, 1),
        'rouge2': rouge_scorer._create_ngrams(tokens, 2),
    }


def _rouge_ngrams(text: str, n: int) -> NgramCounter:
    tokens = _ANSWER_ROUGE_SCORER._tokenizer.tokenize(text)
    return rouge_scorer._create_ngrams(tokens, n)
