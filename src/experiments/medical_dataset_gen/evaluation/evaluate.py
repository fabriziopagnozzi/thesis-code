"""Evaluate retrieval strategies on the synthetic medical benchmark.

This module exists to score top-k, MMR, and facility-location against the gold
facets and distractors generated earlier in the pipeline. It uses shared
candidate-pool logic, per-query metric aggregation, and redundancy-aware
ranking metrics so the benchmark can expose coverage differences rather than
just nearest-neighbor accuracy.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import polars as pl
from rouge_score import rouge_scorer
from tqdm import tqdm

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)
from experiments.medical_dataset_gen.retrieval.embed import load_embedding_arrays
from experiments.medical_dataset_gen.retrieval.utils import (
    build_index_maps,
    candidate_pool_indices,
    retrieval_diagnostics,
    select_indices,
    topn_by_query,
)

ALPHA_NDCG_REDUNDANCY = 0.5
_ANSWER_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2'],
    use_stemmer=True,
)
_TOKEN_RE = re.compile(r'[a-z0-9]+')
_MIN_ANSWER_TOKEN_LEN = 3
_GENERIC_CLINICAL_STOPWORDS = frozenset(
    {
        'about',
        'after',
        'again',
        'also',
        'among',
        'and',
        'are',
        'axis',
        'been',
        'being',
        'between',
        'but',
        'can',
        'care',
        'case',
        'clinical',
        'clinically',
        'common',
        'commonly',
        'compare',
        'compared',
        'corpus',
        'course',
        'data',
        'diagnosed',
        'diagnosis',
        'discharge',
        'documented',
        'during',
        'each',
        'evidence',
        'from',
        'had',
        'has',
        'have',
        'hospital',
        'how',
        'into',
        'medical',
        'more',
        'most',
        'note',
        'often',
        'outcome',
        'outcomes',
        'patient',
        'patients',
        'pattern',
        'rehab',
        'rehabilitation',
        'shows',
        'status',
        'subgroup',
        'synthetic',
        'than',
        'that',
        'the',
        'their',
        'therapy',
        'this',
        'through',
        'treatment',
        'versus',
        'was',
        'were',
        'with',
        'without',
    }
)


def run_evaluate(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    chunk_documents = read_parquet(paths, 'chunk_documents')
    chunk_memberships = read_parquet(paths, 'chunk_memberships')
    queries = read_parquet(paths, 'queries')
    gold_answers = read_parquet(paths, 'gold_answers')
    qrels = read_parquet(paths, 'qrels')
    geometry = read_parquet(paths, 'geometry_stats')
    _assert_pool_scope_match(geometry, cfg.retrieval.pool_scope, table_name='geometry_stats')
    chunk_vectors, query_vectors, chunk_ids, query_ids = load_embedding_arrays(paths)
    maps = build_index_maps(chunk_documents, chunk_memberships, queries, chunk_ids, query_ids)

    facet_gold = _facet_gold_map(qrels)
    qrels_by_query_chunk = _qrels_by_query_chunk(qrels)
    answer_refs_by_query = _answer_refs_by_query(gold_answers)
    gold_by_query = {
        qid: {chunk_id for ids in facet_map.values() for chunk_id in ids}
        for qid, facet_map in facet_gold.items()
    }
    pass_map = dict(
        zip(geometry['query_id'].to_list(), geometry['passes_filter'].to_list(), strict=True)
    )

    rows: list[dict[str, Any]] = []
    compute_answer_rouge = cfg.retrieval.compute_answer_rouge
    for query in tqdm(
        queries.iter_rows(named=True), total=len(queries), desc='Evaluating', dynamic_ncols=True
    ):
        qid = query['query_id']
        if cfg.retrieval.only_pass_geometry and not bool(pass_map.get(qid, False)):
            continue

        query_facet_gold = facet_gold.get(qid)
        query_all_gold = gold_by_query.get(qid)
        answer_refs = answer_refs_by_query.get(qid, {})
        if not query_facet_gold or not query_all_gold:
            continue
        query_terms = (
            _answer_metric_query_terms(str(query.get('query_text') or ''))
            if compute_answer_rouge
            else set()
        )
        answer_rouge_refs = (
            _prepare_answer_rouge_refs(
                answer_refs=answer_refs,
                query_terms=query_terms,
            )
            if compute_answer_rouge
            else None
        )

        qidx = maps['query_id_to_idx'][qid]
        candidate_idx = candidate_pool_indices(
            query_id=qid,
            pool_scope=cfg.retrieval.pool_scope,
            n_chunks=len(chunk_ids),
            chunks_by_source_query=maps['chunks_by_source_query'],
            chunks_by_condition=maps['chunks_by_condition'],
            query_condition_id=query.get('condition_id'),
        )
        topn_global, topn_sims = topn_by_query(
            candidate_indices=candidate_idx,
            chunk_vectors=chunk_vectors,
            query_vector=query_vectors[qidx],
            n=cfg.retrieval.candidate_pool_n,
        )
        if len(topn_global) == 0:
            continue

        candidate_vectors = chunk_vectors[topn_global]
        sim_matrix = candidate_vectors @ candidate_vectors.T
        sim_to_query = topn_sims.astype(np.float32)
        candidate_chunk_ids = [chunk_ids[i] for i in topn_global]
        candidate_rouge_text_by_id = (
            _preprocess_candidate_chunk_texts(
                candidate_chunk_ids=candidate_chunk_ids,
                chunk_by_id=maps['chunk_by_id'],
                query_terms=query_terms,
            )
            if compute_answer_rouge
            else None
        )
        answer_rouge_cache: dict[tuple[str, ...], dict[str, float]] = {}
        topk_by_k: dict[int, np.ndarray] = {}

        for k in cfg.retrieval.k_values:
            if k > len(candidate_chunk_ids):
                continue
            topk_by_k[k] = select_indices('top_k', sim_to_query, sim_matrix, k=k, lam=None)

        for strategy in cfg.retrieval.strategies:
            lam_values: list[None] | list[float] = (
                [None] if strategy == 'top_k' else cfg.retrieval.lambda_values
            )
            for lam in lam_values:
                for k in cfg.retrieval.k_values:
                    if k > len(candidate_chunk_ids):
                        continue
                    selected_local = select_indices(
                        strategy=strategy,
                        sim_to_query=sim_to_query,
                        sim_matrix=sim_matrix,
                        k=k,
                        lam=lam,
                        mmr_window=cfg.retrieval.mmr_window,
                    )
                    selected_chunk_ids = [candidate_chunk_ids[int(i)] for i in selected_local]
                    topk_ref = topk_by_k[k]
                    answer_rouge: dict[str, float] = {}
                    if compute_answer_rouge:
                        assert answer_rouge_refs is not None
                        answer_rouge_key = tuple(str(chunk_id) for chunk_id in selected_chunk_ids)
                        cached_answer_rouge = answer_rouge_cache.get(answer_rouge_key)
                        if cached_answer_rouge is None:
                            cached_answer_rouge = _answer_rouge_metrics(
                                selected_chunk_ids=selected_chunk_ids,
                                candidate_rouge_text_by_id=candidate_rouge_text_by_id or {},
                                reference_ngrams=dict(answer_rouge_refs['answer_ngrams']),
                                facet_reference_rouge1_ngrams=list(
                                    answer_rouge_refs['facet_rouge1_ngrams']
                                ),
                            )
                            answer_rouge_cache[answer_rouge_key] = cached_answer_rouge
                        answer_rouge = cached_answer_rouge
                    row = {
                        'query_id': qid,
                        'query_type': query['query_type'],
                        'condition_id': query['condition_id'],
                        'split': query['split'],
                        'strategy': strategy,
                        'k': k,
                        'lam': lam,
                        'pool_scope': cfg.retrieval.pool_scope,
                        'pool_size': len(candidate_chunk_ids),
                        **_retrieval_metrics(
                            selected_chunk_ids=selected_chunk_ids,
                            chunk_by_id=maps['chunk_by_id'],
                            query_qrels=qrels_by_query_chunk.get(qid, {}),
                            facet_to_gold=query_facet_gold,
                            all_gold_ids=query_all_gold,
                            dominant_facet_id=query['dominant_facet_id'],
                        ),
                        **answer_rouge,
                        **retrieval_diagnostics(
                            selected_local,
                            sim_to_query,
                            sim_matrix,
                            topk_local_indices=topk_ref if strategy != 'top_k' else None,
                        ),
                    }
                    rows.append(row)

    results = pl.DataFrame(rows)
    write_parquet(paths, 'evaluation_results', results)
    stats = summarize_results(results)
    write_parquet(paths, 'evaluation_stats', stats)
    print(stats)
    return results


def _assert_pool_scope_match(
    df: pl.DataFrame,
    expected_pool_scope: str,
    table_name: str,
) -> None:
    if 'pool_scope' not in df.columns or df.is_empty():
        return
    scopes = sorted({str(value) for value in df['pool_scope'].drop_nulls().to_list()})
    if not scopes:
        return
    if scopes != [expected_pool_scope]:
        raise ValueError(
            f'{table_name} was generated with pool_scope={scopes}, '
            f'but the current config expects pool_scope={expected_pool_scope!r}. '
            'Rerun from the geometry stage, or use a config matching the stored artifacts.'
        )


def summarize_results(results: pl.DataFrame) -> pl.DataFrame:
    if len(results) == 0:
        return pl.DataFrame()
    agg_exprs: list[pl.Expr] = [
        pl.col('query_id').n_unique().alias('n_queries'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('gold_recall').mean().alias('Recall@k'),
        pl.col('gold_f1').mean().alias('F1@k'),
        pl.col('average_precision_at_k').mean().alias('MAP@k'),
        pl.col('facet_coverage').mean().alias('MeanFacetHitRate@k'),
        pl.col('facet_coverage').mean().alias('FacetCoverage@k'),
        pl.col('weighted_facet_coverage').mean().alias('MeanFacetRecall@k'),
        pl.col('facet_mrr_at_k').mean().alias('FacetMRR@k'),
        pl.col('alpha_ndcg').mean().alias('alpha-nDCG@k'),
        pl.col('distractor_rate').mean().alias('DistractorRate'),
        pl.col('near_miss_distractor_rate').mean().alias('NearMissDistractorRate'),
        pl.col('background_outlier_rate').mean().alias('BackgroundOutlierRate'),
        pl.col('any_distractor_rate').mean().alias('AnyDistractorRate'),
        pl.col('dominant_facet_rate').mean().alias('DominantFacetRate'),
        pl.col('redundant_gold_rate').mean().alias('RedundantGoldRate'),
        pl.col('fac_cov_score').mean().alias('fac'),
        pl.col('avg_cos').mean().alias('avg_cos'),
        pl.col('jaccard_vs_topk').mean().alias('jac'),
    ]
    optional_rouge_exprs = [
        ('answer_rouge1_recall', 'AnswerROUGE1Recall@k'),
        ('answer_rouge1_precision', 'AnswerROUGE1Precision@k'),
        ('answer_rouge2_recall', 'AnswerROUGE2Recall@k'),
        ('macro_facet_answer_rouge1_recall', 'MacroFacetAnswerROUGE1Recall@k'),
    ]
    agg_exprs.extend(
        pl.col(source_col).mean().alias(target_col)
        for source_col, target_col in optional_rouge_exprs
        if source_col in results.columns
    )
    stats = (
        results.group_by('strategy', 'lam', 'k')
        .agg(agg_exprs)
        .sort('k', 'strategy', 'lam')
    )
    ordered_cols = [
        'strategy',
        'lam',
        'k',
        'n_queries',
        'Precision@k',
        'Recall@k',
        'F1@k',
        'MAP@k',
        'MeanFacetHitRate@k',
        'FacetCoverage@k',
        'MeanFacetRecall@k',
        'FacetMRR@k',
        'alpha-nDCG@k',
        'AnswerROUGE1Recall@k',
        'AnswerROUGE1Precision@k',
        'AnswerROUGE2Recall@k',
        'MacroFacetAnswerROUGE1Recall@k',
        'DistractorRate',
        'NearMissDistractorRate',
        'BackgroundOutlierRate',
        'AnyDistractorRate',
        'DominantFacetRate',
        'RedundantGoldRate',
        'fac',
        'avg_cos',
        'jac',
    ]
    return stats.select([col for col in ordered_cols if col in stats.columns])


def _facet_gold_map(qrels: pl.DataFrame) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in qrels.filter(pl.col('is_gold')).iter_rows(named=True):
        result[row['query_id']][row['facet_id']].append(row['chunk_id'])
    return result


def _qrels_by_query_chunk(qrels: pl.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in qrels.iter_rows(named=True):
        result[str(row['query_id'])][str(row['chunk_id'])] = row
    return result


def _answer_refs_by_query(gold_answers: pl.DataFrame) -> dict[str, dict[str, object]]:
    refs: dict[str, dict[str, object]] = {}
    for row in gold_answers.iter_rows(named=True):
        facet_references = _facet_references_from_answer_row(row)
        refs[str(row['query_id'])] = {
            'answer_text': str(row.get('answer_text') or ''),
            'facet_references': facet_references,
        }
    return refs


def _prepare_answer_rouge_refs(
    answer_refs: dict[str, object],
    query_terms: set[str],
) -> dict[str, object]:
    answer_text = _preprocess_answer_metric_text(
        str(answer_refs.get('answer_text') or ''),
        query_terms=query_terms,
    )
    facet_references = [
        _preprocess_answer_metric_text(str(facet_reference), query_terms=query_terms)
        for facet_reference in list(answer_refs.get('facet_references') or [])
    ]
    return {
        'answer_ngrams': _rouge_ngram_bundle(answer_text),
        'facet_rouge1_ngrams': [
            _rouge_ngrams(facet_reference, n=1)
            for facet_reference in facet_references
            if facet_reference
        ],
    }


def _facet_references_from_answer_row(row: dict[str, Any]) -> list[str]:
    summaries_raw = row.get('facet_summaries_json')
    if summaries_raw:
        try:
            summaries = json.loads(str(summaries_raw))
        except json.JSONDecodeError:
            summaries = {}
        if isinstance(summaries, dict):
            return [str(value) for value in summaries.values() if str(value).strip()]

    facts_raw = row.get('answer_facts_json')
    if not facts_raw:
        return []
    try:
        facts = json.loads(str(facts_raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(facts, list):
        return []
    return [
        str(item.get('summary'))
        for item in facts
        if isinstance(item, dict) and str(item.get('summary') or '').strip()
    ]


def _retrieval_metrics(
    selected_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    query_qrels: dict[str, dict[str, Any]],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
    dominant_facet_id: str,
) -> dict[str, float | int]:
    relevance = _relevance_metrics(
        selected_chunk_ids=selected_chunk_ids,
        all_gold_ids=all_gold_ids,
    )
    facet_coverage = _facet_coverage_metrics(
        selected_chunk_ids=selected_chunk_ids,
        facet_to_gold=facet_to_gold,
    )
    diversified_ranking = _diversified_ranking_metrics(
        selected_chunk_ids=selected_chunk_ids,
        query_qrels=query_qrels,
        facet_to_gold=facet_to_gold,
        all_gold_ids=all_gold_ids,
    )
    redundancy = _redundancy_metrics(
        selected_chunk_ids=selected_chunk_ids,
        query_qrels=query_qrels,
        all_gold_ids=all_gold_ids,
        dominant_facet_id=dominant_facet_id,
        n_selected_gold=int(relevance['n_selected_gold']),
        n_facet_hits=int(facet_coverage['n_unique_gold_facets']),
    )
    selected_rows = [chunk_by_id[cid] for cid in selected_chunk_ids]

    return {
        **relevance,
        **facet_coverage,
        **diversified_ranking,
        **redundancy,
        'n_unique_hadms': len(
            {row.get('admission_id') for row in selected_rows if row.get('admission_id')}
        ),
    }


def _answer_rouge_metrics(
    selected_chunk_ids: list[str],
    candidate_rouge_text_by_id: dict[str, str],
    reference_ngrams: dict[str, Counter[tuple[str, ...]]],
    facet_reference_rouge1_ngrams: list[Counter[tuple[str, ...]]],
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
        'macro_facet_answer_rouge1_recall': float(np.mean(facet_scores))
        if facet_scores
        else 0.0,
    }


def _preprocess_candidate_chunk_texts(
    candidate_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    query_terms: set[str],
) -> dict[str, str]:
    return {
        str(chunk_id): _preprocess_answer_metric_text(
            str(chunk_by_id.get(chunk_id, {}).get('text') or ''),
            query_terms=query_terms,
        )
        for chunk_id in candidate_chunk_ids
    }


def _answer_metric_query_terms(query_text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(query_text.lower())
        if _is_answer_metric_token(token) and token not in _GENERIC_CLINICAL_STOPWORDS
    }


def _preprocess_answer_metric_text(text: str, query_terms: set[str]) -> str:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if not _is_answer_metric_token(token):
            continue
        if token in query_terms or token in _GENERIC_CLINICAL_STOPWORDS:
            continue
        tokens.append(token)
    return ' '.join(tokens)


def _is_answer_metric_token(token: str) -> bool:
    return token.isdigit() or len(token) >= _MIN_ANSWER_TOKEN_LEN


def _rouge_ngram_bundle(text: str) -> dict[str, Counter[tuple[str, ...]]]:
    tokens = _ANSWER_ROUGE_SCORER._tokenizer.tokenize(text)
    return {
        'rouge1': rouge_scorer._create_ngrams(tokens, 1),
        'rouge2': rouge_scorer._create_ngrams(tokens, 2),
    }


def _rouge_ngrams(text: str, n: int) -> Counter[tuple[str, ...]]:
    tokens = _ANSWER_ROUGE_SCORER._tokenizer.tokenize(text)
    return rouge_scorer._create_ngrams(tokens, n)


def _score_answer_rouge_ngrams(
    reference_ngrams: dict[str, Counter[tuple[str, ...]]],
    candidate_ngrams: dict[str, Counter[tuple[str, ...]]],
) -> dict[str, float]:
    rouge1 = rouge_scorer._score_ngrams(
        reference_ngrams['rouge1'],
        candidate_ngrams['rouge1'],
    )
    rouge2 = rouge_scorer._score_ngrams(
        reference_ngrams['rouge2'],
        candidate_ngrams['rouge2'],
    )
    return {
        'rouge1_recall': float(rouge1.recall),
        'rouge1_precision': float(rouge1.precision),
        'rouge2_recall': float(rouge2.recall),
    }


def _relevance_metrics(
    selected_chunk_ids: list[str],
    all_gold_ids: set[str],
) -> dict[str, float | int]:
    n_selected = len(selected_chunk_ids)
    n_selected_gold = sum(1 for chunk_id in selected_chunk_ids if chunk_id in all_gold_ids)
    gold_precision = n_selected_gold / n_selected if n_selected else 0.0
    gold_recall = n_selected_gold / len(all_gold_ids) if all_gold_ids else 0.0
    return {
        'gold_precision': float(gold_precision),
        'gold_recall': float(gold_recall),
        'gold_f1': float(_harmonic_mean(gold_precision, gold_recall)),
        'average_precision_at_k': average_precision_at_k(
            selected_chunk_ids=selected_chunk_ids,
            all_gold_ids=all_gold_ids,
        ),
        'n_selected': n_selected,
        'n_selected_gold': n_selected_gold,
    }


def _facet_coverage_metrics(
    selected_chunk_ids: list[str],
    facet_to_gold: dict[str, list[str]],
) -> dict[str, float | int]:
    selected = set(selected_chunk_ids)
    facet_gold_sets = {
        facet_id: set(gold_ids) for facet_id, gold_ids in facet_to_gold.items() if gold_ids
    }
    facet_hits = {facet_id for facet_id, gold_ids in facet_gold_sets.items() if selected & gold_ids}
    n_facets = len(facet_to_gold)
    n_facet_hits = len(facet_hits)
    facet_coverage = n_facet_hits / n_facets if n_facets else 0.0
    mean_facet_recall = (
        np.mean(
            [
                len(selected & gold_ids) / len(gold_ids)
                for gold_ids in facet_gold_sets.values()
                if gold_ids
            ]
        )
        if facet_gold_sets
        else 0.0
    )
    facet_hit_density = n_facet_hits / len(selected_chunk_ids) if selected_chunk_ids else 0.0
    facet_f1 = _harmonic_mean(facet_hit_density, facet_coverage)

    return {
        'facet_coverage': float(facet_coverage),
        'weighted_facet_coverage': float(mean_facet_recall),
        'facet_hit_density': float(facet_hit_density),
        'unique_facet_rate': float(facet_hit_density),
        'facet_f1': float(facet_f1),
        # Backward-compatible raw names; summary tables no longer label these as AP/AF1.
        'aspect_precision': float(facet_hit_density),
        'aspect_f1': float(facet_f1),
        'n_unique_gold_facets': n_facet_hits,
        'n_total_facets': n_facets,
    }


def _diversified_ranking_metrics(
    selected_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
) -> dict[str, float]:
    facet_mrr_at_k = _facet_mrr(
        selected_chunk_ids=selected_chunk_ids,
        query_qrels=query_qrels,
        facet_ids=list(facet_to_gold),
        all_gold_ids=all_gold_ids,
    )
    return {
        'alpha_ndcg': _alpha_ndcg(
            selected_chunk_ids=selected_chunk_ids,
            query_qrels=query_qrels,
            facet_to_gold=facet_to_gold,
            all_gold_ids=all_gold_ids,
            alpha=ALPHA_NDCG_REDUNDANCY,
        ),
        'facet_mrr_at_k': facet_mrr_at_k,
        'facet_mrr': facet_mrr_at_k,
    }


def _redundancy_metrics(
    selected_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    all_gold_ids: set[str],
    dominant_facet_id: str,
    n_selected_gold: int,
    n_facet_hits: int,
) -> dict[str, float | int]:
    n_selected = len(selected_chunk_ids)
    non_gold_ids = [chunk_id for chunk_id in selected_chunk_ids if chunk_id not in all_gold_ids]
    non_gold_count = len(non_gold_ids)
    background_outlier_count = sum(
        1
        for chunk_id in non_gold_ids
        if query_qrels.get(chunk_id, {}).get('cluster_role') == 'background_outlier'
    )
    near_miss_distractor_count = sum(
        1
        for chunk_id in non_gold_ids
        if _is_query_near_miss_distractor(query_qrels, chunk_id)
    )
    dominant_count = sum(
        1
        for chunk_id in selected_chunk_ids
        if query_qrels.get(chunk_id, {}).get('facet_id') == dominant_facet_id
    )
    selected_facet_counts = Counter(
        query_qrels.get(chunk_id, {}).get('facet_id')
        for chunk_id in selected_chunk_ids
        if chunk_id in all_gold_ids and query_qrels.get(chunk_id, {}).get('facet_id')
    )
    max_facet_concentration = (
        selected_facet_counts.most_common(1)[0][1] / n_selected
        if n_selected and selected_facet_counts
        else 0.0
    )
    redundant_gold_count = max(n_selected_gold - n_facet_hits, 0)
    dominant_facet_rate = dominant_count / n_selected if n_selected else 0.0

    return {
        'distractor_rate': non_gold_count / n_selected if n_selected else 0.0,
        'any_distractor_rate': non_gold_count / n_selected if n_selected else 0.0,
        'near_miss_distractor_rate': (
            near_miss_distractor_count / n_selected if n_selected else 0.0
        ),
        'background_outlier_rate': background_outlier_count / n_selected if n_selected else 0.0,
        'dominant_facet_rate': float(dominant_facet_rate),
        'dominant_cluster_concentration': float(dominant_facet_rate),
        'max_facet_concentration': float(max_facet_concentration),
        'redundant_gold_rate': redundant_gold_count / n_selected if n_selected else 0.0,
        'n_selected_non_gold': non_gold_count,
        'n_selected_near_miss_distractors': near_miss_distractor_count,
        'n_selected_background_outliers': background_outlier_count,
        'n_redundant_gold': redundant_gold_count,
    }


def _is_query_near_miss_distractor(
    query_qrels: dict[str, dict[str, Any]], chunk_id: str
) -> bool:
    row = query_qrels.get(chunk_id)
    return (
        bool(row)
        and not bool(row.get('is_gold'))
        and row.get('cluster_role') != 'background_outlier'
    )


def average_precision_at_k(
    selected_chunk_ids: list[str],
    all_gold_ids: set[str],
    k: int | None = None,
) -> float:
    rank_cutoff = len(selected_chunk_ids) if k is None else k
    denominator = min(len(all_gold_ids), rank_cutoff)
    if denominator <= 0:
        return 0.0

    n_hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(selected_chunk_ids[:rank_cutoff], start=1):
        if chunk_id not in all_gold_ids:
            continue
        n_hits += 1
        precision_sum += n_hits / rank
    return float(precision_sum / denominator)


def _harmonic_mean(left: float, right: float) -> float:
    denom = left + right
    return 0.0 if denom <= 0 else 2 * left * right / denom


def _alpha_ndcg(
    selected_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    facet_to_gold: dict[str, list[str]],
    all_gold_ids: set[str],
    alpha: float,
) -> float:
    """alpha-nDCG with facet_id as the subtopic label.

    Repeated gold chunks from the same facet receive diminishing gain, which
    makes this a ranking-sensitive coverage metric for the synthetic benchmark.
    """
    selected_dcg = _alpha_dcg(
        selected_chunk_ids=selected_chunk_ids,
        query_qrels=query_qrels,
        all_gold_ids=all_gold_ids,
        alpha=alpha,
    )
    ideal_labels = _ideal_alpha_labels(facet_to_gold, k=len(selected_chunk_ids), alpha=alpha)
    ideal_dcg = _alpha_dcg_from_labels(ideal_labels, alpha=alpha)
    return float(selected_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def _alpha_dcg(
    selected_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    all_gold_ids: set[str],
    alpha: float,
) -> float:
    labels = [
        str(query_qrels.get(chunk_id, {}).get('facet_id'))
        if chunk_id in all_gold_ids and query_qrels.get(chunk_id, {}).get('facet_id')
        else None
        for chunk_id in selected_chunk_ids
    ]
    return _alpha_dcg_from_labels(labels, alpha=alpha)


def _alpha_dcg_from_labels(labels: list[str | None], alpha: float) -> float:
    counts: Counter[str] = Counter()
    total = 0.0
    for rank, facet_id in enumerate(labels, start=1):
        if facet_id is None:
            continue
        gain = (1 - alpha) ** counts[facet_id]
        counts[facet_id] += 1
        total += gain / np.log2(rank + 1)
    return float(total)


def _ideal_alpha_labels(
    facet_to_gold: dict[str, list[str]],
    k: int,
    alpha: float,
) -> list[str]:
    remaining = {facet_id: len(gold_ids) for facet_id, gold_ids in facet_to_gold.items()}
    counts: Counter[str] = Counter()
    labels = []
    for _ in range(k):
        candidates = [
            (facet_id, (1 - alpha) ** counts[facet_id])
            for facet_id, n_remaining in remaining.items()
            if n_remaining > 0
        ]
        if not candidates:
            break
        facet_id, _ = max(candidates, key=lambda item: item[1])
        labels.append(facet_id)
        remaining[facet_id] -= 1
        counts[facet_id] += 1
    return labels


def _facet_mrr(
    selected_chunk_ids: list[str],
    query_qrels: dict[str, dict[str, Any]],
    facet_ids: list[str],
    all_gold_ids: set[str],
) -> float:
    if not facet_ids:
        return 0.0
    first_rank: dict[str, int] = {}
    for rank, chunk_id in enumerate(selected_chunk_ids, start=1):
        if chunk_id not in all_gold_ids:
            continue
        facet_id = query_qrels.get(chunk_id, {}).get('facet_id')
        if facet_id:
            first_rank.setdefault(str(facet_id), rank)
    reciprocal_ranks = [
        1 / first_rank[facet_id] if facet_id in first_rank else 0.0 for facet_id in facet_ids
    ]
    return float(np.mean(reciprocal_ranks))


if __name__ == '__main__':
    from experiments.medical_dataset_gen.global_configs import (
        dump_effective_config,
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    dump_effective_config(cfg, paths)
    run_evaluate(cfg, paths)
