"""
Step 4.3: Evaluation framework.

Runs all retrieval strategies on annotated queries and computes metrics.
Primary metric: aspect recall (fraction of facets covered by the retrieved set).

Output: evaluation_results.parquet
"""

import json

import duckdb
import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.mimic.configs import MIMIC_RESULTS_DIR, EvaluateCfg
from experiments.mimic.duck_db_init import (
    connect_mimic_duckdb,
)
from helpers.metrics import avg_cos, fac_cov_score, jaccard
from helpers.query_algorithms import ScoringFunction

from .candidate_pool import (
    CandidatePool,
    CandidatePoolBuilder,
    RetrievalResult,
    run_retrieval,
)

evaluate_cfg = EvaluateCfg.load()


def run_evaluate(
    con: duckdb.DuckDBPyConnection | None = None,
    cfg: EvaluateCfg | None = None,
) -> pl.DataFrame:
    global evaluate_cfg
    if cfg is not None:
        evaluate_cfg = cfg
    if con is None:
        con = connect_mimic_duckdb()

    annotations_df = pl.read_parquet(MIMIC_RESULTS_DIR / 'gold_annotations.parquet')
    annotations_df = annotations_df.filter(pl.col('n_facets') > 0)
    print(f'Loaded {len(annotations_df):,} annotated queries with facets')

    builder = CandidatePoolBuilder(con, cfg=evaluate_cfg, device=evaluate_cfg.device)

    results = evaluate(
        annotations_df,
        builder,
        strategies=evaluate_cfg.strategies,
        k_values=evaluate_cfg.k_values,
        lam_values=evaluate_cfg.lam_values,
    )

    out_path = MIMIC_RESULTS_DIR / 'evaluation_results.parquet'
    results.write_parquet(out_path)
    print(f'\nSaved {len(results):,} result rows to {out_path}')

    store_eval_stats(results)
    return results


def evaluate(
    annotations_df: pl.DataFrame,
    builder: CandidatePoolBuilder,
    strategies: list[ScoringFunction] | None = None,
    k_values: list[int] | None = None,
    lam_values: list[float] | None = None,
    prefilter_n: int | None = None,
) -> pl.DataFrame:
    """Full evaluation across all annotated queries."""
    strategies = strategies or evaluate_cfg.strategies
    k_values = k_values or evaluate_cfg.k_values
    lam_values = lam_values or evaluate_cfg.lam_values
    prefilter_n = prefilter_n or evaluate_cfg.prefilter_n
    all_rows = []

    for row in tqdm(
        annotations_df.iter_rows(named=True), total=len(annotations_df), desc='Evaluating'
    ):
        icd3 = row['icd10_3char']
        query_text = row['query_text']
        modifier_text = row.get('modifier_text')
        facets = json.loads(row['facets_json'])

        if not facets:
            continue

        query_vec = builder.embed_query(query_text)
        pool = builder.for_query_cosine(query_vec, n=prefilter_n)

        query_metrics = evaluate_query(
            pool,
            query_vec,
            facets,
            strategies=strategies,
            k_values=k_values,
            lam_values=lam_values,
        )

        for m in query_metrics:
            all_rows.append(
                {
                    'query_id': row['query_id'],
                    'icd10_3char': icd3,
                    'n_facets': row['n_facets'],
                    **m,
                }
            )

    return pl.DataFrame(all_rows)


def evaluate_query(
    pool: CandidatePool,
    query_vec: np.ndarray,
    facets: dict[str, list[str]],
    strategies: list[ScoringFunction],
    k_values: list[int],
    lam_values: list[float],
) -> list[dict]:
    """Evaluate all strategy x k x λ combos for a single query.

    Pool is assumed to be already prefiltered/stratified.
    Returns list of metric dicts.
    """
    retrieval_results = run_retrieval(
        pool,
        query_vec,
        strategies=strategies,
        k_values=k_values,
        lam_values=lam_values,
        prefilter_n=None,
    )

    # Find top_k results for Jaccard comparison
    topk_by_k: dict[int, RetrievalResult] = {}
    for r in retrieval_results:
        if r.strategy == 'top_k':
            topk_by_k[r.k] = r

    sim_to_query = pool.sim_to_query(query_vec)
    sim_matrix = pool.sim_matrix()

    chunk_id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(pool.chunk_ids)}
    pool_id_set = set(pool.chunk_ids)
    all_gold_ids = {cid for cids in facets.values() for cid in cids}

    metrics = []
    for r in retrieval_results:
        selected_set = set(r.selected_chunk_ids)
        ar = aspect_recall(selected_set, facets)
        war = weighted_aspect_recall(selected_set, facets)
        gp = gold_precision(selected_set, all_gold_ids)
        gr = gold_recall(selected_set, all_gold_ids, pool_id_set)

        eval_indices = np.array(
            [chunk_id_to_idx[cid] for cid in r.selected_chunk_ids if cid in chunk_id_to_idx],
            dtype=np.intp,
        )

        fac = fac_cov_score(eval_indices, sim_matrix) if len(eval_indices) > 0 else 0.0
        ac = avg_cos(eval_indices, sim_to_query) if len(eval_indices) > 0 else 0.0

        topk_ref = topk_by_k.get(r.k)
        if topk_ref is not None and r.strategy != 'top_k':
            topk_eval_idx = np.array(
                [
                    chunk_id_to_idx[cid]
                    for cid in topk_ref.selected_chunk_ids
                    if cid in chunk_id_to_idx
                ],
                dtype=np.intp,
            )
            jac = jaccard(eval_indices, topk_eval_idx)
        else:
            jac = 1.0

        metrics.append(
            {
                'strategy': r.strategy,
                'k': r.k,
                'lam': r.lam,
                'aspect_recall': ar,
                'weighted_aspect_recall': war,
                'gold_precision': gp,
                'gold_recall': gr,
                'fac_cov_score': fac,
                'avg_cos': ac,
                'jaccard_vs_topk': jac,
                'n_unique_hadms': len(set(r.selected_hadm_ids)),
            }
        )

    return metrics


def aspect_recall(selected_chunk_ids: set[str], facets: dict[str, list[str]]) -> float:
    """AR(S) = |{f in F : S ∩ G_f ≠ ∅}| / |F|"""
    if not facets:
        return 0.0
    covered = sum(1 for cids in facets.values() if selected_chunk_ids & set(cids))
    return covered / len(facets)


def weighted_aspect_recall(selected_chunk_ids: set[str], facets: dict[str, list[str]]) -> float:
    """WAR(S) = (1/|F|) * Σ_f |S ∩ G_f| / |G_f|"""
    if not facets:
        return 0.0
    total = 0.0
    for cids in facets.values():
        gold_set = set(cids)
        total += len(selected_chunk_ids & gold_set) / len(gold_set)
    return total / len(facets)


def gold_precision(selected_chunk_ids: set[str], all_gold_ids: set[str]) -> float:
    """Fraction of retrieved chunks that are gold."""
    if not selected_chunk_ids:
        return 0.0
    return len(selected_chunk_ids & all_gold_ids) / len(selected_chunk_ids)


def gold_recall(selected_chunk_ids: set[str], all_gold_ids: set[str], pool_ids: set[str]) -> float:
    """Fraction of gold chunks (that exist in the pool) that are retrieved."""
    reachable_gold = all_gold_ids & pool_ids
    if not reachable_gold:
        return 0.0
    return len(selected_chunk_ids & reachable_gold) / len(reachable_gold)


def store_eval_stats(results_df: pl.DataFrame) -> None:
    print('\n=== Evaluation Summary ===\n')

    summaries = []
    for k in sorted(results_df['k'].unique().to_list()):
        print(f'--- k = {k} ---')
        subset = results_df.filter(pl.col('k') == k)

        summary = (
            subset.group_by('strategy', 'lam')
            .agg(
                pl.col('aspect_recall').mean().alias('AR'),
                pl.col('weighted_aspect_recall').mean().alias('WAR'),
                pl.col('gold_precision').mean().alias('GP'),
                pl.col('gold_recall').mean().alias('GR'),
                pl.col('fac_cov_score').mean().alias('fac'),
                pl.col('avg_cos').mean().alias('cos'),
                pl.col('jaccard_vs_topk').mean().alias('jac'),
                pl.col('aspect_recall').count().alias('n'),
            )
            .sort('strategy', 'lam')
        )
        summaries.append(summary.with_columns(pl.lit(k).alias('k')))
        print(summary)
        print()

    stats_df = pl.concat(summaries)
    stats_path = MIMIC_RESULTS_DIR / 'evaluation_stats.parquet'
    stats_df.write_parquet(stats_path)
    print(f'Saved summary to {stats_path}')


if __name__ == '__main__':
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=4)
    run_evaluate(cfg=EvaluateCfg(**raw['evaluate']))
