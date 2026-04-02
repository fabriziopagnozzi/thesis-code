"""
Step 4.3: Evaluation framework.

Runs all retrieval strategies on annotated queries and computes metrics.
Primary metric: aspect recall (fraction of facets covered by the retrieved set).

Output: evaluation_results.parquet
"""

import json

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
    run_sql_concept_script,
)
from experiments.mimic.phase_4_evaluation.candidate_pool import (
    CandidatePool,
    CandidatePoolBuilder,
    RetrievalResult,
    run_retrieval,
)
from helpers.metrics import avg_cos, fac_cov_score, jaccard
from helpers.query_algorithms import ScoringFunction

DEFAULT_STRATEGIES: list[ScoringFunction] = ['top_k', 'mmr', 'gmmr', 'facility_location']
DEFAULT_K_VALUES = [5, 10, 20]
DEFAULT_LAM_VALUES = [0.3, 0.5, 0.7]


def main():
    con = connect_mimic_duckdb()
    run_sql_concept_script(con, 'demographics/age.sql', 'comorbidity/charlson.sql')

    annotations_df = pl.read_parquet(MIMIC_RESULTS_DIR / 'gold_annotations.parquet')
    annotations_df = annotations_df.filter(pl.col('n_facets') > 0)
    print(f'Loaded {len(annotations_df):,} annotated queries with facets')

    builder = CandidatePoolBuilder(con, device='cuda')

    results = run_evaluation(annotations_df, builder)

    out_path = MIMIC_RESULTS_DIR / 'evaluation_results.parquet'
    results.write_parquet(out_path)
    print(f'\nSaved {len(results):,} result rows to {out_path}')

    print_summary(results)


def aspect_recall(selected_chunk_ids: set[str], facets: dict[str, list[str]]) -> float:
    """AR(S) = |{f in F : S ∩ G_f ≠ ∅}| / |F|"""
    if not facets:
        return 0.0
    covered = sum(1 for cids in facets.values() if selected_chunk_ids & set(cids))
    return covered / len(facets)


def evaluate_query(
    pool: CandidatePool,
    query_vec: np.ndarray,
    facets: dict[str, list[str]],
    strategies: list[ScoringFunction],
    k_values: list[int],
    lam_values: list[float],
    prefilter_n: int = 500,
) -> list[dict]:
    """Evaluate all strategy x k x λ combos for a single query.

    Returns list of metric dicts.
    """
    retrieval_results = run_retrieval(
        pool,
        query_vec,
        strategies=strategies,
        k_values=k_values,
        lam_values=lam_values,
        prefilter_n=prefilter_n,
    )

    # Find top_k results for Jaccard comparison
    topk_by_k: dict[int, RetrievalResult] = {}
    for r in retrieval_results:
        if r.strategy == 'top_k':
            topk_by_k[r.k] = r

    sim_to_query = pool.sim_to_query(query_vec)
    if pool.n > prefilter_n:
        top_indices = np.argsort(sim_to_query)[::-1][:prefilter_n].copy()
        eval_pool = pool.slice(top_indices)
        eval_sim_to_query = sim_to_query[top_indices]
    else:
        eval_pool = pool
        eval_sim_to_query = sim_to_query

    sim_matrix = eval_pool.sim_matrix()

    chunk_id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(eval_pool.chunk_ids)}

    metrics = []
    for r in retrieval_results:
        selected_set = set(r.selected_chunk_ids)
        ar = aspect_recall(selected_set, facets)

        eval_indices = np.array(
            [chunk_id_to_idx[cid] for cid in r.selected_chunk_ids if cid in chunk_id_to_idx],
            dtype=np.intp,
        )

        fac = fac_cov_score(eval_indices, sim_matrix) if len(eval_indices) > 0 else 0.0
        ac = avg_cos(eval_indices, eval_sim_to_query) if len(eval_indices) > 0 else 0.0

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
                'fac_cov_score': fac,
                'avg_cos': ac,
                'jaccard_vs_topk': jac,
                'n_unique_hadms': len(set(r.selected_hadm_ids)),
            }
        )

    return metrics


def run_evaluation(
    annotations_df: pl.DataFrame,
    builder: CandidatePoolBuilder,
    strategies: list[ScoringFunction] = DEFAULT_STRATEGIES,
    k_values: list[int] = DEFAULT_K_VALUES,
    lam_values: list[float] = DEFAULT_LAM_VALUES,
    prefilter_n: int = 500,
) -> pl.DataFrame:
    """Full evaluation across all annotated queries."""
    condition_pools: dict[str, CandidatePool] = {}
    all_rows = []

    for row in tqdm(
        annotations_df.iter_rows(named=True), total=len(annotations_df), desc='Evaluating'
    ):
        icd3 = row['icd10_3char']
        query_text = row['query_text']
        facets = json.loads(row['facets_json'])

        if not facets:
            continue

        if icd3 not in condition_pools:
            condition_pools[icd3] = builder.for_condition(icd3)

        pool = condition_pools[icd3]
        query_vec = builder.embed_query(query_text)

        query_metrics = evaluate_query(
            pool,
            query_vec,
            facets,
            strategies=strategies,
            k_values=k_values,
            lam_values=lam_values,
            prefilter_n=prefilter_n,
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


def print_summary(results_df: pl.DataFrame) -> None:
    print('\n=== Evaluation Summary ===\n')

    for k in sorted(results_df['k'].unique().to_list()):
        print(f'--- k = {k} ---')
        subset = results_df.filter(pl.col('k') == k)

        summary = (
            subset.group_by('strategy', 'lam')
            .agg(
                pl.col('aspect_recall').mean().alias('AR_mean'),
                pl.col('fac_cov_score').mean().alias('fac_mean'),
                pl.col('avg_cos').mean().alias('cos_mean'),
                pl.col('jaccard_vs_topk').mean().alias('jac_mean'),
                pl.col('aspect_recall').count().alias('n_queries'),
            )
            .sort('strategy', 'lam')
        )
        print(summary)
        print()


if __name__ == '__main__':
    main()
