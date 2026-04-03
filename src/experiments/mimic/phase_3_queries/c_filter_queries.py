"""
Step 4.1: Divergence pre-filter.

For each query, run top_k and facility_location on its condition's candidate
pool. Keep only queries where coverage meaningfully diverges from top_k
(i.e., coverage selects different chunks, indicating multi-cluster structure).

TODO: understand if this is good
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
)
from experiments.mimic.phase_4_evaluation.candidate_pool import CandidatePool, CandidatePoolBuilder
from helpers.metrics import fac_cov_score, jaccard
from helpers.query_algorithms import select


def main():
    con = connect_mimic_duckdb()

    queries_df = pl.read_parquet(MIMIC_RESULTS_DIR / 'queries.parquet')
    print(f'Loaded {len(queries_df):,} queries')

    builder = CandidatePoolBuilder(con, device='cuda')
    result = filter_queries(queries_df, builder)

    out_path = MIMIC_RESULTS_DIR / 'divergence_stats.parquet'
    result.write_parquet(out_path)

    n_pass = result.filter(pl.col('passes_filter')).height
    print(f'\nSaved {len(result):,} rows to {out_path}')
    print(f'  Pass filter: {n_pass:,} / {len(result):,} ({100 * n_pass / len(result):.1f}%)')
    print(
        f'  Jaccard div: mean={result["jaccard_div"].mean():.3f}, median={result["jaccard_div"].median():.3f}'
    )
    print(
        f'  Fac gap:     mean={result["fac_gap"].mean():.4f}, median={result["fac_gap"].median():.4f}'
    )


def compute_divergence(
    pool: CandidatePool,
    query_vec: NDArray[np.float32],
    k: int = 10,
    lam: float = 0.5,
    prefilter_n: int = 500,
) -> dict:
    sim_to_query = pool.sim_to_query(query_vec)

    # Prefilter to top N by query similarity
    if pool.n > prefilter_n:
        top_indices = np.argsort(sim_to_query)[::-1][:prefilter_n].copy()
        pool = pool.slice(top_indices)
        sim_to_query = sim_to_query[top_indices]

    sim_matrix = pool.sim_matrix()

    topk_idx = select('top_k', sim_to_query=sim_to_query, k=k)
    fl_idx = select(
        'facility_location',
        sim_to_query=sim_to_query,
        k=k,
        sim_matrix=sim_matrix,
        lam=lam,
    )

    j = jaccard(topk_idx, fl_idx)
    fac_topk = fac_cov_score(topk_idx, sim_matrix)
    fac_fl = fac_cov_score(fl_idx, sim_matrix)

    return {
        'jaccard': j,
        'jaccard_div': 1.0 - j,
        'fac_gap': fac_fl - fac_topk,
        'fac_topk': fac_topk,
        'fac_fl': fac_fl,
        'pool_size': pool.n,
    }


def filter_queries(
    queries_df: pl.DataFrame,
    builder: CandidatePoolBuilder,
    k: int = 10,
    lam: float = 0.5,
    jaccard_threshold: float = 0.7,
    prefilter_n: int = 500,
) -> pl.DataFrame:
    """Run divergence filter on all queries. Returns augmented DataFrame.

    Adds columns: jaccard_div, fac_gap, passes_filter, pool_size.
    Queries with jaccard > jaccard_threshold are filtered out (coverage ≈ top_k).
    """
    results = []
    condition_pools: dict[str, CandidatePool] = {}

    for row in tqdm(
        queries_df.iter_rows(named=True), total=len(queries_df), desc='Divergence filter'
    ):
        icd3 = row['icd10_3char']

        if icd3 not in condition_pools:
            condition_pools[icd3] = builder.for_condition(icd3)

        pool = condition_pools[icd3]
        query_vec = builder.embed_query(row['query_text'])

        div = compute_divergence(pool, query_vec, k=k, lam=lam, prefilter_n=prefilter_n)
        passes = div['jaccard'] < jaccard_threshold

        results.append(
            {
                **{c: row[c] for c in queries_df.columns},
                'jaccard_div': div['jaccard_div'],
                'fac_gap': div['fac_gap'],
                'fac_topk': div['fac_topk'],
                'fac_fl': div['fac_fl'],
                'pool_size': div['pool_size'],
                'passes_filter': passes,
            }
        )

    return pl.DataFrame(results)


if __name__ == '__main__':
    main()
