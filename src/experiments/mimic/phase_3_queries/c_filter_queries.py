"""
Step 4.1: Divergence pre-filter.

For each query, run top_k and facility_location on its condition's candidate
pool. Keep only queries where coverage diverges from top_k
"""

import duckdb
import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.mimic.configs import EvaluateCfg, FilterQueriesCfg, get_parquet_path, global_cfg
from experiments.mimic.duck_db_init import (
    connect_mimic_duckdb,
)
from experiments.mimic.phase_4_evaluation.candidate_pool import CandidatePool, CandidatePoolBuilder
from helpers.metrics import fac_cov_score, jaccard
from helpers.query_algorithms import select

filter_queries_cfg = FilterQueriesCfg.load()


def run_filter_queries(
    con: duckdb.DuckDBPyConnection | None = None,
    cfg: FilterQueriesCfg | None = None,
) -> pl.DataFrame:
    global filter_queries_cfg
    if cfg is not None:
        filter_queries_cfg = cfg
    if con is None:
        con = connect_mimic_duckdb()

    queries_df = pl.read_parquet(get_parquet_path('queries'))
    print(f'Loaded {len(queries_df):,} queries')

    builder = CandidatePoolBuilder(con, cfg=EvaluateCfg.load(), device='cuda')
    result = filter_queries(queries_df, builder)

    out_path = get_parquet_path('divergence_stats')
    result.write_parquet(out_path)

    n_pass = result.filter(pl.col('passes_filter')).height
    print(
        f'\nSaved {len(result):,} rows to {out_path}\n'
        f'  Retained queries: {n_pass:,} / {len(result):,} ({n_pass / len(result) * 100:.2f}%)\n'
        f'  Jaccard div: mean={result["jaccard_div"].mean():.3f}\n'
        f'  Fac gap:     mean={result["fac_gap"].mean():.4f}'
    )
    return result


def filter_queries(
    queries_df: pl.DataFrame,
    builder: CandidatePoolBuilder,
) -> pl.DataFrame:
    """
    Adds columns: jaccard_div, fac_gap, passes_filter, pool_size.
    Divergence is averaged across all (k, lam) combinations in filter_queries_cfg.
    Queries with mean_jaccard > jaccard_threshold are filtered out (coverage ≈ top_k).
    """
    k_values = filter_queries_cfg.k_values
    lam_values = filter_queries_cfg.lam_values
    jaccard_threshold = filter_queries_cfg.jaccard_threshold
    prefilter_n = global_cfg.shared_queries_cfg.prefilter_n

    results = []

    for row in tqdm(
        queries_df.iter_rows(named=True), total=len(queries_df), desc='Divergence filter'
    ):
        query_vec = builder.embed_query(row['query_text'])
        pool = builder.for_query_cosine(query_vec, prefilter_n)

        all_jaccards, all_fac_gaps, all_fac_topks, all_fac_fls = [], [], [], []
        for k in k_values:
            for lam in lam_values:
                div = compute_divergence(pool, query_vec, k=k, lam=lam, prefilter_n=None)
                all_jaccards.append(div['jaccard'])
                all_fac_gaps.append(div['fac_gap'])
                all_fac_topks.append(div['fac_topk'])
                all_fac_fls.append(div['fac_fl'])

        mean_jaccard = float(np.mean(all_jaccards))
        passes = mean_jaccard < jaccard_threshold

        results.append(
            {
                **{c: row[c] for c in queries_df.columns},
                'jaccard_div': 1.0 - mean_jaccard,
                'fac_gap': float(np.mean(all_fac_gaps)),
                'fac_topk': float(np.mean(all_fac_topks)),
                'fac_fl': float(np.mean(all_fac_fls)),
                'pool_size': pool.n,
                'passes_filter': passes,
            }
        )

    return pl.DataFrame(results)


def compute_divergence(
    pool: CandidatePool,
    query_vec: NDArray[np.float32],
    k: int = 10,
    lam: float = 0.5,
    prefilter_n: int | None = None,
) -> dict:
    sim_to_query = pool.sim_to_query(query_vec)

    if prefilter_n is not None and pool.n > prefilter_n:
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


if __name__ == '__main__':
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=3)
    run_filter_queries(cfg=FilterQueriesCfg(**raw['filter_queries']))
