"""
Queries filter.
For each query, run top_k and fac_loc on its condition's candidate
pool. Keep only queries where coverage diverges from top_k
"""

from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.mimic.global_configs import (
    get_table_path,
    global_cfg,
    read_parquet,
    setup_logging,
)
from experiments.mimic.queries.schemas_queries import DivergenceMetrics, FilterQueriesCfg, QueryRow
from experiments.mimic.utils.chunk_pools import ChunkPool, ChunkPoolBuilder
from experiments.mimic.utils.utils import get_vec_col_name
from helpers.metrics import fac_cov_score, jaccard
from helpers.query_algorithms import select

filter_queries_cfg = FilterQueriesCfg.load()
filter_col = f'filter_{get_vec_col_name(global_cfg.embedding_model)}'


def run_filter_queries(cfg: FilterQueriesCfg | None = None) -> pl.DataFrame:
    global filter_queries_cfg
    if cfg is not None:
        filter_queries_cfg = cfg

    queries_df = read_parquet('queries')
    print(f'Loaded {len(queries_df):,} queries')

    builder = ChunkPoolBuilder(model_name=global_cfg.embedding_model)
    result = filter_queries(queries_df, builder)

    out_path = get_table_path('queries')
    result.write_parquet(out_path)

    n_pass = result.select(pl.col(filter_col).sum()).item()
    print(
        f'\nSaved {len(result):,} rows to {out_path}\n'
        f'\tRetained queries: {n_pass:,} / {len(result):,} ({n_pass / len(result) * 100:.2f}%)\n'
        f'\tJaccard div: mean={result["jaccard_div"].mean():.3f}\n'
        f'\tFac gap:     mean={result["fac_gap"].mean():.4f}'
    )
    return result


def filter_queries(
    queries_df: pl.DataFrame,
    builder: ChunkPoolBuilder,
) -> pl.DataFrame:
    """
    Adds columns: jaccard_div, fac_gap, filter_<vec_col>, pool_size.
    Divergence is averaged across all (k, lam) combinations in filter_queries_cfg.
    Queries with mean_jaccard > jaccard_threshold are filtered out (coverage ≈ top_k).
    """
    results = []

    for row in tqdm(
        queries_df.iter_rows(named=True),
        total=len(queries_df),
        desc='Divergence filter',
        dynamic_ncols=True,
    ):
        row = cast(QueryRow, row)
        query_vec = builder.embed_query(row['query_text'])
        pool = builder.topk_cosine_for_condition(
            query_vec, condition_icd10_prefix=row['icd10_3char'], k=global_cfg.prefilter_n
        )

        all_jaccards, all_fac_gaps, all_fac_topks, all_fac_fls = [], [], [], []
        for k in filter_queries_cfg.k_values:
            for lam in filter_queries_cfg.lam_values:
                div = compute_divergence(pool, query_vec, k=k, lam=lam, prefilter_n=None)
                all_jaccards.append(div['jaccard'])
                all_fac_gaps.append(div['fac_gap'])
                all_fac_topks.append(div['fac_topk'])
                all_fac_fls.append(div['fac_fl'])

        mean_jaccard = float(np.mean(all_jaccards))

        results.append(
            {
                filter_col: mean_jaccard < filter_queries_cfg.jaccard_threshold,
                **{c: row[c] for c in queries_df.columns},
                'jaccard_div': 1.0 - mean_jaccard,
                'fac_gap': float(np.mean(all_fac_gaps)),
                'fac_topk': float(np.mean(all_fac_topks)),
                'fac_fl': float(np.mean(all_fac_fls)),
            }
        )

    return pl.DataFrame(results)


def compute_divergence(
    pool: ChunkPool,
    query_vec: NDArray[np.float32],
    k: int = 10,
    lam: float = 0.5,
    prefilter_n: int | None = None,
) -> DivergenceMetrics:
    sim_to_query = pool.sim_scores(query_vec)

    if prefilter_n is not None and pool.n > prefilter_n:
        top_indices = np.argsort(sim_to_query)[::-1][:prefilter_n].copy()
        pool = pool.select_by_indices(top_indices)
        sim_to_query = sim_to_query[top_indices]

    sim_matrix = pool.sim_matrix()

    topk_idx = select('top_k', sim_to_query=sim_to_query, k=k)
    fl_idx = select(
        'fac_loc',
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
    setup_logging()
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='queries')
    run_filter_queries(cfg=FilterQueriesCfg(**raw['filter_queries']))
