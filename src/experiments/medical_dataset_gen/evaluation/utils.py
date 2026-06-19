from collections import defaultdict

import numpy as np
import polars as pl

from experiments.medical_dataset_gen.global_configs import FacLocMmrComparisonKernelsCfg


# Useful mathematical functions
def ci_half_width(values: list[float], z: float = 1.96) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float('nan')
    return z * float(arr.std(ddof=1)) / float(np.sqrt(len(arr)))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-value)))


def pair_kernel_polars_expr(kernel_cfg: FacLocMmrComparisonKernelsCfg) -> pl.Expr:
    if kernel_cfg.pair_aggregation == 'arithmetic_mean':
        return (pl.col('fac_loc_kernel_score') + pl.col('mmr_kernel_score')) / 2.0
    if kernel_cfg.pair_aggregation == 'minimum':
        return pl.min_horizontal('fac_loc_kernel_score', 'mmr_kernel_score')
    return (pl.col('fac_loc_kernel_score') * pl.col('mmr_kernel_score')).sqrt()


def harmonic_mean(left: float, right: float) -> float:
    denom = left + right
    return 0.0 if denom <= 0 else 2 * left * right / denom


def sigmoid_polars_expr(expr: pl.Expr) -> pl.Expr:
    return 1.0 / (1.0 + (-expr).exp())


# Miscellaneous utils
def assert_pool_scope_match(
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


def build_query_to_facet_to_gold_chunks_map(qrels: pl.DataFrame) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for row in qrels.filter(pl.col('is_gold')).iter_rows(named=True):
        result[row['query_id']][row['facet_id']].append(row['chunk_id'])

    return result
