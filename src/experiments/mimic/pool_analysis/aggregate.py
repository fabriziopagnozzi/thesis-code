import numpy as np
import polars as pl

_NON_METRIC = {
    'query_id',
    'icd10_3char',
    'stratum',
    'pool_size',
    'n_modifiers',
    'modifier_labels',
    'n_with_facet',
}


def _metric_cols(stats_df: pl.DataFrame) -> list[str]:
    return [c for c in stats_df.columns if c not in _NON_METRIC and stats_df.schema[c].is_numeric()]


def aggregate_stats(stats_df: pl.DataFrame) -> pl.DataFrame:
    metrics = _metric_cols(stats_df)
    long_rows: list[dict] = []

    for m in metrics:
        arr = stats_df[m].drop_nulls().to_numpy()
        if len(arr) == 0:
            continue
        long_rows.append(
            {
                'group_kind': 'overall',
                'group_value': None,
                'metric': m,
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'p10': float(np.quantile(arr, 0.1)),
                'p90': float(np.quantile(arr, 0.9)),
                'n': len(arr),
            }
        )

    if 'stratum' in stats_df.columns:
        for s_val in sorted(stats_df['stratum'].drop_nulls().unique().to_list()):
            sub = stats_df.filter(pl.col('stratum') == s_val)
            for m in metrics:
                arr = sub[m].drop_nulls().to_numpy()
                if len(arr) == 0:
                    continue
                long_rows.append(
                    {
                        'group_kind': 'stratum',
                        'group_value': str(int(s_val)),
                        'metric': m,
                        'mean': float(arr.mean()),
                        'median': float(np.median(arr)),
                        'p10': float(np.quantile(arr, 0.1)),
                        'p90': float(np.quantile(arr, 0.9)),
                        'n': len(arr),
                    }
                )

    return pl.DataFrame(long_rows)
