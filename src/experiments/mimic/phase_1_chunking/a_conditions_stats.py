"""
Step 1.1: Select target conditions from MIMIC-IV.

Uses Charlson comorbidity buckets as conditions. For each bucket, computes:
  - n_admissions: count of hadm_ids with that Charlson flag > 0
  - mean_comorbidity_count: avg number of OTHER Charlson flags present in those admissions
  - top_comorbidity_mods_json: pre-ranked co-occurring Charlson categories (rate > 0.05),
    used by phase 3.1 without re-querying DuckDB

Sorted by n_admissions descending, filtered by min_admissions.
"""

import json

import duckdb
import polars as pl

from experiments.mimic.configs import (
    ConditionsStatsCfg,
    get_parquet_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.duck_db_init import connect_mimic_duckdb


def run_conditions_stats(
    con: duckdb.DuckDBPyConnection | None = None, cfg: ConditionsStatsCfg | None = None
) -> pl.DataFrame:
    cfg = cfg or ConditionsStatsCfg.load()
    if con is None:
        con = connect_mimic_duckdb()

    df = select_conditions(con=con, min_admissions=cfg.min_admissions)
    out_path = get_parquet_path('conditions_stats')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f'\nSaved {len(df)} conditions to {out_path}')

    return df


def select_conditions(
    con: duckdb.DuckDBPyConnection,
    min_admissions: int,
) -> pl.DataFrame:
    """
    Returns a DataFrame with columns:
        charlson_col, condition_name, n_admissions, mean_comorbidity_count,
        top_comorbidity_mods_json
    """
    charlson_labels = global_cfg.shared_queries_cfg.charlson_labels
    all_cols = list(charlson_labels.keys())

    rows = []
    for col, label in charlson_labels.items():
        other_cols = [c for c in all_cols if c != col]

        result = con.execute(f"""--sql
            SELECT COUNT(DISTINCT hadm_id) AS n_admissions, {', '.join(f'AVG({c}) AS {c}' for c in other_cols)}
            FROM mimiciv_derived.charlson
            WHERE {col} > 0
        """).fetchone()

        if not result or result[0] < min_admissions:
            continue

        n = result[0]
        rates = result[1:]
        scored = [
            {'col': c, 'label': charlson_labels[c], 'rate': round(float(r), 4)}
            for c, r in zip(other_cols, rates, strict=True)
            if r and r > 0.05
        ]
        scored.sort(key=lambda x: x['rate'], reverse=True)

        rows.append(
            {
                'charlson_col': col,
                'condition_name': label,
                'n_admissions': n,
                'mean_comorbidity_count': round(sum(s['rate'] for s in scored), 2),
                'top_comorbidity_mods_json': json.dumps(scored),
            }
        )

    df = pl.DataFrame(rows).sort('n_admissions', descending=True)
    print(
        f'Charlson conditions with >= {min_admissions} admissions: {len(df)}\n{df.select("charlson_col", "n_admissions", "mean_comorbidity_count")}'
    )
    return df


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=1)
    run_conditions_stats(cfg=ConditionsStatsCfg(**raw['conditions_stats']))
