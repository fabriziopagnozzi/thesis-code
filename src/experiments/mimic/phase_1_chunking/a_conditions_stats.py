"""
Step 1.1: Select target conditions from MIMIC-IV.

Uses Charlson comorbidity buckets as conditions (e.g. congestive_heart_failure,
diabetes_without_cc, etc.) rather than ICD-10 3-char prefixes. Each bucket maps
directly to a structured column in mimiciv_derived.charlson, giving clean primary-
condition hadm_id sets for downstream query generation and gold annotation.

Conditions are ranked by n_admissions (count of hadm_ids with that Charlson flag > 0)
and filtered by min_admissions.
"""

import duckdb
import polars as pl

from experiments.mimic.configs import ConditionsStatsCfg, get_parquet_path, global_cfg, setup_logging
from experiments.mimic.duck_db_init import (
    connect_mimic_duckdb,
)


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
        icd10_3char (= Charlson column name), condition_name, n_admissions
    Sorted by n_admissions descending, filtered by min_admissions.
    """
    charlson_labels = global_cfg.shared_queries_cfg.charlson_labels

    rows = []
    for col, label in charlson_labels.items():
        result = con.execute(f"""--sql
            SELECT COUNT(DISTINCT hadm_id) AS n_admissions
            FROM mimiciv_derived.charlson
            WHERE {col} > 0
        """).fetchone()
        n = result[0] if result else 0
        if n >= min_admissions:
            rows.append({'icd10_3char': col, 'condition_name': label, 'n_admissions': n})

    df = pl.DataFrame(rows).sort('n_admissions', descending=True)
    print(f'Charlson conditions with >= {min_admissions} admissions: {len(df)}\n{df}')
    return df


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=1)
    run_conditions_stats(cfg=ConditionsStatsCfg(**raw['conditions_stats']))
