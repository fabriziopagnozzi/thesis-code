"""Step 1.3: Build per-admission metadata table.
Joins patients, admissions, diagnoses, and Charlson comorbidity data
into a single metadata table keyed by hadm_id.
"""

import duckdb
import polars as pl

from experiments.mimic.configs import get_table_path, setup_logging


def run_add_metadata(con: duckdb.DuckDBPyConnection | None = None) -> pl.DataFrame:
    from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb

    if con is None:
        con = connect_mimic_duckdb()

    df = build_admissions_metadata(con)

    out_path = get_table_path('admissions_metadata')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f'Saved to {out_path}')

    return df


def build_admissions_metadata(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    N_ICD_DESCRIPTIONS = 5

    df = con.execute(f"""--sql
        WITH icd_primary AS (
            SELECT
                diagnoses_icd.hadm_id,
                diagnoses_icd.icd_code AS primary_icd_code,
                d_icd_diagnoses.long_title AS primary_icd_description
            FROM diagnoses_icd
            LEFT JOIN d_icd_diagnoses
                ON diagnoses_icd.icd_code = d_icd_diagnoses.icd_code
                AND diagnoses_icd.icd_version = d_icd_diagnoses.icd_version
            WHERE diagnoses_icd.seq_num = 1
        ),
        icd_top AS (
            SELECT
                diagnoses_icd.hadm_id,
                STRING_AGG(d_icd_diagnoses.long_title, '; ' ORDER BY diagnoses_icd.seq_num) AS top_icd_descriptions
            FROM diagnoses_icd
            LEFT JOIN d_icd_diagnoses
                ON diagnoses_icd.icd_code = d_icd_diagnoses.icd_code
                AND diagnoses_icd.icd_version = d_icd_diagnoses.icd_version
            WHERE diagnoses_icd.seq_num <= {N_ICD_DESCRIPTIONS}
            GROUP BY diagnoses_icd.hadm_id
        )
        SELECT
            admissions.hadm_id,
            admissions.subject_id,
            ROUND(age.age, 1) AS age,
            patients.gender,
            admissions.race,
            admissions.insurance,
            admissions.marital_status,
            admissions.admission_type,
            admissions.discharge_location,
            admissions.hospital_expire_flag,
            icd_primary.primary_icd_code,
            icd_primary.primary_icd_description,
            icd_top.top_icd_descriptions,
            charlson.* EXCLUDE (hadm_id, subject_id)
        FROM admissions
        JOIN patients ON admissions.subject_id = patients.subject_id
        LEFT JOIN age ON admissions.hadm_id = age.hadm_id
        LEFT JOIN charlson ON admissions.hadm_id = charlson.hadm_id
        LEFT JOIN icd_primary ON admissions.hadm_id = icd_primary.hadm_id
        LEFT JOIN icd_top ON admissions.hadm_id = icd_top.hadm_id
    """).pl()

    print(f'Built metadata for {len(df):,} admissions')
    return df


if __name__ == '__main__':
    setup_logging()
    run_add_metadata()
