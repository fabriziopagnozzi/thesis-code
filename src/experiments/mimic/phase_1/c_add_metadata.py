"""Step 1.3: Build per-admission metadata table.
Joins patients, admissions, diagnoses, and Charlson comorbidity data
into a single metadata table keyed by hadm_id.
"""

import duckdb
import polars as pl


def build_admissions_metadata(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    N_ICD_DESCRIPTIONS = 5

    df = con.execute(f"""
        WITH icd_primary AS (
            SELECT
                diagnoses_icd.hadm_id,
                diagnoses_icd.icd_code AS primary_icd_code,
                d_icd_diagnoses.long_title AS primary_icd_description
            FROM mimiciv_hosp.diagnoses_icd
            LEFT JOIN mimiciv_hosp.d_icd_diagnoses
                ON diagnoses_icd.icd_code = d_icd_diagnoses.icd_code
                AND diagnoses_icd.icd_version = d_icd_diagnoses.icd_version
            WHERE diagnoses_icd.seq_num = 1
        ),
        icd_top AS (
            SELECT
                diagnoses_icd.hadm_id,
                STRING_AGG(d_icd_diagnoses.long_title, '; ' ORDER BY diagnoses_icd.seq_num) AS top_icd_descriptions
            FROM mimiciv_hosp.diagnoses_icd
            LEFT JOIN mimiciv_hosp.d_icd_diagnoses
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
        FROM mimiciv_hosp.admissions
        JOIN mimiciv_hosp.patients ON admissions.subject_id = patients.subject_id
        LEFT JOIN mimiciv_derived.age ON admissions.hadm_id = age.hadm_id
        LEFT JOIN mimiciv_derived.charlson ON admissions.hadm_id = charlson.hadm_id
        LEFT JOIN icd_primary ON admissions.hadm_id = icd_primary.hadm_id
        LEFT JOIN icd_top ON admissions.hadm_id = icd_top.hadm_id
    """).pl()

    print(f'Built metadata for {len(df):,} admissions')
    return df


if __name__ == '__main__':
    from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR, connect, run_mimic_code_sql

    con = connect()
    run_mimic_code_sql(con, 'demographics/age.sql', 'comorbidity/charlson.sql')
    df = build_admissions_metadata(con)
    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')
    print(f'Saved to {MIMIC_RESULTS_DIR / "admissions_metadata.parquet"}')
