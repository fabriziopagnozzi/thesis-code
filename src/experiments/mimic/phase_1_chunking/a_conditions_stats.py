"""
Step 1.1: Select target conditions from MIMIC-IV based on frequency and comorbidity richness.
Uses the Charlson SQL from mimic-code.

*RATIONALE*
    * Group ICD-10 codes by 3-char prefix (e.g., E11 = Type 2 Diabetes, I50 = Heart Failure). ICD are granular, we want more general conditions to search for comorbidities.

    * Rank by n_admissions * mean_comorbidity_count^2. It favors conditions that are both:
        * Frequent
        * Comorbidity-rich (patients with multiple Charlson categories active --> their discharge notes cover diverse clinical aspects)

    * Filter: only keep conditions with ≥ 200 admissions (ensures enough documents).
"""

import duckdb
import polars as pl

from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
    run_sql_concept_script,
)

# Charlson groups ICD codes into higher level buckets, containing more coarse disease groups among which it makes more sense to look for comorbidities
HIGH_LEVEL_CHARLSON_CONDITIONS = {
    'myocardial_infarct',
    'congestive_heart_failure',
    'peripheral_vascular_disease',
    'cerebrovascular_disease',
    'dementia',
    'chronic_pulmonary_disease',
    'rheumatic_disease',
    'peptic_ulcer_disease',
    'mild_liver_disease',
    'diabetes_without_cc',
    'diabetes_with_cc',
    'paraplegia',
    'renal_disease',
    'malignant_cancer',
    'severe_liver_disease',
    'metastatic_solid_tumor',
    'aids',
}


def select_conditions(
    con: duckdb.DuckDBPyConnection | None = None, min_admissions: int = 200
) -> pl.DataFrame:
    """
    Returns a DataFrame with columns:
        icd10_3char, condition_name, n_admissions, mean_comorbidity_count, score
    """
    if con is None:
        con = connect_mimic_duckdb()
        run_sql_concept_script(con, 'demographics/age.sql', 'comorbidity/charlson.sql')

    df = con.execute(f"""--sql
        WITH charlson_counts AS (
            SELECT hadm_id, {'+'.join(HIGH_LEVEL_CHARLSON_CONDITIONS)} AS n_comorbidity_categories
            FROM mimiciv_derived.charlson
        ),
        icd10_groups AS (
            SELECT DISTINCT hadm_id, SUBSTR(icd_code, 1, 3) AS icd10_3char
            FROM mimiciv_hosp.diagnoses_icd
            WHERE icd_version = 10
        ),
        condition_stats AS (
            SELECT
                icd10_groups.icd10_3char,
                COUNT(DISTINCT icd10_groups.hadm_id) AS n_admissions,
                AVG(charlson_counts.n_comorbidity_categories) AS mean_comorbidity_count
            FROM icd10_groups
            JOIN charlson_counts ON icd10_groups.hadm_id = charlson_counts.hadm_id
            GROUP BY icd10_groups.icd10_3char
            HAVING COUNT(DISTINCT icd10_groups.hadm_id) >= {min_admissions}
        )
        SELECT
            condition_stats.icd10_3char,
            icd_labels.long_title AS condition_name,
            condition_stats.n_admissions,
            ROUND(condition_stats.mean_comorbidity_count, 2) AS mean_comorbidity_count,
            ROUND(
                condition_stats.n_admissions * power(condition_stats.mean_comorbidity_count, 2), 0
            )::INTEGER AS score
        FROM condition_stats
        LEFT JOIN (
            SELECT SUBSTR(icd_code, 1, 3) AS icd10_3char, FIRST(long_title) AS long_title
            FROM mimiciv_hosp.d_icd_diagnoses
            WHERE icd_version = 10
            GROUP BY 1
        ) AS icd_labels ON condition_stats.icd10_3char = icd_labels.icd10_3char
        ORDER BY score DESC
    """).pl()

    # ! NOTE: for now we only select ICD-10 codes. Selecting ICD-9 also would require mapping the codes. Generally the mapping is not 1-to-1, so leave it like this for now. Also we use FIRST(long_title) and get the first match as long_title for the disease group. Check if anything better is possible

    print(f'Conditions with >= {min_admissions} admissions: {len(df)}')
    print('Top 10:')
    print(df.head(10))
    return df


if __name__ == '__main__':
    df = select_conditions()
    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(MIMIC_RESULTS_DIR / 'conditions_stats.parquet')
    print(f'\nSaved {len(df)} conditions to {MIMIC_RESULTS_DIR / "conditions_stats.parquet"}')
