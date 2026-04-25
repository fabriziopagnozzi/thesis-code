import duckdb

from experiments.mimic.configs import (
    MIMIC_IV_DIR,
    get_parquet_path,
    setup_logging,
)
from helpers.dir_paths import (
    BHC_DIR,
    HOSP_DIR,
    ICU_DIR,
    MIMIR_REPO_CODE_DIR,
    NOTE_DIR,
)

INIT_SQL_PATH = MIMIC_IV_DIR / '_mimic_init.sql'
DUCKDB_CONCEPTS_DIR = MIMIR_REPO_CODE_DIR / 'mimic-iv' / 'concepts_duckdb'

HOSP_TABLES = {
    'admissions',
    'd_hcpcs',
    'd_icd_diagnoses',
    'd_icd_procedures',
    'd_labitems',
    'diagnoses_icd',
    'drgcodes',
    'emar',
    'emar_detail',
    'hcpcsevents',
    'labevents',
    'microbiologyevents',
    'omr',
    'patients',
    'pharmacy',
    'poe',
    'poe_detail',
    'prescriptions',
    'procedures_icd',
    'provider',
    'services',
    'transfers',
}
ICU_TABLES = {
    'caregiver',
    'chartevents',
    'd_items',
    'datetimeevents',
    'icustays',
    'ingredientevents',
    'inputevents',
    'outputevents',
    'procedureevents',
}
NOTE_TABLES = {'discharge', 'discharge_detail', 'radiology', 'radiology_detail'}

RESULT_TABLES = {'conditions_stats', 'admissions_metadata', 'chunks'}

# From the MIT repo
DERIVED_CONCEPTS = {
    'age': 'demographics/age.sql',
    'charlson': 'comorbidity/charlson.sql',  # for comorbidity index
}

INIT_SCHEMAS = """--sql
CREATE SCHEMA IF NOT EXISTS mimiciv_hosp;
CREATE SCHEMA IF NOT EXISTS mimiciv_icu;
CREATE SCHEMA IF NOT EXISTS mimiciv_note;
CREATE SCHEMA IF NOT EXISTS mimiciv_derived;
CREATE SCHEMA IF NOT EXISTS mimic_results;
SET search_path = 'mimiciv_hosp,mimiciv_icu,mimiciv_note,mimiciv_derived,mimic_results';
"""


def connect_mimic_duckdb() -> duckdb.DuckDBPyConnection:
    generate_init_sql()
    con = duckdb.connect()
    for statement in INIT_SQL_PATH.read_text().splitlines():
        statement = statement.strip()
        if statement:
            con.execute(statement)

    for table in RESULT_TABLES:
        parquet_file = get_parquet_path(table)
        if parquet_file.exists():
            con.execute(
                f"CREATE VIEW IF NOT EXISTS mimic_results.{table} AS SELECT * FROM read_parquet('{parquet_file}')"
            )

    _load_derived_concepts(con)
    _ensure_unified_diagnoses(con)

    return con


def _load_derived_concepts(con: duckdb.DuckDBPyConnection):
    """Load derived concept tables (age, charlson) from parquet or compute and save them."""
    for table, sql_rel in DERIVED_CONCEPTS.items():
        parquet_path = get_parquet_path(table)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if parquet_path.exists():
            con.execute(
                f"CREATE VIEW IF NOT EXISTS mimiciv_derived.{table} AS SELECT * FROM read_parquet('{parquet_path}')"
            )
        else:
            print(f'Materializing mimiciv_derived.{table} --> {parquet_path.name} ...')
            run_sql_concept_script(con, sql_rel)
            con.execute(f"COPY mimiciv_derived.{table} TO '{parquet_path}' (FORMAT PARQUET)")


def _ensure_unified_diagnoses(con: duckdb.DuckDBPyConnection) -> None:
    """Materialize unified_diagnoses (ICD-9 + ICD-10 → ICD-10) once, then register as view.

    Rows where no ICD-9→10 crosswalk entry exists are dropped, so every row in the
    materialized table has a valid unified_icd10 code.
    """
    parquet_path = get_parquet_path('unified_diagnoses')
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if not parquet_path.exists():
        print('Materializing unified_diagnoses (ICD-9+10 → ICD-10) ...')
        con.execute(f"""--sql
            COPY (
                SELECT hadm_id, unified_icd10
                FROM (
                    SELECT d.hadm_id,
                        CASE
                            WHEN d.icd_version = 10 THEN d.icd_code
                            WHEN d.icd_version = 9  THEN crosswalk.icd10
                        END AS unified_icd10
                    FROM mimiciv_hosp.diagnoses_icd d
                    LEFT JOIN (
                        SELECT icd9cm AS icd9, MIN(icd10cm) AS icd10
                        FROM read_parquet('{get_parquet_path('icd9_to_icd10_cm_gem')}')
                        WHERE regexp_matches(icd10cm, '^[A-Z][0-9]')
                        GROUP BY icd9
                    ) AS crosswalk ON d.icd_code = crosswalk.icd9 AND d.icd_version = 9
                )
                WHERE unified_icd10 IS NOT NULL
                AND regexp_matches(unified_icd10, '^[A-Z][0-9]')
            ) TO '{parquet_path}' (FORMAT PARQUET)
        """)
        print(f'  Saved {parquet_path}')
    con.execute(
        f'CREATE VIEW IF NOT EXISTS unified_diagnoses AS '
        f"SELECT * FROM read_parquet('{parquet_path}')"
    )


def register_result_view(con: duckdb.DuckDBPyConnection, name: str, df) -> None:
    con.execute(f'DROP VIEW IF EXISTS mimic_results.{name}')
    con.register(name, df)


def generate_init_sql(force: bool = False):
    if INIT_SQL_PATH.exists() and not force:
        return
    lines: list[str] = [INIT_SCHEMAS]
    for table in HOSP_TABLES:
        csv = HOSP_DIR / f'{table}.csv'
        if csv.exists():
            lines.append(
                f"CREATE VIEW IF NOT EXISTS mimiciv_hosp.{table} AS SELECT * FROM read_csv_auto('{csv}');"
            )
    for table in ICU_TABLES:
        csv = ICU_DIR / f'{table}.csv'
        if csv.exists():
            lines.append(
                f"CREATE VIEW IF NOT EXISTS mimiciv_icu.{table} AS SELECT * FROM read_csv_auto('{csv}');"
            )
    for table in NOTE_TABLES:
        csv = NOTE_DIR / f'{table}.csv'
        if csv.exists():
            lines.append(
                f"CREATE VIEW IF NOT EXISTS mimiciv_note.{table} AS SELECT * FROM read_csv_auto('{csv}');"
            )
    bhc_csv = BHC_DIR / 'mimic-iv-bhc.csv'
    if bhc_csv.exists():
        lines.append(
            f"CREATE VIEW IF NOT EXISTS mimiciv_note.bhc AS SELECT * FROM read_csv_auto('{bhc_csv}');"
        )
    INIT_SQL_PATH.write_text('\n'.join(lines) + '\n')


def run_sql_concept_script(con: duckdb.DuckDBPyConnection, *relative_paths: str):
    for rel in relative_paths:
        sql_path = DUCKDB_CONCEPTS_DIR / rel
        sql = sql_path.read_text()
        for statement in sql.split(';'):
            statement = statement.strip()
            if statement:
                con.execute(statement)


if __name__ == '__main__':
    setup_logging()
    generate_init_sql()
