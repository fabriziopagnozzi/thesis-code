import duckdb

from helpers.dir_paths import HOSP_DIR, ICU_DIR, MIMIC_CODE_DIR, NOTE_DIR, RESULTS_DIR

DUCKDB_CONCEPTS_DIR = MIMIC_CODE_DIR / 'mimic-iv' / 'concepts_duckdb'
MIMIC_RESULTS_DIR = RESULTS_DIR / 'mimic'

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

# Resulting parquet files saved under RESULTS_DIR
RESULT_TABLES = {'conditions', 'admissions_metadata', 'chunks'}


def connect_mimic_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_hosp')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_icu')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_note')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_derived')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimic_results')

    for table in HOSP_TABLES:
        csv = HOSP_DIR / f'{table}.csv'
        if csv.exists():
            con.execute(f"CREATE VIEW mimiciv_hosp.{table} AS SELECT * FROM read_csv_auto('{csv}')")

    for table in ICU_TABLES:
        csv = ICU_DIR / f'{table}.csv'
        if csv.exists():
            con.execute(f"CREATE VIEW mimiciv_icu.{table} AS SELECT * FROM read_csv_auto('{csv}')")

    for table in NOTE_TABLES:
        csv = NOTE_DIR / f'{table}.csv'
        if csv.exists():
            con.execute(f"CREATE VIEW mimiciv_note.{table} AS SELECT * FROM read_csv_auto('{csv}')")

    for table in RESULT_TABLES:
        parquet = MIMIC_RESULTS_DIR / f'{table}.parquet'
        if parquet.exists():
            con.execute(
                f"CREATE VIEW mimic_results.{table} AS SELECT * FROM read_parquet('{parquet}')"
            )

    con.execute(
        "SET search_path = 'mimiciv_hosp,mimiciv_icu,mimiciv_note,mimiciv_derived,mimic_results'"
    )

    return con


def run_sql_concept_script(con: duckdb.DuckDBPyConnection, *relative_paths: str):
    for rel in relative_paths:
        sql_path = DUCKDB_CONCEPTS_DIR / rel
        sql = sql_path.read_text()
        for statement in sql.split(';'):
            statement = statement.strip()
            if statement:
                con.execute(statement)
