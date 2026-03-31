import duckdb

from helpers.dir_paths import HOSP_DIR, ICU_DIR, MIMIC_CODE_DIR, NOTE_DIR, ROOT_DIR

DUCKDB_CONCEPTS_DIR = MIMIC_CODE_DIR / 'mimic-iv' / 'concepts_duckdb'
MIMIC_RESULTS_DIR = ROOT_DIR / 'results' / 'mimic'

HOSP_TABLES = [
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
]
ICU_TABLES = [
    'caregiver',
    'chartevents',
    'd_items',
    'datetimeevents',
    'icustays',
    'ingredientevents',
    'inputevents',
    'outputevents',
    'procedureevents',
]
NOTE_TABLES = ['discharge', 'discharge_detail', 'radiology', 'radiology_detail']


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_hosp')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_icu')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_note')
    con.execute('CREATE SCHEMA IF NOT EXISTS mimiciv_derived')

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

    return con


def run_mimic_code_sql(con: duckdb.DuckDBPyConnection, *relative_paths: str):
    for rel in relative_paths:
        sql_path = DUCKDB_CONCEPTS_DIR / rel
        sql = sql_path.read_text()
        for statement in sql.split(';'):
            statement = statement.strip()
            if statement:
                con.execute(statement)
