import duckdb

from helpers.dir_paths import BHC_DIR, HOSP_DIR, ICU_DIR, MIMIR_REPO_CODE_DIR, NOTE_DIR, RESULTS_DIR

INIT_SQL_PATH = MIMIR_REPO_CODE_DIR.parent / '_mimic_init.sql'
DUCKDB_CONCEPTS_DIR = MIMIR_REPO_CODE_DIR / 'mimic-iv' / 'concepts_duckdb'
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

    # Result parquets are dynamic
    for table in RESULT_TABLES:
        parquet_file = MIMIC_RESULTS_DIR / f'{table}.parquet'
        if parquet_file.exists():
            con.execute(
                f"CREATE VIEW IF NOT EXISTS mimic_results.{table} AS SELECT * FROM read_parquet('{parquet_file}')"
            )

    return con


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
    generate_init_sql()
