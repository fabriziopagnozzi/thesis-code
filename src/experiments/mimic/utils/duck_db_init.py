import duckdb

from experiments.mimic.configs import (
    get_table_path,
    setup_logging,
)
from experiments.mimic.utils.constants import (
    HOSP_TABLES,
    ICU_TABLES,
    NOTE_TABLES,
    RESULT_TABLES,
    MimicPaths,
    MimicTable,
)

# From the MIT repo
DERIVED_CONCEPTS: dict[MimicTable, str] = {
    'age': 'demographics/age.sql',
    'charlson': 'comorbidity/charlson.sql',
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
    for statement in MimicPaths.init_sql.read_text().splitlines():
        statement = statement.strip()
        if statement:
            con.execute(statement)

    for table in RESULT_TABLES:
        parquet_file = get_table_path(table)
        if parquet_file.exists():
            con.execute(
                f"CREATE VIEW IF NOT EXISTS mimic_results.{table} AS SELECT * FROM read_parquet('{parquet_file}')"
            )

    _load_derived_concepts(con)
    _ensure_unified_diagnoses(con)

    return con


def register_result_view(con: duckdb.DuckDBPyConnection, name: str, df) -> None:
    con.execute(f'DROP VIEW IF EXISTS mimic_results.{name}')
    con.register(name, df)


def generate_init_sql(force: bool = False):
    if MimicPaths.init_sql.exists() and not force:
        return
    lines: list[str] = [INIT_SCHEMAS]

    for table in HOSP_TABLES:
        csv = MimicPaths.hosp / f'{table}.csv'
        if csv.exists():
            lines.append(
                f"CREATE VIEW IF NOT EXISTS mimiciv_hosp.{table} AS SELECT * FROM read_csv_auto('{csv}');"
            )

    for table in ICU_TABLES:
        csv = MimicPaths.icu / f'{table}.csv'
        if csv.exists():
            lines.append(
                f"CREATE VIEW IF NOT EXISTS mimiciv_icu.{table} AS SELECT * FROM read_csv_auto('{csv}');"
            )

    for table in NOTE_TABLES:
        csv = MimicPaths.note / f'{table}.csv'
        if csv.exists():
            lines.append(
                f"CREATE VIEW IF NOT EXISTS mimiciv_note.{table} AS SELECT * FROM read_csv_auto('{csv}');"
            )

    bhc_csv = MimicPaths.bhc / 'mimic-iv-bhc.csv'
    if bhc_csv.exists():
        lines.append(
            f"CREATE VIEW IF NOT EXISTS mimiciv_note.bhc AS SELECT * FROM read_csv_auto('{bhc_csv}');"
        )

    MimicPaths.init_sql.write_text('\n'.join(lines) + '\n')


def run_sql_concept_script(con: duckdb.DuckDBPyConnection, *relative_paths: str):
    for rel in relative_paths:
        sql_path = MimicPaths.duckdb_concepts / rel
        sql = sql_path.read_text()
        for statement in sql.split(';'):
            statement = statement.strip()
            if statement:
                con.execute(statement)


def _load_derived_concepts(con: duckdb.DuckDBPyConnection):
    """Load derived concept tables (age, charlson) from parquet or compute and save them."""
    for table, sql_rel in DERIVED_CONCEPTS.items():
        parquet_path = get_table_path(table)
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
    """Materialize unified_diagnoses (ICD-9 + ICD-10 → ICD-10) once, then register as view."""
    parquet_path = get_table_path('unified_diagnoses')
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
                        FROM read_parquet('{get_table_path('icd9_to_icd10_cm_gem')}')
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


if __name__ == '__main__':
    setup_logging()
    generate_init_sql()
