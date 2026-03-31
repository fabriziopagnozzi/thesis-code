from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR, connect, run_mimic_code_sql

from .a_conditions_stats import select_conditions
from .b_note_chunking import parse_all_notes
from .c_add_metadata import build_admissions_metadata
from .d_dedup import deduplicate

MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    con = connect()

    print('> Building mimic-code derived tables')
    run_mimic_code_sql(con, 'demographics/age.sql', 'comorbidity/charlson.sql')

    print('\n> Step 1.1: Condition selection')
    conditions = select_conditions(con=con)
    conditions.write_parquet(MIMIC_RESULTS_DIR / 'conditions.parquet')

    print('\n> Step 1.2: Parsing discharge notes')
    chunks = parse_all_notes(con)

    print('\n> Step 1.3: Building admissions metadata')
    metadata = build_admissions_metadata(con)
    metadata.write_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')

    print('\n> Step 1.4: Deduplication')
    chunks = deduplicate(chunks, metadata)
    chunks.write_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')

    print(f'\n\nPhase 1 complete. Outputs in {MIMIC_RESULTS_DIR}:')
    print(f'  conditions.parquet:          {len(conditions):>10,} conditions')
    print(f'  admissions_metadata.parquet: {len(metadata):>10,} admissions')
    print(f'  chunks.parquet:              {len(chunks):>10,} chunks')
