from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
    generate_init_sql,
    register_result_view,
)

from .a_conditions_stats import select_conditions
from .b_note_chunking import parse_all_notes
from .c_add_metadata import build_admissions_metadata
from .d_dedup import deduplicate

MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    generate_init_sql(force=True)
    con = connect_mimic_duckdb()

    print('\n> Step 1.1: Condition selection')
    from .a_conditions_stats import _cfg as _conditions_cfg
    conditions = select_conditions(con, min_admissions=_conditions_cfg['min_admissions'])
    conditions.write_parquet(MIMIC_RESULTS_DIR / 'conditions_stats.parquet')
    register_result_view(con, 'conditions', conditions)

    print('\n> Step 1.2: Parsing discharge notes')
    chunks = parse_all_notes(con, output_path=MIMIC_RESULTS_DIR / 'chunks.parquet')
    register_result_view(con, 'chunks', chunks)

    print('\n> Step 1.3: Building admissions metadata')
    metadata = build_admissions_metadata(con)
    metadata.write_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')
    register_result_view(con, 'admissions_metadata', metadata)

    print('\n> Step 1.4: Deduplication')
    chunks = deduplicate(chunks)
    chunks.write_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')
    register_result_view(con, 'chunks', chunks)

    print(
        f'\n\nPhase 1 complete. Outputs in {MIMIC_RESULTS_DIR}:\n'
        f'  conditions_stats.parquet:          {len(conditions):>10,} conditions\n'
        f'  admissions_metadata.parquet: {len(metadata):>10,} admissions\n'
        f'  chunks.parquet:              {len(chunks):>10,} chunks'
    )
