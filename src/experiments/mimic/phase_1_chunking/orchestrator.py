import duckdb

from experiments.mimic.configs import ConditionsStatsCfg, DedupCfg, NoteChunkingCfg
from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
    generate_init_sql,
    register_result_view,
)

from .a_conditions_stats import run_select_conditions
from .b_note_chunking import run_note_chunking
from .c_add_metadata import run_add_metadata
from .d_dedup import run_dedup

MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase_1(
    con: duckdb.DuckDBPyConnection | None = None,
    conditions_cfg: ConditionsStatsCfg | None = None,
    note_chunking_cfg: NoteChunkingCfg | None = None,
    dedup_cfg: DedupCfg | None = None,
):
    if con is None:
        generate_init_sql(force=True)
        con = connect_mimic_duckdb()

    print('\n> Step 1.1: Condition selection')
    conditions = run_select_conditions(con, cfg=conditions_cfg)
    register_result_view(con, 'conditions', conditions)

    print('\n> Step 1.2: Parsing discharge notes')
    chunks = run_note_chunking(con, cfg=note_chunking_cfg)
    register_result_view(con, 'chunks', chunks)

    print('\n> Step 1.3: Building admissions metadata')
    metadata = run_add_metadata(con)
    register_result_view(con, 'admissions_metadata', metadata)

    print('\n> Step 1.4: Deduplication')
    chunks = run_dedup(cfg=dedup_cfg)
    register_result_view(con, 'chunks', chunks)

    print(
        f'\n\nPhase 1 complete. Outputs in {MIMIC_RESULTS_DIR}:\n'
        f'  conditions_stats.parquet:    {len(conditions):>10,} conditions\n'
        f'  admissions_metadata.parquet: {len(metadata):>10,} admissions\n'
        f'  chunks.parquet:              {len(chunks):>10,} chunks'
    )


if __name__ == '__main__':
    run_phase_1()
