import argparse

import duckdb

from experiments.mimic.configs import (
    MIMIC_EXPERIMENT_DIR,
    ConditionsStatsCfg,
    DedupCfg,
    NoteChunkingCfg,
    setup_logging,
)
from experiments.mimic.duck_db_init import (
    connect_mimic_duckdb,
    generate_init_sql,
    register_result_view,
)

from .a_conditions_stats import run_conditions_stats
from .b_note_chunking import run_note_chunking
from .c_dedup import run_dedup

MIMIC_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def run_phase_1(
    con: duckdb.DuckDBPyConnection | None = None,
    conditions_cfg: ConditionsStatsCfg | None = None,
    note_chunking_cfg: NoteChunkingCfg | None = None,
    dedup_cfg: DedupCfg | None = None,
    from_step: int = 1,
    *,
    init_sql: bool,
):
    if con is None:
        if from_step == 1 or init_sql:
            generate_init_sql(force=True)
        con = connect_mimic_duckdb()

    if from_step <= 1:
        print('\n> Step 1.1: Condition selection')
        conditions = run_conditions_stats(con, cfg=conditions_cfg)
        register_result_view(con, 'conditions_stats', conditions)

    if from_step <= 2:
        print('\n> Step 1.2: Parsing discharge notes')
        chunks = run_note_chunking(con, cfg=note_chunking_cfg)
        register_result_view(con, 'chunks', chunks)

    if from_step <= 3:
        print('\n> Step 1.4: Deduplication')
        chunks = run_dedup(cfg=dedup_cfg)
        register_result_view(con, 'chunks', chunks)

    print(f'\n\nPhase 1 complete. Outputs in {MIMIC_EXPERIMENT_DIR}:\n')


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init-sql',
        dest='init_sql',
        type=bool,
        default=True,
        help='Whether to init the _mimic_init.sql file',
    )
    parser.add_argument(
        '--from',
        dest='from_step',
        type=int,
        default=1,
        metavar='STEP',
        help='Resume from this step number (1-3)',
    )
    args = parser.parse_args()

    run_phase_1(from_step=args.from_step, init_sql=args.init_sql)
