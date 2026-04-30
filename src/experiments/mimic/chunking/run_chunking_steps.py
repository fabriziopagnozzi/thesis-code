import argparse

from duckdb import DuckDBPyConnection

from experiments.mimic.chunking.schemas_chunking import (
    ConditionsStatsCfg,
    DedupCfg,
    NoteChunkingCfg,
)
from experiments.mimic.global_configs import MimicPaths, duckdb_con, setup_logging
from experiments.mimic.utils.duck_db_init import (
    register_result_view,
)

from .a_conditions_stats import run_conditions_stats
from .b_note_chunking import run_note_chunking
from .c_dedup_chunks import run_dedup

STEPS = (1, 2, 3)


def run_chunking_subpipeline(
    con: DuckDBPyConnection = duckdb_con,
    conditions_cfg: ConditionsStatsCfg | None = None,
    note_chunking_cfg: NoteChunkingCfg | None = None,
    dedup_cfg: DedupCfg | None = None,
    steps: list[int] | None = None,
):
    if steps is None:
        steps = list(STEPS)

    if 1 in steps:
        print('\n> Step 1.1: Condition selection')
        conditions = run_conditions_stats(con, cfg=conditions_cfg)
        register_result_view(con, 'conditions_stats', conditions)

    if 2 in steps:
        print('\n> Step 1.2: Parsing discharge notes')
        chunks = run_note_chunking(con, cfg=note_chunking_cfg)
        register_result_view(con, 'chunks', chunks)

    if 3 in steps:
        print('\n> Step 1.3: Deduplication')
        chunks = run_dedup(cfg=dedup_cfg)
        register_result_view(con, 'chunks', chunks)

    print(f'\n\nPhase 1 complete. Outputs in {MimicPaths.experiment_dir}:\n')


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=int,
        nargs='*',
        default=list(STEPS),
        choices=STEPS,
        help='Subphase(s) to run (default: all)',
    )
    args = parser.parse_args()

    run_chunking_subpipeline(steps=sorted(args.steps))
