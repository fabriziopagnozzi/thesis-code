"""
MIMIC-IV QA Benchmark pipeline.

Usage:
    python -m experiments.mimic.main 1 2 3 4  # run all phases
    python -m experiments.mimic.main 3 4      # run phases 3 and 4 only
"""

import argparse

from experiments.mimic.configs import setup_logging
from experiments.mimic.phase_1_chunking.orchestrator_1 import run_phase_1
from experiments.mimic.phase_2_embedding.a_embed import run_embed
from experiments.mimic.phase_3_queries.orchestrator_3 import run_phase_3
from experiments.mimic.phase_4_evaluation.a_evaluate import run_evaluate

PHASES = {
    1: lambda: run_phase_1(init_sql=True),
    2: run_embed,
    3: run_phase_3,
    4: run_evaluate,
}


if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description='MIMIC-IV QA Benchmark pipeline')
    parser.add_argument(
        'phases',
        type=int,
        nargs='*',
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help='Phase(s) to run (default: all)',
    )
    args = parser.parse_args()

    for phase_num in sorted(args.phases):
        print(f'\n{"=" * 40}')
        print(f'  Phase {phase_num}')
        print(f'{"=" * 40}\n')
        PHASES[phase_num]()

    print('\nDone.')
