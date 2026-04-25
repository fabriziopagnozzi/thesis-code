import argparse

from experiments.mimic.chunking.run_chunking_steps import run_chunking_subpipeline
from experiments.mimic.configs import setup_logging
from experiments.mimic.embeddings.embed_whole_corpus import run_embed
from experiments.mimic.evaluation.evaluate import run_evaluate
from experiments.mimic.queries.run_queries_steps import run_queries_subpipeline

PHASES = {
    1: lambda: run_chunking_subpipeline(init_sql=True),
    2: run_embed,
    3: run_queries_subpipeline,
    4: run_evaluate,
}

# TODO: don't use run_embed

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description='MIMIC-IV QA Benchmark pipeline')
    parser.add_argument(
        '--phases',
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
