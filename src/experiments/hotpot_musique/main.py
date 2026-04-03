import argparse

from .config import ExperimentConfig
from .experiment import run_experiment, summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Coverage vs Dispersion diversity experiments')

    # Config file (optional - overrides defaults, CLI overrides config)
    p.add_argument('--config', type=str, help='JSON config file path')

    p.add_argument(
        '--dataset',
        type=str,
        choices=['hotpotqa_distractor', 'musique', '2wikimultihopqa'],
    )
    p.add_argument('--dataset-path', type=str)
    p.add_argument(
        '--chunk-mode',
        choices=['sentence', 'document', 'word_window', 'token_window'],
    )
    p.add_argument(
        '--w',
        type=int,
        help='Chunk size in words (word_window mode; Table 6: w in {100, 200})',
    )
    p.add_argument(
        '--chunk-tokens',
        type=int,
        help='Chunk size in LLM tokens (token_window mode; Table 1: ch)',
    )
    p.add_argument(
        '--stride',
        type=int,
        help='Window overlap in tokens (token_window mode; Table 1: st)',
    )
    p.add_argument('--t-max', type=int, help='Token budget for selection')

    p.add_argument('--embedding-model', type=str)
    p.add_argument('--device', type=str)

    p.add_argument('--scoring-functions', nargs='+', type=str)
    p.add_argument('--k-values', nargs='+', type=int)
    p.add_argument('--lambda-values', nargs='+', type=float)
    p.add_argument('--mmr-window', type=int)

    # Pool limits
    p.add_argument('--max-docs', type=int)
    p.add_argument('--max-cands', type=int)

    # Infrastructure
    p.add_argument('--max-records', type=int, help='Limit records (for debugging)')
    p.add_argument('--cache-dir', type=str)
    p.add_argument('--output-dir', type=str)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int)

    # Output control
    p.add_argument('--no-summary', action='store_true', help='Skip printing summary table')

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()

    for field, val in vars(args).items():
        if field not in ('config', 'no_summary') and val is not None:
            setattr(cfg, field, val)

    if args.dataset is not None and args.dataset_path is None:
        default_paths = {
            'hotpotqa_distractor': 'datasets/full-data/HotpotQA/hotpot_dev_distractor_v1.json',
            'musique': 'datasets/full-data/MuSiQue/data/musique_ans_v1.0_dev.jsonl',
            '2wikimultihopqa': 'datasets/full-data/2WikiMultihopQA/data/dev.json',
        }
        cfg.dataset_path = default_paths[args.dataset]

    from pprint import pprint

    pprint(vars(cfg))
    print()

    df = run_experiment(cfg)

    if not args.no_summary:
        summary = summarize(df)
        print('\nAggregated Results\n', summary)


if __name__ == '__main__':
    main()
