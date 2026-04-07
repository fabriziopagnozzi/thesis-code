import argparse
from pathlib import Path

import yaml

_MIMIC_DIR = Path(__file__).parent
path = _MIMIC_DIR / 'global_config.yaml'
with open(path) as f:
    _cfg = yaml.safe_load(f)

N_CONDITIONS: int = _cfg['n_conditions']

_PHASE_DIRS = {
    1: 'phase_1_chunking',
    2: 'phase_2_embedding',
    3: 'phase_3_queries',
    4: 'phase_4_evaluation',
}


def load_config(phase: int, path: str | Path | None = None) -> dict:
    if path is None:
        path = _MIMIC_DIR / _PHASE_DIRS[phase] / f'phase_{phase}_config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def load_config_from_main(phase: int) -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--config', type=str, default=None, help='Path to a custom YAML config file'
    )
    args, _ = parser.parse_known_args()
    return load_config(phase, path=args.config)
