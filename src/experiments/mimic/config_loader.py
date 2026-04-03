import argparse
from pathlib import Path

import yaml

_MIMIC_DIR = Path(__file__).parent

_PHASE_DIRS = {
    1: 'phase_1_chunking',
    2: 'phase_2_embedding',
    3: 'phase_3_queries',
    4: 'phase_4_evaluation',
}


def load_phase_config(phase: int, path: str | Path | None = None) -> dict:
    if path is None:
        path = _MIMIC_DIR / _PHASE_DIRS[phase] / f'phase_{phase}_config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def load_global_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = _MIMIC_DIR / 'global_config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def parse_config_arg(phase: int) -> dict:
    """Parse --config from sys.argv and return the loaded config for the given phase."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None, help='Path to a custom YAML config file')
    args, _ = parser.parse_known_args()
    return load_phase_config(phase, path=args.config)
