import argparse
from pathlib import Path

import yaml

from helpers.dir_paths import MIMIC_IV_DIR

from .config_models import GlobalCfg

_PHASE_DIRS = {
    1: 'phase_1_chunking',
    2: 'phase_2_embedding',
    3: 'phase_3_queries',
    4: 'phase_4_evaluation',
}


def load_global_cfg(path: Path = MIMIC_IV_DIR / 'global_config.yml', cfg: GlobalCfg | None = None):
    global global_cfg

    if cfg is not None:
        global_cfg = cfg
        return

    with open(path) as f:
        _loaded_global_cfg = yaml.safe_load(f)

    global_cfg = GlobalCfg(
        num_conditions=_loaded_global_cfg['n_conditions'],
        results_subdir=_loaded_global_cfg.get('results_subdir'),
    )


load_global_cfg()


def load_default_config(phase: int, path: str | Path | None = None) -> dict:
    if path is None:
        path = MIMIC_IV_DIR / _PHASE_DIRS[phase] / f'phase_{phase}_config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def load_config_from_main(phase: int) -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--config', type=str, default=None, help='Path to a custom YAML config file'
    )
    args, _ = parser.parse_known_args()
    return load_default_config(phase, path=args.config)
