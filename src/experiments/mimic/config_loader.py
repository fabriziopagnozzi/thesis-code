import yaml
from pathlib import Path

_MIMIC_DIR = Path(__file__).parent

_PHASE_DIRS = {
    1: 'phase_1_chunking',
    2: 'phase_2_embedding',
    3: 'phase_3_queries',
    4: 'phase_4_evaluation',
}


def load_phase_config(phase: int) -> dict:
    path = _MIMIC_DIR / _PHASE_DIRS[phase] / f'phase_{phase}_config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def load_global_config() -> dict:
    with open(_MIMIC_DIR / 'global_config.yaml') as f:
        return yaml.safe_load(f)
