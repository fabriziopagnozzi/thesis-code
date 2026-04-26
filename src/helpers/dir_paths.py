from os import getenv
from pathlib import Path


def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    raise RuntimeError('project root not found')


ROOT_DIR = find_project_root()

DATASETS_DIR = (
    ROOT_DIR / 'data' / 'full-data'
    if getenv('LOCAL_MACHINE', 'false') == 'true'
    else Path('/DATA/pagnozzi/datasets')
)

THIRDPARTY_CODE_DIR = ROOT_DIR / 'src' / 'thirdparty'
