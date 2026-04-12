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

# MIMIC-IV code dirs
MIMIC_IV_DIR = ROOT_DIR / 'src' / 'experiments' / 'mimic'
MIMIR_REPO_CODE_DIR = MIMIC_IV_DIR / 'mimic_code'  # MIT Repo with utils

# MIMIC-IV data
HOSP_DIR = DATASETS_DIR / 'mimic-iv' / 'hosp'
ICU_DIR = DATASETS_DIR / 'mimic-iv' / 'icu'
NOTE_DIR = DATASETS_DIR / 'mimic-iv' / 'note'
BHC_DIR = DATASETS_DIR / 'mimic-iv' / 'ext-bhc'
