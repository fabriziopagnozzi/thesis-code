from pathlib import Path


def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    raise RuntimeError('project root not found')


ROOT_DIR = find_project_root()

DATASETS_DIR = ROOT_DIR / 'datasets' / 'full-data'
DATASETS_PREVIEW_DIR = ROOT_DIR / 'datasets' / 'previews'

EMBEDDINGS_CACHE_DIR = ROOT_DIR / '.cached_embeddings'

# MIMIC-IV data
HOSP_DIR = DATASETS_DIR / 'mimic-iv' / 'hosp'
ICU_DIR = DATASETS_DIR / 'mimic-iv' / 'icu'
NOTE_DIR = DATASETS_DIR / 'mimic-iv' / 'note'
BHC_DIR = DATASETS_DIR / 'mimic-iv' / 'ext-bhc'

# MIT repo
MIMIR_REPO_CODE_DIR = ROOT_DIR / 'src' / 'experiments' / 'mimic' / 'mimic_code'
