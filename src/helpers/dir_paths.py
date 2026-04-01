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

# MIMIC-IV stuff
MIMIC_IV_DIR = DATASETS_DIR / 'mimic-iv'
HOSP_DIR = MIMIC_IV_DIR / 'hosp'
ICU_DIR = MIMIC_IV_DIR / 'icu'
NOTE_DIR = MIMIC_IV_DIR / 'note'

# MIT repo
BHC_DIR = DATASETS_DIR / 'mimic-note-Ext-BHC'

MIMIR_REPO_CODE_DIR = ROOT_DIR / 'src' / 'experiments' / 'mimic' / 'mimic_code'

# Experiment results dir
RESULTS_DIR = ROOT_DIR / 'results'
