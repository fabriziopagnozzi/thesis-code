from os import getenv
from typing import Literal, get_args

from helpers.dir_paths import DATASETS_DIR, ROOT_DIR, THIRDPARTY_CODE_DIR


class MimicPaths:
    exp_name = getenv('EXP_NAME', 'default_experiment')
    root = ROOT_DIR / 'src' / 'experiments' / 'mimic'
    results = root / '_results'

    vector_db = results / '_vector_db'
    experiment = results / exp_name
    config = experiment / '_config.yaml'
    logs = experiment / '_logs'

    init_sql = root / '_mimic_init.sql'
    duckdb_concepts = THIRDPARTY_CODE_DIR / 'mimic_code' / 'mimic-iv' / 'concepts_duckdb'

    hosp = DATASETS_DIR / 'mimic-iv' / 'hosp'
    icu = DATASETS_DIR / 'mimic-iv' / 'icu'
    note = DATASETS_DIR / 'mimic-iv' / 'note'
    bhc = DATASETS_DIR / 'mimic-iv' / 'ext-bhc'


for p in [MimicPaths.results, MimicPaths.logs, MimicPaths.vector_db]:
    p.mkdir(parents=True, exist_ok=True)


type HospTable = Literal[
    'admissions',
    'd_hcpcs',
    'd_icd_diagnoses',
    'd_icd_procedures',
    'd_labitems',
    'diagnoses_icd',
    'drgcodes',
    'emar',
    'emar_detail',
    'hcpcsevents',
    'labevents',
    'microbiologyevents',
    'omr',
    'patients',
    'pharmacy',
    'poe',
    'poe_detail',
    'prescriptions',
    'procedures_icd',
    'provider',
    'services',
    'transfers',
]

type IcuTable = Literal[
    'caregiver',
    'chartevents',
    'd_items',
    'datetimeevents',
    'icustays',
    'ingredientevents',
    'inputevents',
    'outputevents',
    'procedureevents',
]

type NoteTable = Literal[
    'discharge',
    'discharge_detail',
    'radiology',
    'radiology_detail',
]

type ResultTable = Literal[
    # shared across all runs ↓
    'admissions_metadata',
    'age',
    'charlson',
    'icd9_to_icd10_cm_gem',
    'unified_diagnoses',
    # per-experiment ↓
    'conditions_stats',
    'condition_filtering',
    'chunks',
    'queries_prompts',
    'queries',
    'divergence_stats',
    'gold_annotations',
    'evaluation_results',
    'evaluation_stats',
    'evaluation_stats_by_stratum',
    'evaluation_best_per_metric',
    'evaluation_best_per_metric_fixed_lam',
]

type MimicTable = HospTable | IcuTable | NoteTable | ResultTable


HOSP_TABLES = set(get_args(HospTable))
ICU_TABLES = set(get_args(IcuTable))
NOTE_TABLES = set(get_args(NoteTable))
RESULT_TABLES = set(get_args(ResultTable))

ALL_TABLES = HOSP_TABLES | ICU_TABLES | NOTE_TABLES | RESULT_TABLES
