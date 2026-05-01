from typing import Literal, get_args

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
    'gold_annotations',
    'gold_answers',
    'evaluation_results',
    'evaluation_stats',
    'evaluation_stats_by_stratum',
    'evaluation_best_per_metric',
    'evaluation_best_per_metric_fixed_lam',
]

type MimicTable = HospTable | IcuTable | NoteTable | ResultTable


HOSP_TABLES: set[HospTable] = set(get_args(HospTable.__value__))
ICU_TABLES: set[IcuTable] = set(get_args(IcuTable.__value__))
NOTE_TABLES: set[NoteTable] = set(get_args(NoteTable.__value__))
RESULT_TABLES: set[ResultTable] = set(get_args(ResultTable.__value__))

ALL_TABLES = HOSP_TABLES | ICU_TABLES | NOTE_TABLES | RESULT_TABLES

# From the MIT repo
DERIVED_CONCEPTS: dict[MimicTable, str] = {
    'age': 'demographics/age.sql',
    'charlson': 'comorbidity/charlson.sql',
}
