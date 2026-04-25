from typing import Literal, get_args

type CharlsonLabel = Literal[
    'myocardial_infarct',
    'congestive_heart_failure',
    'peripheral_vascular_disease',
    'cerebrovascular_disease',
    'dementia',
    'chronic_pulmonary_disease',
    'rheumatic_disease',
    'peptic_ulcer_disease',
    'mild_liver_disease',
    'severe_liver_disease',
    'diabetes_without_cc',
    'diabetes_with_cc',
    'paraplegia',
    'renal_disease',
    'malignant_cancer',
    'metastatic_solid_tumor',
    'aids',
]

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

type NoteTable = Literal['discharge', 'discharge_detail', 'radiology', 'radiology_detail']

type ResultTable = Literal[
    'conditions_stats',
    'condition_filtering',
    'admissions_metadata',
    'chunks',
    'queries_prompts',
    'queries',
    'divergence_stats',
    'gold_annotations',
    'evaluation_results',
    'evaluation_stats',
    'evaluation_best_per_metric',
    'evaluation_best_per_metric_fixed_lam',
]

type MimicTable = HospTable | IcuTable | NoteTable | ResultTable

CHARLSON_LABELS_TO_STR: dict[CharlsonLabel, str] = {
    'myocardial_infarct': 'prior myocardial infarction',
    'congestive_heart_failure': 'congestive heart failure',
    'peripheral_vascular_disease': 'peripheral vascular disease',
    'cerebrovascular_disease': 'cerebrovascular disease',
    'dementia': 'dementia',
    'chronic_pulmonary_disease': 'chronic pulmonary disease (COPD)',
    'rheumatic_disease': 'rheumatic disease',
    'peptic_ulcer_disease': 'peptic ulcer disease',
    'mild_liver_disease': 'mild liver disease',
    'severe_liver_disease': 'severe liver disease (cirrhosis)',
    'diabetes_without_cc': 'diabetes without complications',
    'diabetes_with_cc': 'diabetes with chronic complications',
    'paraplegia': 'hemiplegia or paraplegia',
    'renal_disease': 'chronic kidney disease',
    'malignant_cancer': 'malignancy',
    'metastatic_solid_tumor': 'metastatic cancer',
    'aids': 'HIV/AIDS',
}
HOSP_TABLES = set(get_args(HospTable))
ICU_TABLES = set(get_args(IcuTable))
NOTE_TABLES = set(get_args(NoteTable))
RESULT_TABLES = set(get_args(ResultTable))
ALL_TABLES = HOSP_TABLES | ICU_TABLES | NOTE_TABLES | RESULT_TABLES
