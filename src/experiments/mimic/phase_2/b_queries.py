"""
Step 2.2: Generate multi-aspect queries via enumeration.

For each selected condition, picks modifier axes (comorbidity, demographic) and generates queries using multiple LLM personas.
"""

import itertools

import polars as pl

from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR, connect, run_mimic_code_sql

PERSONAS = {
    'clinician': (
        'You are an experienced attending physician reviewing patient cases. '
        'Generate a clinical management question about {condition} that requires '
        'knowing how treatment or workup changes when {modifier} is present. '
        'The question should require synthesizing information from multiple '
        'patient cases, not textbook symptom lists. Focus on practical management '
        'decisions (drug choice, dosing, monitoring, disposition).'
    ),
    'researcher': (
        'You are a clinical researcher analyzing trends in hospital discharge data. '
        'Generate a research-oriented question about {condition} that investigates '
        'how outcomes or management patterns differ when {modifier} is present. '
        'The question should be answerable by synthesizing discharge summaries '
        'from multiple patients, focusing on observable patterns in care delivery.'
    ),
    'neutral': (
        'Generate a clinical management question about {condition} that requires '
        'knowing how treatment or workup changes when {modifier} is present. '
        'The answer should require synthesizing management decisions from multiple '
        'patient cases, not textbook symptom lists.'
    ),
}


# -- Modifier axis generation --

# Charlson category column names → human-readable labels
CHARLSON_LABELS = {
    'myocardial_infarct': 'prior myocardial infarction',
    'congestive_heart_failure': 'congestive heart failure',
    'peripheral_vascular_disease': 'peripheral vascular disease',
    'cerebrovascular_disease': 'cerebrovascular disease',
    'dementia': 'dementia',
    'chronic_pulmonary_disease': 'chronic pulmonary disease (COPD)',
    'rheumatic_disease': 'rheumatic disease',
    'peptic_ulcer_disease': 'peptic ulcer disease',
    'mild_liver_disease': 'mild liver disease',
    'diabetes_without_cc': 'diabetes without complications',
    'diabetes_with_cc': 'diabetes with chronic complications',
    'paraplegia': 'hemiplegia or paraplegia',
    'renal_disease': 'chronic kidney disease',
    'malignant_cancer': 'malignancy',
    'severe_liver_disease': 'severe liver disease (cirrhosis)',
    'metastatic_solid_tumor': 'metastatic cancer',
    'aids': 'HIV/AIDS',
}

DEMOGRAPHIC_MODIFIERS = [
    'the patient is elderly (age > 75)',
    'the patient is a young adult (age < 40)',
]


def find_top_comorbidity_modifiers(
    con, condition_icd3: str, n: int = 3
) -> list[tuple[str, str, float]]:
    """Find the most common co-occurring Charlson categories for a condition.

    Returns list of (charlson_col, human_label, co_occurrence_rate).
    """
    charlson_cols = list(CHARLSON_LABELS.keys())
    col_list = ', '.join(f'AVG(ch.{c}) AS {c}' for c in charlson_cols)

    rates = con.execute(f"""
        SELECT {col_list}
        FROM mimiciv_hosp.diagnoses_icd di
        JOIN mimiciv_derived.charlson ch ON di.hadm_id = ch.hadm_id
        WHERE di.icd_version = 10 AND SUBSTR(di.icd_code, 1, 3) = '{condition_icd3}'
    """).fetchone()

    scored = []
    for col, rate in zip(charlson_cols, rates, strict=False):
        if rate and rate > 0.05:  # at least 5% co-occurrence
            scored.append((col, CHARLSON_LABELS[col], float(rate)))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:n]


def build_query_specs(conditions: pl.DataFrame, con, max_modifiers: int = 3) -> list[dict]:
    """Build query specifications: (condition, modifier, persona) triples."""
    specs = []

    for row in conditions.head(100).iter_rows(named=True):
        icd3 = row['icd10_3char']
        condition_name = row['condition_name'] or icd3

        # Comorbidity modifiers
        comorbidity_mods = find_top_comorbidity_modifiers(con, icd3, n=max_modifiers)

        modifiers = [(label, 'comorbidity') for _, label, _ in comorbidity_mods]
        modifiers.extend([(m, 'demographic') for m in DEMOGRAPHIC_MODIFIERS])

        for (modifier_text, modifier_type), persona_name in itertools.product(
            modifiers, PERSONAS.keys()
        ):
            specs.append(
                {
                    'icd10_3char': icd3,
                    'condition_name': condition_name,
                    'modifier_text': modifier_text,
                    'modifier_type': modifier_type,
                    'persona': persona_name,
                    'prompt': PERSONAS[persona_name].format(
                        condition=condition_name, modifier=modifier_text
                    ),
                }
            )

    return specs


def main():
    con = connect()
    run_mimic_code_sql(con, 'demographics/age.sql', 'comorbidity/charlson.sql')

    conditions = pl.read_parquet(MIMIC_RESULTS_DIR / 'conditions.parquet')
    print(f'Building query specs for top {min(100, len(conditions))} conditions...')

    specs = build_query_specs(conditions, con)
    df = pl.DataFrame(specs)

    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(MIMIC_RESULTS_DIR / 'query_specs.parquet')

    print(f'Generated {len(df):,} query specs')
    print(f'  Conditions: {df["icd10_3char"].n_unique()}')
    print(f'  Personas: {df["persona"].value_counts()}')
    print(f'  Modifier types: {df["modifier_type"].value_counts()}')
    print(f'\nSample prompt:\n  {df["prompt"][0]}')


if __name__ == '__main__':
    main()
