import polars as pl

from experiments.mimic.embeddings.schemas_embeddings import EmbedJoinedRow
from experiments.mimic.global_configs import read_parquet
from experiments.mimic.utils.charlson import CHARLSON_LABELS
from experiments.mimic.utils.utils import get_age_group, get_charlson_conditions


def add_contextual_prefix_to_df(chunks: pl.DataFrame) -> pl.DataFrame:
    meta_cols = [
        'hadm_id',
        'age',
        'gender',
        'race',
        'primary_icd_description',
        'top_icd_descriptions',
        'charlson_comorbidity_index',
        'admission_type',
        *CHARLSON_LABELS,
    ]
    metadata = read_parquet('admissions_metadata').select(meta_cols).unique(subset=['hadm_id'])
    cols_to_drop = [c for c in meta_cols if c != 'hadm_id']
    return (
        chunks.join(metadata, on='hadm_id', how='left')
        .with_columns(
            contextual_prefix=pl.struct(pl.all()).map_elements(
                build_contextual_prefix, return_dtype=pl.Utf8
            )
        )
        .drop(cols_to_drop)
    )


def build_contextual_prefix(meta_row: EmbedJoinedRow) -> str:
    age = meta_row.get('age')
    if age is not None:
        age_grp = get_age_group(age)
        article = 'an' if age_grp[0] in 'aeiou' else 'a'
        age_part = f'{article} {age_grp} {int(age)}-year-old'
    else:
        age_part = 'a'

    gender = meta_row.get('gender', '')
    if gender == 'F':
        gender_noun, pronoun = 'woman', 'She'
    elif gender == 'M':
        gender_noun, pronoun = 'man', 'He'
    else:
        gender_noun, pronoun = 'patient', 'The patient'

    race = meta_row.get('race', 'unknown')
    primary_dx = meta_row.get('primary_icd_description', 'unknown condition')
    adverb = _admission_adverb(meta_row.get('admission_type'))

    prefix = f'The patient is {age_part} {gender_noun} ({race}), admitted{adverb} for {primary_dx}.'

    chief_complaint = meta_row.get('chief_complaint')
    if chief_complaint and not str(chief_complaint).strip().startswith('"'):
        prefix += f'\nChief complaint: {chief_complaint}.'

    conditions = get_charlson_conditions(meta_row)
    if conditions:
        if len(conditions) == 1:
            cond_str = conditions[0]
        else:
            cond_str = ', '.join(conditions[:-1]) + f', and {conditions[-1]}'
        prefix += f'\n{pronoun} has a history of {cond_str}.'
    else:
        prefix += '\nNo significant chronic comorbidities are recorded.'

    top_icds = meta_row.get('top_icd_descriptions', '')
    if top_icds:
        prefix += f'\nAdditional co-diagnoses from this admission: {top_icds}.'

    return prefix


def _admission_adverb(admission_type: str | None) -> str:
    if not admission_type:
        return ''
    t = admission_type.upper()
    if 'EMER' in t:
        return ' emergently'
    if 'ELECTIVE' in t:
        return ' electively'
    if 'URGENT' in t:
        return ' urgently'
    return ''
