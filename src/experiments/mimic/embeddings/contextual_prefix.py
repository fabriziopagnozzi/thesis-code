import polars as pl

from experiments.mimic.utils.charlson import CHARLSON_LABELS_TO_STR
from experiments.mimic.utils.schemas import EmbedJoinedRow
from experiments.mimic.utils.utils import get_age_group, get_charlson_conditions


def enrich_note_excerpts(
    chunks: pl.DataFrame, metadata: pl.DataFrame
) -> tuple[pl.DataFrame, list[str]]:

    meta_cols = [
        'hadm_id',
        'age',
        'gender',
        'race',
        'primary_icd_description',
        'top_icd_descriptions',
        'charlson_comorbidity_index',
        'admission_type',
        *CHARLSON_LABELS_TO_STR.keys(),
    ]
    meta_subset = metadata.select(meta_cols).unique(subset=['hadm_id'])

    joined = (
        chunks.join(meta_subset, on='hadm_id', how='left')
        .with_columns(
            full_text=pl.struct(pl.all()).map_elements(build_full_text, return_dtype=pl.Utf8)
        )
        .with_columns(text_len=pl.col('full_text').str.len_chars())
        .sort('text_len', descending=False)
    )

    texts = joined['full_text'].to_list()

    return joined.drop(['text_len', 'full_text']), texts


def build_full_text(row_dict: EmbedJoinedRow) -> str:
    prefix = build_contextual_prefix(row_dict)
    return f'{prefix}\nExcerpt from the {row_dict["section_name"]} section of a discharge summary.\n{row_dict["text"]}'


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
