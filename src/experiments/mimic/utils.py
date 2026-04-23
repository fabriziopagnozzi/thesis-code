import re

from pydantic import BaseModel, field_validator


class QueryAspect(BaseModel):
    facet_label: str
    description: str

    @field_validator('facet_label')
    @classmethod
    def normalize_label(cls, v: str) -> str:
        return v.strip().lower().replace(' ', '_')


class TagDecision(BaseModel):
    chunk_id: str
    reason: str = ''


def modifier_to_snake_label(text: str) -> str:
    s = re.sub(r'\([^)]*\)', '', text)
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s.lower())
    s = re.sub(r'\b(?:the|a|an)\b', '', s)
    tokens = [t for t in s.split() if t]
    return '_'.join(tokens[:6])


def aspects_from_modifiers(modifiers_json: list[dict]) -> list[QueryAspect]:
    return [
        QueryAspect(facet_label=modifier_to_snake_label(m['text']), description=m['text'])
        for m in modifiers_json
    ]


def col_for_model(model_name: str) -> str:
    safe = re.sub(r'[/\-.]', '_', model_name)
    return f'vector_{safe}'


CHARLSON_LABELS = {
    'myocardial_infarct': ' prior myocardial infarction',
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
