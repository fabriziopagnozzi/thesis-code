from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AcuteClinicalCoursePayload,
    CareIntensityPayload,
    ChunkTemplateUtils,
    ClinicalFact,
    ComplicationBurdenPayload,
    MedicalOntology,
    RehabOutcomePayload,
    TreatmentDurationPayload,
    parse_axis_payload,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    MedicalDatasetGenPaths,
)

_TEMPLATE_DATA_DIR = MedicalDatasetGenPaths.root / 'data_templates'
_TEMPLATE_DATA_PATH = _TEMPLATE_DATA_DIR / 'chunk_templates.yaml'

_DURATION_RE = re.compile(r'\b\d+\s*[- ]?(?:day|days)\b', re.IGNORECASE)
_AGE_RE = re.compile(r'\b(\d{2,3})\s*[- ]year[- ]old\b', re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(
    r'^\s*(?:brief hospital course|hospital course|discharge summary|'
    r'discharge diagnosis|clinical summary)\s*:\s*',
    re.IGNORECASE,
)


@dataclass
class ChunkValidation:
    hard_errors: list[str]
    soft_warnings: list[str]


def _load_template_utils() -> ChunkTemplateUtils:
    with open(_TEMPLATE_DATA_PATH) as file:
        return ChunkTemplateUtils.model_validate(yaml.safe_load(file) or {})


TEMPLATE_DATA = _load_template_utils()


def render_chunk_text_template(fact: ClinicalFact, ontology: MedicalOntology, rng: Random) -> str:
    payload = parse_axis_payload(fact.axis_payload_json)
    patient_lower = patient_descriptor(fact)
    context = {
        'patient': _sentence_start(patient_lower),
        'patient_lower': patient_lower,
        'condition': fact.condition_display,
        'presentation': rng.choice(ontology.conditions[fact.condition_id].presentations),
        'axis_sentence': _axis_sentence(fact, payload, rng),
        'close': rng.choice(TEMPLATE_DATA.axis_closing_sentences[fact.axis]),
    }
    templates = TEMPLATE_DATA.note_style_templates.get(
        fact.note_style, TEMPLATE_DATA.note_style_templates['brief_hospital_course']
    )
    return squash_whitespaces(rng.choice(templates).format(**context))


def render_duration_chunk(fact: ClinicalFact, rng: Random) -> str:
    payload = parse_axis_payload(fact.axis_payload_json)
    if not isinstance(payload, TreatmentDurationPayload):
        raise TypeError('duration renderer requires TreatmentDurationPayload')
    course_noun = rng.choice(TEMPLATE_DATA.duration_course_nouns)
    duration_phrase = rng.choice([
        template.format(
            treatment=payload.treatment,
            duration_days=payload.duration_days,
            course_noun=course_noun,
        )
        for template in TEMPLATE_DATA.duration_phrase_templates
    ])
    return _sentence_start(
        f'{rng.choice(TEMPLATE_DATA.duration_focus_phrases)}, {duration_phrase}.'
    )


def render_rehab_chunk(fact: ClinicalFact, rng: Random) -> str:
    payload = parse_axis_payload(fact.axis_payload_json)
    if not isinstance(payload, RehabOutcomePayload):
        raise TypeError('rehab renderer requires RehabOutcomePayload')
    return f'The rehabilitation outcome was {payload.outcome}.'


def _axis_sentence(fact: ClinicalFact, payload, rng: Random) -> str:
    if isinstance(payload, TreatmentDurationPayload):
        return render_duration_chunk(fact, rng)
    if isinstance(payload, RehabOutcomePayload):
        return render_rehab_chunk(fact, rng)
    if not isinstance(
        payload,
        (ComplicationBurdenPayload, AcuteClinicalCoursePayload, CareIntensityPayload),
    ):
        raise TypeError(type(payload))
    template = rng.choice(TEMPLATE_DATA.axis_sentence_templates[fact.axis])
    return template.format(axis_value=payload.detail)


def validate_chunk_text(
    text: str, fact: ClinicalFact, ontology: MedicalOntology
) -> ChunkValidation:
    lower = text.lower()
    hard_errors: list[str] = []
    soft_warnings: list[str] = []

    for term in TEMPLATE_DATA.hidden_benchmark_terms:
        if term in lower:
            hard_errors.append(f'contains hidden benchmark term: {term}')
    if _SECTION_HEADER_RE.match(text):
        hard_errors.append('contains leading note-section header')

    if not _contains_condition(text, fact, ontology):
        hard_errors.append(f'missing condition evidence: {fact.condition_display}')
    if not _contains_subgroup_evidence(text, fact, ontology):
        hard_errors.append(f'missing subgroup evidence: {fact.subgroup_label}')

    payload = parse_axis_payload(fact.axis_payload_json)
    required = (
        [str(payload.duration_days), payload.treatment]
        if isinstance(payload, TreatmentDurationPayload)
        else [payload.outcome]
        if isinstance(payload, RehabOutcomePayload)
        else [payload.detail]
    )
    for phrase in required:
        if phrase.lower() not in lower:
            hard_errors.append(f'missing axis payload evidence: {phrase}')

    for other_axis, bins in TEMPLATE_DATA.axis_bin_terms.items():
        if other_axis == fact.axis:
            continue
        foreign_terms = [term for terms in bins.values() for term in terms]
        if any(term.lower() in lower for term in foreign_terms):
            hard_errors.append(f'contains explicit foreign-axis evidence: {other_axis}')

    return ChunkValidation(hard_errors=hard_errors, soft_warnings=soft_warnings)


def patient_descriptor(fact: ClinicalFact) -> str:
    age = int(fact.patient_age)
    noun = 'woman' if fact.patient_sex == 'female' else 'man'
    phrase = fact.clinical_subgroup_phrase
    if fact.subgroup_dimension_id == 'age_band':
        return f'the {age}-year-old {noun}, {phrase}'
    if fact.subgroup_dimension_id == 'sex':
        return f'the {age}-year-old {noun}, described as a {phrase}'
    return f'the {age}-year-old {noun} with {phrase}'


def _sentence_start(text: str) -> str:
    return text[:1].upper() + text[1:]


def _contains_condition(text: str, fact: ClinicalFact, ontology: MedicalOntology) -> bool:
    lower = text.lower()
    if fact.condition_display.lower() in lower:
        return True
    condition = ontology.conditions[fact.condition_id]
    return any(str(term).lower() in lower for term in condition.terms[:3])


def _contains_subgroup_evidence(text: str, fact: ClinicalFact, ontology: MedicalOntology) -> bool:
    lower = text.lower()
    subgroup_id = fact.subgroup_id
    subgroup = ontology.subgroups.get(subgroup_id)

    if subgroup is not None and subgroup.patient_age_range is not None:
        low, high = subgroup.patient_age_range
        if _has_age_in_range(text, low, high):
            return True

    phrase = str(fact.clinical_subgroup_phrase).lower()
    if phrase and phrase in lower:
        return True

    aliases = [str(alias).lower() for alias in (subgroup.aliases if subgroup else [])]
    return any(alias in lower for alias in aliases)


def _has_age_in_range(text: str, low: int, high: int) -> bool:
    return any(low <= int(match.group(1)) <= high for match in _AGE_RE.finditer(text))


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
