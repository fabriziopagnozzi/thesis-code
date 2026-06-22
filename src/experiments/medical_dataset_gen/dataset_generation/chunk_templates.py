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


@dataclass(frozen=True)
class PatientNarrative:
    subject: str
    pronoun: str
    possessive: str

    @property
    def subject_cap(self) -> str:
        return _sentence_start(self.subject)

    @property
    def pronoun_cap(self) -> str:
        return _sentence_start(self.pronoun)

    @property
    def possessive_cap(self) -> str:
        return _sentence_start(self.possessive)


def _load_template_utils() -> ChunkTemplateUtils:
    with open(_TEMPLATE_DATA_PATH) as file:
        return ChunkTemplateUtils.model_validate(yaml.safe_load(file) or {})


TEMPLATE_DATA = _load_template_utils()


def available_note_styles() -> list[str]:
    return list(TEMPLATE_DATA.note_style_templates)


def render_chunk_text_template(fact: ClinicalFact, ontology: MedicalOntology, rng: Random) -> str:
    payload = parse_axis_payload(fact.axis_payload_json)
    patient = patient_narrative(fact)
    axis = ontology.clinical_axes[fact.axis]
    context = {
        'patient': patient.subject_cap,
        'patient_lower': patient.subject,
        'pronoun': patient.pronoun,
        'pronoun_cap': patient.pronoun_cap,
        'possessive': patient.possessive,
        'possessive_cap': patient.possessive_cap,
        'age': fact.patient_age,
        'condition': fact.condition_display.lower(),
        'presentation': rng.choice(ontology.conditions[fact.condition_id].presentations),
        'cohort_sentence': _cohort_sentence(fact, patient, rng),
        'axis_sentence': _axis_sentence(fact, payload, axis_term=axis.label, rng=rng),
    }
    templates = TEMPLATE_DATA.note_style_templates.get(
        fact.note_style, TEMPLATE_DATA.note_style_templates['brief_hospital_course']
    )
    return squash_whitespaces(rng.choice(templates).format(**context))


def _duration_axis_value(payload: TreatmentDurationPayload, rng: Random) -> str:
    return rng.choice(TEMPLATE_DATA.duration_phrase_templates).format(
        treatment=payload.treatment,
        duration_days=payload.duration_days,
    )


def _cohort_sentence(fact: ClinicalFact, patient: PatientNarrative, rng: Random) -> str:
    templates = TEMPLATE_DATA.cohort_evidence_templates
    if fact.subgroup_dimension_id == 'age_band':
        return ''
    if fact.subgroup_dimension_id == 'sex':
        choices = templates.sex
    elif fact.subgroup_is_reference:
        choices = templates.comorbidity_reference
    else:
        choices = templates.comorbidity_present
    return rng.choice(choices).format(
        subgroup_phrase=fact.clinical_subgroup_phrase,
        age=fact.patient_age,
        pronoun=patient.pronoun,
        pronoun_cap=patient.pronoun_cap,
        possessive=patient.possessive,
        possessive_cap=patient.possessive_cap,
    )


def _axis_sentence(
    fact: ClinicalFact,
    payload,
    *,
    axis_term: str,
    rng: Random,
) -> str:
    if isinstance(payload, TreatmentDurationPayload):
        axis_value = _duration_axis_value(payload, rng)
    elif isinstance(payload, RehabOutcomePayload):
        axis_value = payload.outcome
    elif isinstance(
        payload,
        (ComplicationBurdenPayload, AcuteClinicalCoursePayload, CareIntensityPayload),
    ):
        axis_value = payload.detail
    else:
        raise TypeError(type(payload))
    template = rng.choice(TEMPLATE_DATA.axis_sentence_templates[fact.axis])
    return template.format(
        axis_term=axis_term,
        axis_bin_term=fact.axis_bin_term,
        axis_value=axis_value,
    )


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
    if not _contains_axis_evidence(text, fact, ontology):
        hard_errors.append(f'missing target-axis evidence: {fact.axis}')
    if fact.axis_bin_term.lower() not in lower:
        hard_errors.append(f'missing value-bin evidence: {fact.axis_bin_term}')

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

    for other_axis, axis in ontology.clinical_axes.items():
        if other_axis == fact.axis:
            continue
        foreign_terms = [axis.label, *[term for terms in axis.bin_terms.values() for term in terms]]
        if any(term.lower() in lower for term in foreign_terms):
            hard_errors.append(f'contains explicit foreign-axis evidence: {other_axis}')

    return ChunkValidation(hard_errors=hard_errors, soft_warnings=soft_warnings)


def patient_descriptor(fact: ClinicalFact) -> str:
    return patient_narrative(fact).subject


def patient_narrative(fact: ClinicalFact) -> PatientNarrative:
    age = int(fact.patient_age)
    noun = 'woman' if fact.patient_sex == 'female' else 'man'
    pronoun = 'she' if fact.patient_sex == 'female' else 'he'
    possessive = 'her' if fact.patient_sex == 'female' else 'his'
    return PatientNarrative(
        subject=f'the {age}-year-old {noun}',
        pronoun=pronoun,
        possessive=possessive,
    )


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

    lexical_forms = [
        str(alias).lower() for alias in (subgroup.aliases if subgroup else [])
    ] + [str(surface).lower() for surface in (subgroup.surface_phrases if subgroup else [])]
    return any(form in lower for form in lexical_forms)


def _contains_axis_evidence(
    text: str, fact: ClinicalFact, ontology: MedicalOntology
) -> bool:
    lower = text.lower()
    axis = ontology.clinical_axes[fact.axis]
    return any(
        term.lower() in lower for term in [axis.label, *axis.exact_terms, *axis.synonym_terms]
    )


def _has_age_in_range(text: str, low: int, high: int) -> bool:
    return any(low <= int(match.group(1)) <= high for match in _AGE_RE.finditer(text))


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
