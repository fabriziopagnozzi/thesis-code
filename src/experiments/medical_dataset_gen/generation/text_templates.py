from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

import yaml

from experiments.medical_dataset_gen.generation.schemas import (
    ChunkTemplateUtils,
    ClinicalFact,
    MedicalOntology,
)

_TEMPLATE_PHRASES_PATH = Path(__file__).with_name('text_templates_utils.yaml')

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
    with open(_TEMPLATE_PHRASES_PATH) as f:
        return ChunkTemplateUtils.model_validate(yaml.safe_load(f))


TEMPLATE_UTILS = _load_template_utils()


def render_chunk_text_template(fact: ClinicalFact, ontology: MedicalOntology, rng: Random) -> str:
    if fact.axis == 'treatment_duration':
        body = render_duration_chunk(fact, rng)
    elif fact.axis == 'rehab_outcome':
        body = render_rehab_chunk(fact, rng)
    else:
        raise ValueError(f'Unsupported axis: {fact.axis}')

    return squash_whitespaces(body)


def render_duration_chunk(fact: ClinicalFact, rng: Random) -> str:
    patient_lower = patient_descriptor(fact)
    patient = _sentence_start(patient_lower)
    condition = fact.condition_display
    presentation = rng.choice(TEMPLATE_UTILS.condition_presentations[fact.condition_id])
    response = rng.choice(TEMPLATE_UTILS.condition_status_phrases[fact.condition_id])
    course_noun = rng.choice(TEMPLATE_UTILS.duration_course_nouns)
    duration_phrase = rng.choice(
        [
            template.format(
                treatment=fact.treatment,
                duration_days=fact.duration_days,
                course_noun=course_noun,
            )
            for template in TEMPLATE_UTILS.duration_phrase_templates
        ]
    )
    response_verb = rng.choice(TEMPLATE_UTILS.duration_response_verbs)
    close = rng.choice(TEMPLATE_UTILS.duration_closing_sentences[fact.condition_id])
    template = rng.choice(TEMPLATE_UTILS.duration_chunk_templates)

    return template.format(
        patient=patient,
        patient_lower=patient_lower,
        condition=condition,
        presentation=presentation,
        response=response,
        response_verb=response_verb,
        duration_sentence=_sentence_start(f'for treatment duration, {duration_phrase}'),
        duration_sentence_lower=f'for treatment duration, {duration_phrase}',
        close=close,
    )


def render_rehab_chunk(fact: ClinicalFact, rng: Random) -> str:
    patient_lower = patient_descriptor(fact)
    patient = _sentence_start(patient_lower)
    condition = fact.condition_display
    presentation = rng.choice(TEMPLATE_UTILS.condition_presentations[fact.condition_id])
    functional_detail = rng.choice(
        TEMPLATE_UTILS.functional_status_phrases[fact.condition_id][fact.value_bin]
    )
    transition = rng.choice(TEMPLATE_UTILS.rehab_transitions)
    rehab_outcome_verb = rng.choice(TEMPLATE_UTILS.rehab_outcome_verbs)
    if fact.value_bin == 'persistent_deficit':
        close = rng.choice(
            TEMPLATE_UTILS.rehab_closing_sentences.persistent_deficit[fact.condition_id]
        )
    else:
        close = rng.choice(getattr(TEMPLATE_UTILS.rehab_closing_sentences, fact.value_bin))
    template = rng.choice(TEMPLATE_UTILS.rehab_chunk_templates)
    rehab_outcome = f'the rehabilitation outcome as {fact.rehab_outcome}'

    return template.format(
        patient=patient,
        patient_lower=patient_lower,
        condition=condition,
        presentation=presentation,
        transition=transition,
        functional_detail=functional_detail,
        rehab_outcome=rehab_outcome,
        rehab_outcome_verb=rehab_outcome_verb,
        close=close,
    )


def validate_chunk_text(
    text: str, fact: ClinicalFact, ontology: MedicalOntology
) -> ChunkValidation:
    lower = text.lower()
    hard_errors: list[str] = []
    soft_warnings: list[str] = []

    for term in TEMPLATE_UTILS.hidden_benchmark_terms:
        if term in lower:
            hard_errors.append(f'contains hidden benchmark term: {term}')
    if _SECTION_HEADER_RE.match(text):
        hard_errors.append('contains leading note-section header')

    if not _contains_condition(text, fact, ontology):
        hard_errors.append(f'missing condition evidence: {fact.condition_display}')
    if not _contains_subgroup_evidence(text, fact, ontology):
        hard_errors.append(f'missing subgroup evidence: {fact.subgroup_label}')

    if fact.axis == 'treatment_duration':
        duration = str(fact.duration_days)
        treatment = str(fact.treatment).lower()
        if duration not in lower:
            hard_errors.append(f'missing treatment duration days: {duration}')
        if treatment and treatment not in lower:
            hard_errors.append(f'missing treatment: {fact.treatment}')
        extra_treatments = _extra_condition_treatments(text, fact, ontology)
        if extra_treatments:
            soft_warnings.append(f'contains extra treatment(s): {", ".join(extra_treatments)}')
        if _contains_rehab_language(text):
            soft_warnings.append('duration chunk contains rehabilitation-outcome language')
    elif fact.axis == 'rehab_outcome':
        has_exact_rehab = _contains_exact_rehab_outcome(text, fact)
        has_bin_rehab = _contains_rehab_bin_evidence(text, fact, ontology)
        if not has_bin_rehab:
            hard_errors.append(f'missing rehabilitation outcome evidence: {fact.rehab_outcome}')
        elif not has_exact_rehab:
            soft_warnings.append(f'missing exact rehabilitation phrase: {fact.rehab_outcome}')
        if _DURATION_RE.search(text):
            soft_warnings.append('rehab chunk contains explicit duration days')
    else:
        hard_errors.append(f'unsupported axis: {fact.axis}')

    return ChunkValidation(hard_errors=hard_errors, soft_warnings=soft_warnings)


def patient_descriptor(fact: ClinicalFact) -> str:
    age = int(fact.patient_age)
    noun = 'woman' if fact.patient_sex == 'female' else 'man'
    phrase = fact.clinical_subgroup_phrase
    if fact.subgroup_id == 'age_over_75':
        return f'the {age}-year-old {noun} older than 75'
    if fact.subgroup_id == 'age_under_50':
        return f'the {age}-year-old {noun} younger than 50'
    if fact.subgroup_axis == 'demographic':
        return f'the {age}-year-old {noun}'
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

    if subgroup_id == 'age_over_75':
        return (
            _has_age_in_range(text, 76, 120) or 'older than 75' in lower or 'above age 75' in lower
        )
    if subgroup_id == 'age_under_50':
        return (
            _has_age_in_range(text, 18, 49) or 'younger than 50' in lower or 'below age 50' in lower
        )

    phrase = str(fact.clinical_subgroup_phrase).lower()
    if phrase and phrase in lower:
        return True

    subgroup = ontology.subgroups.get(subgroup_id)
    aliases = [str(alias).lower() for alias in (subgroup.aliases if subgroup else [])]
    if any(alias in lower for alias in aliases):
        return True

    subgroup_terms = TEMPLATE_UTILS.subgroup_terms.get(subgroup_id, [])
    return any(term in lower for term in subgroup_terms)


def _has_age_in_range(text: str, low: int, high: int) -> bool:
    return any(low <= int(match.group(1)) <= high for match in _AGE_RE.finditer(text))


def _contains_rehab_language(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in TEMPLATE_UTILS.rehab_language_terms)


def _contains_exact_rehab_outcome(
    text: str,
    fact: ClinicalFact,
) -> bool:
    lower = text.lower()
    outcome = str(fact.rehab_outcome).lower()
    return bool(outcome and outcome in lower)


def _contains_rehab_bin_evidence(text: str, fact: ClinicalFact, ontology: MedicalOntology) -> bool:
    lower = text.lower()
    condition = ontology.conditions[fact.condition_id]
    value_bin = fact.value_bin

    for phrase in condition.rehab_outcomes.get(value_bin, []):
        if _rehab_phrase_matches(lower, str(phrase)):
            return True

    if any(term in lower for term in TEMPLATE_UTILS.rehab_bin_terms.get(value_bin, [])):
        return True

    if value_bin == 'persistent_deficit':
        return _contains_persistent_deficit_evidence(lower)

    return False


def _rehab_phrase_matches(lower_text: str, phrase: str) -> bool:
    phrase_lower = phrase.lower()
    if phrase_lower in lower_text:
        return True

    phrase_tokens = _meaningful_tokens(phrase_lower)
    if len(phrase_tokens) < 2:
        return False

    text_tokens = _meaningful_tokens(lower_text)
    overlap = len(phrase_tokens & text_tokens)
    threshold = 2 if len(phrase_tokens) < 5 else 3
    return overlap >= threshold


def _contains_persistent_deficit_evidence(lower_text: str) -> bool:
    return any(
        term in lower_text for term in TEMPLATE_UTILS.persistent_deficit_descriptor_terms
    ) and any(term in lower_text for term in TEMPLATE_UTILS.persistent_deficit_rehab_terms)


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = set(TEMPLATE_UTILS.meaningful_token_stopwords)
    return {
        token
        for token in re.findall(r'[a-z0-9]+', text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _extra_condition_treatments(
    text: str, fact: ClinicalFact, ontology: MedicalOntology
) -> list[str]:
    lower = text.lower()
    expected = str(fact.treatment).lower()
    extras = []
    for treatment in _duration_treatment_terms(fact, ontology):
        treatment_lower = str(treatment).lower()
        if treatment_lower != expected and treatment_lower in lower:
            extras.append(str(treatment))
    return extras


def _duration_treatment_terms(
    fact: ClinicalFact,
    ontology: MedicalOntology,
) -> list[str]:
    condition = ontology.conditions[fact.condition_id]
    treatments = condition.duration_treatments or condition.treatments
    return [str(treatment) for treatment in treatments]


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
