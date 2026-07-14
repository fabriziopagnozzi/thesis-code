from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from random import Random

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AcuteClinicalCoursePayload,
    CareIntensityPayload,
    ChunkSurfaceGroup,
    ChunkSurfacePolicy,
    ChunkTemplateUtils,
    ChunkTextStyle,
    ClinicalFact,
    ComplicationBurdenPayload,
    DiagnosticEvidencePayload,
    MedicalOntology,
    RehabOutcomePayload,
    TreatmentDurationPayload,
    parse_axis_payload,
)
from experiments.medical_dataset_gen.utils.global_utils import (
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
class ChunkTemplateProvenance:
    outer_template_family: str | None
    outer_template_id: str | None
    axis_template_family: str | None
    axis_template_id: str | None


@dataclass(frozen=True)
class RenderedChunkTemplate:
    text: str
    provenance: ChunkTemplateProvenance


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


def available_note_styles(surface_group: ChunkSurfaceGroup | None = None) -> list[str]:
    if surface_group is None:
        return list(TEMPLATE_DATA.note_style_templates)
    return [
        family
        for family, templates in TEMPLATE_DATA.note_style_templates.items()
        if templates.templates_for_group(surface_group)
    ]


def select_chunk_surface_group(
    split: str,
    policy: ChunkSurfacePolicy = 'split_heldout',
) -> ChunkSurfaceGroup:
    if policy == 'seen_only':
        return 'seen'
    if policy == 'heldout_only':
        return 'heldout'
    return 'heldout' if split == 'test' else 'seen'


def render_chunk_text_template(
    fact: ClinicalFact,
    ontology: MedicalOntology,
    rng: Random,
    text_style: ChunkTextStyle = 'semantic_hardened',
    surface_group: ChunkSurfaceGroup | None = None,
) -> str:
    return render_chunk_text_template_result(
        fact,
        ontology,
        rng,
        text_style=text_style,
        surface_group=surface_group,
    ).text


def render_chunk_text_template_result(
    fact: ClinicalFact,
    ontology: MedicalOntology,
    rng: Random,
    text_style: ChunkTextStyle = 'semantic_hardened',
    surface_group: ChunkSurfaceGroup | None = None,
) -> RenderedChunkTemplate:
    payload = parse_axis_payload(fact.axis_payload_json)
    patient = patient_narrative(fact)
    axis = ontology.clinical_axes[fact.axis]
    resolved_surface_group = surface_group or fact.chunk_surface_group
    axis_sentence, axis_template_family, axis_template_id = _axis_sentence(
        fact,
        payload,
        patient=patient,
        axis_term=axis.label,
        rng=rng,
        text_style=text_style,
        surface_group=resolved_surface_group,
    )
    outer_template_family, outer_template_id, outer_template = _outer_template(
        fact,
        resolved_surface_group,
        text_style=text_style,
    )
    context = {
        'patient': patient.subject_cap,
        'patient_lower': patient.subject,
        'pronoun': patient.pronoun,
        'pronoun_cap': patient.pronoun_cap,
        'possessive': patient.possessive,
        'possessive_cap': patient.possessive_cap,
        'age': fact.patient_age,
        'condition': fact.condition_display,
        'presentation': rng.choice(ontology.conditions[fact.condition_id].presentations),
        'cohort_sentence': _cohort_sentence(fact, patient, rng, resolved_surface_group),
        'axis_sentence': axis_sentence,
    }
    return RenderedChunkTemplate(
        text=squash_whitespaces(outer_template.format(**context)),
        provenance=ChunkTemplateProvenance(
            outer_template_family=outer_template_family,
            outer_template_id=outer_template_id,
            axis_template_family=axis_template_family,
            axis_template_id=axis_template_id,
        ),
    )


def _outer_template(
    fact: ClinicalFact,
    surface_group: ChunkSurfaceGroup,
    *,
    text_style: ChunkTextStyle,
) -> tuple[str | None, str | None, str]:
    if text_style == 'ontology_explicit':
        legacy_templates = [
            '{patient} was admitted with {condition} after {presentation}. '
            '{cohort_sentence} {axis_sentence}',
            '{patient} was hospitalized for {condition} after presenting with {presentation}. '
            '{cohort_sentence} {axis_sentence}',
            '{cohort_sentence} {patient} then received inpatient care for {condition} '
            'after {presentation}. {axis_sentence}',
            '{axis_sentence} {patient} had been admitted for {condition} after {presentation}. '
            '{cohort_sentence}',
            '{patient} presented with {presentation} and was treated for {condition}. '
            'The chart also noted routine nursing review. {cohort_sentence} {axis_sentence}',
        ]
        index = _stable_index(fact, 'seen', 'outer_explicit', len(legacy_templates))
        return None, f'ontology_explicit_outer_{index + 1}', legacy_templates[index]

    family = _canonical_outer_family(fact.note_style)
    bucket = TEMPLATE_DATA.note_style_templates.get(family)
    if bucket is None:
        family = next(iter(TEMPLATE_DATA.note_style_templates))
        bucket = TEMPLATE_DATA.note_style_templates[family]
    choices = bucket.templates_for_group(surface_group)
    index = _stable_index(fact, surface_group, f'outer:{family}', len(choices))
    selected = choices[index]
    return family, selected.id, selected.template


def _duration_axis_value(payload: TreatmentDurationPayload, rng: Random) -> str:
    return rng.choice(TEMPLATE_DATA.duration_phrase_templates).format(
        treatment=payload.treatment,
        duration_days=payload.duration_days,
    )


def _treatment_completion_clause(payload: TreatmentDurationPayload) -> str:
    treatment = payload.treatment.lower()
    course_id = (payload.treatment_course_id or '').lower()
    if 'plasma' in treatment or 'exchange' in treatment:
        return 'no further exchange sessions were scheduled'
    if any(
        token in treatment
        for token in (
            'corticosteroid',
            'dexamethasone',
            'hydrocortisone',
            'methylprednisolone',
            'prednisone',
            'steroid',
        )
    ):
        return 'no additional inpatient steroid doses were planned'
    if 'antibiotic' in course_id:
        return 'no further inpatient antibiotic doses were scheduled'
    if 'antiviral' in course_id:
        return 'no further inpatient antiviral doses were scheduled'
    if 'prophylaxis' in course_id and 'ceftriaxone' in treatment:
        return 'no further inpatient prophylaxis doses were scheduled'
    if any(token in course_id for token in ('anticoagulation', 'antithrombotic')):
        return 'the inpatient antithrombotic transition was documented as complete'
    if 'decongestion' in course_id:
        return 'IV decongestion was stepped down afterward'
    if 'albumin' in treatment:
        return 'the albumin infusion interval was documented as complete'
    if 'encephalopathy' in course_id:
        return 'the inpatient encephalopathy regimen was documented as complete'
    if any(token in course_id for token in ('immunosuppression', 'antiinflammatory')):
        return 'the inpatient immunosuppression plan was finalized'
    if 'supportive_care' in course_id:
        return 'supportive measures were tapered afterward'
    if 'infusion' in course_id:
        return 'the inpatient infusion order was documented as complete'
    return 'the active inpatient order was documented as complete'


def _cohort_sentence(
    fact: ClinicalFact,
    patient: PatientNarrative,
    rng: Random,
    surface_group: ChunkSurfaceGroup,
) -> str:
    templates = TEMPLATE_DATA.cohort_evidence_templates
    # Age and sex are already expressed by the patient descriptor. Repeating
    # them as a separate sentence produces unnatural prose without adding evidence.
    if fact.subgroup_dimension_id in {'age_band', 'sex'}:
        return ''
    if fact.subgroup_is_reference:
        choices = templates.comorbidity_reference.templates_for_group(surface_group)
    else:
        choices = templates.comorbidity_present.templates_for_group(surface_group)
    return rng.choice(choices).template.format(
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
    patient: PatientNarrative,
    axis_term: str,
    rng: Random,
    text_style: ChunkTextStyle,
    surface_group: ChunkSurfaceGroup,
) -> tuple[str, str | None, str | None]:
    if isinstance(payload, TreatmentDurationPayload):
        axis_value = _duration_axis_value(payload, rng)
        treatment_completion_clause = _treatment_completion_clause(payload)
    elif isinstance(payload, RehabOutcomePayload):
        axis_value = payload.outcome
        treatment_completion_clause = ''
    elif isinstance(
        payload,
        (
            ComplicationBurdenPayload,
            AcuteClinicalCoursePayload,
            CareIntensityPayload,
            DiagnosticEvidencePayload,
        ),
    ):
        axis_value = payload.detail
        treatment_completion_clause = ''
    else:
        raise TypeError(type(payload))
    if text_style == 'ontology_explicit':
        templates = TEMPLATE_DATA.axis_sentence_templates.ontology_explicit[fact.axis][
            fact.value_bin
        ]
        index = _stable_index(fact, 'seen', 'axis_explicit', len(templates))
        template = templates[index]
        return (
            template.format(
                axis_term=axis_term,
                axis_bin_term=fact.axis_bin_term,
                axis_value=axis_value,
                pronoun=patient.pronoun,
                pronoun_cap=patient.pronoun_cap,
                possessive=patient.possessive,
                possessive_cap=patient.possessive_cap,
                treatment_completion_clause=treatment_completion_clause,
                treatment_completion_clause_cap=_sentence_start(treatment_completion_clause),
            ),
            None,
            f'ontology_explicit_{fact.axis}_{fact.value_bin}_{index + 1}',
        )

    semantic_templates = TEMPLATE_DATA.axis_sentence_templates.semantic_hardened[fact.axis][
        fact.value_bin
    ].template_specs(surface_group)
    index = _stable_index(fact, surface_group, 'axis_semantic', len(semantic_templates))
    family, template_spec = semantic_templates[index]
    return (
        template_spec.template.format(
            axis_term=axis_term,
            axis_bin_term=fact.axis_bin_term,
            axis_value=axis_value,
            pronoun=patient.pronoun,
            pronoun_cap=patient.pronoun_cap,
            possessive=patient.possessive,
            possessive_cap=patient.possessive_cap,
            treatment_completion_clause=treatment_completion_clause,
            treatment_completion_clause_cap=_sentence_start(treatment_completion_clause),
        ),
        family,
        template_spec.id,
    )


def _canonical_outer_family(note_style: str) -> str:
    legacy_styles = {
        'brief_hospital_course': 'admission_course',
        'progress_note': 'embedded_course',
        'discharge_summary': 'cohort_first',
    }
    return legacy_styles.get(note_style, note_style)


def _stable_index(
    fact: ClinicalFact,
    surface_group: ChunkSurfaceGroup,
    purpose: str,
    size: int,
) -> int:
    if size < 1:
        raise ValueError(f'cannot choose from an empty template list for {purpose}')
    payload = {
        'condition_id': fact.condition_id,
        'subgroup_id': fact.subgroup_id,
        'axis': fact.axis,
        'value_bin': fact.value_bin,
        'axis_payload_json': fact.axis_payload_json,
        'surface_group': surface_group,
        'purpose': purpose,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return int(digest[:16], 16) % size


def validate_chunk_text(
    text: str,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    text_style: ChunkTextStyle = 'semantic_hardened',
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
    if text_style == 'ontology_explicit' and not _contains_axis_evidence(text, fact, ontology):
        hard_errors.append(f'missing target-axis evidence: {fact.axis}')
    if text_style == 'ontology_explicit' and fact.axis_bin_term.lower() not in lower:
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

    if subgroup is not None and subgroup.patient_sex is not None:
        sex_nouns = ('woman', 'female') if subgroup.patient_sex == 'female' else ('man', 'male')
        if any(re.search(rf'\b{term}\b', lower) for term in sex_nouns):
            return True

    if subgroup is not None and subgroup.patient_age_range is not None:
        low, high = subgroup.patient_age_range
        if _has_age_in_range(text, low, high):
            return True

    phrase = str(fact.clinical_subgroup_phrase).lower()
    if phrase and phrase in lower:
        return True

    lexical_forms = [str(alias).lower() for alias in (subgroup.aliases if subgroup else [])] + [
        str(surface).lower() for surface in (subgroup.surface_phrases if subgroup else [])
    ]
    return any(form in lower for form in lexical_forms)


def _contains_axis_evidence(text: str, fact: ClinicalFact, ontology: MedicalOntology) -> bool:
    lower = text.lower()
    axis = ontology.clinical_axes[fact.axis]
    return any(
        term.lower() in lower
        for term in [
            axis.label,
            *axis.exact_terms,
            *axis.synonym_terms,
            *axis.bin_terms[fact.value_bin],
        ]
    )


def _has_age_in_range(text: str, low: int, high: int) -> bool:
    return any(low <= int(match.group(1)) <= high for match in _AGE_RE.finditer(text))


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
