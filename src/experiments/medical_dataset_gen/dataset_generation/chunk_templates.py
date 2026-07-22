from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from random import Random

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AcuteClinicalCoursePayload,
    AxisFactPayload,
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
        axis_query_label=axis.query_label,
        rng=rng,
        text_style=text_style,
        surface_group=resolved_surface_group,
    )
    outer_template_family, outer_template_id, outer_template = _outer_template(
        fact,
        resolved_surface_group,
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
        'presentation': _presentation_without_axis_repetition(
            fact,
            payload,
            ontology,
            resolved_surface_group,
        ),
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


def _presentation_without_axis_repetition(
    fact: ClinicalFact,
    payload: AxisFactPayload,
    ontology: MedicalOntology,
    surface_group: ChunkSurfaceGroup,
) -> str:
    """Select a condition presentation that does not restate the axis evidence.

    Some condition presentations intentionally contain diagnostic detail that can
    also be selected as an axis payload. Keeping both sentences makes a chunk
    repetitive and violates the one-evidence-span rendering invariant.
    """
    presentations = ontology.conditions[fact.condition_id].presentations
    evidence_phrases = _presentation_exclusion_phrases(payload, fact.condition_display)
    nonrepeating_presentations = [
        presentation
        for presentation in presentations
        if not any(_phrase_count(presentation, phrase) for phrase in evidence_phrases)
    ]
    choices = nonrepeating_presentations or presentations
    index = _stable_index(fact, surface_group, 'presentation', len(choices))
    return _context_free_surface(choices[index], fact.condition_display)


def _presentation_exclusion_phrases(
    payload: AxisFactPayload,
    condition_display: str,
) -> list[str]:
    if isinstance(payload, TreatmentDurationPayload):
        return [payload.treatment]
    if isinstance(payload, RehabOutcomePayload):
        return [_context_free_surface(payload.outcome, condition_display)]
    if isinstance(
        payload,
        (
            ComplicationBurdenPayload,
            AcuteClinicalCoursePayload,
            CareIntensityPayload,
            DiagnosticEvidencePayload,
        ),
    ):
        return [_context_free_surface(payload.detail, condition_display)]
    raise TypeError(type(payload))


def _outer_template(
    fact: ClinicalFact,
    surface_group: ChunkSurfaceGroup,
) -> tuple[str | None, str | None, str]:
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
    templates = TEMPLATE_DATA.duration_phrase_templates
    if 'therapy' in payload.treatment.casefold():
        templates = [template for template in templates if 'therapy with' not in template]
    return rng.choice(templates).format(
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
    axis_query_label: str,
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
    axis_value = _context_free_surface(axis_value, fact.condition_display)
    paired_templates = TEMPLATE_DATA.axis_sentence_templates.paired[fact.axis]
    template_specs = paired_templates.template_specs(surface_group)
    index = _stable_index(fact, surface_group, 'axis_sentence', len(template_specs))
    family, template_spec = template_specs[index]
    context = {
        'axis_query_label': axis_query_label,
        'axis_bin_term': fact.axis_bin_term,
        'axis_value': axis_value,
        'pronoun': patient.pronoun,
        'pronoun_cap': patient.pronoun_cap,
        'possessive': patient.possessive,
        'possessive_cap': patient.possessive_cap,
        'treatment_completion_clause': treatment_completion_clause,
        'treatment_completion_clause_cap': _sentence_start(treatment_completion_clause),
    }
    text = template_spec.template.format(**context)
    if text_style == 'ontology_explicit':
        bin_in_base = bool(_phrase_count(text, fact.axis_bin_term))
        label_in_bin = axis_query_label.casefold() in fact.axis_bin_term.casefold()
        if bin_in_base and label_in_bin:
            suffix = 'This category was recorded in the chart.'
        elif bin_in_base:
            suffix = 'This {axis_query_label} was recorded in the chart.'
        elif label_in_bin:
            suffix = 'This category corresponded to {axis_bin_term}.'
        else:
            suffix = paired_templates.ontology_explicit_suffix
        text = f'{text} {suffix.format(**context)}'
    return text, family, template_spec.id


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
        'chunk_reuse_key': fact.chunk_reuse_key,
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


def _context_free_surface(text: str, condition_display: str) -> str:
    """Remove only a repeated full diagnosis from a local evidence phrase.

    The outer note already identifies the diagnosis. Removing an exact repeat
    keeps condition-specific clinical detail while avoiding the common
    ``condition ... condition`` construction in a single short chunk.
    """
    cleaned = re.sub(re.escape(condition_display), '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(' ,;')
    cleaned = re.sub(r'\b(?:after|during|for|with|of)\s*(?=[.,;]|$)', '', cleaned)
    return cleaned.strip(' ,;')


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

    condition_occurrences = _phrase_count(text, fact.condition_display)
    if condition_occurrences != 1:
        hard_errors.append(
            f'condition display must occur exactly once; found {condition_occurrences}: '
            f'{fact.condition_display}'
        )
    if not _contains_subgroup_evidence(text, fact, ontology):
        hard_errors.append(f'missing subgroup evidence: {fact.subgroup_label}')
    axis = ontology.clinical_axes[fact.axis]
    if text_style == 'ontology_explicit':
        if _phrase_count(text, axis.query_label) != 1:
            hard_errors.append(f'axis query label must occur exactly once: {axis.query_label}')
        if _phrase_count(text, fact.axis_bin_term) != 1:
            hard_errors.append(f'value-bin term must occur exactly once: {fact.axis_bin_term}')
    else:
        for term in (axis.query_label, axis.label):
            if _phrase_count(text, term):
                hard_errors.append(f'contains explicit target-axis terminology: {term}')
        if _phrase_count(text, fact.axis_bin_term):
            soft_warnings.append(
                f'payload overlaps the canonical value-bin term: {fact.axis_bin_term}'
            )

    payload = parse_axis_payload(fact.axis_payload_json)
    if isinstance(payload, TreatmentDurationPayload):
        duration_occurrences = _duration_evidence_count(text, payload.duration_days)
        if duration_occurrences != 1:
            hard_errors.append(
                'treatment-duration evidence must occur exactly once; '
                f'found {duration_occurrences}: {payload.duration_days}-day'
            )
        required = [payload.treatment]
    elif isinstance(payload, RehabOutcomePayload):
        required = [_context_free_surface(payload.outcome, fact.condition_display)]
    else:
        required = [_context_free_surface(payload.detail, fact.condition_display)]
    for phrase in required:
        occurrences = _phrase_count(text, phrase)
        if occurrences != 1:
            hard_errors.append(
                f'axis payload evidence must occur exactly once; found {occurrences}: {phrase}'
            )

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


def _phrase_count(text: str, phrase: str) -> int:
    if not phrase.strip():
        return 0
    return len(re.findall(rf'(?<!\w){re.escape(phrase)}(?!\w)', text, flags=re.IGNORECASE))


def _duration_evidence_count(text: str, duration_days: int) -> int:
    """Count a duration only when it is attached to a day unit.

    Bare values such as ``3`` or ``56`` also occur in ages and unrelated
    measurements, so they are unsuitable as a standalone rendering check.
    """
    duration = re.escape(str(duration_days))
    return len(
        re.findall(
            rf'(?<!\w){duration}(?:\s*-\s*|\s+)day(?:s)?\b',
            text,
            flags=re.IGNORECASE,
        )
    )


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
            axis.query_label,
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
