from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from random import Random
from string import Formatter

import yaml

from experiments.medical_dataset_gen.dataset_generation.schemas import (
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

TEMPLATE_DATA_DIR = MedicalDatasetGenPaths.root / 'data_templates'


def _load_template_utils() -> ChunkTemplateUtils:
    with open(TEMPLATE_DATA_DIR / 'chunk_templates.yaml') as file:
        return ChunkTemplateUtils.model_validate(yaml.safe_load(file) or {})


CHUNK_TEMPLATE_DATA = _load_template_utils()


@dataclass
class ChunkValidation:
    hard_errors: list[str]


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


def available_note_styles(surface_group: ChunkSurfaceGroup | None = None) -> list[str]:
    templates = CHUNK_TEMPLATE_DATA.note_style_templates.outer_template
    if surface_group is None:
        return list(templates)
    return [
        family for family, bucket in templates.items() if bucket.templates_for_group(surface_group)
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
) -> tuple[str | None, str | None, str]:
    family = _canonical_outer_family(fact.note_style)
    templates = CHUNK_TEMPLATE_DATA.note_style_templates.templates_for_anchor(fact.condition_anchor)
    bucket = templates.get(family)
    if bucket is None:
        family = next(iter(templates))
        bucket = templates[family]
    choices = bucket.templates_for_group(surface_group)
    index = _stable_index(fact, surface_group, f'outer:{family}', len(choices))
    selected = choices[index]
    return family, selected.id, selected.template


def _duration_phrase(payload: TreatmentDurationPayload) -> str:
    return f'{payload.duration_days} days of {payload.treatment}'


def _cohort_sentence(
    fact: ClinicalFact,
    patient: PatientNarrative,
    rng: Random,
    surface_group: ChunkSurfaceGroup,
) -> str:
    templates = CHUNK_TEMPLATE_DATA.cohort_evidence_templates
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
        text, family, template_id = _treatment_duration_sentence(
            fact,
            payload,
            rng=rng,
            surface_group=surface_group,
        )
        if text_style == 'ontology_explicit':
            text = f'{text} {_simple_interpretation_sentence(fact, surface_group=surface_group)}'
        return text, family, template_id
    elif isinstance(payload, RehabOutcomePayload):
        axis_value = payload.outcome
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
    else:
        raise TypeError(type(payload))
    paired_templates = CHUNK_TEMPLATE_DATA.axis_sentence_templates.paired[fact.axis]
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
    }
    text = template_spec.template.format(**context)
    if text_style == 'ontology_explicit':
        text = f'{text} {_simple_interpretation_sentence(fact, surface_group=surface_group)}'
    return text, family, template_spec.id


def _treatment_duration_sentence(
    fact: ClinicalFact,
    payload: TreatmentDurationPayload,
    *,
    rng: Random,
    surface_group: ChunkSurfaceGroup,
) -> tuple[str, str, str]:
    course_id = payload.treatment_course_id
    if course_id is None:
        raise ValueError(f'treatment duration payload for {fact.fact_id} lacks treatment_course_id')
    bucket = CHUNK_TEMPLATE_DATA.treatment_course_templates.get(course_id)
    if bucket is None:
        raise ValueError(f'missing treatment-course template for {course_id!r}')
    choices = bucket.templates_for_group(surface_group)
    index = _stable_index(fact, surface_group, f'treatment_course:{course_id}', len(choices))
    selected = choices[index]
    text = selected.template.format(
        duration_days=payload.duration_days,
        treatment=payload.treatment,
        duration_phrase=_duration_phrase(payload),
    )
    return text, f'treatment_course:{course_id}', selected.id


def _simple_interpretation_sentence(
    fact: ClinicalFact,
    *,
    surface_group: ChunkSurfaceGroup,
) -> str:
    try:
        bucket = CHUNK_TEMPLATE_DATA.simple_interpretations[fact.axis][fact.value_bin]
    except KeyError as exc:
        raise ValueError(
            f'missing simple interpretation for {fact.axis!r}/{fact.value_bin!r}'
        ) from exc
    choices = bucket.templates_for_group(surface_group)
    index = _stable_index(fact, surface_group, 'simple_interpretation', len(choices))
    return choices[index].template


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


def validate_chunk_text(
    text: str,
    fact: ClinicalFact,
    ontology: MedicalOntology,
    text_style: ChunkTextStyle = 'semantic_hardened',
    surface_group: ChunkSurfaceGroup | None = None,
) -> ChunkValidation:
    lower = text.lower()
    hard_errors: list[str] = []

    for term in CHUNK_TEMPLATE_DATA.hidden_benchmark_terms:
        if term in lower:
            hard_errors.append(f'contains hidden benchmark term: {term}')

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
        interpretation = _simple_interpretation_sentence(
            fact,
            surface_group=surface_group or fact.chunk_surface_group,
        )
        if _phrase_count(text, interpretation) != 1:
            hard_errors.append(
                f'simple interpretation sentence must occur exactly once: {interpretation}'
            )
    else:
        for term in (axis.query_label, axis.label):
            if _phrase_count(text, term):
                hard_errors.append(f'contains explicit target-axis terminology: {term}')

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
        required = [payload.outcome]
    else:
        required = [payload.detail]
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

    return ChunkValidation(hard_errors=hard_errors)


def validate_chunk_template_sources(ontology: MedicalOntology) -> None:
    """Fail before generation if the ontology/template cross-product is incomplete."""
    errors: list[str] = []
    _validate_treatment_course_template_coverage(ontology, errors)
    _validate_simple_interpretation_coverage(ontology, errors)
    _validate_template_placeholders(errors)
    if errors:
        raise ValueError('invalid v4 chunk template sources: ' + '; '.join(errors))


def _validate_treatment_course_template_coverage(
    ontology: MedicalOntology,
    errors: list[str],
) -> None:
    course_ids = {
        course_id
        for condition in ontology.conditions.values()
        for course_id in condition.axis_values['treatment_duration'].treatments  # type: ignore
    }
    template_ids = set(CHUNK_TEMPLATE_DATA.treatment_course_templates)
    missing = sorted(course_ids - template_ids)
    extra = sorted(template_ids - course_ids)
    if missing:
        errors.append(f'missing treatment_course_templates: {missing}')
    if extra:
        errors.append(f'treatment_course_templates define unknown courses: {extra}')


def _validate_simple_interpretation_coverage(
    ontology: MedicalOntology,
    errors: list[str],
) -> None:
    for axis_id, axis in ontology.clinical_axes.items():
        expected = set(axis.bins)
        actual = set(CHUNK_TEMPLATE_DATA.simple_interpretations.get(axis_id, {}))
        axis_name = axis_id.replace('_', ' ')
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            errors.append(f'simple_interpretations.{axis_id}: missing={missing}, extra={extra}')
        for value_bin, bucket in CHUNK_TEMPLATE_DATA.simple_interpretations.get(
            axis_id, {}
        ).items():
            for spec in [*bucket.seen, *bucket.heldout]:
                if _phrase_count(spec.template, axis_name) != 1:
                    errors.append(
                        f'simple_interpretations.{axis_id}.{value_bin}/{spec.id} '
                        f'must contain the axis name exactly once: {axis_name!r}'
                    )


def _validate_template_placeholders(errors: list[str]) -> None:
    treatment_fields = {'duration_days', 'treatment', 'duration_phrase'}
    outer_fields = {
        'patient',
        'patient_lower',
        'pronoun',
        'pronoun_cap',
        'possessive',
        'possessive_cap',
        'age',
        'condition',
        'cohort_sentence',
        'axis_sentence',
    }
    for condition_anchor in ('outer_template', 'axis_evidence'):
        templates = CHUNK_TEMPLATE_DATA.note_style_templates.templates_for_anchor(condition_anchor)
        for family, bucket in templates.items():
            for spec in [*bucket.seen, *bucket.heldout]:
                fields = _template_fields(spec.template)
                unknown = sorted(fields - outer_fields)
                if unknown:
                    errors.append(f'{family}/{spec.id} has unknown placeholders: {unknown}')
                condition_count = sum(
                    1
                    for _, field_name, _, _ in Formatter().parse(spec.template)
                    if field_name == 'condition'
                )
                expected_count = 1 if condition_anchor == 'outer_template' else 0
                if condition_count != expected_count:
                    errors.append(
                        f'{family}/{spec.id} must contain {{condition}} {expected_count} time(s)'
                    )
                if 'axis_sentence' not in fields:
                    errors.append(f'{family}/{spec.id} must contain {{axis_sentence}}')
    for course_id, bucket in CHUNK_TEMPLATE_DATA.treatment_course_templates.items():
        for spec in [*bucket.seen, *bucket.heldout]:
            fields = _template_fields(spec.template)
            unknown = sorted(fields - treatment_fields)
            if unknown:
                errors.append(f'{course_id}/{spec.id} has unknown placeholders: {unknown}')
    for axis_id, interpretations_by_bin in CHUNK_TEMPLATE_DATA.simple_interpretations.items():
        for value_bin, bucket in interpretations_by_bin.items():
            for spec in [*bucket.seen, *bucket.heldout]:
                fields = _template_fields(spec.template)
                if fields:
                    errors.append(
                        f'simple_interpretations.{axis_id}.{value_bin}/{spec.id} '
                        f'must be a complete authored sentence without placeholders: '
                        f'{sorted(fields)}'
                    )


def _template_fields(template: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None and field_name
    }


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


def _phrase_count(text: str, phrase: str) -> int:
    if not phrase.strip():
        return 0
    return len(re.findall(rf'(?<!\w){re.escape(phrase)}(?!\w)', text, flags=re.IGNORECASE))


def _duration_evidence_count(text: str, duration_days: int) -> int:
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


def _has_age_in_range(text: str, low: int, high: int) -> bool:
    return any(
        low <= int(match.group(1)) <= high
        for match in re.compile(r'\b(\d{2,3})\s*[- ]year[- ]old\b', re.IGNORECASE).finditer(text)
    )


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
