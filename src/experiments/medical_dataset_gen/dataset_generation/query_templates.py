from __future__ import annotations

import re

import yaml

from experiments.medical_dataset_gen.dataset_generation.chunk_templates import TEMPLATE_DATA_DIR
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    ClinicalAxis,
    MedicalOntology,
    QueryFocusMode,
    QueryPlan,
    QueryStructure,
    QueryTemplateData,
    QueryTemplateSpec,
    canonical_query_focus_mode,
)


def _load_query_template_data() -> QueryTemplateData:
    with open(TEMPLATE_DATA_DIR / 'query_answer_templates.yaml') as f:
        return QueryTemplateData.model_validate(yaml.safe_load(f) or {})


QUERY_TEMPLATE_DATA = _load_query_template_data()


def render_query_template(
    plan: QueryPlan,
    ontology: MedicalOntology,
    *,
    template_id: str | None = None,
    focus_mode: QueryFocusMode = 'natural',
    query_structure: QueryStructure = 'unbalanced',
) -> str:
    focus_mode = canonical_query_focus_mode(query_structure, focus_mode)
    resolved_template_id = template_id
    if resolved_template_id is None and plan.template_id != 'deferred':
        resolved_template_id = plan.template_id
    if resolved_template_id is None:
        resolved_template_id = query_template_ids(query_structure, focus_mode)[0]
    template = query_template_spec(
        resolved_template_id,
        query_structure=query_structure,
        focus_mode=focus_mode,
    ).template
    primary_axis = ontology.clinical_axes[plan.primary_axis]
    secondary_axis = ontology.clinical_axes[plan.secondary_axis]

    context = {
        'condition': plan.condition_display,
        'condition_id': plan.condition_id,
        'subgroup_a': plan.subgroup_a_label,
        'subgroup_a_id': plan.subgroup_a_id,
        'subgroup_b': plan.subgroup_b_label,
        'subgroup_b_id': plan.subgroup_b_id,
        'primary_axis_label': axis_query_label(plan.primary_axis, ontology),
        'primary_axis_focus': primary_axis.query_focus.text_for(focus_mode),
        'secondary_axis_label': axis_query_label(plan.secondary_axis, ontology),
        'secondary_axis_focus': secondary_axis.query_focus.text_for(focus_mode),
    }

    return squash_whitespaces(template.format(**context))


def render_answer_template(
    plan: QueryPlan,
    *,
    subgroup_a_primary: str,
    subgroup_a_secondary: str,
    subgroup_b_primary: str,
    subgroup_b_secondary: str,
    ontology: MedicalOntology,
) -> str:
    template = QUERY_TEMPLATE_DATA.answer_template.template
    context = {
        'condition': plan.condition_display,
        'condition_id': plan.condition_id,
        'subgroup_a': plan.subgroup_a_label,
        'subgroup_a_id': plan.subgroup_a_id,
        'subgroup_b': plan.subgroup_b_label,
        'subgroup_b_id': plan.subgroup_b_id,
        'primary_axis_label': axis_query_label(plan.primary_axis, ontology),
        'secondary_axis_label': axis_query_label(plan.secondary_axis, ontology),
        'subgroup_a_primary': subgroup_a_primary,
        'subgroup_a_secondary': subgroup_a_secondary,
        'subgroup_b_primary': subgroup_b_primary,
        'subgroup_b_secondary': subgroup_b_secondary,
    }
    return squash_whitespaces(template.format(**context))


def query_template_ids(
    query_structure: QueryStructure = 'unbalanced',
    focus_mode: QueryFocusMode = 'natural',
) -> list[str]:
    focus_mode = canonical_query_focus_mode(query_structure, focus_mode)
    return [spec.id for spec in QUERY_TEMPLATE_DATA.query_templates[query_structure][focus_mode]]


def query_template_spec(
    template_id: str,
    *,
    query_structure: QueryStructure = 'unbalanced',
    focus_mode: QueryFocusMode = 'natural',
) -> QueryTemplateSpec:
    focus_mode = canonical_query_focus_mode(query_structure, focus_mode)
    for spec in QUERY_TEMPLATE_DATA.query_templates[query_structure][focus_mode]:
        if spec.id == template_id:
            return spec
    raise KeyError(f'unknown query template id for {query_structure}/{focus_mode}: {template_id}')


def axis_query_label(axis: ClinicalAxis, ontology: MedicalOntology | None = None) -> str:
    """Return the ontology-owned query wording, with a legacy-safe fallback."""
    if ontology is not None:
        return ontology.clinical_axes[axis].query_label
    return axis.replace('_', ' ')


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
