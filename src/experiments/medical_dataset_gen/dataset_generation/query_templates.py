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
    TemplateSpec,
)
from experiments.medical_dataset_gen.utils.deterministic_ids import stable_id, stable_int


def _load_query_template_data() -> QueryTemplateData:
    with open(TEMPLATE_DATA_DIR / 'query_answer_templates.yaml') as f:
        return QueryTemplateData.model_validate(yaml.safe_load(f) or {})


QUERY_TEMPLATE_DATA = _load_query_template_data()


def render_query_template(
    plan: QueryPlan,
    ontology: MedicalOntology,
    *,
    template_id: str,
    focus_mode: QueryFocusMode = 'natural',
    query_structure: QueryStructure = 'unbalanced',
) -> str:
    template = query_template_spec(
        template_id,
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
    return [spec.id for spec in QUERY_TEMPLATE_DATA.query_templates[query_structure][focus_mode]]


def query_template_spec(
    template_id: str,
    *,
    query_structure: QueryStructure = 'unbalanced',
    focus_mode: QueryFocusMode = 'natural',
) -> TemplateSpec:
    for spec in QUERY_TEMPLATE_DATA.query_templates[query_structure][focus_mode]:
        if spec.id == template_id:
            return spec
    raise KeyError(f'unknown query template id for {query_structure}/{focus_mode}: {template_id}')


def select_query_template_id(
    plan: QueryPlan,
    *,
    dataset_schema_version: int,
    global_seed: int,
    query_structure: QueryStructure,
    focus_mode: QueryFocusMode,
) -> str:
    """Select the authored frame deterministically for one query realization."""
    template_ids = query_template_ids(query_structure, focus_mode)
    query_key = stable_id(
        'qv4',
        dataset_schema_version,
        global_seed,
        plan.evidence_profile_id,
        plan.primary_axis,
        plan.secondary_axis,
    )
    return template_ids[stable_int(query_key, 'template') % len(template_ids)]


def axis_query_label(axis: ClinicalAxis, ontology: MedicalOntology) -> str:
    return ontology.clinical_axes[axis].query_label


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
