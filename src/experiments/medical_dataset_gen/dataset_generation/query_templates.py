from __future__ import annotations

import re

import yaml

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AnswerTemplateSpec,
    MedicalOntology,
    QueryPlan,
    QueryTemplateData,
    QueryTemplateSpec,
    QueryType,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    MedicalDatasetGenPaths,
)

_QUERY_TEMPLATE_PATH = (
    MedicalDatasetGenPaths.root / 'data_templates' / 'query_answer_templates.yaml'
)


def _load_query_template_data() -> QueryTemplateData:
    with open(_QUERY_TEMPLATE_PATH) as f:
        return QueryTemplateData.model_validate(yaml.safe_load(f) or {})


QUERY_TEMPLATE_DATA = _load_query_template_data()


def render_query_template(plan: QueryPlan, ontology: MedicalOntology) -> str:
    template = query_template_spec(plan.query_type, plan.template_id).template

    context = {
        'condition': plan.condition_display,
        'condition_id': plan.condition_id,
        'subgroup_a': plan.subgroup_a_label,
        'subgroup_a_id': plan.subgroup_a_id,
        'subgroup_b': plan.subgroup_b_label,
        'subgroup_b_id': plan.subgroup_b_id,
        'primary_axis_label': ontology.clinical_axes[plan.primary_axis].label,
        'secondary_axis_label': ontology.clinical_axes[plan.secondary_axis].label,
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
    template = answer_template_spec(plan.query_type).template
    context = {
        'condition': plan.condition_display,
        'condition_id': plan.condition_id,
        'subgroup_a': plan.subgroup_a_label,
        'subgroup_a_id': plan.subgroup_a_id,
        'subgroup_b': plan.subgroup_b_label,
        'subgroup_b_id': plan.subgroup_b_id,
        'primary_axis_label': ontology.clinical_axes[plan.primary_axis].label,
        'secondary_axis_label': ontology.clinical_axes[plan.secondary_axis].label,
        'subgroup_a_primary': subgroup_a_primary,
        'subgroup_a_secondary': subgroup_a_secondary,
        'subgroup_b_primary': subgroup_b_primary,
        'subgroup_b_secondary': subgroup_b_secondary,
    }
    return squash_whitespaces(template.format(**context))


def query_template_ids(query_type: QueryType) -> list[str]:
    return [spec.id for spec in QUERY_TEMPLATE_DATA.query_templates[query_type]]


def query_template_spec(query_type: QueryType, template_id: str) -> QueryTemplateSpec:
    for spec in QUERY_TEMPLATE_DATA.query_templates[query_type]:
        if spec.id == template_id:
            return spec
    raise KeyError(f'unknown query template id for {query_type}: {template_id}')


def answer_template_spec(query_type: QueryType) -> AnswerTemplateSpec:
    try:
        return QUERY_TEMPLATE_DATA.answer_templates[query_type]
    except KeyError as exc:
        raise KeyError(f'unknown answer template for query type: {query_type}') from exc


def squash_whitespaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
