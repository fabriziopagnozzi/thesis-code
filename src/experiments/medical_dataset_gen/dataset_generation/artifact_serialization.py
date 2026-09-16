"""Explicit parquet-row serialization for benchmark construction artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping

from experiments.medical_dataset_gen.dataset_generation.artifact_schemas import (
    AnswerFact,
    GoldAnswerOutputRow,
    QueryOutputRow,
    QueryPlan,
)


def query_plan_from_parquet_row(row: Mapping[str, object]) -> QueryPlan:
    payload = dict(row)
    facets_json = _required_json_column(payload, 'facets_json')
    logical_form_json = _required_json_column(payload, 'logical_form_json')
    payload['facets'] = json.loads(facets_json)
    payload['logical_form'] = json.loads(logical_form_json)
    return QueryPlan.model_validate(payload)


def query_plan_to_parquet_row(plan: QueryPlan) -> dict[str, object]:
    row = plan.model_dump(mode='python', exclude={'facets', 'logical_form'})
    facets_json, logical_form_json = _query_plan_json_columns(plan)
    row['facets_json'] = facets_json
    row['logical_form_json'] = logical_form_json
    return row


def query_output_row(
    plan: QueryPlan,
    *,
    query_text: str,
    template_id: str,
) -> QueryOutputRow:
    facets_json, logical_form_json = _query_plan_json_columns(plan)
    return {
        'query_id': plan.query_id,
        'evidence_profile_id': plan.evidence_profile_id,
        'pool_id': plan.pool_id,
        'outcome_profile_id': plan.outcome_profile_id,
        'query_type': plan.query_type,
        'template_id': template_id,
        'condition_id': plan.condition_id,
        'condition_display': plan.condition_display,
        'subgroup_a_id': plan.subgroup_a_id,
        'subgroup_a_label': plan.subgroup_a_label,
        'subgroup_b_id': plan.subgroup_b_id,
        'subgroup_b_label': plan.subgroup_b_label,
        'cohort_contrast_id': plan.cohort_contrast_id,
        'cohort_contrast_family': plan.cohort_contrast_family,
        'cohort_dimension_id': plan.cohort_dimension_id,
        'primary_axis': plan.primary_axis,
        'secondary_axis': plan.secondary_axis,
        'dominant_primary_facet_id': plan.dominant_primary_facet_id,
        'split': plan.split,
        'n_facets': len(plan.facets),
        'facets_json': facets_json,
        'logical_form_json': logical_form_json,
        'query_text': query_text,
    }


def gold_answer_output_row(
    plan: QueryPlan,
    *,
    answer_text: str,
    facet_summaries: dict[str, str],
    facet_answer_objects: list[AnswerFact],
    supporting_fact_ids: list[str],
) -> GoldAnswerOutputRow:
    return {
        'query_id': plan.query_id,
        'evidence_profile_id': plan.evidence_profile_id,
        'pool_id': plan.pool_id,
        'answer_text': answer_text,
        'facet_summaries_json': json.dumps(facet_summaries, sort_keys=True),
        'answer_facts_json': json.dumps(
            [fact.model_dump(mode='json') for fact in facet_answer_objects], sort_keys=True
        ),
        'supporting_fact_ids_json': json.dumps(supporting_fact_ids, sort_keys=True),
        'supporting_facet_ids_json': json.dumps(
            [facet.facet_id for facet in plan.facets], sort_keys=True
        ),
    }


def _query_plan_json_columns(plan: QueryPlan) -> tuple[str, str]:
    facets_json = json.dumps(
        [facet.model_dump(mode='json') for facet in plan.facets], sort_keys=True
    )
    logical_form_json = json.dumps(
        plan.logical_form.model_dump(mode='json', by_alias=True), sort_keys=True
    )
    return facets_json, logical_form_json


def _required_json_column(payload: dict[str, object], name: str) -> str:
    try:
        value = payload.pop(name)
    except KeyError as exc:
        raise ValueError(f'query-plan row is missing {name!r}') from exc
    if not isinstance(value, str):
        raise TypeError(f'query-plan column {name!r} must be a JSON string')
    return value
