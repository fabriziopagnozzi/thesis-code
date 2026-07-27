from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import polars as pl

from experiments.medical_dataset_gen.dataset_generation.ontology_utils import load_ontology
from experiments.medical_dataset_gen.dataset_generation.query_templates import (
    render_answer_template,
    render_query_template,
)
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    AcuteClinicalCoursePayload,
    AnswerFact,
    AnswerSourceFact,
    CareIntensityPayload,
    ClinicalAxis,
    ComplicationBurdenPayload,
    DiagnosticEvidencePayload,
    GoldAnswerOutputRow,
    MedicalOntology,
    QueryFocusMode,
    QueryOutputRow,
    QueryPlan,
    QueryPlanFacet,
    QueryStructure,
    RehabOutcomePayload,
    TreatmentDurationPayload,
    parse_axis_payload,
)
from experiments.medical_dataset_gen.utils.global_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


def run_make_queries_answers(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ontology = load_ontology(cfg)
    plans_df = read_parquet(paths, 'query_plans')

    plans_by_query: dict[str, QueryPlan] = {}
    for plan_row in plans_df.iter_rows(named=True):
        plan = QueryPlan.model_validate(plan_row)
        plans_by_query[plan.query_id] = plan

    query_rows: list[QueryOutputRow] = []
    answer_rows: list[GoldAnswerOutputRow] = []

    current_query_id: str | None = None
    current_fact_rows: list[AnswerSourceFact] = []
    retained_memberships = (
        read_parquet(paths, 'chunk_memberships')
        .filter(pl.col('is_gold'))
        .select(
            'query_id',
            'facet_id',
            'axis',
            'value_bin',
            'axis_payload_json',
            'facet_priority',
            'fact_id',
        )
    )
    for raw_fact_row in retained_memberships.iter_rows(named=True):
        fact_row = AnswerSourceFact.model_validate(raw_fact_row)
        query_id = fact_row.query_id
        if current_query_id is None:
            current_query_id = query_id
        elif query_id != current_query_id:
            _finalize_query(
                query_id=current_query_id,
                fact_rows=current_fact_rows,
                plans_by_query=plans_by_query,
                ontology=ontology,
                cfg=cfg,
                query_rows=query_rows,
                answer_rows=answer_rows,
            )
            current_fact_rows = []
            current_query_id = query_id
        current_fact_rows.append(fact_row)

    if current_query_id is not None:
        _finalize_query(
            query_id=current_query_id,
            fact_rows=current_fact_rows,
            plans_by_query=plans_by_query,
            ontology=ontology,
            cfg=cfg,
            query_rows=query_rows,
            answer_rows=answer_rows,
        )

    queries = pl.from_dicts(query_rows, infer_schema_length=None)
    answers = pl.from_dicts(answer_rows, infer_schema_length=None)
    expected_queries = len(plans_by_query)
    if len(queries) != expected_queries or len(answers) != expected_queries:
        raise RuntimeError(
            'planned-versus-written query count mismatch: '
            f'plans={expected_queries}, queries={len(queries)}, answers={len(answers)}'
        )
    write_parquet(paths, 'queries', queries)
    write_parquet(paths, 'gold_answers', answers)
    return queries, answers


def _finalize_query(
    *,
    query_id: str,
    fact_rows: list[AnswerSourceFact],
    plans_by_query: dict[str, QueryPlan],
    ontology: MedicalOntology,
    cfg: ExperimentCfg,
    query_rows: list[QueryOutputRow],
    answer_rows: list[GoldAnswerOutputRow],
) -> None:
    plan = plans_by_query.get(query_id)
    if plan is None or not fact_rows:
        return

    query_text = render_query(
        plan,
        ontology,
        focus_mode=cfg.generation.focus_mode,
        query_structure=cfg.generation.query_structure,
    )

    facet_summaries, facet_answer_objects = _facet_summaries(plan.facets, fact_rows)
    answer_text = _canonical_answer(plan, facet_summaries, ontology)

    query_rows.append(plan.to_query_row(query_text))
    answer_rows.append(
        plan.to_answer_row(
            answer_text=answer_text,
            facet_summaries=facet_summaries,
            facet_answer_objects=facet_answer_objects,
            supporting_fact_ids=[fact.fact_id for fact in fact_rows],
        )
    )


def render_query(
    plan: QueryPlan,
    ontology: MedicalOntology,
    *,
    template_id: str | None = None,
    focus_mode: QueryFocusMode = 'natural',
    query_structure: QueryStructure = 'unbalanced',
) -> str:
    return render_query_template(
        plan,
        ontology,
        template_id=template_id,
        focus_mode=focus_mode,
        query_structure=query_structure,
    )


def _canonical_answer(
    plan: QueryPlan, facet_summaries: dict[str, str], ontology: MedicalOntology
) -> str:
    return render_answer_template(
        plan,
        subgroup_a_primary=facet_summaries[
            _profile_facet_key(plan.subgroup_a_id, plan.primary_axis)
        ],
        subgroup_a_secondary=facet_summaries[
            _profile_facet_key(plan.subgroup_a_id, plan.secondary_axis)
        ],
        subgroup_b_primary=facet_summaries[
            _profile_facet_key(plan.subgroup_b_id, plan.primary_axis)
        ],
        subgroup_b_secondary=facet_summaries[
            _profile_facet_key(plan.subgroup_b_id, plan.secondary_axis)
        ],
        ontology=ontology,
    )


def _profile_facet_key(subgroup_id: str, axis: ClinicalAxis) -> str:
    # Profile-level keys are invariant across the two query-specific facet-ID spaces.
    return f'{subgroup_id}|{axis}'


def _facet_summaries(
    facets: list[QueryPlanFacet],
    facts: Sequence[AnswerSourceFact],
) -> tuple[dict[str, str], list[AnswerFact]]:
    by_facet: dict[str, list[AnswerSourceFact]] = defaultdict(list)
    for fact in facts:
        by_facet[fact.facet_id].append(fact)

    summaries: dict[str, str] = {}
    answer_facts: list[AnswerFact] = []
    for facet in facets:
        facet_id = facet.facet_id
        rows = by_facet[facet_id]
        if not rows:
            summaries[_profile_facet_key(facet.subgroup_id, facet.axis)] = 'no generated evidence'
            continue

        if any(row.axis != facet.axis for row in rows):
            raise ValueError(f'facet {facet_id!r} mixes clinical axes')
        # Opposite-focus views always share local_idx=0, making this canonical
        # payload summary invariant to their different membership prefix sizes.
        payload = parse_axis_payload(rows[0].axis_payload_json)
        text = _summarize_payload(facet.value_bin, payload)

        summaries[_profile_facet_key(facet.subgroup_id, facet.axis)] = text
        answer_facts.append(
            AnswerFact(
                facet_id=facet_id,
                subgroup_label=facet.subgroup_label,
                axis=facet.axis,
                summary=text,
                supporting_fact_ids=[row.fact_id for row in rows],
            )
        )

    return summaries, answer_facts


def _with_indefinite_article(phrase: str) -> str:
    article = 'an' if phrase[:1].lower() in {'a', 'e', 'i', 'o', 'u'} else 'a'
    return f'{article} {phrase}'


def _summarize_payload(value_bin: str, payload) -> str:
    bin_label = value_bin.replace('_', ' ')
    if isinstance(payload, TreatmentDurationPayload):
        return f'{_with_indefinite_article(bin_label)} course of {payload.duration_days} days with {payload.treatment}'
    if isinstance(payload, RehabOutcomePayload):
        return f'{_with_indefinite_article(bin_label)} pattern: {payload.outcome}'
    if isinstance(payload, ComplicationBurdenPayload):
        return f'{_with_indefinite_article(bin_label)} pattern: {payload.detail}'
    if isinstance(payload, AcuteClinicalCoursePayload):
        return f'{_with_indefinite_article(bin_label)} trajectory: {payload.detail}'
    if isinstance(payload, CareIntensityPayload):
        return f'{_with_indefinite_article(bin_label)} level: {payload.detail}'
    if isinstance(payload, DiagnosticEvidencePayload):
        return f'{_with_indefinite_article(bin_label)} evidence source: {payload.detail}'
    raise TypeError(type(payload))


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.logging_utils import (
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_make_queries_answers(cfg, paths)
