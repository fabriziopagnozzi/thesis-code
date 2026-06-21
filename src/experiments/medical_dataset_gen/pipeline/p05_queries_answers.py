from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import polars as pl
import pyarrow.parquet as pq

from experiments.medical_dataset_gen.dataset_generation.ontology_utils import load_ontology
from experiments.medical_dataset_gen.dataset_generation.prompts_default import (
    MedicalDatasetGenDefaultPrompts,
)
from experiments.medical_dataset_gen.dataset_generation.query_templates import (
    render_answer_template,
    render_query_template,
)
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AnswerFact,
    AnswerSourceFact,
    ClinicalAxis,
    ClinicalAxisOntology,
    GoldAnswerOutputRow,
    MedicalOntology,
    QueryOutputRow,
    QueryPlan,
    QueryPlanFacet,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet
from helpers.ollama_client import generate


def run_make_queries_answers(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ontology = load_ontology(cfg)
    plans_df = read_parquet(paths, 'query_plans')
    failed_query_ids = _failed_query_ids(paths)

    plans_by_query: dict[str, QueryPlan] = {}
    for plan_row in plans_df.iter_rows(named=True):
        plan = QueryPlan.model_validate(plan_row)
        if plan.query_id not in failed_query_ids:
            plans_by_query[plan.query_id] = plan

    query_rows: list[QueryOutputRow] = []
    answer_rows: list[GoldAnswerOutputRow] = []

    current_query_id: str | None = None
    current_fact_rows: list[AnswerSourceFact] = []
    facts_file = pq.ParquetFile(paths.table_path('clinical_facts'))
    for batch in facts_file.iter_batches(  # type: ignore
        columns=[
            'query_id',
            'is_gold',
            'facet_id',
            'axis',
            'value_bin',
            'duration_days',
            'treatment',
            'rehab_outcome',
            'fact_id',
        ],
        batch_size=65536,
    ):
        for raw_fact_row in batch.to_pylist():
            if not raw_fact_row['is_gold']:
                continue
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
    write_parquet(paths, 'queries', queries)
    write_parquet(paths, 'gold_answers', answers)
    if failed_query_ids:
        print(
            f'[queries_answers] skipped {len(failed_query_ids):,} failed query/queries from chunk generation'
        )
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

    query_text = render_query(plan, ontology)
    query_text = maybe_paraphrase_query(
        query_text=query_text,
        plan=plan,
        ontology=ontology,
        llm_name=cfg.generation.llm_name,
        use_llm=cfg.generation.use_llm_query_paraphrase,
        temperature=cfg.generation.llm_temperature,
        num_ctx=cfg.generation.llm_num_ctx,
    )

    facet_summaries, facet_answer_objects = _facet_summaries(plan.facets, fact_rows)
    answer_text = _canonical_answer(plan, facet_summaries)

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
) -> str:
    return render_query_template(plan, ontology)


def maybe_paraphrase_query(
    query_text: str,
    plan: QueryPlan,
    ontology: MedicalOntology,
    llm_name: str,
    use_llm: bool,
    temperature: float,
    num_ctx: int,
) -> str:
    if not use_llm:
        return query_text

    prompt = MedicalDatasetGenDefaultPrompts.query_paraphrase_prompt(query_text, plan)
    paraphrase = generate(prompt, model=llm_name, temperature=temperature, num_ctx=num_ctx).strip()
    required = [plan.condition_display, plan.subgroup_a_label, plan.subgroup_b_label]
    lower = paraphrase.lower()
    has_required_entities = all(label.lower() in lower for label in required)
    has_balanced_axes = all(
        _contains_axis_language(lower, ontology.clinical_axes[axis_id])
        for axis_id in ('treatment_duration', 'rehab_outcome')
    )
    if paraphrase and has_required_entities and has_balanced_axes:
        return paraphrase
    return query_text


def _contains_axis_language(lower_text: str, axis: ClinicalAxisOntology) -> bool:
    terms = [axis.label, *axis.exact_terms, *axis.synonym_terms]
    return any(str(term).lower() in lower_text for term in terms)


def _canonical_answer(plan: QueryPlan, facet_summaries: dict[str, str]) -> str:
    subgroup_a_duration = facet_summaries[
        facets_by(plan, plan.subgroup_a_label, 'treatment_duration')
    ]
    subgroup_a_rehab = facet_summaries[facets_by(plan, plan.subgroup_a_label, 'rehab_outcome')]
    subgroup_b_duration = facet_summaries[
        facets_by(plan, plan.subgroup_b_label, 'treatment_duration')
    ]
    subgroup_b_rehab = facet_summaries[facets_by(plan, plan.subgroup_b_label, 'rehab_outcome')]
    return render_answer_template(
        plan,
        subgroup_a_duration=subgroup_a_duration,
        subgroup_a_rehab=subgroup_a_rehab,
        subgroup_b_duration=subgroup_b_duration,
        subgroup_b_rehab=subgroup_b_rehab,
    )


def facets_by(plan: QueryPlan, subgroup_label: str, axis: ClinicalAxis) -> str:
    for facet in plan.facets:
        if facet.subgroup_label == subgroup_label and facet.axis == axis:
            return facet.facet_id
    raise KeyError((subgroup_label, axis))


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
            summaries[facet_id] = 'no generated evidence'
            continue

        bins = Counter(row.value_bin for row in rows)
        mode_bin = bins.most_common(1)[0][0]
        if facet.axis == 'treatment_duration':
            duration_rows = [row for row in rows if row.axis == 'treatment_duration']
            if len(duration_rows) != len(rows):
                raise ValueError(f'facet {facet_id!r} mixes treatment and rehabilitation facts')
            durations = [
                row.duration_days for row in duration_rows if row.duration_days is not None
            ]
            avg_duration = round(sum(durations) / len(durations), 1)
            treatment = Counter(
                row.treatment for row in duration_rows if row.treatment is not None
            ).most_common(1)[0][0]
            text = (
                f'{_with_indefinite_article(mode_bin.replace("_", " "))} course, averaging {avg_duration} days, '
                f'most often with {treatment}'
            )
        else:
            rehab_rows = [row for row in rows if row.axis == 'rehab_outcome']
            if len(rehab_rows) != len(rows):
                raise ValueError(f'facet {facet_id!r} mixes treatment and rehabilitation facts')
            example = Counter(
                row.rehab_outcome for row in rehab_rows if row.rehab_outcome is not None
            ).most_common(1)[0][0]
            text = (
                f'{_with_indefinite_article(mode_bin.replace("_", " "))} pattern, '
                f'commonly described as {example}'
            )

        summaries[facet_id] = text
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


def _failed_query_ids(paths: MedicalDatasetGenPaths) -> set[str]:
    reject_path = paths.table_path('generation_rejects')
    if not reject_path.exists():
        return set()
    rejects = read_parquet(paths, 'generation_rejects')
    if 'query_id' not in rejects.columns or rejects.height == 0:
        return set()
    return {str(query_id) for query_id in rejects['query_id'].drop_nulls().to_list()}


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_make_queries_answers(cfg, paths)
