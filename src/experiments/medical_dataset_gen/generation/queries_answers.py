import json
from collections import Counter, defaultdict
from typing import Any

import polars as pl

from experiments.medical_dataset_gen.generation.ontology import load_ontology
from experiments.medical_dataset_gen.generation.text_templates import (
    canonical_answer,
    maybe_paraphrase_query,
    render_query,
)
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    json_dumps,
    read_parquet,
    write_parquet,
)


def run_make_queries_answers(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ontology = load_ontology(cfg)
    plans_df = read_parquet(paths, 'query_plans')
    facts_df = read_parquet(paths, 'clinical_facts')
    failed_query_ids = _failed_query_ids(paths)

    facts_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts_df.filter(pl.col('is_gold')).iter_rows(named=True):
        facts_by_query[fact['query_id']].append(dict(fact))

    query_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []

    for plan_row in plans_df.iter_rows(named=True):
        plan = dict(plan_row)
        if plan['query_id'] in failed_query_ids:
            continue
        plan['facets'] = json.loads(plan['facets_json'])
        query_text = render_query(plan, ontology)
        query_text = maybe_paraphrase_query(
            query_text=query_text,
            plan=plan,
            llm_name=cfg.generation.llm_name,
            use_llm=cfg.generation.use_llm_query_paraphrase,
            temperature=cfg.generation.llm_temperature,
            num_ctx=cfg.generation.llm_num_ctx,
        )

        facet_summaries, facet_answer_objects = _facet_summaries(
            plan['facets'],
            facts_by_query[plan['query_id']],
        )
        answer_text = canonical_answer(plan, facet_summaries)

        query_rows.append(_query_row(plan, query_text))
        answer_rows.append(
            _answer_row(
                plan=plan,
                answer_text=answer_text,
                facet_summaries=facet_summaries,
                facet_answer_objects=facet_answer_objects,
                supporting_facts=facts_by_query[plan['query_id']],
            )
        )

    queries = pl.DataFrame(query_rows)
    answers = pl.DataFrame(answer_rows)
    write_parquet(paths, 'queries', queries)
    write_parquet(paths, 'gold_answers', answers)
    if failed_query_ids:
        print(f'[queries_answers] skipped {len(failed_query_ids):,} failed query/queries from chunk generation')
    return queries, answers


def _facet_summaries(
    facets: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_facet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        by_facet[fact['facet_id']].append(fact)

    summaries = {}
    answer_facts = []
    for facet in facets:
        facet_id = facet['facet_id']
        rows = by_facet[facet_id]
        if not rows:
            summaries[facet_id] = 'no generated evidence'
            continue

        bins = Counter(row['value_bin'] for row in rows)
        mode_bin = bins.most_common(1)[0][0]
        if facet['axis'] == 'treatment_duration':
            durations = [int(row['duration_days']) for row in rows if row['duration_days'] is not None]
            avg_duration = round(sum(durations) / len(durations), 1)
            treatment = Counter(row['treatment'] for row in rows).most_common(1)[0][0]
            text = f'a {mode_bin.replace("_", " ")} course, averaging {avg_duration} days, most often with {treatment}'
        else:
            example = Counter(row['rehab_outcome'] for row in rows).most_common(1)[0][0]
            text = f'a {mode_bin.replace("_", " ")} pattern, commonly described as {example}'

        summaries[facet_id] = text
        answer_facts.append({
            'facet_id': facet_id,
            'subgroup_label': facet['subgroup_label'],
            'axis': facet['axis'],
            'summary': text,
            'supporting_fact_ids': [row['fact_id'] for row in rows],
        })

    return summaries, answer_facts


def _query_row(plan: dict[str, Any], query_text: str) -> dict[str, Any]:
    return _pick(
        plan,
        'query_id',
        'query_type',
        'template_id',
        'condition_id',
        'condition_display',
        'subgroup_a_id',
        'subgroup_a_label',
        'subgroup_b_id',
        'subgroup_b_label',
        'dominant_facet_id',
        'split',
        'n_facets',
        'facets_json',
        'logical_form_json',
    ) | {'query_text': query_text}


def _answer_row(
    plan: dict[str, Any],
    answer_text: str,
    facet_summaries: dict[str, str],
    facet_answer_objects: list[dict[str, Any]],
    supporting_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        'query_id': plan['query_id'],
        'answer_text': answer_text,
        'facet_summaries_json': json_dumps(facet_summaries),
        'answer_facts_json': json_dumps(facet_answer_objects),
        'supporting_fact_ids_json': json_dumps([fact['fact_id'] for fact in supporting_facts]),
        'supporting_facet_ids_json': json_dumps([facet['facet_id'] for facet in plan['facets']]),
    }


def _pick(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys}


def _failed_query_ids(paths: MedicalDatasetGenPaths) -> set[str]:
    reject_path = paths.table_path('generation_rejects')
    if not reject_path.exists():
        return set()
    rejects = read_parquet(paths, 'generation_rejects')
    if 'query_id' not in rejects.columns or rejects.height == 0:
        return set()
    return {str(query_id) for query_id in rejects['query_id'].drop_nulls().to_list()}


if __name__ == '__main__':
    from experiments.medical_dataset_gen.global_configs import (
        dump_effective_config,
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    dump_effective_config(cfg, paths)
    run_make_queries_answers(cfg, paths)
