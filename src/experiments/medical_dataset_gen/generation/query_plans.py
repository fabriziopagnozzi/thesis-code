"""Build the hidden multi-aspect query plans for the synthetic benchmark.

This module exists to define the benchmark geometry before any text is
rendered, so each query has explicit facets, conditions, and subgroup
comparisons. It uses ontology-driven enumeration, seeded randomness, and
balanced plan materialization to keep the query set controlled and reproducible.
"""

from random import Random
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.generation.ontology import (
    axis_ids,
    load_ontology,
    selected_conditions,
    subgroup_pairs,
)
from experiments.medical_dataset_gen.generation.schemas import (
    ClinicalConditionOntology,
    ClinicalSubgroupOntology,
    FacetAxis,
    QueryLogicalForm,
    QueryPlan,
    QueryPlanFacet,
    QueryPlanSpec,
    QueryType,
    Split,
)
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    write_parquet,
)


def run_make_query_plans(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    conditions = selected_conditions(ontology, cfg.global_.conditions)
    pairs = subgroup_pairs(ontology)
    axes = axis_ids(ontology)
    if set(axes) != {'treatment_duration', 'rehab_outcome'}:
        raise ValueError(f'MVP expects treatment_duration and rehab_outcome axes, got {axes}')

    rng = Random(cfg.global_.seed)
    rows: list[dict[str, object]] = []
    plan_idx = 0
    per_condition_specs = [
        _plan_specs_for_condition(condition_id, condition, pairs, cfg.generation.query_types)
        for condition_id, condition in conditions
    ]
    offsets = [0] * len(per_condition_specs)
    while len(rows) < cfg.global_.n_queries:
        emitted = False
        for condition_idx, specs in enumerate(per_condition_specs):
            offset = offsets[condition_idx]
            if offset >= len(specs):
                continue
            spec = specs[offset]
            offsets[condition_idx] += 1
            emitted = True
            plan_idx += 1
            rows.append(_materialize_plan_row(cfg, rng, plan_idx, spec).to_row())
            if len(rows) >= cfg.global_.n_queries:
                break
        if not emitted:
            break

    df = pl.DataFrame(rows)
    write_parquet(paths, 'query_plans', df)
    return df


def _plan_specs_for_condition(
    condition_id: str,
    condition: ClinicalConditionOntology,
    pairs: list[tuple[tuple[str, ClinicalSubgroupOntology], tuple[str, ClinicalSubgroupOntology]]],
    query_types: list[str],
) -> list[QueryPlanSpec]:
    specs: list[QueryPlanSpec] = []
    for (subgroup_a_id, subgroup_a), (subgroup_b_id, subgroup_b) in pairs:
        for query_type in query_types:
            specs.append(
                QueryPlanSpec(
                    query_type=cast(QueryType, query_type),
                    condition_id=condition_id,
                    condition_display=condition.display,
                    subgroup_a_id=subgroup_a_id,
                    subgroup_a=subgroup_a,
                    subgroup_b_id=subgroup_b_id,
                    subgroup_b=subgroup_b,
                )
            )
    return specs


def _materialize_plan_row(
    cfg: ExperimentCfg, rng: Random, plan_idx: int, spec: QueryPlanSpec
) -> QueryPlan:
    query_id = f'q{plan_idx:05d}'
    dominant_slot = (plan_idx - 1) % 4
    facets = _facets_for_plan(
        query_id=query_id,
        condition_id=spec.condition_id,
        condition_display=spec.condition_display,
        subgroup_a_id=spec.subgroup_a_id,
        subgroup_a=spec.subgroup_a,
        subgroup_b_id=spec.subgroup_b_id,
        subgroup_b=spec.subgroup_b,
        dominant_slot=dominant_slot,
        dominant_size=cfg.generation.gold_chunks_dominant,
        complementary_size=cfg.generation.gold_chunks_complementary,
    )
    dominant_facet_id = facets[dominant_slot].facet_id
    logical_form = QueryLogicalForm(
        type=spec.query_type,
        condition=spec.condition_id,
        subgroups=[spec.subgroup_a_id, spec.subgroup_b_id],
        axes=['treatment_duration', 'rehab_outcome'],
        facets=[facet.facet_id for facet in facets],
        dominant_facet_id=dominant_facet_id,
    )
    return QueryPlan(
        query_id=query_id,
        plan_seed=rng.randint(0, 2**31 - 1),
        split=_split_for_index(plan_idx),
        query_type=spec.query_type,
        template_id=spec.query_type,
        condition_id=spec.condition_id,
        condition_display=spec.condition_display,
        **_subgroup_fields('subgroup_a', spec.subgroup_a_id, spec.subgroup_a),
        **_subgroup_fields('subgroup_b', spec.subgroup_b_id, spec.subgroup_b),
        dominant_facet_id=dominant_facet_id,
        n_facets=len(facets),
        gold_chunks_total=sum(facet.target_gold_chunks for facet in facets),
        distractor_chunks=cfg.generation.distractors_per_query,
        facets=facets,
        logical_form=logical_form,
    )


def _facets_for_plan(
    query_id: str,
    condition_id: str,
    condition_display: str,
    subgroup_a_id: str,
    subgroup_a: ClinicalSubgroupOntology,
    subgroup_b_id: str,
    subgroup_b: ClinicalSubgroupOntology,
    dominant_slot: int,
    dominant_size: int,
    complementary_size: int,
) -> list[QueryPlanFacet]:
    subgroup_a_duration, subgroup_b_duration = (
        ('short', 'prolonged') if dominant_slot in {0, 1} else ('prolonged', 'short')
    )
    subgroup_a_rehab, subgroup_b_rehab = (
        ('home_rehab', 'inpatient_rehab')
        if dominant_slot in {0, 2}
        else ('persistent_deficit', 'home_rehab')
    )
    raw_facets = [
        (subgroup_a_id, subgroup_a, 'treatment_duration', subgroup_a_duration),
        (subgroup_a_id, subgroup_a, 'rehab_outcome', subgroup_a_rehab),
        (subgroup_b_id, subgroup_b, 'treatment_duration', subgroup_b_duration),
        (subgroup_b_id, subgroup_b, 'rehab_outcome', subgroup_b_rehab),
    ]
    facets = []
    for idx, (subgroup_id, subgroup, axis, value_bin) in enumerate(raw_facets):
        facet_id = f'{query_id}_f{idx + 1}'
        is_dominant = idx == dominant_slot
        facets.append(
            QueryPlanFacet(
                facet_id=facet_id,
                condition_id=condition_id,
                condition_display=condition_display,
                subgroup_id=subgroup_id,
                subgroup_label=subgroup.label,
                subgroup_axis=subgroup.axis,
                subgroup_field=subgroup.field,
                subgroup_value=subgroup.value,
                axis=cast(FacetAxis, axis),
                value_bin=value_bin,
                cluster_id=f'{query_id}_c{idx + 1}',
                cluster_role='dominant_gold' if is_dominant else 'complementary_gold',
                target_gold_chunks=dominant_size if is_dominant else complementary_size,
            )
        )
    return facets


def _split_for_index(plan_idx: int) -> Split:
    bucket = plan_idx % 20
    if bucket in {0, 1, 2}:
        return 'test'
    if bucket in {3, 4, 5}:
        return 'validation'
    return 'train'


def _subgroup_fields(
    prefix: str, subgroup_id: str, subgroup: ClinicalSubgroupOntology
) -> dict[str, object]:
    return {
        f'{prefix}_id': subgroup_id,
        f'{prefix}_label': subgroup.label,
        f'{prefix}_axis': subgroup.axis,
        f'{prefix}_field': subgroup.field,
        f'{prefix}_value': subgroup.value,
    }


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
    run_make_query_plans(cfg, paths)
