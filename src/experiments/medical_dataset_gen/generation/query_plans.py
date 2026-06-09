from random import Random
from typing import Any

import polars as pl

from experiments.medical_dataset_gen.generation.ontology import (
    axis_ids,
    load_ontology,
    selected_conditions,
    subgroup_pairs,
)
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    json_dumps,
    write_parquet,
)


def run_make_query_plans(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    conditions = selected_conditions(ontology, cfg.global_.conditions)
    pairs = subgroup_pairs(ontology)
    axes = axis_ids(ontology)
    if set(axes) != {'treatment_duration', 'rehab_outcome'}:
        raise ValueError(f'MVP expects treatment_duration and rehab_outcome axes, got {axes}')

    rng = Random(cfg.global_.seed)
    rows: list[dict[str, Any]] = []
    plan_idx = 0
    for condition_id, condition in conditions:
        for (subgroup_a_id, subgroup_a), (subgroup_b_id, subgroup_b) in pairs:
            for query_type in cfg.generation.query_types:
                if len(rows) >= cfg.global_.n_queries:
                    break
                plan_idx += 1
                query_id = f'q{plan_idx:05d}'
                dominant_slot = (plan_idx - 1) % 4
                facets = _facets_for_plan(
                    query_id=query_id,
                    condition_id=condition_id,
                    condition_display=condition['display'],
                    subgroup_a_id=subgroup_a_id,
                    subgroup_a=subgroup_a,
                    subgroup_b_id=subgroup_b_id,
                    subgroup_b=subgroup_b,
                    dominant_slot=dominant_slot,
                    dominant_size=cfg.generation.gold_chunks_dominant,
                    complementary_size=cfg.generation.gold_chunks_complementary,
                )
                dominant_facet_id = facets[dominant_slot]['facet_id']
                logical_form = {
                    'type': query_type,
                    'condition': condition_id,
                    'subgroups': [subgroup_a_id, subgroup_b_id],
                    'axes': ['treatment_duration', 'rehab_outcome'],
                    'facets': [f['facet_id'] for f in facets],
                    'dominant_facet_id': dominant_facet_id,
                }
                rows.append({
                    'query_id': query_id,
                    'plan_seed': rng.randint(0, 2**31 - 1),
                    'split': _split_for_index(plan_idx),
                    'query_type': query_type,
                    'template_id': query_type,
                    'condition_id': condition_id,
                    'condition_display': condition['display'],
                    **_subgroup_fields('subgroup_a', subgroup_a_id, subgroup_a),
                    **_subgroup_fields('subgroup_b', subgroup_b_id, subgroup_b),
                    'dominant_facet_id': dominant_facet_id,
                    'n_facets': len(facets),
                    'gold_chunks_total': sum(f['target_gold_chunks'] for f in facets),
                    'distractor_chunks': cfg.generation.distractors_per_query,
                    'facets_json': json_dumps(facets),
                    'logical_form_json': json_dumps(logical_form),
                })
            if len(rows) >= cfg.global_.n_queries:
                break
        if len(rows) >= cfg.global_.n_queries:
            break

    df = pl.DataFrame(rows)
    write_parquet(paths, 'query_plans', df)
    return df


def _facets_for_plan(
    query_id: str,
    condition_id: str,
    condition_display: str,
    subgroup_a_id: str,
    subgroup_a: dict[str, Any],
    subgroup_b_id: str,
    subgroup_b: dict[str, Any],
    dominant_slot: int,
    dominant_size: int,
    complementary_size: int,
) -> list[dict[str, Any]]:
    raw_facets = [
        (subgroup_a_id, subgroup_a, 'treatment_duration'),
        (subgroup_a_id, subgroup_a, 'rehab_outcome'),
        (subgroup_b_id, subgroup_b, 'treatment_duration'),
        (subgroup_b_id, subgroup_b, 'rehab_outcome'),
    ]
    facets = []
    for idx, (subgroup_id, subgroup, axis) in enumerate(raw_facets):
        facet_id = f'{query_id}_f{idx + 1}'
        is_dominant = idx == dominant_slot
        facets.append({
            'facet_id': facet_id,
            'condition_id': condition_id,
            'condition_display': condition_display,
            'subgroup_id': subgroup_id,
            'subgroup_label': subgroup['label'],
            'subgroup_axis': subgroup['axis'],
            'subgroup_field': subgroup['field'],
            'subgroup_value': subgroup['value'],
            'axis': axis,
            'cluster_id': f'{query_id}_c{idx + 1}',
            'cluster_role': 'dominant_gold' if is_dominant else 'complementary_gold',
            'target_gold_chunks': dominant_size if is_dominant else complementary_size,
        })
    return facets


def _split_for_index(plan_idx: int) -> str:
    bucket = plan_idx % 20
    if bucket in {0, 1, 2}:
        return 'test'
    if bucket in {3, 4, 5}:
        return 'validation'
    return 'train'


def _subgroup_fields(prefix: str, subgroup_id: str, subgroup: dict[str, Any]) -> dict[str, Any]:
    return {
        f'{prefix}_id': subgroup_id,
        f'{prefix}_label': subgroup['label'],
        f'{prefix}_axis': subgroup['axis'],
        f'{prefix}_field': subgroup['field'],
        f'{prefix}_value': subgroup['value'],
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
