import json
from random import Random
from typing import Any

import polars as pl

from experiments.medical_dataset_gen.generation.ontology import (
    load_ontology,
    other_conditions,
    other_subgroups,
)
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    json_dumps,
    read_parquet,
    write_parquet,
)


def run_make_facts(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    plans = read_parquet(paths, 'query_plans')
    rows: list[dict[str, Any]] = []

    for plan in plans.iter_rows(named=True):
        plan_seed = int(plan['plan_seed'])
        rng = Random(plan_seed)
        facets = json.loads(plan['facets_json'])
        for facet in facets:
            for local_idx in range(int(facet['target_gold_chunks'])):
                rows.append(
                    _gold_fact(
                        plan=plan,
                        facet=facet,
                        ontology=ontology,
                        local_idx=local_idx,
                        rng=rng,
                    )
                )

        rows.extend(_distractor_facts(plan=plan, ontology=ontology, rng=rng, n=cfg.generation.distractors_per_query))

    df = pl.DataFrame(rows)
    write_parquet(paths, 'clinical_facts', df)
    return df


def _gold_fact(
    plan: dict[str, Any],
    facet: dict[str, Any],
    ontology: dict[str, Any],
    local_idx: int,
    rng: Random,
) -> dict[str, Any]:
    return _base_fact(
        plan=plan,
        facet=facet,
        ontology=ontology,
        rng=rng,
        local_idx=local_idx,
        is_gold=True,
        distractor_type=None,
        condition_id=facet['condition_id'],
        condition_display=facet['condition_display'],
        subgroup_id=facet['subgroup_id'],
        subgroup_label=facet['subgroup_label'],
        subgroup_axis=facet['subgroup_axis'],
        subgroup_field=facet['subgroup_field'],
        subgroup_value=facet['subgroup_value'],
        axis=facet['axis'],
        cluster_id=facet['cluster_id'],
        cluster_role=facet['cluster_role'],
    )


def _distractor_facts(
    plan: dict[str, Any],
    ontology: dict[str, Any],
    rng: Random,
    n: int,
) -> list[dict[str, Any]]:
    facets = json.loads(plan['facets_json'])
    excluded_subgroups = {plan['subgroup_a_id'], plan['subgroup_b_id']}
    distractor_types = [
        'same_condition_wrong_subgroup',
        'same_subgroup_wrong_condition',
        'same_axis_wrong_condition',
    ]
    rows = []
    for local_idx in range(n):
        dtype = distractor_types[local_idx % len(distractor_types)]
        target_facet = facets[local_idx % len(facets)]

        condition_id = plan['condition_id']
        condition_display = plan['condition_display']
        subgroup_id = target_facet['subgroup_id']
        subgroup_label = target_facet['subgroup_label']
        subgroup_axis = target_facet['subgroup_axis']
        subgroup_field = target_facet['subgroup_field']
        subgroup_value = target_facet['subgroup_value']
        axis = target_facet['axis']

        if dtype == 'same_condition_wrong_subgroup':
            choices = other_subgroups(ontology, excluded_subgroups)
            subgroup_id, subgroup = choices[local_idx % len(choices)]
            subgroup_label = subgroup['label']
            subgroup_axis = subgroup['axis']
            subgroup_field = subgroup['field']
            subgroup_value = subgroup['value']
        elif dtype == 'same_subgroup_wrong_condition':
            choices = other_conditions(ontology, condition_id)
            condition_id, condition = choices[local_idx % len(choices)]
            condition_display = condition['display']
        elif dtype == 'same_axis_wrong_condition':
            condition_choices = other_conditions(ontology, condition_id)
            subgroup_choices = other_subgroups(ontology, excluded_subgroups)
            condition_id, condition = condition_choices[local_idx % len(condition_choices)]
            condition_display = condition['display']
            subgroup_id, subgroup = subgroup_choices[(local_idx + 1) % len(subgroup_choices)]
            subgroup_label = subgroup['label']
            subgroup_axis = subgroup['axis']
            subgroup_field = subgroup['field']
            subgroup_value = subgroup['value']

        rows.append(
            _base_fact(
                plan=plan,
                facet=target_facet,
                ontology=ontology,
                rng=rng,
                local_idx=local_idx,
                is_gold=False,
                distractor_type=dtype,
                condition_id=condition_id,
                condition_display=condition_display,
                subgroup_id=subgroup_id,
                subgroup_label=subgroup_label,
                subgroup_axis=subgroup_axis,
                subgroup_field=subgroup_field,
                subgroup_value=subgroup_value,
                axis=axis,
                cluster_id=f'{plan["query_id"]}_d{local_idx + 1:02d}',
                cluster_role='hard_distractor',
            )
        )
    return rows


def _base_fact(
    plan: dict[str, Any],
    facet: dict[str, Any],
    ontology: dict[str, Any],
    rng: Random,
    local_idx: int,
    is_gold: bool,
    distractor_type: str | None,
    condition_id: str,
    condition_display: str,
    subgroup_id: str,
    subgroup_label: str,
    subgroup_axis: str,
    subgroup_field: str,
    subgroup_value: str,
    axis: str,
    cluster_id: str,
    cluster_role: str,
) -> dict[str, Any]:
    value_bin, duration_days, treatment, rehab_outcome = _axis_values(
        ontology=ontology,
        condition_id=condition_id,
        axis=axis,
        rng=rng,
        local_idx=local_idx,
    )
    query_id = plan['query_id']
    support_facet_id = facet['facet_id'] if is_gold else None
    fact_id = f'{query_id}_{"g" if is_gold else "d"}_{len(cluster_id)}_{local_idx:03d}_{rng.randint(0, 9999):04d}'
    admission_id = f'adm_{query_id}_{cluster_id}_{local_idx:03d}'
    patient_id = f'pat_{query_id}_{cluster_id}_{local_idx // 2:03d}'
    note_style = rng.choice(['brief_hospital_course', 'brief_hospital_course', 'discharge_diagnosis'])
    patient_age = _patient_age(subgroup_id, rng)
    patient_sex = rng.choice(['female', 'male'])
    clinical_subgroup_phrase = _clinical_subgroup_phrase(subgroup_id, rng)

    must_mention = [condition_display, subgroup_label]
    if axis == 'treatment_duration':
        must_mention.extend([str(duration_days), 'treatment duration'])
    else:
        must_mention.extend([value_bin.replace('_', ' '), 'rehabilitation'])

    must_not_mention = []
    if subgroup_label != plan['subgroup_a_label']:
        must_not_mention.append(plan['subgroup_a_label'])
    if subgroup_label != plan['subgroup_b_label']:
        must_not_mention.append(plan['subgroup_b_label'])

    return {
        'query_id': query_id,
        'source_query_id': query_id,
        'fact_id': fact_id,
        'facet_id': support_facet_id,
        'target_facet_id': facet['facet_id'],
        'cluster_id': cluster_id,
        'cluster_role': cluster_role,
        'condition_id': condition_id,
        'condition_display': condition_display,
        'subgroup_id': subgroup_id,
        'subgroup_label': subgroup_label,
        'subgroup_axis': subgroup_axis,
        'subgroup_field': subgroup_field,
        'subgroup_value': subgroup_value,
        'axis': axis,
        'value_bin': value_bin,
        'duration_days': duration_days,
        'treatment': treatment,
        'rehab_outcome': rehab_outcome,
        'is_gold': is_gold,
        'distractor_type': distractor_type,
        'admission_id': admission_id,
        'patient_id': patient_id,
        'patient_age': patient_age,
        'patient_sex': patient_sex,
        'clinical_subgroup_phrase': clinical_subgroup_phrase,
        'note_style': note_style,
        'split': plan['split'],
        'must_mention_json': json_dumps(must_mention),
        'must_not_mention_json': json_dumps(must_not_mention),
    }


def _axis_values(
    ontology: dict[str, Any],
    condition_id: str,
    axis: str,
    rng: Random,
    local_idx: int,
) -> tuple[str, int | None, str | None, str | None]:
    condition = ontology['conditions'][condition_id]
    if axis == 'treatment_duration':
        bins = list(condition['duration_days'])
        value_bin = bins[(local_idx + rng.randint(0, 2)) % len(bins)]
        low, high = condition['duration_days'][value_bin]
        duration_days = rng.randint(int(low), int(high))
        duration_treatments = condition.get('duration_treatments') or condition['treatments']
        treatment = rng.choice(duration_treatments)
        return value_bin, duration_days, treatment, None

    bins = list(condition['rehab_outcomes'])
    value_bin = bins[(local_idx + rng.randint(0, 2)) % len(bins)]
    rehab_outcome = rng.choice(condition['rehab_outcomes'][value_bin])
    return value_bin, None, None, rehab_outcome


def _patient_age(subgroup_id: str, rng: Random) -> int:
    if subgroup_id == 'age_over_75':
        return rng.randint(76, 90)
    if subgroup_id == 'age_under_50':
        return rng.randint(24, 49)
    return rng.randint(52, 74)


def _clinical_subgroup_phrase(subgroup_id: str, rng: Random) -> str:
    phrases = {
        'age_over_75': ['older adult', 'patient above age 75', 'elderly patient'],
        'age_under_50': ['younger adult', 'adult below age 50', 'younger patient'],
        'uncomplicated_diabetes': [
            'uncomplicated type 2 diabetes',
            'diabetes without documented end-organ complications',
        ],
        'chronic_kidney_disease': [
            'stage 3 chronic kidney disease',
            'baseline chronic kidney disease',
        ],
        'copd': [
            'COPD treated with maintenance inhalers',
            'chronic obstructive pulmonary disease',
        ],
        'immunosuppression': [
            'immunosuppression from chronic steroid therapy',
            'immunosuppression after transplant medication use',
        ],
    }
    choices = phrases.get(subgroup_id)
    if not choices:
        return subgroup_id.replace('_', ' ')
    return rng.choice(choices)


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
    run_make_facts(cfg, paths)
