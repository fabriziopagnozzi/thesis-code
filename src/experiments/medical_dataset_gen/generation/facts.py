"""Generate the hidden clinical facts that support each benchmark query.

This module exists to populate each query plan with gold evidence chunks and
hard distractors before any natural-language rendering happens. It uses seeded
sampling over the ontology plus per-facet cluster construction so the benchmark
has redundant positives and plausible negatives by design.
"""

from __future__ import annotations

import hashlib
import json
from random import Random
from typing import Literal, cast

import polars as pl
import pyarrow.parquet as pq

from experiments.medical_dataset_gen.generation.ontology import (
    load_ontology,
    other_conditions,
    other_subgroups,
)
from experiments.medical_dataset_gen.generation.schemas import (
    ClinicalFact,
    MedicalOntology,
    QueryPlan,
    QueryPlanFacet,
    SubgroupAxis,
)
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
)


def run_make_facts(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    plans = read_parquet(paths, 'query_plans')
    path = paths.table_path('clinical_facts')
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    last_df = pl.DataFrame()

    try:
        for plan_row in plans.iter_rows(named=True):
            plan = QueryPlan.model_validate(plan_row)
            rng = Random(plan.plan_seed)
            rows: list[dict[str, object]] = []

            for facet in plan.facets:
                for local_idx in range(int(facet.target_gold_chunks)):
                    rows.append(
                        make_gold_fact(
                            plan=plan,
                            facet=facet,
                            ontology=ontology,
                            local_idx=local_idx,
                            rng=rng,
                        ).model_dump(mode='python')
                    )

            rows.extend(
                fact.model_dump(mode='python')
                for fact in make_distractor_facts(
                    plan=plan,
                    ontology=ontology,
                    rng=rng,
                    n=cfg.generation.distractors_per_query,
                )
            )

            df = _facts_frame(rows)
            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)

            total_rows += len(df)
            last_df = df
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise ValueError('no query plans available to generate clinical facts')
    print(f'[write] clinical_facts: {total_rows:,} rows -> {path}')
    return last_df


def _facts_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.from_dicts(rows, infer_schema_length=None)


def make_gold_fact(
    plan: QueryPlan,
    facet: QueryPlanFacet,
    ontology: MedicalOntology,
    local_idx: int,
    rng: Random,
) -> ClinicalFact:
    return make_base_fact(
        plan=plan,
        facet=facet,
        ontology=ontology,
        rng=rng,
        local_idx=local_idx,
        is_gold=True,
        distractor_type=None,
        condition_id=facet.condition_id,
        condition_display=facet.condition_display,
        subgroup_id=facet.subgroup_id,
        subgroup_label=facet.subgroup_label,
        subgroup_axis=facet.subgroup_axis,
        subgroup_field=facet.subgroup_field,
        subgroup_value=facet.subgroup_value,
        axis=facet.axis,
        cluster_id=facet.cluster_id,
        cluster_role=facet.cluster_role,
    )


def make_distractor_facts(
    plan: QueryPlan,
    ontology: MedicalOntology,
    rng: Random,
    n: int,
) -> list[ClinicalFact]:
    facets = plan.facets
    excluded_subgroups = {plan.subgroup_a_id, plan.subgroup_b_id}
    distractor_types = [
        'same_condition_wrong_subgroup',
        'same_subgroup_wrong_condition',
        'same_axis_wrong_condition',
    ]
    rows: list[ClinicalFact] = []
    for local_idx in range(n):
        dtype = distractor_types[local_idx % len(distractor_types)]
        target_facet = facets[local_idx % len(facets)]

        condition_id = plan.condition_id
        condition_display = plan.condition_display
        subgroup_id = target_facet.subgroup_id
        subgroup_label = target_facet.subgroup_label
        subgroup_axis = target_facet.subgroup_axis
        subgroup_field = target_facet.subgroup_field
        subgroup_value = target_facet.subgroup_value
        axis = target_facet.axis

        if dtype == 'same_condition_wrong_subgroup':
            choices = other_subgroups(ontology, excluded_subgroups)
            subgroup_id, subgroup = choices[local_idx % len(choices)]
            subgroup_label = subgroup.label
            subgroup_axis = subgroup.axis
            subgroup_field = subgroup.field
            subgroup_value = subgroup.value
        elif dtype == 'same_subgroup_wrong_condition':
            choices = other_conditions(ontology, condition_id)
            condition_id, condition = choices[local_idx % len(choices)]
            condition_display = condition.display
        elif dtype == 'same_axis_wrong_condition':
            condition_choices = other_conditions(ontology, condition_id)
            subgroup_choices = other_subgroups(ontology, excluded_subgroups)
            condition_id, condition = condition_choices[(local_idx + 1) % len(condition_choices)]
            condition_display = condition.display
            subgroup_id, subgroup = subgroup_choices[(local_idx + 1) % len(subgroup_choices)]
            subgroup_label = subgroup.label
            subgroup_axis = subgroup.axis
            subgroup_field = subgroup.field
            subgroup_value = subgroup.value

        rows.append(
            make_base_fact(
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
                cluster_id=f'{plan.query_id}_d{local_idx + 1:02d}',
                cluster_role='hard_distractor',
            )
        )
    return rows


def make_base_fact(
    plan: QueryPlan,
    facet: QueryPlanFacet,
    ontology: MedicalOntology,
    rng: Random,
    local_idx: int,
    is_gold: bool,
    distractor_type: str | None,
    condition_id: str,
    condition_display: str,
    subgroup_id: str,
    subgroup_label: str,
    subgroup_axis: SubgroupAxis,
    subgroup_field: str,
    subgroup_value: str,
    axis: str,
    cluster_id: str,
    cluster_role: str,
) -> ClinicalFact:
    value_bin = _axis_value_bin(
        ontology=ontology,
        condition_id=condition_id,
        axis=axis,
        target_value_bin=facet.value_bin,
        local_idx=local_idx,
    )
    chunk_reuse_key = _chunk_reuse_key(
        condition_id=condition_id,
        subgroup_id=subgroup_id,
        axis=axis,
        value_bin=value_bin,
        local_idx=local_idx,
    )
    surface_rng = Random(_stable_seed(chunk_reuse_key))
    duration_days, treatment, rehab_outcome = _axis_values(
        ontology=ontology,
        condition_id=condition_id,
        axis=axis,
        value_bin=value_bin,
        rng=surface_rng,
    )
    query_id = plan.query_id
    support_facet_id = facet.facet_id if is_gold else None
    fact_id = (
        f'{query_id}_{"g" if is_gold else "d"}_{len(cluster_id)}_{local_idx:03d}_'
        f'{rng.randint(0, 9999):04d}'
    )
    admission_id = f'adm_{query_id}_{cluster_id}_{local_idx:03d}'
    patient_id = f'pat_{query_id}_{cluster_id}_{local_idx // 2:03d}'
    note_style = surface_rng.choice(
        [
            'brief_hospital_course',
            'brief_hospital_course',
            'discharge_diagnosis',
        ]
    )
    patient_age = _patient_age(subgroup_id, surface_rng)
    patient_sex = surface_rng.choice(['female', 'male'])
    clinical_subgroup_phrase = _clinical_subgroup_phrase(subgroup_id, surface_rng)

    must_mention = [condition_display, subgroup_label]
    if axis == 'treatment_duration':
        must_mention.extend([str(duration_days), 'treatment duration'])
    else:
        must_mention.extend([value_bin.replace('_', ' '), 'rehabilitation'])

    must_not_mention = []
    if subgroup_label != plan.subgroup_a_label:
        must_not_mention.append(plan.subgroup_a_label)
    if subgroup_label != plan.subgroup_b_label:
        must_not_mention.append(plan.subgroup_b_label)

    return ClinicalFact(
        query_id=query_id,
        source_query_id=query_id,
        fact_id=fact_id,
        chunk_reuse_key=chunk_reuse_key,
        facet_id=support_facet_id,
        target_facet_id=facet.facet_id,
        cluster_id=cluster_id,
        cluster_role=cast(
            Literal['dominant_gold', 'complementary_gold', 'hard_distractor'],
            cluster_role,
        ),
        condition_id=condition_id,
        condition_display=condition_display,
        subgroup_id=subgroup_id,
        subgroup_label=subgroup_label,
        subgroup_axis=subgroup_axis,
        subgroup_field=subgroup_field,
        subgroup_value=subgroup_value,
        axis=cast(Literal['treatment_duration', 'rehab_outcome'], axis),
        value_bin=value_bin,
        duration_days=duration_days,
        treatment=treatment,
        rehab_outcome=rehab_outcome,
        is_gold=is_gold,
        distractor_type=distractor_type,
        admission_id=admission_id,
        patient_id=patient_id,
        patient_age=patient_age,
        patient_sex=cast(Literal['female', 'male'], patient_sex),
        clinical_subgroup_phrase=clinical_subgroup_phrase,
        note_style=note_style,
        split=plan.split,
        must_mention=must_mention,
        must_not_mention=must_not_mention,
    )


def _axis_value_bin(
    ontology: MedicalOntology,
    condition_id: str,
    axis: str,
    target_value_bin: str | None,
    local_idx: int,
) -> str:
    condition = ontology.conditions[condition_id]
    if axis == 'treatment_duration':
        bins: list[str] = list(condition.duration_days)
        if target_value_bin in condition.duration_days:
            return str(target_value_bin)
        return bins[local_idx % len(bins)]

    bins = list(condition.rehab_outcomes)
    if target_value_bin in condition.rehab_outcomes:
        return str(target_value_bin)
    return bins[local_idx % len(bins)]


def _axis_values(
    ontology: MedicalOntology,
    condition_id: str,
    axis: str,
    value_bin: str,
    rng: Random,
) -> tuple[int | None, str | None, str | None]:
    condition = ontology.conditions[condition_id]
    if axis == 'treatment_duration':
        low, high = condition.duration_days[value_bin]
        duration_days = rng.randint(int(low), int(high))
        duration_treatments = condition.duration_treatments or condition.treatments
        treatment = rng.choice(duration_treatments)
        return duration_days, treatment, None

    rehab_outcome = rng.choice(condition.rehab_outcomes[value_bin])
    return None, None, rehab_outcome


def _chunk_reuse_key(
    condition_id: str,
    subgroup_id: str,
    axis: str,
    value_bin: str,
    local_idx: int,
) -> str:
    payload = {
        'schema': 'medical_chunk_reuse_v1',
        'condition_id': condition_id,
        'subgroup_id': subgroup_id,
        'axis': axis,
        'value_bin': value_bin,
        'local_idx': local_idx,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


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
        'obesity': [
            'obesity with limited baseline exercise tolerance',
            'elevated body mass index with chronic mobility strain',
        ],
        'malignancy': [
            'active malignancy on recent systemic therapy',
            'ongoing cancer treatment with baseline frailty',
        ],
        'atrial_fibrillation': [
            'chronic atrial fibrillation on long-term rate control',
            'baseline atrial fibrillation treated with anticoagulation',
        ],
        'chronic_liver_disease': [
            'chronic liver disease with prior hepatic decompensation',
            'cirrhosis with baseline hepatic dysfunction',
        ],
        'dementia': [
            'baseline dementia with memory impairment',
            'chronic cognitive impairment from dementia',
        ],
        'frailty': [
            'baseline frailty with reduced reserve',
            'frailty with limited pre-hospital mobility',
        ],
        'peripheral_vascular_disease': [
            'peripheral vascular disease with chronic limb symptoms',
            'baseline peripheral artery disease',
        ],
        'autoimmune_disease': [
            'systemic autoimmune disease on chronic immunomodulatory therapy',
            'chronic inflammatory autoimmune disease',
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
