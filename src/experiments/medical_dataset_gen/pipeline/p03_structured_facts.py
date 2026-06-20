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
from typing import cast

import polars as pl
import pyarrow.parquet as pq

from experiments.medical_dataset_gen.dataset_generation.ontology_utils import (
    load_ontology,
    other_conditions,
    other_subgroups,
)
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    DISTRACTOR_TYPES,
    ClinicalAxis,
    ClinicalFact,
    ClusterRole,
    MedicalOntology,
    PatientSex,
    QueryPlan,
    QueryPlanFacet,
    SubgroupAxis,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    unreachable_code,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet


def run_make_facts(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    if (
        cfg.generation.dominance_mode == 'embedding_calibrated'
        and not paths.table_path('query_plan_calibration').exists()
    ):
        raise FileNotFoundError(
            'embedding_calibrated dominance requires query_plan_calibration.parquet. '
            'Run the calibrate_plans stage before facts.'
        )

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
            rows.extend(
                fact.model_dump(mode='python')
                for fact in make_background_outlier_facts(
                    plan=plan,
                    ontology=ontology,
                    rng=rng,
                    n_clusters=cfg.generation.background_outlier_clusters_per_query,
                    cluster_size=cfg.generation.background_outlier_cluster_size,
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

    rows: list[ClinicalFact] = []
    for local_idx in range(n):
        distractor_type = DISTRACTOR_TYPES[local_idx % len(DISTRACTOR_TYPES)]
        target_facet = facets[local_idx % len(facets)]

        condition_id = plan.condition_id
        condition_display = plan.condition_display
        subgroup_id = target_facet.subgroup_id
        subgroup_label = target_facet.subgroup_label
        subgroup_axis = target_facet.subgroup_axis
        subgroup_field = target_facet.subgroup_field
        subgroup_value = target_facet.subgroup_value
        axis = target_facet.axis

        if distractor_type == 'same_condition_wrong_subgroup':
            choices = other_subgroups(ontology, excluded_subgroups)
            subgroup_id, subgroup = choices[local_idx % len(choices)]
            subgroup_label = subgroup.label
            subgroup_axis = subgroup.axis
            subgroup_field = subgroup.field
            subgroup_value = subgroup.value
        elif distractor_type == 'same_subgroup_wrong_condition':
            choices = other_conditions(ontology, condition_id)
            condition_id, condition = choices[local_idx % len(choices)]
            condition_display = condition.display
        elif distractor_type == 'same_axis_wrong_condition':
            condition_choices = other_conditions(ontology, condition_id)
            subgroup_choices = other_subgroups(ontology, excluded_subgroups)
            condition_id, condition = condition_choices[(local_idx + 1) % len(condition_choices)]
            condition_display = condition.display
            subgroup_id, subgroup = subgroup_choices[(local_idx + 1) % len(subgroup_choices)]
            subgroup_label = subgroup.label
            subgroup_axis = subgroup.axis
            subgroup_field = subgroup.field
            subgroup_value = subgroup.value
        else:
            unreachable_code(f'Distractor type not covered: {distractor_type}')

        rows.append(
            make_base_fact(
                plan=plan,
                facet=target_facet,
                ontology=ontology,
                rng=rng,
                local_idx=local_idx,
                is_gold=False,
                distractor_type=distractor_type,
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


def make_background_outlier_facts(
    plan: QueryPlan,
    ontology: MedicalOntology,
    rng: Random,
    n_clusters: int,
    cluster_size: int,
) -> list[ClinicalFact]:
    """Create coherent non-gold clinical islands outside the query facet structure."""
    if n_clusters <= 0 or cluster_size <= 0:
        return []

    condition_choices = other_conditions(ontology, plan.condition_id)
    if not condition_choices:
        return []

    excluded_subgroups = {plan.subgroup_a_id, plan.subgroup_b_id}
    subgroup_choices = other_subgroups(ontology, excluded_subgroups)
    if not subgroup_choices:
        subgroup_choices = list(ontology.subgroups.items())

    rows: list[ClinicalFact] = []
    for cluster_idx in range(n_clusters):
        condition_id, condition = condition_choices[
            (plan.plan_seed + cluster_idx) % len(condition_choices)
        ]
        subgroup_id, subgroup = subgroup_choices[
            (plan.plan_seed + cluster_idx * 3) % len(subgroup_choices)
        ]
        axis = 'treatment_duration' if cluster_idx % 2 == 0 else 'rehab_outcome'
        value_bin = _background_value_bin(ontology, condition_id, axis, cluster_idx)
        cluster_id = f'{plan.query_id}_bg{cluster_idx + 1:02d}'

        for local_idx in range(cluster_size):
            rows.append(
                make_base_fact(
                    plan=plan,
                    facet=None,
                    ontology=ontology,
                    rng=rng,
                    local_idx=local_idx,
                    is_gold=False,
                    distractor_type='background_clinical_cluster',
                    condition_id=condition_id,
                    condition_display=condition.display,
                    subgroup_id=subgroup_id,
                    subgroup_label=subgroup.label,
                    subgroup_axis=subgroup.axis,
                    subgroup_field=subgroup.field,
                    subgroup_value=subgroup.value,
                    axis=axis,
                    cluster_id=cluster_id,
                    cluster_role='background_outlier',
                    target_value_bin=value_bin,
                )
            )
    return rows


def make_base_fact(
    plan: QueryPlan,
    facet: QueryPlanFacet | None,
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
    axis: ClinicalAxis,
    cluster_id: str,
    cluster_role: str,
    target_value_bin: str | None = None,
) -> ClinicalFact:
    value_bin = _axis_value_bin(
        ontology=ontology,
        condition_id=condition_id,
        axis=axis,
        target_value_bin=target_value_bin
        if target_value_bin is not None
        else facet.value_bin
        if facet is not None
        else None,
        local_idx=local_idx,
    )
    chunk_reuse_key = _chunk_reuse_key(
        condition_id=condition_id,
        subgroup_id=subgroup_id,
        axis=axis,
        value_bin=value_bin,
        local_idx=local_idx,
        reuse_scope=_chunk_reuse_scope(
            is_gold=is_gold,
            distractor_type=distractor_type,
            cluster_role=cluster_role,
        ),
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
    support_facet_id = facet.facet_id if is_gold and facet is not None else None
    target_facet_id = facet.facet_id if facet is not None else None
    fact_id = (
        f'{query_id}_{"g" if is_gold else "d"}_{len(cluster_id)}_{local_idx:03d}_'
        f'{rng.randint(0, 9999):04d}'
    )
    admission_id = f'adm_{query_id}_{cluster_id}_{local_idx:03d}'
    patient_id = f'pat_{query_id}_{cluster_id}_{local_idx // 2:03d}'
    note_style = surface_rng.choice([
        'brief_hospital_course',
        'brief_hospital_course',
        'discharge_diagnosis',
    ])
    patient_age = _patient_age(subgroup_id, surface_rng)
    patient_sex = surface_rng.choice(['female', 'male'])
    clinical_subgroup_phrase = _clinical_subgroup_phrase(ontology, subgroup_id, surface_rng)

    must_mention = [
        condition_display,
        _subgroup_required_mention(
            subgroup_id=subgroup_id,
            subgroup_label=subgroup_label,
            patient_age=patient_age,
        ),
    ]
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
        target_facet_id=target_facet_id,
        cluster_id=cluster_id,
        cluster_role=cast(ClusterRole, cluster_role),
        condition_id=condition_id,
        condition_display=condition_display,
        subgroup_id=subgroup_id,
        subgroup_label=subgroup_label,
        subgroup_axis=subgroup_axis,
        subgroup_field=subgroup_field,
        subgroup_value=subgroup_value,
        axis=cast(ClinicalAxis, axis),
        value_bin=value_bin,
        duration_days=duration_days,
        treatment=treatment,
        rehab_outcome=rehab_outcome,
        is_gold=is_gold,
        distractor_type=distractor_type,
        admission_id=admission_id,
        patient_id=patient_id,
        patient_age=patient_age,
        patient_sex=cast(PatientSex, patient_sex),
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
    axis: ClinicalAxis,
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


def _background_value_bin(
    ontology: MedicalOntology,
    condition_id: str,
    axis: str,
    cluster_idx: int,
) -> str:
    condition = ontology.conditions[condition_id]
    if axis == 'treatment_duration':
        preferred = ['standard', 'short', 'prolonged']
        available = condition.duration_days
    else:
        preferred = ['home_rehab', 'inpatient_rehab', 'persistent_deficit']
        available = condition.rehab_outcomes

    for value_bin in preferred[cluster_idx:] + preferred[:cluster_idx]:
        if value_bin in available:
            return value_bin
    return next(iter(available))


def _chunk_reuse_key(
    condition_id: str,
    subgroup_id: str,
    axis: str,
    value_bin: str,
    local_idx: int,
    reuse_scope: str,
) -> str:
    payload = {
        'schema': 'medical_chunk_reuse_v2',
        'reuse_scope': reuse_scope,
        'condition_id': condition_id,
        'subgroup_id': subgroup_id,
        'axis': axis,
        'value_bin': value_bin,
        'local_idx': local_idx,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _chunk_reuse_scope(
    *,
    is_gold: bool,
    distractor_type: str | None,
    cluster_role: str,
) -> str:
    if is_gold:
        return 'gold'
    if distractor_type:
        return f'distractor:{distractor_type}'
    return f'non_gold:{cluster_role}'


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def _patient_age(subgroup_id: str, rng: Random) -> int:
    if subgroup_id == 'age_over_75':
        return rng.randint(76, 90)
    if subgroup_id == 'age_under_50':
        return rng.randint(24, 49)
    return rng.randint(52, 74)


def _clinical_subgroup_phrase(ontology: MedicalOntology, subgroup_id: str, rng: Random) -> str:
    subgroup = ontology.subgroups[subgroup_id]
    if subgroup.surface_phrases:
        return rng.choice(subgroup.surface_phrases)
    if subgroup.aliases:
        return rng.choice(subgroup.aliases)
    return subgroup.label.removeprefix('patients with ').removeprefix('patients ').strip()


def _subgroup_required_mention(subgroup_id: str, subgroup_label: str, patient_age: int) -> str:
    if subgroup_id in {'age_over_75', 'age_under_50'}:
        return f'{patient_age}-year-old'
    return subgroup_label


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_make_facts(cfg, paths)
