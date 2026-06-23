"""Expand schema-v2 plans into deterministic, typed clinical facts."""

from __future__ import annotations

import hashlib
import json
from random import Random
from typing import cast

import polars as pl
import pyarrow.parquet as pq

from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    available_note_styles,
)
from experiments.medical_dataset_gen.dataset_generation.ontology_utils import (
    get_axis_bins,
    load_ontology,
    other_conditions,
    other_subgroups,
)
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    CLINICAL_AXIS_LIST,
    DISTRACTOR_TYPES,
    AcuteClinicalCoursePayload,
    AxisFactPayload,
    CareIntensityPayload,
    ClinicalAxis,
    ClinicalFact,
    ComplicationBurdenPayload,
    ConditionKey,
    MedicalOntology,
    PatientSex,
    QueryPlan,
    QueryPlanFacet,
    RehabOutcomePayload,
    SubgroupAxis,
    TreatmentDurationAxisValues,
    TreatmentDurationPayload,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet


def run_make_facts(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    if (
        cfg.generation.calibration_mode == 'embedding_calibrated'
        and not paths.table_path('query_plan_calibration').exists()
    ):
        raise FileNotFoundError('run calibrate_plans before facts in embedding-calibrated mode')
    ontology = load_ontology(cfg)
    plans = read_parquet(paths, 'query_plans')
    path = paths.table_path('clinical_facts')
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total = 0
    last = pl.DataFrame()
    try:
        for plan_row in plans.iter_rows(named=True):
            plan = QueryPlan.model_validate(plan_row)
            rng = Random(plan.plan_seed)
            facts: list[ClinicalFact] = []
            for facet in plan.facets:
                facts.extend(
                    make_gold_fact(plan, facet, ontology, local_idx, rng)
                    for local_idx in range(facet.target_gold_chunks)
                )
            facts.extend(
                make_distractor_facts(plan, ontology, rng, cfg.generation.distractors_per_query)
            )
            facts.extend(
                make_background_outlier_facts(
                    plan,
                    ontology,
                    rng,
                    cfg.generation.background_outlier_clusters_per_query,
                    cfg.generation.background_outlier_cluster_size,
                )
            )
            facts.extend(
                make_single_isolated_outlier_facts(
                    plan,
                    ontology,
                    rng,
                    cfg.generation.single_isolated_outliers_per_query,
                )
            )
            frame = _facts_frame([fact.model_dump(mode='python') for fact in facts])
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
            total += len(frame)
            last = frame
    finally:
        if writer is not None:
            writer.close()
    if total == 0:
        raise ValueError('no query plans available to generate facts')
    print(f'[write] clinical_facts: {total:,} rows -> {path}')
    return last


NOTE_STYLE_IDS = available_note_styles()


def _facts_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.from_dicts(rows, infer_schema_length=None)


def make_gold_fact(
    plan: QueryPlan,
    facet: QueryPlanFacet,
    ontology: MedicalOntology,
    local_idx: int,
    rng: Random,
) -> ClinicalFact:
    cohort = ontology.subgroups[facet.subgroup_id]
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
        subgroup_dimension_id=cohort.dimension_id,
        subgroup_level_id=cohort.level_id,
        subgroup_is_reference=cohort.is_reference,
        axis=facet.axis,
        value_bin=facet.value_bin,
        cluster_id=facet.cluster_id,
        cluster_role=facet.cluster_role,
    )


def make_distractor_facts(
    plan: QueryPlan, ontology: MedicalOntology, rng: Random, n: int
) -> list[ClinicalFact]:
    excluded = {plan.subgroup_a_id, plan.subgroup_b_id}
    rows: list[ClinicalFact] = []
    for local_idx in range(n):
        target = plan.facets[local_idx % len(plan.facets)]
        distractor_type = DISTRACTOR_TYPES[local_idx % len(DISTRACTOR_TYPES)]
        condition_id = plan.condition_id
        condition_display = plan.condition_display
        subgroup_id = target.subgroup_id
        subgroup = ontology.subgroups[subgroup_id]
        if distractor_type == 'same_condition_wrong_subgroup':
            subgroup_id, subgroup = other_subgroups(ontology, excluded)[
                local_idx % (len(ontology.subgroups) - 2)
            ]
        elif distractor_type == 'same_subgroup_wrong_condition':
            condition_id, condition = other_conditions(ontology, plan.condition_id)[
                local_idx % (len(ontology.conditions) - 1)
            ]
            condition_display = condition.display
        else:
            condition_id, condition = other_conditions(ontology, plan.condition_id)[
                (local_idx + 1) % (len(ontology.conditions) - 1)
            ]
            condition_display = condition.display
            subgroup_id, subgroup = other_subgroups(ontology, excluded)[
                (local_idx + 1) % (len(ontology.subgroups) - 2)
            ]
        rows.append(
            make_base_fact(
                plan=plan,
                facet=target,
                ontology=ontology,
                rng=rng,
                local_idx=local_idx,
                is_gold=False,
                distractor_type=distractor_type,
                condition_id=condition_id,
                condition_display=condition_display,
                subgroup_id=subgroup_id,
                subgroup_label=subgroup.label,
                subgroup_axis=subgroup.axis,
                subgroup_field=subgroup.field,
                subgroup_value=subgroup.value,
                subgroup_dimension_id=subgroup.dimension_id,
                subgroup_level_id=subgroup.level_id,
                subgroup_is_reference=subgroup.is_reference,
                axis=target.axis,
                value_bin=target.value_bin,
                cluster_id=f'{plan.pool_id}_d{local_idx + 1:02d}',
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
    rows: list[ClinicalFact] = []
    excluded = {plan.subgroup_a_id, plan.subgroup_b_id}
    conditions = other_conditions(ontology, plan.condition_id)
    cohorts = other_subgroups(ontology, excluded)
    for cluster_idx in range(n_clusters):
        condition_id, condition = conditions[
            _stable_seed(plan.evidence_profile_id, 'background_condition', cluster_idx)
            % len(conditions)
        ]
        subgroup_id, subgroup = cohorts[
            _stable_seed(plan.evidence_profile_id, 'background_cohort', cluster_idx) % len(cohorts)
        ]
        axis = plan.secondary_axis
        bins = get_axis_bins(ontology, axis)
        value_bin = bins[cluster_idx % len(bins)]
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
                    subgroup_dimension_id=subgroup.dimension_id,
                    subgroup_level_id=subgroup.level_id,
                    subgroup_is_reference=subgroup.is_reference,
                    axis=axis,
                    value_bin=value_bin,
                    cluster_id=f'{plan.pool_id}_bg{cluster_idx + 1:02d}',
                    cluster_role='background_outlier',
                )
            )
    return rows


def make_single_isolated_outlier_facts(
    plan: QueryPlan,
    ontology: MedicalOntology,
    rng: Random,
    n: int,
) -> list[ClinicalFact]:
    """Create singleton satellites on the query manifold but off the answer axes.

    Each point retains the target facet's condition and cohort anchor but swaps to
    a clinical axis that the query never asks about. That makes the point a true
    singleton outlier relative to the answer-focused facets without turning it into
    the same wrong-condition/wrong-subgroup near miss used for hard distractors.
    """
    dense_facets = sorted(
        (facet for facet in plan.facets if facet.priority == 'primary'),
        key=lambda facet: (-facet.target_gold_chunks, facet.facet_id),
    )
    non_query_axes = [
        axis for axis in CLINICAL_AXIS_LIST if axis not in {plan.primary_axis, plan.secondary_axis}
    ]
    if not non_query_axes:
        raise ValueError('single isolated outliers require at least one clinical axis outside query')
    rows: list[ClinicalFact] = []
    for local_idx in range(n):
        target = dense_facets[local_idx % len(dense_facets)]
        axis = non_query_axes[
            _stable_seed(plan.query_id, 'single_isolated_axis', target.facet_id, local_idx)
            % len(non_query_axes)
        ]
        bins = get_axis_bins(ontology, axis)
        value_bin = bins[
            _stable_seed(plan.query_id, 'single_isolated_value_bin', target.facet_id, local_idx)
            % len(bins)
        ]
        cohort = ontology.subgroups[target.subgroup_id]
        rows.append(
            make_base_fact(
                plan=plan,
                facet=target,
                ontology=ontology,
                rng=rng,
                local_idx=local_idx,
                is_gold=False,
                distractor_type='single_isolated_outlier',
                condition_id=target.condition_id,
                condition_display=target.condition_display,
                subgroup_id=target.subgroup_id,
                subgroup_label=target.subgroup_label,
                subgroup_axis=target.subgroup_axis,
                subgroup_field=target.subgroup_field,
                subgroup_value=target.subgroup_value,
                subgroup_dimension_id=cohort.dimension_id,
                subgroup_level_id=cohort.level_id,
                subgroup_is_reference=cohort.is_reference,
                axis=axis,
                value_bin=value_bin,
                cluster_id=f'{plan.pool_id}_iso{local_idx + 1:02d}',
                cluster_role='single_isolated_outlier',
            )
        )
    return rows


def make_base_fact(
    *,
    plan: QueryPlan,
    facet: QueryPlanFacet | None,
    ontology: MedicalOntology,
    rng: Random,
    local_idx: int,
    is_gold: bool,
    distractor_type: str | None,
    condition_id: ConditionKey,
    condition_display: str,
    subgroup_id: str,
    subgroup_label: str,
    subgroup_axis: SubgroupAxis,
    subgroup_field: str,
    subgroup_value: str,
    subgroup_dimension_id: str,
    subgroup_level_id: str,
    subgroup_is_reference: bool,
    axis: ClinicalAxis,
    value_bin: str,
    cluster_id: str,
    cluster_role,
) -> ClinicalFact:
    payload = _axis_payload(ontology, condition_id, axis, value_bin, local_idx)
    payload_json = json.dumps(payload.model_dump(mode='json'), sort_keys=True)
    reuse_key = _chunk_reuse_key(
        condition_id,
        subgroup_id,
        axis,
        value_bin,
        payload_json,
        local_idx,
        'gold' if is_gold else f'distractor:{distractor_type}',
    )
    surface_rng = Random(_stable_seed(reuse_key))
    subgroup = ontology.subgroups[subgroup_id]
    age = _patient_age(subgroup.patient_age_range, ontology.patient_defaults.age_range, surface_rng)
    sex: PatientSex = _patient_sex(subgroup.patient_sex, surface_rng)
    phrase = surface_rng.choice(subgroup.surface_phrases)
    # A shared human-readable anchor keeps documents from the same semantic bin
    # cohesive while their condition-specific payloads and note styles still vary.
    axis_bin_term = ontology.clinical_axes[axis].bin_terms[value_bin][0]
    support_facet_id = facet.facet_id if is_gold and facet is not None else None
    target_facet_id = facet.facet_id if facet is not None else None
    fact_id = (
        f'{plan.query_id}_{"g" if is_gold else "d"}_{local_idx:03d}_{rng.randint(0, 9999):04d}'
    )
    required_payload = _payload_required_phrase(payload)
    must_mention = [condition_display, ontology.clinical_axes[axis].label, axis_bin_term, required_payload]
    if subgroup_dimension_id != 'age_band':
        must_mention.insert(1, phrase)
    must_not_mention = [
        label for label in (plan.subgroup_a_label, plan.subgroup_b_label) if label != subgroup_label
    ]
    return ClinicalFact(
        query_id=plan.query_id,
        evidence_profile_id=plan.evidence_profile_id,
        pool_id=plan.pool_id,
        primary_axis=plan.primary_axis,
        secondary_axis=plan.secondary_axis,
        calibrated_primary_facet_id=plan.calibrated_primary_facet_id,
        fact_id=fact_id,
        chunk_reuse_key=reuse_key,
        facet_id=support_facet_id,
        target_facet_id=target_facet_id,
        cluster_id=cluster_id,
        cluster_role=cluster_role,
        condition_id=condition_id,
        condition_display=condition_display,
        subgroup_id=subgroup_id,
        subgroup_label=subgroup_label,
        subgroup_axis=subgroup_axis,
        subgroup_field=subgroup_field,
        subgroup_value=subgroup_value,
        subgroup_dimension_id=subgroup_dimension_id,
        subgroup_level_id=subgroup_level_id,
        subgroup_is_reference=subgroup_is_reference,
        axis=axis,
        value_bin=value_bin,
        axis_bin_term=axis_bin_term,
        axis_payload_json=payload_json,
        facet_priority=facet.priority if facet is not None else None,
        is_gold=is_gold,
        distractor_type=distractor_type,
        admission_id=f'adm_{plan.pool_id}_{cluster_id}_{local_idx:03d}',
        patient_id=f'pat_{plan.evidence_profile_id}_{subgroup_id}_{local_idx // 2:03d}',
        patient_age=age,
        patient_sex=sex,
        clinical_subgroup_phrase=phrase,
        note_style=surface_rng.choice(NOTE_STYLE_IDS),
        split=plan.split,
        must_mention=must_mention,
        must_not_mention=must_not_mention,
    )


def _axis_payload(
    ontology: MedicalOntology,
    condition_id: ConditionKey,
    axis: ClinicalAxis,
    value_bin: str,
    local_idx: int,
) -> AxisFactPayload:
    condition = ontology.conditions[condition_id]
    axis_values = condition.axis_values[axis]
    seed = _stable_seed(condition_id, axis, value_bin, local_idx)
    rng = Random(seed)
    if axis == 'treatment_duration':
        if not isinstance(axis_values, TreatmentDurationAxisValues):
            raise TypeError(f'{condition_id}/{axis} has the wrong axis-values payload')
        low, high = axis_values.bins[value_bin]
        return TreatmentDurationPayload(
            axis=axis,
            duration_days=rng.randint(low, high),
            treatment=rng.choice(axis_values.treatments),
        )
    values = axis_values.bins[value_bin]
    if axis == 'rehab_outcome':
        return RehabOutcomePayload(
            axis=axis,
            outcome=rng.choice(values),
        )
    detail = rng.choice(values)
    if axis == 'complication_burden':
        return ComplicationBurdenPayload(axis=axis, detail=detail)
    if axis == 'acute_clinical_course':
        return AcuteClinicalCoursePayload(axis=axis, detail=detail)
    return CareIntensityPayload(axis='care_intensity', detail=detail)


def _payload_required_phrase(payload: AxisFactPayload) -> str:
    if isinstance(payload, TreatmentDurationPayload):
        return f'{payload.duration_days} days of {payload.treatment}'
    if isinstance(payload, RehabOutcomePayload):
        return payload.outcome
    return payload.detail


def _patient_age(
    cohort_range: tuple[int, int] | None,
    default_range: tuple[int, int],
    rng: Random,
) -> int:
    low, high = cohort_range or default_range
    return rng.randint(low, high)


def _patient_sex(cohort_sex: PatientSex | None, rng: Random) -> PatientSex:
    if cohort_sex is not None:
        return cohort_sex
    return cast(PatientSex, rng.choice(['female', 'male']))


def _chunk_reuse_key(
    condition_id: str,
    subgroup_id: str,
    axis: str,
    value_bin: str,
    payload_json: str,
    local_idx: int,
    reuse_scope: str,
) -> str:
    raw = json.dumps(
        {
            'schema': 2,
            'condition_id': condition_id,
            'subgroup_id': subgroup_id,
            'axis': axis,
            'value_bin': value_bin,
            'payload': payload_json,
            'local_idx': local_idx,
            'reuse_scope': reuse_scope,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _stable_seed(*values: object) -> int:
    return int(hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:16], 16)


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    config = load_config_from_cli()
    output_paths = paths_for(config)
    setup_logging(output_paths)
    run_make_facts(config, output_paths)
