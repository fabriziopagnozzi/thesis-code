"""Expand schema-v4 plans into deterministic, typed clinical facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from random import Random
from typing import cast

import polars as pl
import pyarrow.parquet as pq

from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    available_note_styles,
    select_chunk_surface_group,
)
from experiments.medical_dataset_gen.dataset_generation.ontology_utils import (
    get_axis_bins,
    load_ontology,
    other_conditions,
    outlier_subgroups,
)
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    CLINICAL_AXIS_LIST,
    AcuteClinicalCoursePayload,
    AxisFactPayload,
    CareIntensityPayload,
    ChunkSurfacePolicy,
    ClinicalAxis,
    ClinicalFact,
    ComplicationBurdenPayload,
    ConditionKey,
    DiagnosticEvidencePayload,
    MedicalOntology,
    PatientSex,
    QueryPlan,
    QueryPlanFacet,
    RehabOutcomePayload,
    SubgroupAxis,
    TreatmentDurationAxisValues,
    TreatmentDurationPayload,
    axis_payload_required_phrase,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    BackgroundDistractorSpec,
    ChunkPoolsCfg,
    DistractorSpec,
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet

_FACTS_WRITE_BATCH_ROWS = 131_072


def run_make_facts(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    plans = read_parquet(paths, 'query_plans')
    path = paths.table_path('clinical_facts')
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total = 0
    last = pl.DataFrame()
    pending_facts: list[ClinicalFact] = []

    def flush_pending_facts() -> None:
        nonlocal writer, last
        if not pending_facts:
            return
        frame = _facts_frame(pending_facts)
        table = frame.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        last = frame
        pending_facts.clear()

    try:
        for plan_row in plans.iter_rows(named=True):
            plan = QueryPlan.model_validate(plan_row)
            rng = Random(plan.plan_seed)
            facts: list[ClinicalFact] = []
            for facet in plan.facets:
                facts.extend(
                    make_gold_fact(
                        plan,
                        facet,
                        ontology,
                        local_idx,
                        rng,
                        chunk_surface_policy=cfg.generation.chunk_surface_policy,
                    )
                    for local_idx in range(facet.target_gold_chunks)
                )
            facts.extend(
                make_distractor_facts(
                    plan,
                    ontology,
                    rng,
                    cfg.generation.chunk_pools,
                    chunk_surface_policy=cfg.generation.chunk_surface_policy,
                )
            )
            facts.extend(
                make_background_outlier_facts(
                    plan,
                    ontology,
                    rng,
                    cfg.generation.chunk_pools.background_outliers,
                    chunk_surface_policy=cfg.generation.chunk_surface_policy,
                )
            )
            _assert_query_local_chunk_reuse_keys(plan.query_id, facts)
            pending_facts.extend(facts)
            total += len(facts)
            # Batching avoids thousands of tiny Parquet row groups while preserving
            # plan order, which downstream answer construction relies on.
            if len(pending_facts) >= _FACTS_WRITE_BATCH_ROWS:
                flush_pending_facts()
        flush_pending_facts()
    finally:
        if writer is not None:
            writer.close()
    if total == 0:
        raise ValueError('no query plans available to generate facts')
    print(f'[write] clinical_facts: {total:,} rows -> {path}')
    return last


NOTE_STYLE_IDS = available_note_styles()


def _facts_frame(facts: list[ClinicalFact]) -> pl.DataFrame:
    return pl.from_dicts(
        [
            fact.model_dump(mode='python') if isinstance(fact, ClinicalFact) else fact
            for fact in facts
        ],
        infer_schema_length=None,
    )


def _assert_query_local_chunk_reuse_keys(query_id: str, facts: list[ClinicalFact]) -> None:
    seen_by_reuse_key: dict[str, ClinicalFact] = {}
    duplicate_examples: list[dict[str, str]] = []
    duplicate_count = 0

    for fact in facts:
        previous = seen_by_reuse_key.get(fact.chunk_reuse_key)
        if previous is None:
            seen_by_reuse_key[fact.chunk_reuse_key] = fact
            continue
        duplicate_count += 1
        if len(duplicate_examples) < 5:
            duplicate_examples.append({
                'chunk_reuse_key': fact.chunk_reuse_key,
                'previous_fact_id': previous.fact_id,
                'previous_cluster_id': previous.cluster_id,
                'fact_id': fact.fact_id,
                'cluster_id': fact.cluster_id,
            })

    if duplicate_count:
        raise RuntimeError(
            'query-local chunk_reuse_key collision in structured facts; '
            f'query_id={query_id!r}, duplicates={duplicate_count:,}, '
            f'examples={duplicate_examples}'
        )


def make_gold_fact(
    plan: QueryPlan,
    facet: QueryPlanFacet,
    ontology: MedicalOntology,
    local_idx: int,
    rng: Random,
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout',
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
        chunk_surface_policy=chunk_surface_policy,
    )


def make_distractor_facts(
    plan: QueryPlan,
    ontology: MedicalOntology,
    rng: Random,
    chunk_pools: ChunkPoolsCfg,
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout',
) -> list[ClinicalFact]:
    """Create the configured mix of point distractors for one query-local pool."""
    rows: list[ClinicalFact] = []
    for facet in plan.facets:
        local_cfg = _local_distractor_config_for_facet(chunk_pools, facet)
        rows.extend(
            make_local_distractor_facts(
                plan,
                facet,
                ontology,
                rng,
                local_cfg,
                chunk_surface_policy=chunk_surface_policy,
            )
        )
    return rows


def _local_distractor_config_for_facet(
    chunk_pools: ChunkPoolsCfg,
    facet: QueryPlanFacet,
) -> list[DistractorSpec]:
    if facet.cluster_role == 'dominant_primary_gold':
        return chunk_pools.dominant_primary.distractors
    if facet.cluster_role == 'primary_gold':
        return chunk_pools.other_primary.distractors
    if facet.cluster_role == 'niche_gold':
        return chunk_pools.niche.distractors
    return chunk_pools.secondary.distractors


def make_local_distractor_facts(
    plan: QueryPlan,
    target: QueryPlanFacet,
    ontology: MedicalOntology,
    rng: Random,
    distractor_config: list[DistractorSpec],
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout',
) -> list[ClinicalFact]:
    rows: list[ClinicalFact] = []
    for spec_idx, spec in enumerate(distractor_config):
        for local_idx in range(spec.size):
            rows.append(
                make_local_distractor_fact(
                    plan=plan,
                    target=target,
                    ontology=ontology,
                    rng=rng,
                    spec=spec,
                    spec_idx=spec_idx,
                    local_idx=local_idx,
                    chunk_surface_policy=chunk_surface_policy,
                )
            )
    return rows


def make_local_distractor_fact(
    *,
    plan: QueryPlan,
    target: QueryPlanFacet,
    ontology: MedicalOntology,
    rng: Random,
    spec: DistractorSpec,
    spec_idx: int,
    local_idx: int,
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout',
) -> ClinicalFact:
    distractor_type = _distractor_type_slug(spec)
    scope = _distractor_scope(distractor_type, spec_idx)
    resolved = _resolve_distractor_fields(
        plan=plan,
        target=target,
        ontology=ontology,
        spec=spec,
        scope=scope,
        selection_idx=local_idx,
    )
    condition_id, condition_display, subgroup_id, subgroup, axis, value_bin = resolved
    return make_base_fact(
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
        axis=axis,
        value_bin=value_bin,
        cluster_id=_distractor_cluster_id(plan, target, spec_idx, local_idx),
        cluster_role='hard_distractor',
        # The target facet becomes part of the reuse scope so per-facet local
        # distractor pools stay distinct even when their semantic shells match.
        reuse_scope=f'distractor:{scope}:target_{target.facet_id}',
        chunk_surface_policy=chunk_surface_policy,
    )


def _resolve_distractor_fields(
    *,
    plan: QueryPlan,
    target: QueryPlanFacet,
    ontology: MedicalOntology,
    spec: DistractorSpec,
    scope: str,
    selection_idx: int,
):
    condition_id = target.condition_id
    condition_display = target.condition_display
    if spec.changes_condition():
        condition_id, condition = _cycled_other_condition(
            ontology,
            target.condition_id,
            plan.query_id,
            target.facet_id,
            scope,
            selection_idx,
        )
        condition_display = condition.display

    subgroup_id = target.subgroup_id
    subgroup = ontology.subgroups[subgroup_id]
    if spec.changes_subgroup():
        subgroup_id, subgroup = _cycled_other_subgroup(
            ontology,
            {plan.subgroup_a_id, plan.subgroup_b_id},
            plan.query_id,
            target.facet_id,
            scope,
            selection_idx,
        )

    axis = target.axis
    value_bin = target.value_bin
    if spec.changes_axis():
        axis = _cycled_non_query_axis(plan, target.facet_id, scope, selection_idx)
        value_bin = _cycled_axis_bin(
            ontology,
            axis,
            None,
            plan.query_id,
            target.facet_id,
            scope,
            selection_idx,
        )
    elif spec.changes_axis_value_bin():
        value_bin = _cycled_axis_bin(
            ontology,
            axis,
            target.value_bin,
            plan.query_id,
            target.facet_id,
            scope,
            selection_idx,
        )

    return condition_id, condition_display, subgroup_id, subgroup, axis, value_bin


def _cycled_other_subgroup(
    ontology: MedicalOntology,
    excluded_ids: set[str],
    query_id: str,
    target_facet_id: str,
    scope: str,
    local_idx: int,
):
    alternatives = outlier_subgroups(ontology, excluded_ids)
    if not alternatives:
        raise ValueError('no eligible outlier subgroups are available')
    offset = _stable_seed(query_id, target_facet_id, scope, 'subgroup') % len(alternatives)
    return alternatives[(offset + local_idx) % len(alternatives)]


def _cycled_other_condition(
    ontology: MedicalOntology,
    excluded_condition_id: ConditionKey,
    query_id: str,
    target_facet_id: str,
    scope: str,
    local_idx: int,
):
    alternatives = other_conditions(ontology, excluded_condition_id)
    if not alternatives:
        raise ValueError('no eligible outlier conditions are available')
    offset = _stable_seed(query_id, target_facet_id, scope, 'condition') % len(alternatives)
    return alternatives[(offset + local_idx) % len(alternatives)]


def _cycled_non_query_axis(
    plan: QueryPlan,
    target_facet_id: str,
    scope: str,
    selection_idx: int,
) -> ClinicalAxis:
    non_query_axes: list[ClinicalAxis] = [
        axis for axis in CLINICAL_AXIS_LIST if axis not in {plan.primary_axis, plan.secondary_axis}
    ]
    if not non_query_axes:
        raise ValueError('different-axis distractors require at least one non-query clinical axis')
    return non_query_axes[
        _stable_seed(plan.query_id, target_facet_id, scope, 'axis', selection_idx)
        % len(non_query_axes)
    ]


def _cycled_axis_bin(
    ontology: MedicalOntology,
    axis: ClinicalAxis,
    excluded_value_bin: str | None,
    query_id: str,
    target_facet_id: str,
    scope: str,
    local_idx: int,
) -> str:
    bins = [
        value_bin for value_bin in get_axis_bins(ontology, axis) if value_bin != excluded_value_bin
    ]
    if not bins:
        raise ValueError(f'no eligible value bins are available for axis {axis!r}')
    offset = _stable_seed(query_id, target_facet_id, scope, 'value_bin') % len(bins)
    return bins[(offset + local_idx) % len(bins)]


def _distractor_cluster_id(
    plan: QueryPlan,
    target: QueryPlanFacet,
    spec_idx: int,
    local_idx: int,
) -> str:
    return f'{plan.pool_id}_{target.facet_id}_d{spec_idx + 1:02d}_{local_idx + 1:02d}'


def _distractor_type_slug(spec: DistractorSpec) -> str:
    axis_slug = 'diff' if spec.changes_axis() else 'same'
    value_slug = 'diff' if spec.changes_axis_value_bin() else 'same'
    parts = [
        f'c_{"diff" if spec.changes_condition() else "same"}',
        f's_{"diff" if spec.changes_subgroup() else "same"}',
        f'a_{axis_slug}',
    ]
    if not spec.changes_axis():
        parts.append(f'v_{value_slug}')
    return '__'.join(parts)


def _distractor_scope(distractor_type: str, spec_idx: int) -> str:
    return f'{distractor_type}:spec_{spec_idx + 1:02d}'


def make_background_outlier_facts(
    plan: QueryPlan,
    ontology: MedicalOntology,
    rng: Random,
    specs: list[BackgroundDistractorSpec],
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout',
) -> list[ClinicalFact]:
    rows: list[ClinicalFact] = []
    for spec_idx, spec in enumerate(specs):
        distractor_type = _distractor_type_slug(spec)
        scope = _distractor_scope(distractor_type, spec_idx)
        for cluster_idx in range(spec.num_clusters):
            target = _background_anchor_facet(plan, spec_idx, cluster_idx)
            resolved = _resolve_distractor_fields(
                plan=plan,
                target=target,
                ontology=ontology,
                spec=spec,
                scope=f'{scope}:cluster_{cluster_idx + 1:02d}',
                selection_idx=cluster_idx,
            )
            condition_id, condition_display, subgroup_id, subgroup, axis, value_bin = resolved
            cluster_id = f'{plan.pool_id}_bg_s{spec_idx + 1:02d}_c{cluster_idx + 1:02d}'
            for local_idx in range(spec.size):
                reuse_scope = (
                    f'distractor:background_clinical_cluster:{scope}:cluster_{cluster_idx + 1:02d}'
                )
                rows.append(
                    make_base_fact(
                        plan=plan,
                        facet=None,
                        ontology=ontology,
                        rng=rng,
                        local_idx=local_idx,
                        is_gold=False,
                        distractor_type=distractor_type,
                        reuse_scope=reuse_scope,
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
                        axis=axis,
                        value_bin=value_bin,
                        cluster_id=cluster_id,
                        cluster_role='background_outlier',
                        chunk_surface_policy=chunk_surface_policy,
                    )
                )
    return rows


def _background_anchor_facet(
    plan: QueryPlan,
    spec_idx: int,
    cluster_idx: int,
) -> QueryPlanFacet:
    return plan.facets[
        _stable_seed(plan.query_id, 'background_anchor', spec_idx, cluster_idx) % len(plan.facets)
    ]


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
    reuse_scope: str | None = None,
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout',
) -> ClinicalFact:
    payload = _axis_payload(ontology, condition_id, axis, value_bin, local_idx)
    payload_json = json.dumps(payload.model_dump(mode='json'), sort_keys=True)
    resolved_reuse_scope = reuse_scope or ('gold' if is_gold else f'distractor:{distractor_type}')
    chunk_surface_group = select_chunk_surface_group(plan.split, chunk_surface_policy)
    reuse_key = _chunk_reuse_key(
        condition_id,
        subgroup_id,
        axis,
        value_bin,
        payload_json,
        local_idx,
        resolved_reuse_scope,
        chunk_surface_group,
    )
    surface_rng = Random(_stable_seed(reuse_key))
    template_rng = Random(
        _stable_seed(
            'template',
            condition_id,
            subgroup_id,
            axis,
            value_bin,
            payload_json,
            local_idx,
            chunk_surface_group,
        )
    )
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
    condition_anchor = (
        'axis_evidence'
        if condition_display.casefold() in required_payload.casefold()
        else 'outer_template'
    )
    must_mention = [
        condition_display,
        ontology.clinical_axes[axis].label,
        axis_bin_term,
        required_payload,
    ]
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
        dominant_primary_facet_id=plan.dominant_primary_facet_id,
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
        condition_anchor=condition_anchor,
        facet_priority=facet.priority if facet is not None else None,
        is_gold=is_gold,
        distractor_type=distractor_type,
        admission_id=f'adm_{plan.pool_id}_{cluster_id}_{local_idx:03d}',
        patient_id=f'pat_{plan.evidence_profile_id}_{subgroup_id}_{local_idx // 2:03d}',
        patient_age=age,
        patient_sex=sex,
        clinical_subgroup_phrase=phrase,
        note_style=template_rng.choice(NOTE_STYLE_IDS),
        chunk_surface_group=chunk_surface_group,
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
        treatment_course_id, treatment_course = rng.choice(list(axis_values.treatments.items()))
        duration_days = rng.choice(treatment_course.bins[value_bin])
        return TreatmentDurationPayload(
            axis=axis,
            duration_days=duration_days,
            treatment=rng.choice(treatment_course.surface_forms),
            treatment_course_id=treatment_course_id,
        )
    values = cast(Sequence[str], axis_values.bins[value_bin])
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
    if axis == 'care_intensity':
        return CareIntensityPayload(axis=axis, detail=detail)
    return DiagnosticEvidencePayload(axis='diagnostic_evidence_type', detail=detail)


def _payload_required_phrase(payload: AxisFactPayload) -> str:
    return axis_payload_required_phrase(payload)


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
    chunk_surface_group: str,
) -> str:
    raw = json.dumps(
        {
            'schema': 4,
            'condition_id': condition_id,
            'subgroup_id': subgroup_id,
            'axis': axis,
            'value_bin': value_bin,
            'payload': payload_json,
            'local_idx': local_idx,
            'reuse_scope': reuse_scope,
            'chunk_surface_group': chunk_surface_group,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _stable_seed(*values: object) -> int:
    return int(hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:16], 16)


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.logging_utils import (
        setup_logging,
    )

    config = load_config_from_cli()
    output_paths = paths_for(config)
    setup_logging(output_paths)
    run_make_facts(config, output_paths)
