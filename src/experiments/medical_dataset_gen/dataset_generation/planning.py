"""Build deterministic schema-v4 query plans from explicit evidence profiles.

Pair-level ontology policies can restrict which axis may be dominant for a
given joint profile, preventing clinically entangled primary-axis queries from
materializing at all.
"""

from __future__ import annotations

from itertools import combinations

import polars as pl

from experiments.medical_dataset_gen.dataset_generation.ontology_utils import (
    get_axis_pair_profiles,
    get_selected_conditions,
    load_ontology,
    make_subgroup_pairs_for_condition,
    resolve_axis_pair_generation_policy,
)
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    CLINICAL_AXIS_LIST,
    ClinicalAxis,
    DataSplit,
    MedicalOntology,
    QueryLogicalForm,
    QueryPlan,
    QueryPlanFacet,
    QueryPlanSpec,
    QueryType,
    SubgroupOntology,
)
from experiments.medical_dataset_gen.utils.deterministic_ids import stable_id, stable_int
from experiments.medical_dataset_gen.utils.global_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import write_parquet

QUERY_TYPE: QueryType = 'prioritized_subgroup_comparison'
_PROFILE_SPLIT_BUCKET_COUNT = 10
_TEST_PROFILE_BUCKET_COUNT = 5


def run_make_query_plans(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    conditions = get_selected_conditions(ontology, cfg.global_.conditions)
    excluded_axes = set(cfg.generation.excluded_clinical_axes)
    axis_pairs = [
        pair for pair in combinations(CLINICAL_AXIS_LIST, 2) if not set(pair) & excluded_axes
    ]
    specs: list[QueryPlanSpec] = []
    plans: list[QueryPlan] = []
    next_query_number = 1

    for condition_key, condition in conditions:
        contrasts = make_subgroup_pairs_for_condition(ontology, condition_key)
        for contrast, (cohort_a_id, cohort_a), (cohort_b_id, cohort_b) in contrasts:
            for axis_a, axis_b in axis_pairs:
                policy = resolve_axis_pair_generation_policy(
                    ontology,
                    condition_id=condition_key,
                    left=axis_a,
                    right=axis_b,
                )
                profiles = [
                    profile
                    for profile in get_axis_pair_profiles(ontology, axis_a, axis_b)
                    if profile.id not in policy.blocked_profile_ids
                ]
                if not profiles:
                    continue
                primary_axes: list[ClinicalAxis] = [
                    axis
                    for axis in (axis_a, axis_b)
                    if (
                        ontology.clinical_axes[axis].allow_as_primary
                        and axis in policy.allowed_primary_axes
                    )
                ]
                if not primary_axes:
                    continue
                for profile in profiles:
                    evidence_profile_id = stable_id(
                        'epv4',
                        cfg.dataset_schema_version,
                        cfg.global_.seed,
                        condition_key,
                        contrast.id,
                        contrast.family,
                        contrast.dimension_id,
                        cohort_a_id,
                        cohort_b_id,
                        axis_a,
                        axis_b,
                        profile.id,
                        profile.cohort_a_bins,
                        profile.cohort_b_bins,
                    )
                    spec = QueryPlanSpec(
                        evidence_profile_id=evidence_profile_id,
                        cohort_contrast_id=contrast.id,
                        cohort_contrast_family=contrast.family,
                        cohort_dimension_id=contrast.dimension_id,
                        axis_a=axis_a,
                        axis_b=axis_b,
                        profile_id=profile.id,
                        cohort_a_bins=profile.cohort_a_bins,
                        cohort_b_bins=profile.cohort_b_bins,
                        condition_key=condition_key,
                        condition_display=condition.display,
                        subgroup_a_id=cohort_a_id,
                        subgroup_a=cohort_a,
                        subgroup_b_id=cohort_b_id,
                        subgroup_b=cohort_b,
                    )
                    specs.append(spec)
                    if len(primary_axes) == 2:
                        plans.extend(
                            (
                                _materialize_plan(
                                    cfg,
                                    ontology,
                                    spec,
                                    axis_a,
                                    axis_b,
                                    query_id=f'q{next_query_number}',
                                ),
                                _materialize_plan(
                                    cfg,
                                    ontology,
                                    spec,
                                    axis_b,
                                    axis_a,
                                    query_id=f'q{next_query_number + 1}',
                                ),
                            )
                        )
                        next_query_number += 2
                        continue

                    # If one axis is unsuitable as the dominant retrieval target,
                    # emit one query per surviving joint profile with the allowed
                    # primary axis. This keeps the richer profile inventory while
                    # avoiding clinically weak dominant-axis orientations.
                    primary_axis = primary_axes[0]
                    secondary_axis = axis_b if primary_axis == axis_a else axis_a
                    plans.append(
                        _materialize_plan(
                            cfg,
                            ontology,
                            spec,
                            primary_axis,
                            secondary_axis,
                            query_id=f'q{next_query_number}',
                        )
                    )
                    next_query_number += 1

    rows = [plan.to_row() for plan in plans]
    if cfg.generation.query_limit is not None:
        rows = rows[: int(cfg.generation.query_limit)]

    df = pl.from_dicts(rows, infer_schema_length=None)
    if df['query_id'].n_unique() != len(df):
        raise RuntimeError('query IDs must be unique')
    write_parquet(paths, 'query_plans', df)
    print(f'[plans] {len(specs):,} evidence profiles -> {len(df):,} prioritized queries')
    return df


def _materialize_plan(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    spec: QueryPlanSpec,
    primary_axis: ClinicalAxis,
    secondary_axis: ClinicalAxis,
    *,
    query_id: str,
) -> QueryPlan:
    query_key = stable_id(
        'qv4',
        cfg.dataset_schema_version,
        cfg.global_.seed,
        spec.evidence_profile_id,
        primary_axis,
        secondary_axis,
    )
    pool_id = stable_id(
        'poolv4',
        cfg.dataset_schema_version,
        cfg.global_.seed,
        spec.evidence_profile_id,
        primary_axis,
        secondary_axis,
    )
    split = _split_for_profile(spec.evidence_profile_id)
    primary_candidates = [
        (spec.subgroup_a_id, primary_axis),
        (spec.subgroup_b_id, primary_axis),
    ]
    selected_subgroup_id, _ = primary_candidates[
        stable_int(cfg.global_.seed, query_key, 'initial_primary') % 2
    ]

    bin_by_cohort_axis = {
        (spec.subgroup_a_id, spec.axis_a): spec.cohort_a_bins[0],
        (spec.subgroup_a_id, spec.axis_b): spec.cohort_a_bins[1],
        (spec.subgroup_b_id, spec.axis_a): spec.cohort_b_bins[0],
        (spec.subgroup_b_id, spec.axis_b): spec.cohort_b_bins[1],
    }
    raw_facets: list[tuple[str, SubgroupOntology, ClinicalAxis]] = [
        (spec.subgroup_a_id, spec.subgroup_a, spec.axis_a),
        (spec.subgroup_a_id, spec.subgroup_a, spec.axis_b),
        (spec.subgroup_b_id, spec.subgroup_b, spec.axis_a),
        (spec.subgroup_b_id, spec.subgroup_b, spec.axis_b),
    ]
    secondary_indices = [
        index for index, (_, _, axis) in enumerate(raw_facets, start=1) if axis == secondary_axis
    ]
    # Rotate which complementary facet is niche without introducing sampling state.
    niche_indices = set(
        sorted(
            secondary_indices,
            key=lambda index: stable_int(query_key, 'niche_gold', index),
        )[: cfg.generation.chunk_pools.niche.num_clusters_per_query]
    )
    facets: list[QueryPlanFacet] = []
    for index, (cohort_id, cohort, axis) in enumerate(raw_facets, start=1):
        facet_id = f'{query_id}_f{index}'
        is_primary = axis == primary_axis
        is_dominant = is_primary and cohort_id == selected_subgroup_id
        is_niche = index in niche_indices
        target = (
            cfg.generation.chunk_pools.dominant_primary.size
            if is_dominant
            else cfg.generation.chunk_pools.other_primary.size
            if is_primary
            else cfg.generation.chunk_pools.niche.size
            if is_niche
            else cfg.generation.chunk_pools.secondary.size
        )
        facets.append(
            QueryPlanFacet(
                facet_id=facet_id,
                condition_id=spec.condition_key,
                condition_display=spec.condition_display,
                subgroup_id=cohort_id,
                subgroup_label=cohort.label,
                subgroup_axis=cohort.axis,
                subgroup_field=cohort.field,
                subgroup_value=cohort.value,
                axis=axis,
                value_bin=bin_by_cohort_axis[(cohort_id, axis)],
                cluster_id=f'{pool_id}_c{index}',
                cluster_role=(
                    'dominant_primary_gold'
                    if is_dominant
                    else 'primary_gold'
                    if is_primary
                    else 'niche_gold'
                    if is_niche
                    else 'secondary_gold'
                ),
                target_gold_chunks=target,
                priority='primary' if is_primary else 'secondary',
            )
        )
    dominant_id = next(
        facet.facet_id for facet in facets if facet.cluster_role == 'dominant_primary_gold'
    )
    logical_form = QueryLogicalForm(
        type=QUERY_TYPE,
        condition=spec.condition_key,
        subgroups=[spec.subgroup_a_id, spec.subgroup_b_id],
        axes=[primary_axis, secondary_axis],
        facets=[facet.facet_id for facet in facets],
        cohort_contrast_family=spec.cohort_contrast_family,
        primary_axis=primary_axis,
        secondary_axis=secondary_axis,
        dominant_primary_facet_id=dominant_id,
    )
    return QueryPlan(
        query_id=query_id,
        evidence_profile_id=spec.evidence_profile_id,
        pool_id=pool_id,
        outcome_profile_id=spec.profile_id,
        plan_seed=stable_int(cfg.global_.seed, query_key) % (2**31 - 1),
        split=split,
        query_type=QUERY_TYPE,
        # Surface-template selection belongs to queries_answers so plans can be shared
        # across query wording modes. The field remains for archived schema compatibility.
        template_id='deferred',
        condition_id=spec.condition_key,
        condition_display=spec.condition_display,
        **spec.subgroup_a.prefixed_fields('subgroup_a', spec.subgroup_a_id),  # type: ignore[arg-type]
        **spec.subgroup_b.prefixed_fields('subgroup_b', spec.subgroup_b_id),  # type: ignore[arg-type]
        cohort_contrast_id=spec.cohort_contrast_id,
        cohort_contrast_family=spec.cohort_contrast_family,
        cohort_dimension_id=spec.cohort_dimension_id,
        primary_axis=primary_axis,
        secondary_axis=secondary_axis,
        dominant_primary_facet_id=dominant_id,
        n_facets=4,
        gold_chunks_total=sum(facet.target_gold_chunks for facet in facets),
        distractor_chunks=cfg.generation.total_distractor_chunks(),
        facets=facets,
        logical_form=logical_form,
    )


def _split_for_profile(evidence_profile_id: str) -> DataSplit:
    """Assign the benchmark's fixed 50/50 profile-level split once in p01."""
    bucket = stable_int(evidence_profile_id, 'split') % _PROFILE_SPLIT_BUCKET_COUNT
    return 'test' if bucket < _TEST_PROFILE_BUCKET_COUNT else 'validation'
