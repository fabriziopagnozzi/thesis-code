"""Build deterministic schema-v2 query plans from explicit evidence profiles."""

from __future__ import annotations

import hashlib
from itertools import combinations

import polars as pl

from experiments.medical_dataset_gen.dataset_generation.ontology_utils import (
    get_axis_pair_profiles,
    get_selected_conditions,
    load_ontology,
    make_subgroup_pairs,
)
from experiments.medical_dataset_gen.dataset_generation.query_templates import query_template_ids
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    CLINICAL_AXIS_LIST,
    ClinicalAxis,
    DataSplit,
    MedicalOntology,
    QueryLogicalForm,
    QueryPlan,
    QueryPlanFacet,
    QueryPlanSpec,
    QueryType,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import write_parquet

QUERY_TYPE: QueryType = 'prioritized_subgroup_comparison'
EXPECTED_FULL_PROFILE_COUNT = 2_720
EXPECTED_FULL_QUERY_COUNT = 5_440


def run_make_query_plans(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    conditions = get_selected_conditions(ontology, cfg.global_.conditions)
    contrasts = make_subgroup_pairs(ontology)
    axis_pairs = list(combinations(CLINICAL_AXIS_LIST, 2))
    specs: list[QueryPlanSpec] = []

    for condition_index, (condition_key, condition) in enumerate(conditions):
        for contrast_index, (
            contrast,
            (cohort_a_id, cohort_a),
            (cohort_b_id, cohort_b),
        ) in enumerate(contrasts):
            for axis_pair_index, (axis_a, axis_b) in enumerate(axis_pairs):
                profiles = get_axis_pair_profiles(ontology, axis_a, axis_b)
                # Explicit parity rotation prevents a cohort level or clinical axis
                # from being systematically assigned the less favorable profile.
                profile = profiles[
                    (
                        cfg.global_.seed
                        + condition_index
                        + contrast_index
                        + axis_pair_index
                    )
                    % len(profiles)
                ]
                evidence_profile_id = _stable_id(
                    'epv2',
                    cfg.dataset_schema_version,
                    cfg.global_.seed,
                    condition_key,
                    contrast.id,
                    contrast.dimension_id,
                    cohort_a_id,
                    cohort_b_id,
                    axis_a,
                    axis_b,
                    profile.id,
                    profile.cohort_a_bins,
                    profile.cohort_b_bins,
                )
                specs.append(
                    QueryPlanSpec(
                        evidence_profile_id=evidence_profile_id,
                        cohort_contrast_id=contrast.id,
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
                )

    rows = [
        plan.to_row()
        for spec in specs
        for plan in (
            _materialize_plan(cfg, ontology, spec, spec.axis_a, spec.axis_b),
            _materialize_plan(cfg, ontology, spec, spec.axis_b, spec.axis_a),
        )
    ]
    if cfg.generation.query_limit is not None:
        rows = rows[: int(cfg.generation.query_limit)]

    if (
        cfg.global_.conditions == len(ontology.conditions)
        and cfg.generation.query_limit is None
        and (len(specs) != EXPECTED_FULL_PROFILE_COUNT or len(rows) != EXPECTED_FULL_QUERY_COUNT)
    ):
        raise RuntimeError(f'v2 cardinality mismatch: profiles={len(specs)}, queries={len(rows)}')

    df = pl.from_dicts(rows, infer_schema_length=None)
    if df['query_id'].n_unique() != len(df):
        raise RuntimeError('stable v2 query IDs must be unique')
    write_parquet(paths, 'query_plans', df)
    print(f'[plans] {len(specs):,} evidence profiles -> {len(df):,} prioritized queries')
    return df


def _materialize_plan(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    spec: QueryPlanSpec,
    primary_axis: ClinicalAxis,
    secondary_axis: ClinicalAxis,
) -> QueryPlan:
    query_id = _stable_id(
        'qv2',
        cfg.dataset_schema_version,
        cfg.global_.seed,
        spec.evidence_profile_id,
        primary_axis,
        secondary_axis,
    )
    pool_id = _stable_id(
        'poolv2',
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
        _stable_int(cfg.global_.seed, query_id, 'initial_primary') % 2
    ]

    bin_by_cohort_axis = {
        (spec.subgroup_a_id, spec.axis_a): spec.cohort_a_bins[0],
        (spec.subgroup_a_id, spec.axis_b): spec.cohort_a_bins[1],
        (spec.subgroup_b_id, spec.axis_a): spec.cohort_b_bins[0],
        (spec.subgroup_b_id, spec.axis_b): spec.cohort_b_bins[1],
    }
    raw_facets = [
        (spec.subgroup_a_id, spec.subgroup_a, spec.axis_a),
        (spec.subgroup_a_id, spec.subgroup_a, spec.axis_b),
        (spec.subgroup_b_id, spec.subgroup_b, spec.axis_a),
        (spec.subgroup_b_id, spec.subgroup_b, spec.axis_b),
    ]
    facets: list[QueryPlanFacet] = []
    for index, (cohort_id, cohort, axis) in enumerate(raw_facets, start=1):
        facet_id = f'{query_id}_f{index}'
        is_primary = axis == primary_axis
        is_calibrated = is_primary and cohort_id == selected_subgroup_id
        target = (
            cfg.generation.gold_chunks_calibrated_primary
            if is_calibrated
            else cfg.generation.gold_chunks_other_primary
            if is_primary
            else cfg.generation.gold_chunks_secondary
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
                    'calibrated_primary_gold'
                    if is_calibrated
                    else 'primary_gold'
                    if is_primary
                    else 'secondary_gold'
                ),
                target_gold_chunks=target,
                priority='primary' if is_primary else 'secondary',
            )
        )
    calibrated_id = next(
        facet.facet_id for facet in facets if facet.cluster_role == 'calibrated_primary_gold'
    )
    template_ids = query_template_ids()
    template_id = template_ids[_stable_int(query_id, 'template') % len(template_ids)]
    logical_form = QueryLogicalForm(
        type=QUERY_TYPE,
        condition=spec.condition_key,
        subgroups=[spec.subgroup_a_id, spec.subgroup_b_id],
        axes=[primary_axis, secondary_axis],
        facets=[facet.facet_id for facet in facets],
        primary_axis=primary_axis,
        secondary_axis=secondary_axis,
        calibrated_primary_facet_id=calibrated_id,
    )
    return QueryPlan(
        query_id=query_id,
        evidence_profile_id=spec.evidence_profile_id,
        pool_id=pool_id,
        outcome_profile_id=spec.profile_id,
        plan_seed=_stable_int(cfg.global_.seed, query_id) % (2**31 - 1),
        split=split,
        query_type=QUERY_TYPE,
        template_id=template_id,
        condition_id=spec.condition_key,
        condition_display=spec.condition_display,
        **spec.subgroup_a.prefixed_fields('subgroup_a', spec.subgroup_a_id),  # type: ignore[arg-type]
        **spec.subgroup_b.prefixed_fields('subgroup_b', spec.subgroup_b_id),  # type: ignore[arg-type]
        cohort_contrast_id=spec.cohort_contrast_id,
        cohort_dimension_id=spec.cohort_dimension_id,
        primary_axis=primary_axis,
        secondary_axis=secondary_axis,
        calibrated_primary_facet_id=calibrated_id,
        n_facets=4,
        gold_chunks_total=sum(facet.target_gold_chunks for facet in facets),
        distractor_chunks=(
            cfg.generation.distractors_per_query
            + cfg.generation.background_outlier_clusters_per_query
            * cfg.generation.background_outlier_cluster_size
        ),
        facets=facets,
        logical_form=logical_form,
    )


def _stable_id(prefix: str, *parts: object) -> str:
    raw = '|'.join(str(part) for part in parts)
    return f'{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:16]}'


def _stable_int(*parts: object) -> int:
    raw = '|'.join(str(part) for part in parts)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16)


def _split_for_profile(evidence_profile_id: str) -> DataSplit:
    bucket = _stable_int(evidence_profile_id, 'split') % 20
    if bucket < 3:
        return 'test'
    if bucket < 6:
        return 'validation'
    return 'train'


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    config = load_config_from_cli()
    output_paths = paths_for(config)
    setup_logging(output_paths)
    run_make_query_plans(config, output_paths)
