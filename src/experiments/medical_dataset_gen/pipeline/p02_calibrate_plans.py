"""Calibrate query emphasis and reject semantically entangled plans early."""

from __future__ import annotations

import json
from random import Random
from typing import TypedDict

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.dataset_generation.chunk_rendering import (
    render_canonical_chunk_text,
)
from experiments.medical_dataset_gen.dataset_generation.ontology_utils import load_ontology
from experiments.medical_dataset_gen.dataset_generation.query_templates import query_template_ids
from experiments.medical_dataset_gen.pipeline.p03_facts import make_gold_fact
from experiments.medical_dataset_gen.pipeline.p05_queries_answers import render_query
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ChunkTextStyle,
    MedicalOntology,
    QueryPlan,
    QueryPlanFacet,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


class FacetProbe(TypedDict):
    texts: list[str]
    offsets: dict[str, tuple[int, int]]
    labels: list[str]


def run_calibrate_query_plans(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    plans_df = read_parquet(paths, 'query_plans')
    plans = [QueryPlan.model_validate(row) for row in plans_df.iter_rows(named=True)]
    if cfg.generation.calibration_mode == 'rotating':
        rows = [_calibration_row_without_embeddings(plan) for plan in plans]
        write_parquet(paths, 'query_plan_calibration', pl.from_dicts(rows))
        return plans_df

    ontology = load_ontology(cfg)
    from helpers.embedder import Embedder

    embedder = Embedder(
        model_name=cfg.embeddings.model_name,
        batch_size=cfg.embeddings.batch_size,
        query_prompt=cfg.embeddings.query_prompt,
        document_prompt=cfg.embeddings.document_prompt,
        device=cfg.embeddings.device,
        devices=cfg.embeddings.devices,
    )
    updated: list[QueryPlan] = []
    calibration_rows: list[dict[str, object]] = []
    probe_n = int(cfg.generation.calibration_probe_chunks_per_facet)
    template_ids = query_template_ids(
        cfg.generation.query_structure,
        cfg.generation.focus_mode,
    )
    max_facets_per_plan = max(len(plan.facets) for plan in plans)
    plans_per_batch = max(1, min(128, 4096 // max(1, probe_n * max_facets_per_plan)))
    try:
        for start in tqdm(
            range(0, len(plans), plans_per_batch),
            desc='Calibrating query emphasis',
            dynamic_ncols=True,
        ):
            batch = plans[start : start + plans_per_batch]
            probes = [
                _prepare_probe(plan, ontology, probe_n, cfg.generation.chunk_text_style)
                for plan in batch
            ]
            query_vectors = embedder.embed_queries(
                [
                    render_query(
                        plan,
                        ontology,
                        template_id=template_id,
                        focus_mode=cfg.generation.focus_mode,
                        query_structure=cfg.generation.query_structure,
                    )
                    for plan in batch
                    for template_id in template_ids
                ],
                normalize=cfg.embeddings.normalize,
            )
            probe_vectors = embedder.embed_docs(
                [text for probe in probes for text in probe['texts']],
                normalize=cfg.embeddings.normalize,
            )
            probe_offset = 0
            query_offset = 0
            for plan, probe in zip(batch, probes, strict=True):
                probe_end = probe_offset + len(probe['texts'])
                query_end = query_offset + len(template_ids)
                plan_query_vectors = np.asarray(query_vectors[query_offset:query_end])
                plan_probe_vectors = np.asarray(probe_vectors[probe_offset:probe_end])
                selected_facet_id, selected_template_id, row = _select_calibration(
                    plan=plan,
                    probe=probe,
                    template_ids=template_ids,
                    query_vectors=plan_query_vectors,
                    probe_vectors=plan_probe_vectors,
                    probe_n=probe_n,
                )
                semantic_gate = _semantic_probe_gate(
                    cfg=cfg,
                    plan=plan,
                    probe=probe,
                    query_vector=plan_query_vectors[template_ids.index(selected_template_id)],
                    probe_vectors=plan_probe_vectors,
                )
                probe_offset = probe_end
                query_offset = query_end
                if bool(semantic_gate['passes_semantic_gate']):
                    updated.append(
                        _with_dominant_primary_query(
                            cfg,
                            plan,
                            selected_facet_id=selected_facet_id,
                            selected_template_id=selected_template_id,
                        )
                    )
                calibration_rows.append(row | semantic_gate)
    finally:
        embedder.release()

    updated_rows = [plan.to_row() for plan in updated]
    updated_df = (
        pl.from_dicts(updated_rows, infer_schema_length=None) if updated_rows else plans_df.head(0)
    )
    write_parquet(paths, 'query_plans', updated_df)
    write_parquet(
        paths,
        'query_plan_calibration',
        pl.from_dicts(calibration_rows, infer_schema_length=None),
    )
    print(f'[calibrate_plans] retained {len(updated):,}/{len(plans):,} plans after semantic gate')
    return updated_df


def _prepare_probe(
    plan: QueryPlan,
    ontology: MedicalOntology,
    probe_n: int,
    chunk_text_style: ChunkTextStyle,
) -> FacetProbe:
    if len(plan.facets) != plan.n_facets:
        raise ValueError(
            f'query {plan.query_id} declares {plan.n_facets} facets but contains {len(plan.facets)}'
        )
    rng = Random(plan.plan_seed)
    texts: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    labels: list[str] = []
    for facet in plan.facets:
        start = len(texts)
        for local_idx in range(probe_n):
            fact = make_gold_fact(plan, facet, ontology, local_idx, rng)
            texts.append(
                render_canonical_chunk_text(
                    fact,
                    ontology,
                    chunk_text_style,
                    surface_group='seen',
                )
            )
            labels.append(facet.facet_id)
        offsets[facet.facet_id] = (start, len(texts))
    return {'texts': texts, 'offsets': offsets, 'labels': labels}


def _select_calibration(
    *,
    plan: QueryPlan,
    probe: FacetProbe,
    template_ids: list[str],
    query_vectors: np.ndarray,
    probe_vectors: np.ndarray,
    probe_n: int,
) -> tuple[str, str, dict[str, str | float]]:
    if len(template_ids) != len(query_vectors):
        raise ValueError('each query template must have one calibration embedding')
    primary_facets = [facet for facet in plan.facets if facet.priority == 'primary']
    secondary_facets = [facet for facet in plan.facets if facet.priority == 'secondary']
    if len(primary_facets) != 2 or len(secondary_facets) != 2:
        raise ValueError(f'query {plan.query_id} must have two facets per axis priority')
    offsets = probe['offsets']
    candidate_rows: list[dict[str, str | float]] = []
    candidate_facet_stats: list[list[dict[str, str | float]]] = []
    for template_index, (template_id, query_vector) in enumerate(
        zip(template_ids, query_vectors, strict=True)
    ):
        similarities = probe_vectors @ query_vector
        facet_stats: list[dict[str, str | float]] = []
        for facet in plan.facets:
            start, end = offsets[facet.facet_id]
            values = similarities[start:end]
            p25, median, p75 = np.percentile(values, [25, 50, 75])
            facet_stats.append({
                'facet_id': facet.facet_id,
                'subgroup_id': facet.subgroup_id,
                'axis': facet.axis,
                'value_bin': facet.value_bin,
                'priority': facet.priority,
                'mean_query_sim': float(values.mean()),
                'p25_query_sim': float(p25),
                'median_query_sim': float(median),
                'p75_query_sim': float(p75),
            })
        primary_stats = [row for row in facet_stats if row['priority'] == 'primary']
        secondary_stats = [row for row in facet_stats if row['priority'] == 'secondary']
        primary_axis_margin = min(float(row['p25_query_sim']) for row in primary_stats) - max(
            float(row['p75_query_sim']) for row in secondary_stats
        )
        primary_cohort_mean_gap = abs(
            float(primary_stats[0]['mean_query_sim']) - float(primary_stats[1]['mean_query_sim'])
        )
        candidate_rows.append({
            'template_id': template_id,
            'template_index': template_index,
            'primary_axis_probe_margin': primary_axis_margin,
            'primary_cohort_mean_gap': primary_cohort_mean_gap,
            'secondary_mean_query_sim': float(
                np.mean([float(row['mean_query_sim']) for row in secondary_stats])
            ),
        })
        candidate_facet_stats.append(facet_stats)

    selected_index = max(
        range(len(candidate_rows)),
        key=lambda index: (
            float(candidate_rows[index]['primary_axis_probe_margin']),
            -float(candidate_rows[index]['primary_cohort_mean_gap']),
            -index,
        ),
    )
    selected_candidate = candidate_rows[selected_index]
    selected_stats = candidate_facet_stats[selected_index]
    selected_primary_stats = [row for row in selected_stats if row['priority'] == 'primary']
    selected_facet = max(selected_primary_stats, key=lambda item: float(item['mean_query_sim']))
    other_primary = next(
        item for item in selected_primary_stats if item['facet_id'] != selected_facet['facet_id']
    )
    primary_cohort_margin = float(selected_facet['p25_query_sim']) - float(
        other_primary['p75_query_sim']
    )
    primary_axis_margin = float(selected_candidate['primary_axis_probe_margin'])
    warning = primary_axis_margin < 0.0 or primary_cohort_margin < 0.0
    return (
        str(selected_facet['facet_id']),
        str(selected_candidate['template_id']),
        {
            'query_id': plan.query_id,
            'evidence_profile_id': plan.evidence_profile_id,
            'primary_axis': plan.primary_axis,
            'secondary_axis': plan.secondary_axis,
            'previous_template_id': plan.template_id,
            'selected_template_id': str(selected_candidate['template_id']),
            'previous_dominant_primary_facet_id': plan.dominant_primary_facet_id,
            'dominant_primary_facet_id': str(selected_facet['facet_id']),
            'dominant_primary_subgroup_id': str(selected_facet['subgroup_id']),
            'probe_chunks_per_facet': probe_n,
            'selected_mean_query_sim': float(selected_facet['mean_query_sim']),
            'selected_probe_margin': primary_cohort_margin,
            'primary_axis_probe_margin': primary_axis_margin,
            'primary_cohort_mean_gap': float(selected_candidate['primary_cohort_mean_gap']),
            'secondary_mean_query_sim': float(selected_candidate['secondary_mean_query_sim']),
            'calibration_warning': warning,
            'facet_stats_json': json.dumps(selected_stats, sort_keys=True),
            'template_stats_json': json.dumps(candidate_rows, sort_keys=True),
        },
    )


def _semantic_probe_gate(
    *,
    cfg: ExperimentCfg,
    plan: QueryPlan,
    probe: FacetProbe,
    query_vector: np.ndarray,
    probe_vectors: np.ndarray,
) -> dict[str, object]:
    if not probe['texts']:
        failures = {'fail_missing_probe_texts': True}
        return {
            'semantic_gate_applied': True,
            'passes_semantic_gate': False,
            'semantic_gate_failures_json': json.dumps(failures, sort_keys=True),
            'semantic_gate_failure_reasons_json': json.dumps(list(failures)),
            'probe_topk_k': 0,
            'probe_primary_axis_target': 0,
            'probe_primary_axis_topk_count': 0,
            'probe_topk_retrieved_facets': 0,
            'probe_mean_in_facet_similarity': 0.0,
            'probe_mean_cross_facet_similarity': 0.0,
            'probe_in_minus_cross_similarity': 0.0,
            'probe_same_axis_cohort_gap': 0.0,
            'probe_same_cohort_axis_gap': 0.0,
        }

    sims = probe_vectors @ query_vector
    chunk_pools = cfg.generation.chunk_pools
    competitive_pool_mass = (
        chunk_pools.gold_chunks_per_query() + chunk_pools.near_miss_distractors_per_query()
    )
    stress_horizon_k = cfg.geometry_filter.stress_horizon(
        competitive_pool_mass=competitive_pool_mass
    )
    topk_k = min(stress_horizon_k, len(probe['texts']))
    ranked_idx = np.argsort(-sims)[:topk_k]
    facet_by_id = {facet.facet_id: facet for facet in plan.facets}
    topk_labels = [probe['labels'][int(index)] for index in ranked_idx]
    primary_axis_topk_count = sum(
        1 for facet_id in topk_labels if facet_by_id[facet_id].axis == plan.primary_axis
    )
    n_topk_retrieved_facets = len(set(topk_labels))
    primary_axis_fraction = primary_axis_topk_count / max(topk_k, 1)
    retrieved_facet_fraction = n_topk_retrieved_facets / max(len(plan.facets), 1)
    separation = _probe_facet_separation(plan, probe, probe_vectors)
    failures = {
        'fail_excess_stress_horizon_facet_coverage': (
            retrieved_facet_fraction > cfg.geometry_filter.max_retrieved_facet_fraction
        ),
        'fail_weak_primary_axis_dominance': (
            primary_axis_fraction < cfg.geometry_filter.min_primary_axis_fraction
        ),
    }
    failure_reasons = [name for name, failed in failures.items() if failed]
    return {
        'semantic_gate_applied': True,
        'passes_semantic_gate': not failure_reasons,
        'semantic_gate_failures_json': json.dumps(failures, sort_keys=True),
        'semantic_gate_failure_reasons_json': json.dumps(failure_reasons),
        'probe_topk_k': topk_k,
        'probe_primary_axis_target': cfg.geometry_filter.min_primary_axis_fraction * topk_k,
        'probe_primary_axis_topk_count': primary_axis_topk_count,
        'probe_topk_retrieved_facets': n_topk_retrieved_facets,
        'probe_mean_in_facet_similarity': separation['mean_in_facet_similarity'],
        'probe_mean_cross_facet_similarity': separation['mean_cross_facet_similarity'],
        'probe_in_minus_cross_similarity': separation['in_minus_cross_similarity'],
        'probe_same_axis_cohort_gap': separation['same_axis_cohort_gap'],
        'probe_same_cohort_axis_gap': separation['same_cohort_axis_gap'],
    }


def _probe_facet_separation(
    plan: QueryPlan,
    probe: FacetProbe,
    probe_vectors: np.ndarray,
) -> dict[str, float]:
    labels = np.array(probe['labels'])
    if len(labels) < 2:
        return {
            'mean_in_facet_similarity': 0.0,
            'mean_cross_facet_similarity': 0.0,
            'in_minus_cross_similarity': 0.0,
            'same_axis_cohort_gap': 0.0,
            'same_cohort_axis_gap': 0.0,
        }
    sim = probe_vectors @ probe_vectors.T
    same = labels[:, None] == labels[None, :]
    not_self = ~np.eye(len(labels), dtype=bool)
    in_vals = sim[same & not_self]
    cross_vals = sim[~same & not_self]
    in_sim = float(in_vals.mean()) if len(in_vals) else 0.0
    cross_sim = float(cross_vals.mean()) if len(cross_vals) else 0.0
    facet_by_id = {facet.facet_id: facet for facet in plan.facets}
    same_axis_diff_cohort: list[float] = []
    same_cohort_diff_axis: list[float] = []
    for left in range(len(labels)):
        left_facet = facet_by_id[str(labels[left])]
        for right in range(left + 1, len(labels)):
            if labels[left] == labels[right]:
                continue
            right_facet = facet_by_id[str(labels[right])]
            value = float(sim[left, right])
            if left_facet.axis == right_facet.axis:
                same_axis_diff_cohort.append(value)
            elif left_facet.subgroup_id == right_facet.subgroup_id:
                same_cohort_diff_axis.append(value)
    same_axis_mean = float(np.mean(same_axis_diff_cohort)) if same_axis_diff_cohort else 0.0
    same_cohort_mean = float(np.mean(same_cohort_diff_axis)) if same_cohort_diff_axis else 0.0
    return {
        'mean_in_facet_similarity': in_sim,
        'mean_cross_facet_similarity': cross_sim,
        'in_minus_cross_similarity': in_sim - cross_sim,
        'same_axis_cohort_gap': in_sim - same_axis_mean,
        'same_cohort_axis_gap': in_sim - same_cohort_mean,
    }


def _with_dominant_primary_query(
    cfg: ExperimentCfg,
    plan: QueryPlan,
    *,
    selected_facet_id: str,
    selected_template_id: str,
) -> QueryPlan:
    facets: list[QueryPlanFacet] = []
    for facet in plan.facets:
        selected = facet.facet_id == selected_facet_id
        primary = facet.priority == 'primary'
        facets.append(
            facet.model_copy(
                update={
                    'cluster_role': (
                        'dominant_primary_gold'
                        if selected
                        else 'primary_gold'
                        if primary
                        else facet.cluster_role
                    ),
                    'target_gold_chunks': (
                        cfg.generation.chunk_pools.dominant_primary.size
                        if selected
                        else cfg.generation.chunk_pools.other_primary.size
                        if primary
                        else facet.target_gold_chunks
                    ),
                }
            )
        )
    logical = plan.logical_form.model_copy(update={'dominant_primary_facet_id': selected_facet_id})
    return plan.model_copy(
        update={
            'template_id': selected_template_id,
            'dominant_primary_facet_id': selected_facet_id,
            'gold_chunks_total': sum(f.target_gold_chunks for f in facets),
            'facets': facets,
            'logical_form': logical,
        }
    )


def _calibration_row_without_embeddings(plan: QueryPlan) -> dict[str, object]:
    return {
        'query_id': plan.query_id,
        'evidence_profile_id': plan.evidence_profile_id,
        'primary_axis': plan.primary_axis,
        'secondary_axis': plan.secondary_axis,
        'previous_template_id': plan.template_id,
        'selected_template_id': plan.template_id,
        'previous_dominant_primary_facet_id': plan.dominant_primary_facet_id,
        'dominant_primary_facet_id': plan.dominant_primary_facet_id,
        'dominant_primary_subgroup_id': next(
            f.subgroup_id for f in plan.facets if f.facet_id == plan.dominant_primary_facet_id
        ),
        'probe_chunks_per_facet': 0,
        'selected_mean_query_sim': None,
        'selected_probe_margin': None,
        'primary_axis_probe_margin': None,
        'primary_cohort_mean_gap': None,
        'secondary_mean_query_sim': None,
        'calibration_warning': False,
        'facet_stats_json': '[]',
        'template_stats_json': '[]',
        'semantic_gate_applied': False,
        'passes_semantic_gate': None,
        'semantic_gate_failures_json': '{}',
        'semantic_gate_failure_reasons_json': '[]',
        'probe_topk_k': None,
        'probe_primary_axis_target': None,
        'probe_primary_axis_topk_count': None,
        'probe_topk_retrieved_facets': None,
        'probe_mean_in_facet_similarity': None,
        'probe_mean_cross_facet_similarity': None,
        'probe_in_minus_cross_similarity': None,
        'probe_same_axis_cohort_gap': None,
        'probe_same_cohort_axis_gap': None,
    }


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.logging import (
        setup_logging,
    )

    config = load_config_from_cli()
    output_paths = paths_for(config)
    setup_logging(output_paths)
    run_calibrate_query_plans(config, output_paths)
