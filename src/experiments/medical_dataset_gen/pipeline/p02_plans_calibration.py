"""Calibrate query emphasis and the larger cluster within the declared primary axis."""

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
from experiments.medical_dataset_gen.pipeline.p03_structured_facts import make_gold_fact
from experiments.medical_dataset_gen.pipeline.p05_queries_answers import render_query
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    MedicalOntology,
    QueryPlan,
    QueryPlanFacet,
)
from experiments.medical_dataset_gen.utils.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


class FacetProbe(TypedDict):
    texts: list[str]
    offsets: dict[str, tuple[int, int]]


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
        device=cfg.embeddings.device,
        devices=cfg.embeddings.devices,
    )
    updated: list[QueryPlan] = []
    calibration_rows: list[dict[str, object]] = []
    probe_n = int(cfg.generation.calibration_probe_chunks_per_facet)
    template_ids = query_template_ids()
    max_facets_per_plan = max(len(plan.facets) for plan in plans)
    plans_per_batch = max(1, min(128, 4096 // max(1, probe_n * max_facets_per_plan)))
    try:
        for start in tqdm(
            range(0, len(plans), plans_per_batch),
            desc='Calibrating query emphasis',
            dynamic_ncols=True,
        ):
            batch = plans[start : start + plans_per_batch]
            probes = [_prepare_probe(plan, ontology, probe_n) for plan in batch]
            query_vectors = embedder.embed_queries(
                [
                    render_query(plan, ontology, template_id=template_id)
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
                selected_facet_id, selected_template_id, row = _select_calibration(
                    plan=plan,
                    probe=probe,
                    template_ids=template_ids,
                    query_vectors=np.asarray(query_vectors[query_offset:query_end]),
                    probe_vectors=np.asarray(probe_vectors[probe_offset:probe_end]),
                    probe_n=probe_n,
                )
                probe_offset = probe_end
                query_offset = query_end
                updated.append(
                    _with_calibrated_query(
                        cfg,
                        plan,
                        selected_facet_id=selected_facet_id,
                        selected_template_id=selected_template_id,
                    )
                )
                calibration_rows.append(row)
    finally:
        embedder.release()

    updated_df = pl.from_dicts([plan.to_row() for plan in updated], infer_schema_length=None)
    write_parquet(paths, 'query_plans', updated_df)
    write_parquet(
        paths,
        'query_plan_calibration',
        pl.from_dicts(calibration_rows, infer_schema_length=None),
    )
    return updated_df


def _prepare_probe(
    plan: QueryPlan, ontology: MedicalOntology, probe_n: int
) -> FacetProbe:
    if len(plan.facets) != plan.n_facets:
        raise ValueError(
            f'query {plan.query_id} declares {plan.n_facets} facets but contains {len(plan.facets)}'
        )
    rng = Random(plan.plan_seed)
    texts: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    for facet in plan.facets:
        start = len(texts)
        for local_idx in range(probe_n):
            fact = make_gold_fact(plan, facet, ontology, local_idx, rng)
            texts.append(render_canonical_chunk_text(fact, ontology))
        offsets[facet.facet_id] = (start, len(texts))
    return {'texts': texts, 'offsets': offsets}


def _select_calibration(
    *,
    plan: QueryPlan,
    probe: FacetProbe,
    template_ids: list[str],
    query_vectors: np.ndarray,
    probe_vectors: np.ndarray,
    probe_n: int,
) -> tuple[str, str, dict[str, object]]:
    if len(template_ids) != len(query_vectors):
        raise ValueError('each query template must have one calibration embedding')
    primary_facets = [facet for facet in plan.facets if facet.priority == 'primary']
    secondary_facets = [facet for facet in plan.facets if facet.priority == 'secondary']
    if len(primary_facets) != 2 or len(secondary_facets) != 2:
        raise ValueError(f'query {plan.query_id} must have two facets per axis priority')
    offsets = probe['offsets']
    candidate_rows: list[dict[str, object]] = []
    candidate_facet_stats: list[list[dict[str, object]]] = []
    for template_index, (template_id, query_vector) in enumerate(
        zip(template_ids, query_vectors, strict=True)
    ):
        similarities = probe_vectors @ query_vector
        facet_stats: list[dict[str, object]] = []
        for facet in plan.facets:
            start, end = offsets[facet.facet_id]
            values = similarities[start:end]
            p25, median, p75 = np.percentile(values, [25, 50, 75])
            facet_stats.append(
                {
                    'facet_id': facet.facet_id,
                    'subgroup_id': facet.subgroup_id,
                    'axis': facet.axis,
                    'value_bin': facet.value_bin,
                    'priority': facet.priority,
                    'mean_query_sim': float(values.mean()),
                    'p25_query_sim': float(p25),
                    'median_query_sim': float(median),
                    'p75_query_sim': float(p75),
                }
            )
        primary_stats = [row for row in facet_stats if row['priority'] == 'primary']
        secondary_stats = [row for row in facet_stats if row['priority'] == 'secondary']
        primary_axis_margin = min(float(row['p25_query_sim']) for row in primary_stats) - max(
            float(row['p75_query_sim']) for row in secondary_stats
        )
        primary_cohort_mean_gap = abs(
            float(primary_stats[0]['mean_query_sim'])
            - float(primary_stats[1]['mean_query_sim'])
        )
        candidate_rows.append(
            {
                'template_id': template_id,
                'template_index': template_index,
                'primary_axis_probe_margin': primary_axis_margin,
                'primary_cohort_mean_gap': primary_cohort_mean_gap,
                'secondary_mean_query_sim': float(
                    np.mean([float(row['mean_query_sim']) for row in secondary_stats])
                ),
            }
        )
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
    selected_facet = max(
        selected_primary_stats, key=lambda item: float(item['mean_query_sim'])
    )
    other_primary = next(
        item for item in selected_primary_stats if item['facet_id'] != selected_facet['facet_id']
    )
    primary_cohort_margin = float(selected_facet['p25_query_sim']) - float(
        other_primary['p75_query_sim']
    )
    primary_axis_margin = float(selected_candidate['primary_axis_probe_margin'])
    warning = primary_axis_margin < 0.0 or primary_cohort_margin < 0.0
    return str(selected_facet['facet_id']), str(selected_candidate['template_id']), {
        'query_id': plan.query_id,
        'evidence_profile_id': plan.evidence_profile_id,
        'primary_axis': plan.primary_axis,
        'secondary_axis': plan.secondary_axis,
        'previous_template_id': plan.template_id,
        'selected_template_id': str(selected_candidate['template_id']),
        'previous_calibrated_primary_facet_id': plan.calibrated_primary_facet_id,
        'calibrated_primary_facet_id': str(selected_facet['facet_id']),
        'calibrated_primary_subgroup_id': str(selected_facet['subgroup_id']),
        'probe_chunks_per_facet': probe_n,
        'selected_mean_query_sim': float(selected_facet['mean_query_sim']),
        'selected_probe_margin': primary_cohort_margin,
        'primary_axis_probe_margin': primary_axis_margin,
        'primary_cohort_mean_gap': float(selected_candidate['primary_cohort_mean_gap']),
        'secondary_mean_query_sim': float(selected_candidate['secondary_mean_query_sim']),
        'calibration_warning': warning,
        'facet_stats_json': json.dumps(selected_stats, sort_keys=True),
        'template_stats_json': json.dumps(candidate_rows, sort_keys=True),
    }


def _with_calibrated_query(
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
                        'calibrated_primary_gold'
                        if selected
                        else 'primary_gold'
                        if primary
                        else facet.cluster_role
                    ),
                    'target_gold_chunks': (
                        cfg.generation.gold_chunks_calibrated_primary
                        if selected
                        else cfg.generation.gold_chunks_other_primary
                        if primary
                        else facet.target_gold_chunks
                    ),
                }
            )
        )
    logical = plan.logical_form.model_copy(
        update={'calibrated_primary_facet_id': selected_facet_id}
    )
    return plan.model_copy(
        update={
            'template_id': selected_template_id,
            'calibrated_primary_facet_id': selected_facet_id,
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
        'previous_calibrated_primary_facet_id': plan.calibrated_primary_facet_id,
        'calibrated_primary_facet_id': plan.calibrated_primary_facet_id,
        'calibrated_primary_subgroup_id': next(
            f.subgroup_id for f in plan.facets if f.facet_id == plan.calibrated_primary_facet_id
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
    }


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    config = load_config_from_cli()
    output_paths = paths_for(config)
    setup_logging(output_paths)
    run_calibrate_query_plans(config, output_paths)
