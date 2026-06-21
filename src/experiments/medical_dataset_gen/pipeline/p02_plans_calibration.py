"""Calibrate the larger cluster within a query's declared primary axis."""

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


class PrimaryFacetProbe(TypedDict):
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
    plans_per_batch = max(1, min(128, 4096 // max(1, probe_n * 2)))
    try:
        for start in tqdm(
            range(0, len(plans), plans_per_batch),
            desc='Calibrating primary facets',
            dynamic_ncols=True,
        ):
            batch = plans[start : start + plans_per_batch]
            probes = [_prepare_probe(plan, ontology, probe_n) for plan in batch]
            query_vectors = embedder.embed_queries(
                [render_query(plan, ontology) for plan in batch],
                normalize=cfg.embeddings.normalize,
            )
            probe_vectors = embedder.embed_docs(
                [text for probe in probes for text in probe['texts']],
                normalize=cfg.embeddings.normalize,
            )
            offset = 0
            for plan, probe, query_vector in zip(batch, probes, query_vectors, strict=True):
                end = offset + len(probe['texts'])
                selected, row = _select_primary_facet(
                    plan=plan,
                    probe=probe,
                    query_vector=np.asarray(query_vector),
                    vectors=np.asarray(probe_vectors[offset:end]),
                    probe_n=probe_n,
                )
                offset = end
                updated.append(_with_calibrated_primary_facet(cfg, plan, selected))
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
) -> PrimaryFacetProbe:
    primary_facets = [facet for facet in plan.facets if facet.priority == 'primary']
    if len(primary_facets) != 2:
        raise ValueError(f'query {plan.query_id} must contain two primary facets')
    rng = Random(plan.plan_seed)
    texts: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    for facet in primary_facets:
        start = len(texts)
        for local_idx in range(probe_n):
            fact = make_gold_fact(plan, facet, ontology, local_idx, rng)
            texts.append(render_canonical_chunk_text(fact, ontology))
        offsets[facet.facet_id] = (start, len(texts))
    return {'texts': texts, 'offsets': offsets}


def _select_primary_facet(
    *,
    plan: QueryPlan,
    probe: PrimaryFacetProbe,
    query_vector: np.ndarray,
    vectors: np.ndarray,
    probe_n: int,
):
    primary_facets = [facet for facet in plan.facets if facet.priority == 'primary']
    offsets = probe['offsets']
    similarities = vectors @ query_vector
    stats: list[dict[str, object]] = []
    for facet in primary_facets:
        start, end = offsets[facet.facet_id]
        values = similarities[start:end]
        p25, median, p75 = np.percentile(values, [25, 50, 75])
        stats.append(
            {
                'facet_id': facet.facet_id,
                'subgroup_id': facet.subgroup_id,
                'mean_query_sim': float(values.mean()),
                'p25_query_sim': float(p25),
                'median_query_sim': float(median),
                'p75_query_sim': float(p75),
            }
        )
    selected = max(stats, key=lambda item: float(item['mean_query_sim']))
    other = next(item for item in stats if item['facet_id'] != selected['facet_id'])
    margin = float(selected['p25_query_sim']) - float(other['p75_query_sim'])
    warning = margin < 0.0
    return str(selected['facet_id']), {
        'query_id': plan.query_id,
        'evidence_profile_id': plan.evidence_profile_id,
        'primary_axis': plan.primary_axis,
        'previous_calibrated_primary_facet_id': plan.calibrated_primary_facet_id,
        'calibrated_primary_facet_id': str(selected['facet_id']),
        'calibrated_primary_subgroup_id': str(selected['subgroup_id']),
        'probe_chunks_per_facet': probe_n,
        'selected_mean_query_sim': float(selected['mean_query_sim']),
        'selected_probe_margin': margin,
        'calibration_warning': warning,
        'facet_stats_json': json.dumps(stats, sort_keys=True),
    }


def _with_calibrated_primary_facet(
    cfg: ExperimentCfg, plan: QueryPlan, selected_id: str
) -> QueryPlan:
    facets: list[QueryPlanFacet] = []
    for facet in plan.facets:
        selected = facet.facet_id == selected_id
        primary = facet.priority == 'primary'
        facets.append(
            facet.model_copy(
                update={
                    'cluster_role': (
                        'calibrated_primary_gold'
                        if selected
                        else 'primary_gold'
                        if primary
                        else 'secondary_gold'
                    ),
                    'target_gold_chunks': (
                        cfg.generation.gold_chunks_calibrated_primary
                        if selected
                        else cfg.generation.gold_chunks_other_primary
                        if primary
                        else cfg.generation.gold_chunks_secondary
                    ),
                }
            )
        )
    logical = plan.logical_form.model_copy(update={'calibrated_primary_facet_id': selected_id})
    return plan.model_copy(
        update={
            'calibrated_primary_facet_id': selected_id,
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
        'previous_calibrated_primary_facet_id': plan.calibrated_primary_facet_id,
        'calibrated_primary_facet_id': plan.calibrated_primary_facet_id,
        'calibrated_primary_subgroup_id': next(
            f.subgroup_id for f in plan.facets if f.facet_id == plan.calibrated_primary_facet_id
        ),
        'probe_chunks_per_facet': 0,
        'selected_mean_query_sim': None,
        'selected_probe_margin': None,
        'calibration_warning': False,
        'facet_stats_json': '[]',
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
