"""Calibrate dominant facets from neutral probe embeddings.

This stage keeps query wording natural while making the hidden dominant facet
an embedding-space property. It probes each planned facet with deterministic
role-neutral chunk text, embeds those probes against the normal query text, and
then oversamples the facet that is naturally closest to the query.
"""

from __future__ import annotations

import json
from random import Random
from typing import Any

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    MedicalOntology,
    QueryPlan,
    QueryPlanFacet,
)

from .chunk_rendering import (
    render_canonical_chunk_text,
)
from .facts import make_gold_fact
from .ontology import load_ontology
from .queries_answers import render_query


def run_calibrate_query_plans(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    plans_df = read_parquet(paths, 'query_plans')
    if cfg.generation.dominance_mode == 'rotating':
        calibration = _rotating_calibration_frame(plans_df)
        write_parquet(paths, 'query_plan_calibration', calibration)
        print('[calibrate_plans] dominance_mode=rotating; kept planned dominant facets')
        return plans_df

    ontology = load_ontology(cfg)
    plans = [QueryPlan.model_validate(row) for row in plans_df.iter_rows(named=True)]
    if not plans:
        calibration = _calibration_frame([])
        write_parquet(paths, 'query_plan_calibration', calibration)
        return plans_df

    from helpers.embedder import Embedder

    updated_plans: list[QueryPlan] = []
    calibration_rows: list[dict[str, object]] = []
    probe_n = int(cfg.generation.dominance_probe_chunks_per_facet)
    plans_per_batch = max(1, min(128, 4096 // max(1, probe_n * 4)))

    embedder = Embedder(
        model_name=cfg.embeddings.model_name,
        batch_size=cfg.embeddings.batch_size,
        query_prompt=cfg.embeddings.query_prompt,
        device=cfg.embeddings.device,
        devices=cfg.embeddings.devices,
    )
    try:
        for start in tqdm(
            range(0, len(plans), plans_per_batch),
            desc='Calibrating plans',
            dynamic_ncols=True,
        ):
            batch = plans[start : start + plans_per_batch]
            probe_batch = [_probe_plan(plan, ontology, probe_n) for plan in batch]
            query_vectors = embedder.embed_queries(
                [probe['query_text'] for probe in probe_batch],
                normalize=cfg.embeddings.normalize,
            )
            probe_vectors = embedder.embed_docs(
                [text for probe in probe_batch for text in probe['probe_texts']],
                normalize=cfg.embeddings.normalize,
            )

            offset = 0
            for plan, probe, query_vector in zip(batch, probe_batch, query_vectors, strict=True):
                n_probe_texts = len(probe['probe_texts'])
                plan_probe_vectors = probe_vectors[offset : offset + n_probe_texts]
                offset += n_probe_texts

                selected, row = _select_dominant_facet(
                    cfg=cfg,
                    plan=plan,
                    probe=probe,
                    query_vector=np.asarray(query_vector, dtype=np.float32),
                    probe_vectors=np.asarray(plan_probe_vectors, dtype=np.float32),
                    probe_n=probe_n,
                )
                updated_plans.append(_with_dominant_facet(cfg, plan, selected))
                calibration_rows.append(row)
    finally:
        embedder.release()

    updated_df = pl.from_dicts([plan.to_row() for plan in updated_plans], infer_schema_length=None)
    calibration = _calibration_frame(calibration_rows)
    write_parquet(paths, 'query_plans', updated_df)
    write_parquet(paths, 'query_plan_calibration', calibration)
    _print_calibration_summary(calibration)
    return updated_df


def _probe_plan(plan: QueryPlan, ontology: MedicalOntology, probe_n: int) -> dict[str, Any]:
    rng = Random(plan.plan_seed)
    query_text = render_query(plan, ontology)
    probe_texts: list[str] = []
    facet_offsets: dict[str, tuple[int, int]] = {}

    for facet in plan.facets:
        start = len(probe_texts)
        for local_idx in range(probe_n):
            fact = make_gold_fact(
                plan=plan,
                facet=facet,
                ontology=ontology,
                local_idx=local_idx,
                rng=rng,
            )
            probe_texts.append(render_canonical_chunk_text(fact, ontology))
        facet_offsets[facet.facet_id] = (start, len(probe_texts))

    return {
        'query_text': query_text,
        'probe_texts': probe_texts,
        'facet_offsets': facet_offsets,
    }


def _select_dominant_facet(
    *,
    cfg: ExperimentCfg,
    plan: QueryPlan,
    probe: dict[str, Any],
    query_vector: np.ndarray,
    probe_vectors: np.ndarray,
    probe_n: int,
) -> tuple[str, dict[str, object]]:
    sims = probe_vectors @ query_vector
    facet_stats: list[dict[str, int | float | str]] = []

    for facet in plan.facets:
        start, end = probe['facet_offsets'][facet.facet_id]
        facet_sims = np.asarray(sims[start:end], dtype=np.float32)
        p25, p50, p75 = np.percentile(facet_sims, [25, 50, 75])
        facet_stats.append({
            'facet_id': facet.facet_id,
            'subgroup_id': facet.subgroup_id,
            'subgroup_label': facet.subgroup_label,
            'axis': facet.axis,
            'value_bin': facet.value_bin,
            'probe_text_count': probe_n,
            'mean_query_sim': float(facet_sims.mean()),
            'min_query_sim': float(facet_sims.min()),
            'p25_query_sim': float(p25),
            'median_query_sim': float(p50),
            'p75_query_sim': float(p75),
            'max_query_sim': float(facet_sims.max()),
        })

    for stat in facet_stats:
        complement_p75 = [
            float(other['p75_query_sim'])
            for other in facet_stats
            if other['facet_id'] != stat['facet_id']
        ]
        best_complement_p75 = max(complement_p75) if complement_p75 else 0.0
        stat['best_complement_p75_query_sim'] = best_complement_p75
        stat['probe_margin_p25_gt_best_complement_p75'] = (
            float(stat['p25_query_sim']) - best_complement_p75
        )

    selected = max(
        facet_stats,
        key=lambda stat: (
            float(stat['probe_margin_p25_gt_best_complement_p75']),
            float(stat['mean_query_sim']),
        ),
    )
    margin = float(selected['probe_margin_p25_gt_best_complement_p75'])
    min_margin = cfg.generation.calibration_min_probe_margin
    passes_margin = min_margin is None or margin >= min_margin

    return str(selected['facet_id']), {
        'query_id': plan.query_id,
        'dominance_mode': cfg.generation.dominance_mode,
        'previous_dominant_facet_id': plan.dominant_facet_id,
        'selected_dominant_facet_id': str(selected['facet_id']),
        'selected_axis': str(selected['axis']),
        'selected_subgroup_id': str(selected['subgroup_id']),
        'selected_value_bin': str(selected['value_bin']),
        'probe_chunks_per_facet': probe_n,
        'selected_mean_query_sim': float(selected['mean_query_sim']),
        'selected_p25_query_sim': float(selected['p25_query_sim']),
        'selected_p75_query_sim': float(selected['p75_query_sim']),
        'best_complement_p75_query_sim': float(selected['best_complement_p75_query_sim']),
        'selected_probe_margin': margin,
        'calibration_min_probe_margin': min_margin,
        'passes_calibration_margin': passes_margin,
        'query_text': str(probe['query_text']),
        'facet_stats_json': json.dumps(facet_stats, sort_keys=True),
    }


def _with_dominant_facet(
    cfg: ExperimentCfg,
    plan: QueryPlan,
    dominant_facet_id: str,
) -> QueryPlan:
    facets: list[QueryPlanFacet] = []
    for facet in plan.facets:
        is_dominant = facet.facet_id == dominant_facet_id
        facets.append(
            facet.model_copy(
                update={
                    'cluster_role': 'dominant_gold' if is_dominant else 'complementary_gold',
                    'target_gold_chunks': (
                        cfg.generation.gold_chunks_dominant
                        if is_dominant
                        else cfg.generation.gold_chunks_complementary
                    ),
                }
            )
        )

    logical_form = plan.logical_form.model_copy(update={'dominant_facet_id': dominant_facet_id})
    return plan.model_copy(
        update={
            'dominant_facet_id': dominant_facet_id,
            'gold_chunks_total': sum(facet.target_gold_chunks for facet in facets),
            'facets': facets,
            'logical_form': logical_form,
        }
    )


def _rotating_calibration_frame(plans_df: pl.DataFrame) -> pl.DataFrame:
    if plans_df.is_empty():
        return _calibration_frame([])
    rows = [
        {
            'query_id': str(row['query_id']),
            'dominance_mode': 'rotating',
            'previous_dominant_facet_id': str(row['dominant_facet_id']),
            'selected_dominant_facet_id': str(row['dominant_facet_id']),
            'selected_axis': None,
            'selected_subgroup_id': None,
            'selected_value_bin': None,
            'probe_chunks_per_facet': 0,
            'selected_mean_query_sim': None,
            'selected_p25_query_sim': None,
            'selected_p75_query_sim': None,
            'best_complement_p75_query_sim': None,
            'selected_probe_margin': None,
            'calibration_min_probe_margin': None,
            'passes_calibration_margin': True,
            'query_text': None,
            'facet_stats_json': '[]',
        }
        for row in plans_df.iter_rows(named=True)
    ]
    return _calibration_frame(rows)


def _calibration_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        'query_id': pl.String,
        'dominance_mode': pl.String,
        'previous_dominant_facet_id': pl.String,
        'selected_dominant_facet_id': pl.String,
        'selected_axis': pl.String,
        'selected_subgroup_id': pl.String,
        'selected_value_bin': pl.String,
        'probe_chunks_per_facet': pl.Int64,
        'selected_mean_query_sim': pl.Float64,
        'selected_p25_query_sim': pl.Float64,
        'selected_p75_query_sim': pl.Float64,
        'best_complement_p75_query_sim': pl.Float64,
        'selected_probe_margin': pl.Float64,
        'calibration_min_probe_margin': pl.Float64,
        'passes_calibration_margin': pl.Boolean,
        'query_text': pl.String,
        'facet_stats_json': pl.String,
    }
    return pl.from_dicts(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _print_calibration_summary(calibration: pl.DataFrame) -> None:
    if calibration.is_empty():
        print('[calibrate_plans] no query plans available')
        return
    selected_slots = (
        calibration
        .with_columns(pl.col('selected_dominant_facet_id').str.extract(r'_f(\d)$', 1).alias('slot'))
        .group_by('slot')
        .agg(pl.len().alias('n'))
        .sort('slot')
    )
    n_pass_margin = int(calibration['passes_calibration_margin'].sum())
    print(f'[calibrate_plans] calibrated {len(calibration):,} query plan(s)')
    print(f'[calibrate_plans] margin pass: {n_pass_margin:,}/{len(calibration):,}')
    print(selected_slots)


if __name__ == '__main__':
    from experiments.medical_dataset_gen.global_configs import (
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_calibrate_query_plans(cfg, paths)
