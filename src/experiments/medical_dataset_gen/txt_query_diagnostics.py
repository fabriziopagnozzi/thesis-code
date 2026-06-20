from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    load_config,
)
from experiments.medical_dataset_gen.retrieval.embed import load_embedding_arrays
from experiments.medical_dataset_gen.retrieval.utils import select_indices

_QUERY_COLUMNS = [
    'query_id',
    'query_type',
    'template_id',
    'condition_id',
    'condition_display',
    'subgroup_a_id',
    'subgroup_a_label',
    'subgroup_b_id',
    'subgroup_b_label',
    'dominant_facet_id',
    'split',
    'n_facets',
    'facets_json',
    'logical_form_json',
    'query_text',
]
_PLAN_COLUMNS = [
    'query_id',
    'plan_seed',
    'split',
    'query_type',
    'template_id',
    'condition_id',
    'condition_display',
    'subgroup_a_id',
    'subgroup_a_label',
    'subgroup_a_axis',
    'subgroup_a_field',
    'subgroup_a_value',
    'subgroup_b_id',
    'subgroup_b_label',
    'subgroup_b_axis',
    'subgroup_b_field',
    'subgroup_b_value',
    'dominant_facet_id',
    'n_facets',
    'gold_chunks_total',
    'distractor_chunks',
    'facets_json',
    'logical_form_json',
]
_CALIBRATION_COLUMNS = [
    'query_id',
    'dominance_mode',
    'previous_dominant_facet_id',
    'selected_dominant_facet_id',
    'selected_axis',
    'selected_subgroup_id',
    'selected_value_bin',
    'probe_chunks_per_facet',
    'selected_mean_query_sim',
    'selected_p25_query_sim',
    'selected_p75_query_sim',
    'best_complement_p75_query_sim',
    'selected_probe_margin',
    'calibration_min_probe_margin',
    'passes_calibration_margin',
    'facet_stats_json',
]
_ANSWER_COLUMNS = [
    'query_id',
    'answer_text',
    'facet_summaries_json',
    'answer_facts_json',
    'supporting_fact_ids_json',
    'supporting_facet_ids_json',
]
_MEMBERSHIP_COLUMNS = [
    'membership_id',
    'chunk_id',
    'query_id',
    'source_query_id',
    'fact_id',
    'facet_id',
    'target_facet_id',
    'cluster_id',
    'cluster_role',
    'is_gold',
    'distractor_type',
    'split',
]
_QREL_COLUMNS = [
    'query_id',
    'chunk_id',
    'fact_id',
    'facet_id',
    'target_facet_id',
    'cluster_id',
    'cluster_role',
    'is_gold',
    'distractor_type',
    'relevance_grade',
    'support_type',
]
_FACT_COLUMNS = [
    'query_id',
    'source_query_id',
    'fact_id',
    'chunk_reuse_key',
    'facet_id',
    'target_facet_id',
    'cluster_id',
    'cluster_role',
    'condition_id',
    'condition_display',
    'subgroup_id',
    'subgroup_label',
    'subgroup_axis',
    'subgroup_field',
    'subgroup_value',
    'axis',
    'value_bin',
    'duration_days',
    'treatment',
    'rehab_outcome',
    'is_gold',
    'distractor_type',
    'admission_id',
    'patient_id',
    'patient_age',
    'patient_sex',
    'clinical_subgroup_phrase',
    'note_style',
    'split',
    'must_mention',
    'must_not_mention',
]
_CHUNK_COLUMNS = [
    'chunk_id',
    'chunk_reuse_key',
    'text',
    'approx_words',
    'text_generation_source',
    'llm_attempted',
    'llm_rejected',
    'condition_id',
    'condition_display',
    'subgroup_id',
    'subgroup_label',
    'subgroup_axis',
    'subgroup_field',
    'subgroup_value',
    'axis',
    'value_bin',
    'duration_days',
    'treatment',
    'rehab_outcome',
    'patient_age',
    'patient_sex',
    'clinical_subgroup_phrase',
    'note_style',
    'validation_soft_warning_count',
    'validation_soft_warnings_json',
]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.exp)
    paths = MedicalDatasetGenPaths(
        cfg.global_.output_experiment,
        result_dir_overrides=cfg.global_.result_dir_overrides,
    )

    requested_query_ids = _requested_query_ids(args, paths)
    if not requested_query_ids:
        raise ValueError(
            'provide at least one query id with --query-id/--query-ids/--query-id-file, '
            'or use --from-plot-group bad'
        )
    comparison_query_ids = _comparison_query_ids(args, paths, requested_query_ids)

    diagnostics = build_text_diagnostics(
        cfg=cfg,
        paths=paths,
        query_ids=requested_query_ids,
        comparison_query_ids=comparison_query_ids,
        pool_n=args.pool_n,
        extra_top_ks=_parse_top_ks(args),
        chunk_text_chars=args.chunk_text_chars,
        chunk_text_mode=args.chunk_text_mode,
        detail=args.detail,
        rank_trace_limit=args.rank_trace_limit,
        top_chunk_texts=args.top_chunk_texts,
        representative_chunks_per_label=args.representative_chunks_per_label,
        max_representative_chunks=args.max_representative_chunks,
    )

    out_path = args.out or _default_out_path(paths, requested_query_ids, args.from_plot_group)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(diagnostics)
    print(f'[write] query diagnostics -> {out_path}')


def build_text_diagnostics(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    query_ids: list[str],
    comparison_query_ids: list[str],
    pool_n: int | None,
    extra_top_ks: list[int],
    chunk_text_chars: int,
    chunk_text_mode: str,
    detail: str,
    rank_trace_limit: int,
    top_chunk_texts: int,
    representative_chunks_per_label: int,
    max_representative_chunks: int,
) -> str:
    query_ids = _dedupe(query_ids)
    comparison_query_ids = _dedupe(comparison_query_ids)
    all_query_ids = _dedupe([*comparison_query_ids, *query_ids])
    pool_n = pool_n or cfg.retrieval.candidate_pool_n

    queries = _collect_query_table(paths, 'queries', all_query_ids, _QUERY_COLUMNS)
    present_query_ids = queries['query_id'].to_list() if 'query_id' in queries.columns else []
    missing_query_ids = [
        query_id for query_id in all_query_ids if query_id not in present_query_ids
    ]
    if not present_query_ids:
        raise ValueError(f'none of the requested query ids exist in {paths.table_path("queries")}')

    query_plans = _collect_query_table(paths, 'query_plans', present_query_ids, _PLAN_COLUMNS)
    calibration = _collect_query_table(
        paths, 'query_plan_calibration', present_query_ids, _CALIBRATION_COLUMNS
    )
    gold_answers = _collect_query_table(paths, 'gold_answers', present_query_ids, _ANSWER_COLUMNS)
    geometry = _collect_query_table(paths, 'geometry_stats', present_query_ids)
    plot_stats = _collect_query_table(paths, 'embedding_geometry_query_stats', present_query_ids)
    eval_results = _collect_query_table(paths, 'evaluation_results', present_query_ids)
    memberships = _collect_query_table(
        paths, 'chunk_memberships', present_query_ids, _MEMBERSHIP_COLUMNS
    )
    qrels = _collect_query_table(paths, 'qrels', present_query_ids, _QREL_COLUMNS)
    facts = _collect_query_table(paths, 'clinical_facts', present_query_ids, _FACT_COLUMNS)

    chunk_vectors, query_vectors, raw_chunk_ids, raw_query_ids = load_embedding_arrays(paths)
    chunk_ids = [str(value) for value in raw_chunk_ids]
    query_embedding_ids = [str(value) for value in raw_query_ids]
    chunk_id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
    query_id_to_idx = {query_id: idx for idx, query_id in enumerate(query_embedding_ids)}

    candidate_ids_by_query = _candidate_ids_by_query(
        cfg=cfg,
        paths=paths,
        queries=queries,
        query_ids=present_query_ids,
        memberships=memberships,
        all_chunk_ids=chunk_ids,
    )
    ranked_by_query = _rank_candidate_pools(
        queries=queries,
        query_ids=present_query_ids,
        candidate_ids_by_query=candidate_ids_by_query,
        chunk_id_to_idx=chunk_id_to_idx,
        query_id_to_idx=query_id_to_idx,
        chunk_vectors=chunk_vectors,
        query_vectors=query_vectors,
        pool_n=pool_n,
    )
    needed_chunk_ids = {
        str(row['chunk_id'])
        for ranked_rows in ranked_by_query.values()
        for row in ranked_rows
        if row.get('chunk_id')
    }
    chunk_documents = _collect_chunk_documents(paths, needed_chunk_ids)

    render_ctx = _RenderContext(
        cfg=cfg,
        paths=paths,
        pool_n=pool_n,
        missing_query_ids=missing_query_ids,
        queries_by_id=_rows_by_key(queries, 'query_id'),
        plans_by_id=_rows_by_key(query_plans, 'query_id'),
        calibration_by_id=_rows_by_key(calibration, 'query_id'),
        answers_by_id=_rows_by_key(gold_answers, 'query_id'),
        geometry_by_id=_rows_by_key(geometry, 'query_id'),
        plot_stats_by_id=_rows_by_key(plot_stats, 'query_id'),
        eval_results=eval_results,
        memberships_by_query_chunk=_rows_by_pair(memberships, 'query_id', 'chunk_id'),
        qrels_by_query_chunk=_rows_by_pair(qrels, 'query_id', 'chunk_id'),
        facts_by_fact_id=_rows_by_key(facts, 'fact_id'),
        chunk_by_id=_rows_by_key(chunk_documents, 'chunk_id'),
        ranked_by_query=ranked_by_query,
        chunk_vectors=chunk_vectors,
        query_vectors=query_vectors,
        chunk_id_to_idx=chunk_id_to_idx,
        query_id_to_idx=query_id_to_idx,
        extra_top_ks=extra_top_ks,
        chunk_text_chars=chunk_text_chars,
        chunk_text_mode=chunk_text_mode,
        detail=detail,
        rank_trace_limit=rank_trace_limit,
        top_chunk_texts=top_chunk_texts,
        representative_chunks_per_label=representative_chunks_per_label,
        max_representative_chunks=max_representative_chunks,
    )
    comparison_present = [
        query_id for query_id in comparison_query_ids if query_id in present_query_ids
    ]
    target_present = [
        query_id
        for query_id in query_ids
        if query_id in present_query_ids and query_id not in comparison_present
    ]
    return _render_diagnostics(render_ctx, target_present, comparison_present)


class _RenderContext:
    def __init__(
        self,
        *,
        cfg: ExperimentCfg,
        paths: MedicalDatasetGenPaths,
        pool_n: int,
        missing_query_ids: list[str],
        queries_by_id: dict[str, dict[str, Any]],
        plans_by_id: dict[str, dict[str, Any]],
        calibration_by_id: dict[str, dict[str, Any]],
        answers_by_id: dict[str, dict[str, Any]],
        geometry_by_id: dict[str, dict[str, Any]],
        plot_stats_by_id: dict[str, dict[str, Any]],
        eval_results: pl.DataFrame,
        memberships_by_query_chunk: dict[tuple[str, str], dict[str, Any]],
        qrels_by_query_chunk: dict[tuple[str, str], dict[str, Any]],
        facts_by_fact_id: dict[str, dict[str, Any]],
        chunk_by_id: dict[str, dict[str, Any]],
        ranked_by_query: dict[str, list[dict[str, Any]]],
        chunk_vectors: NDArray[np.float32],
        query_vectors: NDArray[np.float32],
        chunk_id_to_idx: dict[str, int],
        query_id_to_idx: dict[str, int],
        extra_top_ks: list[int],
        chunk_text_chars: int,
        chunk_text_mode: str,
        detail: str,
        rank_trace_limit: int,
        top_chunk_texts: int,
        representative_chunks_per_label: int,
        max_representative_chunks: int,
    ):
        self.cfg = cfg
        self.paths = paths
        self.pool_n = pool_n
        self.missing_query_ids = missing_query_ids
        self.queries_by_id = queries_by_id
        self.plans_by_id = plans_by_id
        self.calibration_by_id = calibration_by_id
        self.answers_by_id = answers_by_id
        self.geometry_by_id = geometry_by_id
        self.plot_stats_by_id = plot_stats_by_id
        self.eval_results = eval_results
        self.memberships_by_query_chunk = memberships_by_query_chunk
        self.qrels_by_query_chunk = qrels_by_query_chunk
        self.facts_by_fact_id = facts_by_fact_id
        self.chunk_by_id = chunk_by_id
        self.ranked_by_query = ranked_by_query
        self.chunk_vectors = chunk_vectors
        self.query_vectors = query_vectors
        self.chunk_id_to_idx = chunk_id_to_idx
        self.query_id_to_idx = query_id_to_idx
        self.extra_top_ks = extra_top_ks
        self.chunk_text_chars = chunk_text_chars
        self.chunk_text_mode = chunk_text_mode
        self.detail = detail
        self.rank_trace_limit = rank_trace_limit
        self.top_chunk_texts = top_chunk_texts
        self.representative_chunks_per_label = representative_chunks_per_label
        self.max_representative_chunks = max_representative_chunks


def _render_diagnostics(
    ctx: _RenderContext,
    target_query_ids: list[str],
    comparison_query_ids: list[str],
) -> str:
    lines: list[str] = []
    _h1(lines, 'MEDICAL DATASET QUERY TEXT DIAGNOSTICS')
    _kv(
        lines,
        [
            ('experiment', ctx.paths.exp_name),
            ('experiment_dir', str(ctx.paths.experiment_dir)),
            ('pool_scope', ctx.cfg.retrieval.pool_scope),
            ('candidate_pool_n_used', ctx.pool_n),
            ('embedding_model', ctx.cfg.embeddings.model_name),
            ('generation.dominance_mode', ctx.cfg.generation.dominance_mode),
            ('geometry.topk_dominance_k', ctx.cfg.geometry.topk_dominance_k),
            ('geometry.primary_topk_dominance_k', ctx.cfg.geometry.primary_topk_dominance_k),
            ('geometry.max_topk_retrieved_facets', ctx.cfg.geometry.max_topk_retrieved_facets),
            ('geometry.min_topk_dominant_count', ctx.cfg.geometry.min_topk_dominant_count),
            ('retrieval.k_values', ctx.cfg.retrieval.k_values),
            ('retrieval.lambda_values', ctx.cfg.retrieval.lambda_values),
            ('detail', ctx.detail),
            ('chunk_text_mode', ctx.chunk_text_mode),
            ('chunk_text_chars', ctx.chunk_text_chars),
            ('comparison_query_ids', comparison_query_ids),
            ('target_query_ids', target_query_ids),
        ],
    )
    if ctx.missing_query_ids:
        lines.append('')
        lines.append(f'Missing requested query ids: {_json(ctx.missing_query_ids)}')

    lines.append('')
    lines.append(
        'Diagnostic target: inspect whether poor queries are filtered because nearest-neighbor '
        'top-k already represents too many hidden facets, and whether that looks driven by '
        'semantic non-separability, query wording, or chunk quality.'
    )

    for query_id in comparison_query_ids:
        lines.append('')
        _render_query(lines, ctx, query_id, heading_prefix='GOOD COMPARISON')

    if comparison_query_ids and target_query_ids:
        lines.append('')
        _h1(lines, 'TARGET BAD QUERIES')
    for query_id in target_query_ids:
        lines.append('')
        _render_query(lines, ctx, query_id, heading_prefix='TARGET')
    return '\n'.join(lines).rstrip() + '\n'


def _render_query(
    lines: list[str],
    ctx: _RenderContext,
    query_id: str,
    *,
    heading_prefix: str,
) -> None:
    query = ctx.queries_by_id[query_id]
    plan = ctx.plans_by_id.get(query_id, query)
    answer = ctx.answers_by_id.get(query_id, {})
    ranked_rows = _annotated_ranked_rows(ctx, query_id)
    facets = _facets(query)
    facet_labels = _facet_labels(facets)
    dominant_facet_id = str(query.get('dominant_facet_id') or '')
    dominant_target = _dominant_target_chunks(facets, dominant_facet_id)

    _h1(lines, f'{heading_prefix} QUERY {query_id}')
    lines.append('')
    lines.append('Query text:')
    lines.append(str(query.get('query_text') or '').strip())
    lines.append('')
    _kv(
        lines,
        [
            ('query_type', query.get('query_type')),
            ('template_id', query.get('template_id')),
            ('split', query.get('split')),
            ('condition_id', query.get('condition_id')),
            ('condition_display', query.get('condition_display')),
            ('subgroup_a', query.get('subgroup_a_label')),
            ('subgroup_b', query.get('subgroup_b_label')),
            ('dominant_facet_id', dominant_facet_id),
            ('dominant_target_gold_chunks', dominant_target),
        ],
    )

    _h2(lines, 'Hidden Facet Plan')
    _table(
        lines,
        [
            {
                'facet_id': facet.get('facet_id'),
                'role': facet.get('cluster_role'),
                'cluster_id': facet.get('cluster_id'),
                'subgroup': facet.get('subgroup_label'),
                'axis': facet.get('axis'),
                'value_bin': facet.get('value_bin'),
                'target_gold_chunks': facet.get('target_gold_chunks'),
            }
            for facet in facets
        ],
        ['facet_id', 'role', 'cluster_id', 'subgroup', 'axis', 'value_bin', 'target_gold_chunks'],
    )

    calibration = ctx.calibration_by_id.get(query_id)
    if calibration:
        _h2(lines, 'Dominance Calibration')
        _kv(
            lines,
            [
                ('dominance_mode', calibration.get('dominance_mode')),
                ('previous_dominant_facet_id', calibration.get('previous_dominant_facet_id')),
                ('selected_dominant_facet_id', calibration.get('selected_dominant_facet_id')),
                ('selected_axis', calibration.get('selected_axis')),
                ('selected_subgroup_id', calibration.get('selected_subgroup_id')),
                ('selected_value_bin', calibration.get('selected_value_bin')),
                ('probe_chunks_per_facet', calibration.get('probe_chunks_per_facet')),
                ('selected_mean_query_sim', calibration.get('selected_mean_query_sim')),
                ('selected_p25_query_sim', calibration.get('selected_p25_query_sim')),
                (
                    'best_complement_p75_query_sim',
                    calibration.get('best_complement_p75_query_sim'),
                ),
                ('selected_probe_margin', calibration.get('selected_probe_margin')),
                ('calibration_min_probe_margin', calibration.get('calibration_min_probe_margin')),
                ('passes_calibration_margin', calibration.get('passes_calibration_margin')),
            ],
        )
        if ctx.detail == 'full':
            facet_stats = _json_loads(calibration.get('facet_stats_json'), [])
            if facet_stats:
                _table(
                    lines,
                    facet_stats,
                    [
                        'facet_id',
                        'subgroup_id',
                        'axis',
                        'value_bin',
                        'mean_query_sim',
                        'p25_query_sim',
                        'p75_query_sim',
                        'probe_margin_p25_gt_best_complement_p75',
                    ],
                )

    logical_form = _json_loads(plan.get('logical_form_json') or query.get('logical_form_json'), {})
    if logical_form and ctx.detail == 'full':
        lines.append('')
        lines.append('Logical form JSON:')
        lines.append(_json(logical_form, indent=2))

    _h2(lines, 'Gold Answer')
    if answer:
        facet_summaries = _json_loads(answer.get('facet_summaries_json'), {})
        if ctx.detail == 'full' or not facet_summaries:
            lines.append(str(answer.get('answer_text') or '').strip())
        if facet_summaries:
            lines.append('Facet summaries:')
            for facet_id, summary in facet_summaries.items():
                lines.append(f'- {facet_id} ({facet_labels.get(str(facet_id), "")}): {summary}')
    else:
        lines.append('No gold answer row found.')

    _h2(lines, 'Stored Geometry Rows')
    geometry = ctx.geometry_by_id.get(query_id)
    if geometry:
        _kv(lines, _geometry_items(geometry, compact=ctx.detail == 'compact'))
    else:
        lines.append('No geometry_stats row found.')
    plot_stats = ctx.plot_stats_by_id.get(query_id)
    if plot_stats:
        lines.append('')
        lines.append('Embedding geometry plot stats:')
        _kv(lines, _plot_stat_items(plot_stats, compact=ctx.detail == 'compact'))

    _h2(lines, 'Recomputed Top-K Coverage')
    coverage_rows = _topk_coverage_rows(ctx, query_id, ranked_rows, facets, dominant_facet_id)
    _table(
        lines,
        coverage_rows,
        [
            'k',
            'k_lt_dominant_target',
            'gold',
            'non_gold',
            'facets_hit',
            'n_facets_hit',
            'dominant_count',
            'top_facet',
            'facet_counts',
            'distractor_counts',
        ],
    )

    all_facet_rank = _rank_all_facets_covered(ranked_rows)
    lines.append('')
    _kv(
        lines,
        [
            ('rank_where_all_gold_facets_first_covered', all_facet_rank),
            ('dominant_target_gold_chunks', dominant_target),
            ('all_facets_covered_before_dominant_exhausted', _lt(all_facet_rank, dominant_target)),
        ],
    )

    _h2(lines, 'First Ranks And Query Similarity By Facet')
    _table(
        lines,
        _facet_rank_rows(ranked_rows, facets),
        [
            'facet_id',
            'label',
            'role',
            'target_gold_chunks',
            'first_rank',
            'first_five_ranks',
            'n_in_pool',
            'mean_query_sim',
            'max_query_sim',
            'min_query_sim',
        ],
    )

    _h2(lines, 'Pool Composition')
    composition_df = _composition_frame(ranked_rows)
    _render_composition(lines, composition_df, detail=ctx.detail)

    _h2(lines, 'Embedding Separability')
    _render_embedding_separability(lines, ctx, query_id, ranked_rows, facets)

    if ctx.detail == 'full':
        _h2(lines, 'Selection Snapshot')
        _render_selection_snapshot(lines, ctx, query_id, ranked_rows)
        trace_rows = ranked_rows
        chunk_rows = ranked_rows
        chunk_title = 'Full Candidate Chunk Texts In Query-Similarity Order'
    else:
        trace_rows = ranked_rows[: ctx.rank_trace_limit]
        if ctx.chunk_text_mode == 'all':
            chunk_rows = ranked_rows
            chunk_title = 'All Candidate Chunk Texts In Query-Similarity Order'
        else:
            chunk_rows = _representative_chunk_rows(
                ranked_rows,
                top_n=ctx.top_chunk_texts,
                per_label=ctx.representative_chunks_per_label,
                max_rows=ctx.max_representative_chunks,
            )
            chunk_title = 'Representative Chunk Texts'

    _h2(lines, f'Rank Trace Top {len(trace_rows)} Of {len(ranked_rows)}')
    _table(
        lines,
        [_rank_trace_row(row, facet_labels) for row in trace_rows],
        [
            'rank',
            'sim',
            'label',
            'cluster_role',
            'cluster_id',
            'condition',
            'subgroup',
            'axis',
            'value_bin',
            'chunk_id',
        ],
    )

    _h2(lines, chunk_title)
    if ctx.chunk_text_mode == 'all' or ctx.detail == 'full':
        lines.append(f'Showing all {len(chunk_rows)} candidate chunks with full text.')
    else:
        lines.append(
            f'Showing {len(chunk_rows)} representative chunks. '
            'Use --chunk-text-mode all for every candidate chunk.'
        )
    for row in chunk_rows:
        _render_chunk_text(
            lines,
            row,
            facet_labels,
            ctx.chunk_text_chars,
            compact=ctx.detail == 'compact',
        )


def _geometry_items(row: dict[str, Any], *, compact: bool) -> list[tuple[str, Any]]:
    if compact:
        preferred = [
            'passes_filter',
            'pool_size',
            'primary_topk_dominance_k',
            'n_facets_present',
            'topk_dominant_count',
            'planned_dominant_facet_id',
            'planned_topk_dominant_count',
            'planned_topk_dominant_fraction',
            'n_topk_retrieved_facets',
            'max_topk_retrieved_facets',
            'rank_where_all_facets_first_covered',
            'all_facets_covered_before_primary_k',
            'mean_in_facet_similarity',
            'mean_cross_facet_similarity',
            'in_minus_cross_similarity',
            'fail_weak_topk_dominance',
            'fail_too_many_topk_facets',
            'fail_weak_facet_separation',
            'n_near_miss_distractors_in_pool',
            'n_background_outliers_in_pool',
            'query_to_gold_mean',
            'query_to_background_outlier_mean',
            'background_outlier_first_rank',
            'jaccard_topk_facloc',
        ]
        return [(key, row.get(key)) for key in preferred if key in row]

    preferred = [
        'passes_filter',
        'pool_scope',
        'pool_size',
        'topk_dominance_k',
        'primary_topk_dominance_k',
        'n_facets',
        'n_facets_present',
        'all_facets_present',
        'topk_dominant_count',
        'planned_dominant_facet_id',
        'planned_topk_dominant_count',
        'planned_topk_dominant_fraction',
        'n_topk_retrieved_facets',
        'max_topk_retrieved_facets',
        'rank_where_all_facets_first_covered',
        'all_facets_covered_before_primary_k',
        'n_distractors_in_pool',
        'n_near_miss_distractors_in_pool',
        'mean_in_facet_similarity',
        'mean_cross_facet_similarity',
        'in_minus_cross_similarity',
        'fail_missing_facet',
        'fail_weak_topk_dominance',
        'fail_too_many_topk_facets',
        'fail_weak_facet_separation',
        'fail_too_few_near_miss_distractors',
        'fail_missing_or_malformed_background_outlier',
        'facets_present_json',
        'topk_retrieved_facets_json',
        'n_background_outliers_in_pool',
        'n_background_outlier_clusters_in_pool',
        'background_outlier_complete',
        'background_outlier_mean_in_cluster_similarity',
        'query_to_background_outlier_mean',
        'query_to_gold_mean',
        'gold_minus_background_outlier_similarity_margin',
        'background_outlier_first_rank',
        'background_outlier_median_rank',
        'fac_topk',
        'fac_facloc',
        'avg_cos_topk',
        'avg_cos_facloc',
        'jaccard_topk_facloc',
    ]
    return [(key, row.get(key)) for key in preferred if key in row]


def _plot_stat_items(row: dict[str, Any], *, compact: bool) -> list[tuple[str, Any]]:
    if compact:
        preferred = [
            'selection_group',
            'plot_k',
            'gold_silhouette_cosine',
            'hdbscan_ari_hidden',
            'hdbscan_nmi_hidden',
            'top_k_n_facets_selected',
            'top_k_gold_precision',
            'top_k_distractor_rate',
            'top_k_dominant_fraction',
            'fac_loc_n_facets_selected',
            'fac_loc_gold_precision',
            'fac_loc_distractor_rate',
            'fac_loc_dominant_fraction',
        ]
        return [(key, row.get(key)) for key in preferred if key in row]

    preferred = [
        'selection_group',
        'plot_k',
        'pool_size',
        'n_hidden_labels',
        'n_gold_points',
        'n_distractor_points',
        'gold_silhouette_cosine',
        'mean_in_facet_similarity',
        'mean_cross_facet_similarity',
        'in_minus_cross_similarity',
        'query_to_gold_mean',
        'query_to_distractor_mean',
        'hdbscan_n_clusters',
        'hdbscan_noise_rate',
        'hdbscan_ari_hidden',
        'hdbscan_nmi_hidden',
        'top_k_n_facets_selected',
        'top_k_gold_precision',
        'top_k_distractor_rate',
        'top_k_dominant_fraction',
        'mmr_n_facets_selected',
        'mmr_gold_precision',
        'mmr_distractor_rate',
        'mmr_dominant_fraction',
        'fac_loc_n_facets_selected',
        'fac_loc_gold_precision',
        'fac_loc_distractor_rate',
        'fac_loc_dominant_fraction',
    ]
    return [(key, row.get(key)) for key in preferred if key in row]


def _topk_coverage_rows(
    ctx: _RenderContext,
    query_id: str,
    ranked_rows: list[dict[str, Any]],
    facets: list[dict[str, Any]],
    dominant_facet_id: str,
) -> list[dict[str, Any]]:
    del query_id
    top_ks = _diagnostic_top_ks(ctx, len(ranked_rows))
    dominant_target = _dominant_target_chunks(facets, dominant_facet_id)
    rows = []
    for k in top_ks:
        selected = ranked_rows[:k]
        facet_counts = Counter(
            str(row['facet_id']) for row in selected if row.get('is_gold') and row.get('facet_id')
        )
        distractor_counts = Counter(
            str(row['distractor_type'] or row['cluster_role'] or row['label_id'])
            for row in selected
            if not row.get('is_gold')
        )
        top_facet, _top_facet_count = _counter_top(facet_counts)
        rows.append({
            'k': k,
            'k_lt_dominant_target': _lt(k, dominant_target),
            'gold': sum(1 for row in selected if row.get('is_gold')),
            'non_gold': sum(1 for row in selected if not row.get('is_gold')),
            'facets_hit': ', '.join(sorted(facet_counts)),
            'n_facets_hit': len(facet_counts),
            'dominant_count': facet_counts.get(dominant_facet_id, 0),
            'top_facet': top_facet,
            'facet_counts': _json(dict(sorted(facet_counts.items()))),
            'distractor_counts': _json(dict(sorted(distractor_counts.items()))),
        })
    return rows


def _facet_rank_rows(
    ranked_rows: list[dict[str, Any]], facets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    by_facet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        if row.get('is_gold') and row.get('facet_id'):
            by_facet[str(row['facet_id'])].append(row)
    for facet in facets:
        facet_id = str(facet.get('facet_id') or '')
        facet_rows = by_facet.get(facet_id, [])
        sims = [float(row['sim_to_query']) for row in facet_rows]
        ranks = [int(row['rank']) for row in facet_rows]
        rows.append({
            'facet_id': facet_id,
            'label': _facet_label(facet),
            'role': facet.get('cluster_role'),
            'target_gold_chunks': facet.get('target_gold_chunks'),
            'first_rank': min(ranks) if ranks else None,
            'first_five_ranks': ', '.join(str(rank) for rank in sorted(ranks)[:5]),
            'n_in_pool': len(facet_rows),
            'mean_query_sim': float(np.mean(sims)) if sims else None,
            'max_query_sim': max(sims) if sims else None,
            'min_query_sim': min(sims) if sims else None,
        })
    return rows


def _render_composition(lines: list[str], composition_df: pl.DataFrame, *, detail: str) -> None:
    if composition_df.is_empty():
        lines.append('No ranked rows to summarize.')
        return

    groups = [
        ('By cluster role', ['cluster_role']),
        ('By label id', ['label_id']),
    ]
    if detail == 'full':
        groups.extend([
            ('By facet/target/role', ['facet_id', 'target_facet_id', 'cluster_role']),
            (
                'By condition/subgroup/axis/value',
                ['condition_display', 'subgroup_label', 'axis', 'value_bin'],
            ),
            (
                'By text generation and soft warnings',
                ['text_generation_source', 'validation_soft_warning_count'],
            ),
        ])

    for title, keys in groups:
        lines.append('')
        lines.append(f'{title}:')
        group = (
            composition_df
            .group_by(keys)
            .agg(
                pl.len().alias('n'),
                pl.col('rank').min().alias('first_rank'),
                pl.col('rank').median().alias('median_rank'),
                pl.col('sim_to_query').mean().alias('mean_query_sim'),
                pl.col('is_gold').cast(pl.Int64).sum().alias('gold_n'),
            )
            .sort(['first_rank', 'n'], descending=[False, True])
        )
        _table(
            lines,
            group.to_dicts(),
            [*keys, 'n', 'gold_n', 'first_rank', 'median_rank', 'mean_query_sim'],
        )


def _render_embedding_separability(
    lines: list[str],
    ctx: _RenderContext,
    query_id: str,
    ranked_rows: list[dict[str, Any]],
    facets: list[dict[str, Any]],
) -> None:
    if query_id not in ctx.query_id_to_idx:
        lines.append('No query embedding found.')
        return

    gold_by_facet: dict[str, list[str]] = defaultdict(list)
    for row in ranked_rows:
        if row.get('is_gold') and row.get('facet_id') and row['chunk_id'] in ctx.chunk_id_to_idx:
            gold_by_facet[str(row['facet_id'])].append(str(row['chunk_id']))

    separation = _gold_similarity_separation(ctx, gold_by_facet)
    _kv(
        lines,
        [
            ('recomputed_mean_in_facet_similarity', separation['mean_in_facet_similarity']),
            ('recomputed_mean_cross_facet_similarity', separation['mean_cross_facet_similarity']),
            ('recomputed_in_minus_cross_similarity', separation['in_minus_cross_similarity']),
        ],
    )

    query_vector = ctx.query_vectors[ctx.query_id_to_idx[query_id]]
    per_facet_rows = []
    centroids: dict[str, NDArray[np.float32]] = {}
    for facet in facets:
        facet_id = str(facet.get('facet_id') or '')
        ids = gold_by_facet.get(facet_id, [])
        vectors = ctx.chunk_vectors[[ctx.chunk_id_to_idx[chunk_id] for chunk_id in ids]]
        if len(vectors):
            sims = vectors @ query_vector
            centroid = np.asarray(vectors.mean(axis=0), dtype=np.float32)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
                centroids[facet_id] = centroid
        else:
            sims = np.asarray([], dtype=np.float32)
        per_facet_rows.append({
            'facet_id': facet_id,
            'label': _facet_label(facet),
            'role': facet.get('cluster_role'),
            'n': len(ids),
            'mean_query_sim': float(sims.mean()) if len(sims) else None,
            'max_query_sim': float(sims.max()) if len(sims) else None,
            'min_query_sim': float(sims.min()) if len(sims) else None,
            'query_to_centroid': float(centroids[facet_id] @ query_vector)
            if facet_id in centroids
            else None,
        })
    lines.append('')
    lines.append('Per-facet query similarity:')
    _table(
        lines,
        per_facet_rows,
        [
            'facet_id',
            'label',
            'role',
            'n',
            'mean_query_sim',
            'max_query_sim',
            'min_query_sim',
            'query_to_centroid',
        ],
    )

    if centroids:
        lines.append('')
        lines.append('Facet centroid cosine matrix:')
        facet_ids = [
            str(facet.get('facet_id') or '')
            for facet in facets
            if str(facet.get('facet_id') or '') in centroids
        ]
        matrix_rows = []
        for left in facet_ids:
            row = {'facet_id': left}
            for right in facet_ids:
                row[right] = float(centroids[left] @ centroids[right])
            matrix_rows.append(row)
        _table(lines, matrix_rows, ['facet_id', *facet_ids])


def _render_selection_snapshot(
    lines: list[str],
    ctx: _RenderContext,
    query_id: str,
    ranked_rows: list[dict[str, Any]],
) -> None:
    if not ranked_rows:
        lines.append('No ranked rows.')
        return
    candidate_vectors = ctx.chunk_vectors[
        [ctx.chunk_id_to_idx[str(row['chunk_id'])] for row in ranked_rows]
    ]
    sim_to_query = np.asarray([float(row['sim_to_query']) for row in ranked_rows], dtype=np.float32)
    sim_matrix = candidate_vectors @ candidate_vectors.T
    snapshot_ks = [
        value
        for value in _dedupe([
            *ctx.cfg.retrieval.k_values,
            ctx.cfg.embedding_geometry.plot_k,
            ctx.cfg.geometry.topk_dominance_k,
        ])
        if value <= len(ranked_rows)
    ]
    lambdas = _snapshot_lambdas(ctx.cfg)
    rows = []
    for k in snapshot_ks:
        topk = select_indices('top_k', sim_to_query, sim_matrix, k=k, lam=None)
        rows.append(_selection_summary_row('top_k', None, k, topk, topk, ranked_rows, sim_to_query))

        for strategy in ctx.cfg.retrieval.strategies.difference({'top_k'}):
            for lam in lambdas:
                selected = select_indices(
                    strategy,
                    sim_to_query,
                    sim_matrix,
                    k=k,
                    lam=lam,
                    mmr_window=ctx.cfg.retrieval.mmr_window,
                )
                rows.append(
                    _selection_summary_row(
                        strategy,
                        lam,
                        k,
                        selected,
                        topk,
                        ranked_rows,
                        sim_to_query,
                    )
                )
    _table(
        lines,
        rows,
        [
            'strategy',
            'lam',
            'k',
            'gold',
            'non_gold',
            'n_facets',
            'facet_counts',
            'distractor_counts',
            'top_facet',
            'max_facet_fraction',
            'avg_query_sim',
            'jaccard_vs_topk',
        ],
    )

    if ctx.eval_results.height > 0 and 'query_id' in ctx.eval_results.columns:
        eval_rows = (
            ctx.eval_results
            .filter(pl.col('query_id') == query_id)
            .select([
                col
                for col in [
                    'strategy',
                    'lam',
                    'k',
                    'gold_precision',
                    'facet_coverage',
                    'weighted_facet_coverage',
                    'distractor_rate',
                    'dominant_facet_rate',
                    'max_facet_concentration',
                    'redundant_gold_rate',
                    'avg_cos',
                    'jaccard_vs_topk',
                ]
                if col in ctx.eval_results.columns
            ])
            .sort(['k', 'strategy', 'lam'])
        )
        if eval_rows.height:
            lines.append('')
            lines.append('Stored evaluation_results rows for this query:')
            _table(lines, eval_rows.to_dicts(), eval_rows.columns)


def _selection_summary_row(
    strategy: str,
    lam: float | None,
    k: int,
    selected: NDArray[np.intp],
    topk: NDArray[np.intp],
    ranked_rows: list[dict[str, Any]],
    sim_to_query: NDArray[np.float32],
) -> dict[str, Any]:
    selected_rows = [ranked_rows[int(idx)] for idx in selected]
    facet_counts = Counter(
        str(row['facet_id']) for row in selected_rows if row.get('is_gold') and row.get('facet_id')
    )
    distractor_counts = Counter(
        str(row['distractor_type'] or row['cluster_role'] or row['label_id'])
        for row in selected_rows
        if not row.get('is_gold')
    )
    top_facet, top_facet_count = _counter_top(facet_counts)
    return {
        'strategy': strategy,
        'lam': lam,
        'k': k,
        'gold': sum(1 for row in selected_rows if row.get('is_gold')),
        'non_gold': sum(1 for row in selected_rows if not row.get('is_gold')),
        'n_facets': len(facet_counts),
        'facet_counts': _json(dict(sorted(facet_counts.items()))),
        'distractor_counts': _json(dict(sorted(distractor_counts.items()))),
        'top_facet': top_facet,
        'max_facet_fraction': top_facet_count / len(selected_rows) if selected_rows else 0.0,
        'avg_query_sim': float(sim_to_query[selected].mean()) if len(selected) else 0.0,
        'jaccard_vs_topk': _jaccard(selected, topk),
    }


def _render_chunk_text(
    lines: list[str],
    row: dict[str, Any],
    facet_labels: dict[str, str],
    chunk_text_chars: int,
    *,
    compact: bool,
) -> None:
    lines.append('')
    lines.append(
        f'--- rank {row["rank"]:03d} | {row["chunk_id"]} | sim={_fmt(row["sim_to_query"])} ---'
    )
    if compact:
        meta_parts = [
            ('label', _label_for_row(row, facet_labels)),
            ('gold', row.get('is_gold')),
            ('role', row.get('cluster_role')),
            ('distractor', row.get('distractor_type')),
            ('condition', row.get('condition_display')),
            ('subgroup', row.get('subgroup_label')),
            ('axis', row.get('axis')),
            ('value_bin', row.get('value_bin')),
            ('days', row.get('duration_days')),
            ('treatment', row.get('treatment')),
            ('rehab', row.get('rehab_outcome')),
            ('warnings', row.get('validation_soft_warning_count')),
        ]
        lines.append(
            'meta: '
            + '; '.join(
                f'{key}={_fmt(value)}' for key, value in meta_parts if value not in [None, '']
            )
        )
        text = str(row.get('text') or '').strip()
        if chunk_text_chars > 0 and len(text) > chunk_text_chars:
            text = text[:chunk_text_chars].rstrip() + ' ... [truncated]'
        lines.append('text: ' + text)
        return

    metadata = [
        ('label', _label_for_row(row, facet_labels)),
        ('is_gold', row.get('is_gold')),
        ('cluster_role', row.get('cluster_role')),
        ('distractor_type', row.get('distractor_type')),
        ('condition', row.get('condition_display')),
        ('subgroup', row.get('subgroup_label')),
        ('axis', row.get('axis')),
        ('value_bin', row.get('value_bin')),
        ('duration_days', row.get('duration_days')),
        ('treatment', row.get('treatment')),
        ('rehab_outcome', row.get('rehab_outcome')),
        ('validation_soft_warning_count', row.get('validation_soft_warning_count')),
    ]
    if not compact:
        metadata.extend([
            ('facet_id', row.get('facet_id')),
            ('target_facet_id', row.get('target_facet_id')),
            ('cluster_id', row.get('cluster_id')),
            ('patient_age', row.get('patient_age')),
            ('patient_sex', row.get('patient_sex')),
            ('note_style', row.get('note_style')),
            ('approx_words', row.get('approx_words')),
            ('text_generation_source', row.get('text_generation_source')),
            ('validation_soft_warnings_json', row.get('validation_soft_warnings_json')),
            ('fact_id', row.get('fact_id')),
            ('must_mention', row.get('must_mention')),
            ('must_not_mention', row.get('must_not_mention')),
        ])
    _kv(lines, metadata)
    text = str(row.get('text') or '').strip()
    if chunk_text_chars > 0 and len(text) > chunk_text_chars:
        text = text[:chunk_text_chars].rstrip() + ' ... [truncated]'
    lines.append('text:')
    lines.append(text)


def _representative_chunk_rows(
    ranked_rows: list[dict[str, Any]],
    *,
    top_n: int,
    per_label: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    for row in ranked_rows[: max(0, top_n)]:
        selected[str(row['chunk_id'])] = row

    seen_by_label: Counter[str] = Counter()
    for row in ranked_rows:
        label_id = str(row.get('label_id') or row.get('cluster_role') or '')
        if seen_by_label[label_id] >= per_label:
            continue
        selected[str(row['chunk_id'])] = row
        seen_by_label[label_id] += 1
        if len(selected) >= max_rows:
            break

    return sorted(selected.values(), key=lambda row: int(row['rank']))[:max_rows]


def _rank_trace_row(row: dict[str, Any], facet_labels: dict[str, str]) -> dict[str, Any]:
    return {
        'rank': row.get('rank'),
        'sim': row.get('sim_to_query'),
        'label': _label_for_row(row, facet_labels),
        'cluster_role': row.get('cluster_role'),
        'cluster_id': row.get('cluster_id'),
        'condition': row.get('condition_display'),
        'subgroup': row.get('subgroup_label'),
        'axis': row.get('axis'),
        'value_bin': row.get('value_bin'),
        'chunk_id': row.get('chunk_id'),
    }


def _annotated_ranked_rows(ctx: _RenderContext, query_id: str) -> list[dict[str, Any]]:
    query = ctx.queries_by_id[query_id]
    rows = []
    for rank_row in ctx.ranked_by_query.get(query_id, []):
        chunk_id = str(rank_row['chunk_id'])
        qrel = ctx.qrels_by_query_chunk.get((query_id, chunk_id), {})
        membership = ctx.memberships_by_query_chunk.get((query_id, chunk_id), {})
        fact_id = str(qrel.get('fact_id') or membership.get('fact_id') or '')
        fact = ctx.facts_by_fact_id.get(fact_id, {})
        chunk = ctx.chunk_by_id.get(chunk_id, {})
        row = {
            **chunk,
            **rank_row,
            'fact_id': fact_id or None,
            'facet_id': qrel.get('facet_id') or membership.get('facet_id'),
            'target_facet_id': qrel.get('target_facet_id') or membership.get('target_facet_id'),
            'cluster_id': qrel.get('cluster_id') or membership.get('cluster_id'),
            'cluster_role': qrel.get('cluster_role') or membership.get('cluster_role'),
            'is_gold': bool(qrel.get('is_gold') or membership.get('is_gold')),
            'distractor_type': qrel.get('distractor_type') or membership.get('distractor_type'),
            'relevance_grade': qrel.get('relevance_grade'),
            'support_type': qrel.get('support_type'),
            'must_mention': fact.get('must_mention'),
            'must_not_mention': fact.get('must_not_mention'),
        }
        row['label_id'] = _label_id(row, query)
        rows.append(row)
    return rows


def _label_id(row: dict[str, Any], query: dict[str, Any]) -> str:
    if row.get('is_gold') and row.get('facet_id'):
        return str(row['facet_id'])
    if row.get('distractor_type'):
        return str(row['distractor_type'])
    if row.get('cluster_role'):
        return str(row['cluster_role'])
    if row.get('condition_id') == query.get('condition_id'):
        return 'off_query_same_condition'
    return 'off_query_wrong_condition'


def _label_for_row(row: dict[str, Any], facet_labels: dict[str, str]) -> str:
    label_id = str(row.get('label_id') or '')
    if label_id in facet_labels:
        return f'{label_id}: {facet_labels[label_id]}'
    return label_id.replace('_', ' ')


def _composition_frame(ranked_rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not ranked_rows:
        return pl.DataFrame()
    keys = [
        'rank',
        'sim_to_query',
        'is_gold',
        'label_id',
        'facet_id',
        'target_facet_id',
        'cluster_id',
        'cluster_role',
        'distractor_type',
        'condition_display',
        'subgroup_label',
        'axis',
        'value_bin',
        'text_generation_source',
        'validation_soft_warning_count',
    ]
    return pl.DataFrame([{key: row.get(key) for key in keys} for row in ranked_rows])


def _candidate_ids_by_query(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    queries: pl.DataFrame,
    query_ids: list[str],
    memberships: pl.DataFrame,
    all_chunk_ids: list[str],
) -> dict[str, list[str]]:
    if cfg.retrieval.pool_scope == 'full_corpus':
        return {query_id: all_chunk_ids for query_id in query_ids}

    if cfg.retrieval.pool_scope == 'query_local':
        result: dict[str, list[str]] = {query_id: [] for query_id in query_ids}
        if memberships.is_empty() or not {'query_id', 'chunk_id'}.issubset(memberships.columns):
            return result
        for row in memberships.select('query_id', 'chunk_id').iter_rows(named=True):
            result.setdefault(str(row['query_id']), []).append(str(row['chunk_id']))
        return {query_id: _dedupe(result.get(query_id, [])) for query_id in query_ids}

    if cfg.retrieval.pool_scope != 'same_condition':
        raise ValueError(f'Unsupported pool_scope: {cfg.retrieval.pool_scope}')

    query_condition = {
        str(row['query_id']): str(row['condition_id'])
        for row in queries.select('query_id', 'condition_id').iter_rows(named=True)
    }
    condition_ids = sorted(set(query_condition.values()))
    chunks = _collect_chunk_documents_by_condition(
        paths, condition_ids, ['chunk_id', 'condition_id']
    )
    ids_by_condition: dict[str, list[str]] = defaultdict(list)
    for row in chunks.iter_rows(named=True):
        ids_by_condition[str(row['condition_id'])].append(str(row['chunk_id']))
    return {
        query_id: ids_by_condition.get(query_condition.get(query_id, ''), [])
        for query_id in query_ids
    }


def _rank_candidate_pools(
    *,
    queries: pl.DataFrame,
    query_ids: list[str],
    candidate_ids_by_query: dict[str, list[str]],
    chunk_id_to_idx: dict[str, int],
    query_id_to_idx: dict[str, int],
    chunk_vectors: NDArray[np.float32],
    query_vectors: NDArray[np.float32],
    pool_n: int,
) -> dict[str, list[dict[str, Any]]]:
    query_rows = _rows_by_key(queries, 'query_id')
    idx_to_chunk_id = {idx: chunk_id for chunk_id, idx in chunk_id_to_idx.items()}
    ranked: dict[str, list[dict[str, Any]]] = {}
    for query_id in query_ids:
        if query_id not in query_rows or query_id not in query_id_to_idx:
            ranked[query_id] = []
            continue
        candidate_indices = np.asarray(
            [
                chunk_id_to_idx[chunk_id]
                for chunk_id in candidate_ids_by_query.get(query_id, [])
                if chunk_id in chunk_id_to_idx
            ],
            dtype=np.intp,
        )
        if len(candidate_indices) == 0:
            ranked[query_id] = []
            continue
        sims = chunk_vectors[candidate_indices] @ query_vectors[query_id_to_idx[query_id]]
        order = np.argsort(sims)[::-1][: min(pool_n, len(sims))]
        rows = []
        for rank, local_idx in enumerate(order, start=1):
            chunk_idx = int(candidate_indices[int(local_idx)])
            rows.append({
                'rank': rank,
                'chunk_id': str(idx_to_chunk_id[chunk_idx]),
                'chunk_idx': chunk_idx,
                'sim_to_query': float(sims[int(local_idx)]),
                'candidate_pool_size_before_topn': len(candidate_indices),
            })
        ranked[query_id] = rows
    return ranked


def _collect_query_table(
    paths: MedicalDatasetGenPaths,
    table: str,
    query_ids: list[str],
    columns: list[str] | None = None,
) -> pl.DataFrame:
    path = paths.experiment_dir / f'{table}.parquet'
    if not path.exists():
        return pl.DataFrame()
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    if 'query_id' in schema:
        lf = lf.filter(pl.col('query_id').is_in(query_ids))
    selected = _available_columns(schema, columns)
    if selected:
        lf = lf.select(selected)
    return lf.collect()


def _collect_chunk_documents(paths: MedicalDatasetGenPaths, chunk_ids: set[str]) -> pl.DataFrame:
    path = paths.table_path('chunk_documents')
    if not path.exists() or not chunk_ids:
        return pl.DataFrame()
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    selected = _available_columns(schema, _CHUNK_COLUMNS)
    return lf.filter(pl.col('chunk_id').is_in(sorted(chunk_ids))).select(selected).collect()


def _collect_chunk_documents_by_condition(
    paths: MedicalDatasetGenPaths, condition_ids: list[str], columns: list[str]
) -> pl.DataFrame:
    path = paths.table_path('chunk_documents')
    if not path.exists() or not condition_ids:
        return pl.DataFrame()
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    selected = _available_columns(schema, columns)
    return lf.filter(pl.col('condition_id').is_in(condition_ids)).select(selected).collect()


def _available_columns(schema: pl.Schema, columns: list[str] | None) -> list[str]:
    if columns is None:
        return list(schema.names())
    return [column for column in columns if column in schema]


def _rows_by_key(df: pl.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    if df.is_empty() or key not in df.columns:
        return {}
    return {str(row[key]): row for row in df.iter_rows(named=True)}


def _rows_by_pair(df: pl.DataFrame, left: str, right: str) -> dict[tuple[str, str], dict[str, Any]]:
    if df.is_empty() or left not in df.columns or right not in df.columns:
        return {}
    return {(str(row[left]), str(row[right])): row for row in df.iter_rows(named=True)}


def _facets(query: dict[str, Any]) -> list[dict[str, Any]]:
    raw = query.get('facets_json')
    facets = _json_loads(raw, [])
    return facets if isinstance(facets, list) else []


def _facet_labels(facets: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(facet.get('facet_id')): _facet_label(facet) for facet in facets if facet.get('facet_id')
    }


def _facet_label(facet: dict[str, Any]) -> str:
    subgroup = str(facet.get('subgroup_label') or '')
    axis = str(facet.get('axis') or '')
    value_bin = str(facet.get('value_bin') or '')
    return f'{subgroup} / {axis} / {value_bin}'


def _dominant_target_chunks(facets: list[dict[str, Any]], dominant_facet_id: str) -> int | None:
    for facet in facets:
        if facet.get('facet_id') == dominant_facet_id:
            value = facet.get('target_gold_chunks')
            return int(value) if value is not None else None
    return None


def _diagnostic_top_ks(ctx: _RenderContext, pool_size: int) -> list[int]:
    values = [
        *ctx.cfg.retrieval.k_values,
        ctx.cfg.embedding_geometry.plot_k,
        ctx.cfg.geometry.topk_dominance_k,
        *ctx.extra_top_ks,
    ]
    return [value for value in sorted(set(values)) if 0 < value <= pool_size]


def _rank_all_facets_covered(ranked_rows: list[dict[str, Any]]) -> int | None:
    all_facets = {
        str(row['facet_id']) for row in ranked_rows if row.get('is_gold') and row.get('facet_id')
    }
    seen: set[str] = set()
    for row in ranked_rows:
        if row.get('is_gold') and row.get('facet_id'):
            seen.add(str(row['facet_id']))
        if all_facets and seen == all_facets:
            return int(row['rank'])
    return None


def _gold_similarity_separation(
    ctx: _RenderContext, gold_by_facet: dict[str, list[str]]
) -> dict[str, float]:
    chunk_ids = []
    labels = []
    for facet_id, ids in gold_by_facet.items():
        for chunk_id in ids:
            if chunk_id in ctx.chunk_id_to_idx:
                chunk_ids.append(chunk_id)
                labels.append(facet_id)
    if len(chunk_ids) < 2:
        return {
            'mean_in_facet_similarity': 0.0,
            'mean_cross_facet_similarity': 0.0,
            'in_minus_cross_similarity': 0.0,
        }
    vectors = ctx.chunk_vectors[[ctx.chunk_id_to_idx[chunk_id] for chunk_id in chunk_ids]]
    sim = vectors @ vectors.T
    labels_arr = np.asarray(labels)
    same = labels_arr[:, None] == labels_arr[None, :]
    not_self = ~np.eye(len(labels_arr), dtype=bool)
    in_vals = sim[same & not_self]
    cross_vals = sim[~same & not_self]
    in_sim = float(in_vals.mean()) if len(in_vals) else 0.0
    cross_sim = float(cross_vals.mean()) if len(cross_vals) else 0.0
    return {
        'mean_in_facet_similarity': in_sim,
        'mean_cross_facet_similarity': cross_sim,
        'in_minus_cross_similarity': in_sim - cross_sim,
    }


def _counter_top(counter: Counter[str]) -> tuple[str | None, int]:
    if not counter:
        return None, 0
    key, count = counter.most_common(1)[0]
    return key, count


def _snapshot_lambdas(cfg: ExperimentCfg) -> list[float]:
    values = sorted({float(value) for value in cfg.retrieval.lambda_values})
    if len(values) <= 3:
        return values
    return _dedupe([values[0], values[len(values) // 2], values[-1]])


def _jaccard(left: NDArray[np.intp], right: NDArray[np.intp]) -> float:
    left_set = {int(value) for value in left}
    right_set = {int(value) for value in right}
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _requested_query_ids(args: argparse.Namespace, paths: MedicalDatasetGenPaths) -> list[str]:
    query_ids = []
    for value in args.query_id or []:
        query_ids.extend(_split_query_ids(value))
    query_ids.extend(_split_query_ids(args.query_ids or ''))
    if args.query_id_file:
        query_ids.extend(_query_ids_from_file(args.query_id_file))
    if args.from_plot_group:
        query_ids.extend(_query_ids_from_plot_group(paths, args.from_plot_group))
    return _dedupe(query_ids)


def _comparison_query_ids(
    args: argparse.Namespace,
    paths: MedicalDatasetGenPaths,
    target_query_ids: list[str],
) -> list[str]:
    if args.no_comparison:
        return []

    query_ids = []
    for value in args.comparison_query_id or []:
        query_ids.extend(_split_query_ids(value))
    query_ids.extend(_split_query_ids(args.comparison_query_ids or ''))

    if not query_ids and args.from_plot_group == 'bad' and args.comparison_plot_group:
        query_ids.extend(_query_ids_from_plot_group(paths, args.comparison_plot_group))

    target_set = set(target_query_ids)
    query_ids = [query_id for query_id in _dedupe(query_ids) if query_id not in target_set]
    return query_ids[: max(0, args.comparison_n)]


def _query_ids_from_file(path: Path) -> list[str]:
    values = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        values.extend(_split_query_ids(line))
    return values


def _query_ids_from_plot_group(paths: MedicalDatasetGenPaths, group: str) -> list[str]:
    root = paths.figures_dir / 'embedding_geometry' / group
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _parse_top_ks(args: argparse.Namespace) -> list[int]:
    values = []
    for value in args.top_k or []:
        values.append(int(value))
    for value in _split_query_ids(args.top_ks or ''):
        values.append(int(value))
    return sorted({value for value in values if value > 0})


def _split_query_ids(value: str) -> list[str]:
    return [part.strip() for part in value.replace(',', ' ').split() if part.strip()]


def _default_out_path(
    paths: MedicalDatasetGenPaths, query_ids: list[str], from_plot_group: str | None
) -> Path:
    if from_plot_group:
        stem = from_plot_group
    elif len(query_ids) <= 4:
        stem = '_'.join(query_ids)
    else:
        stem = f'{query_ids[0]}_and_{len(query_ids) - 1}_more'
    return paths.experiment_dir / '_diagnostics' / f'txt_query_diagnostics_{stem}.txt'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Dump cohesive text diagnostics for generated medical RAG queries.'
    )
    parser.add_argument('--exp', default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    parser.add_argument('--query-id', action='append', default=[])
    parser.add_argument('--query-ids', default='')
    parser.add_argument('--query-id-file', type=Path, default=None)
    parser.add_argument(
        '--from-plot-group',
        choices=['good', 'mid', 'bad', 'manual'],
        default=None,
        help='Add query ids from _figures/embedding_geometry/<group> subdirectories.',
    )
    parser.add_argument(
        '--comparison-query-id',
        action='append',
        default=[],
        help='Explicit query id to render before target queries as a comparison case.',
    )
    parser.add_argument(
        '--comparison-query-ids',
        default='',
        help='Comma/space-separated comparison query ids rendered before target queries.',
    )
    parser.add_argument(
        '--comparison-plot-group',
        choices=['good', 'mid', 'bad', 'manual'],
        default='good',
        help='Plot group to sample comparison queries from when --from-plot-group bad is used.',
    )
    parser.add_argument(
        '--comparison-n',
        type=int,
        default=1,
        help='Number of comparison queries to prepend.',
    )
    parser.add_argument(
        '--no-comparison',
        action='store_true',
        help='Do not prepend a good comparison query.',
    )
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--pool-n', type=int, default=None)
    parser.add_argument('--top-k', action='append', type=int, default=[])
    parser.add_argument('--top-ks', default='')
    parser.add_argument(
        '--detail',
        choices=['compact', 'full'],
        default='compact',
        help='compact keeps only LLM-sized diagnostic evidence; full dumps every candidate row.',
    )
    parser.add_argument(
        '--chunk-text-mode',
        choices=['all', 'representative'],
        default='all',
        help='In compact mode, include all chunk texts or only representative examples.',
    )
    parser.add_argument(
        '--rank-trace-limit',
        type=int,
        default=8,
        help='Number of ranked candidates to show in compact mode.',
    )
    parser.add_argument(
        '--top-chunk-texts',
        type=int,
        default=2,
        help='Always include text for the first N ranked chunks in compact mode.',
    )
    parser.add_argument(
        '--representative-chunks-per-label',
        type=int,
        default=1,
        help='Include up to this many chunks per facet/distractor label in compact mode.',
    )
    parser.add_argument(
        '--max-representative-chunks',
        type=int,
        default=4,
        help='Maximum chunk text blocks per query in compact mode.',
    )
    parser.add_argument(
        '--chunk-text-chars',
        type=int,
        default=0,
        help='Truncate chunk text to this many characters. Use 0 for full text.',
    )
    return parser.parse_args()


def _h1(lines: list[str], text: str) -> None:
    lines.append('=' * 5)
    lines.append(text)
    lines.append('=' * 5)


def _h2(lines: list[str], text: str) -> None:
    lines.append('')
    lines.append('-' * 5)
    lines.append(text)
    lines.append('-' * 5)


def _kv(lines: list[str], items: list[tuple[str, Any]]) -> None:
    for key, value in items:
        if value is None or value == '':
            continue
        lines.append(f'{key}: {_fmt(value)}')


def _table(lines: list[str], rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        lines.append('(none)')
        return
    lines.append('| ' + ' | '.join(columns) + ' |')
    lines.append('| ' + ' | '.join('---' for _ in columns) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(_cell(row.get(column)) for column in columns) + ' |')


def _cell(value: Any) -> str:
    return _fmt(value).replace('\n', '<br>').replace('|', '\\|')


def _fmt(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        if math.isnan(value):
            return 'nan'
        return f'{value:.4f}'
    if isinstance(value, np.floating):
        return _fmt(float(value))
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, list | tuple | dict):
        return _json(value)
    return str(value)


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == '':
        return default
    if isinstance(value, list | dict):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _json(value: Any, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _dedupe(values: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _lt(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return left < right


if __name__ == '__main__':
    main()
