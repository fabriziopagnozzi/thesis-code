from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.embedding_geometry.plots import (
    plot_cluster_quality_overview,
    plot_full_strategy_selection_overlay,
    plot_query_overview_4panel,
    plot_strategy_overlay,
)
from experiments.medical_dataset_gen.evaluation.retrieval_utils import select_indices
from experiments.mimic.global_configs import MimicPaths, get_pool_analysis_path, read_parquet
from experiments.mimic.pool_analysis.schemas_pool_analysis import PoolAnalysisCfg
from experiments.mimic.queries.schemas_queries import QueryModifier, QueryRow
from experiments.mimic.utils.chunk_pools import (
    ChunkPool,
    ChunkPoolBuilder,
)

_DEFAULT_LAMBDA_VALUES = [0.3, 0.5, 0.7]
_DEFAULT_STRATEGIES = ['top_k', 'mmr', 'fac_loc']
_SOFT_DISTRACTOR_LABEL = 'soft distractor: same condition, wrong subgroup'
_BACKGROUND_LABEL = 'background outlier: clinical cluster'


def render_embedding_geometry_figures(
    cfg: PoolAnalysisCfg,
    stats_df: pl.DataFrame,
    points_df: pl.DataFrame,
    *,
    pool_builder: ChunkPoolBuilder | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if stats_df.is_empty() or points_df.is_empty():
        print('[embedding_geometry] skipping; pool-analysis tables are empty')
        return pl.DataFrame(), pl.DataFrame()

    queries_df = read_parquet('queries')
    if queries_df.is_empty():
        print('[embedding_geometry] skipping; queries.parquet is empty')
        return pl.DataFrame(), pl.DataFrame()

    eval_results = _maybe_read_eval_table('evaluation_results')
    k_values = _available_k_values(eval_results)
    diagnostic_k = _diagnostic_k(k_values, stats_df)
    lambda_values = _available_lambda_values(eval_results)
    query_groups = _query_groups(
        ranked=_rank_queries_for_embedding_geometry(stats_df, eval_results, diagnostic_k),
        n_queries=min(cfg.n_figures, stats_df.height),
    )
    if not query_groups:
        print('[embedding_geometry] no queries selected for plotting')
        return pl.DataFrame(), pl.DataFrame()

    if pool_builder is None:
        from experiments.mimic.global_configs import global_cfg

        pool_builder = ChunkPoolBuilder(model_name=global_cfg.embedding_model)

    out_root = MimicPaths.figures_dir / 'pool_analysis' / 'embedding_geometry'
    out_root.mkdir(parents=True, exist_ok=True)

    stats_by_qid = {
        int(row['query_id']): row for row in stats_df.iter_rows(named=True)
    }
    points_by_qid = {
        int(df['query_id'][0]): df.sort('cos_to_query', descending=True)
        for df in points_df.partition_by('query_id')
        if df.height > 0
    }
    query_rows = {
        int(row['query_id']): cast(QueryRow, row)
        for row in queries_df.iter_rows(named=True)
        if row.get('query_id') is not None
    }

    point_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    for group, query_ids in query_groups.items():
        for query_id in query_ids:
            query_row = query_rows.get(query_id)
            stats_row = stats_by_qid.get(query_id)
            query_points = points_by_qid.get(query_id)
            if query_row is None or stats_row is None or query_points is None:
                continue

            rendered = _render_query(
                cfg=cfg,
                query_row=query_row,
                stats_row=stats_row,
                points_df=query_points,
                eval_results=eval_results,
                pool_builder=pool_builder,
                lambda_values=lambda_values,
                k_values=k_values,
                diagnostic_k=diagnostic_k,
                group=group,
                out_root=out_root,
            )
            if rendered is None:
                continue
            query_point_rows, query_stat_row = rendered
            point_rows.extend(query_point_rows)
            stat_rows.append(query_stat_row)

    points_out = pl.DataFrame(point_rows) if point_rows else pl.DataFrame()
    stats_out = pl.DataFrame(stat_rows) if stat_rows else pl.DataFrame()

    if not points_out.is_empty():
        get_pool_analysis_path('embedding_geometry_points.parquet', ensure_parent=True)
        points_out.write_parquet(
            get_pool_analysis_path('embedding_geometry_points.parquet', ensure_parent=True)
        )
    if not stats_out.is_empty():
        stats_out.write_parquet(
            get_pool_analysis_path('embedding_geometry_query_stats.parquet', ensure_parent=True)
        )
        plot_cluster_quality_overview(stats_out, out_root)

    print(f'[embedding_geometry] saved figures to {out_root}')
    return points_out, stats_out


def _render_query(
    *,
    cfg: PoolAnalysisCfg,
    query_row: QueryRow,
    stats_row: dict[str, Any],
    points_df: pl.DataFrame,
    eval_results: pl.DataFrame,
    pool_builder: ChunkPoolBuilder,
    lambda_values: list[float],
    k_values: list[int],
    diagnostic_k: int,
    group: str,
    out_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    pool = _pool_from_points(points_df, pool_builder)
    if pool is None or pool.n == 0:
        return None

    modifiers = QueryModifier.parse_list(query_row.get('modifiers_json', '') or '')
    if not modifiers:
        return None

    facet_onehot = _facet_onehot(
        pool=pool,
        icd10_3char=query_row['icd10_3char'],
        modifiers=modifiers,
        builder=pool_builder,
    )
    if not facet_onehot.any():
        return None

    qid = int(query_row['query_id'])
    qkey = _query_key(qid)
    sim_to_query = points_df['cos_to_query'].to_numpy().astype(np.float32)
    sim_matrix = pool.sim_matrix()
    coords = np.column_stack(
        [points_df['umap_x'].to_numpy(), points_df['umap_y'].to_numpy()]
    ).astype(np.float32)
    query_coord = _query_coord(coords, sim_to_query)
    cluster_labels = points_df['hdbscan_cluster'].to_numpy().astype(np.int32)
    lof = points_df['lof'].to_numpy().astype(np.float32) if 'lof' in points_df.columns else None

    label_payload = _label_payload(
        qkey=qkey,
        facet_onehot=facet_onehot,
        modifiers=modifiers,
        cluster_labels=cluster_labels,
        lof=lof,
    )
    selections = _choose_overlay_selections(
        sim_to_query=sim_to_query,
        sim_matrix=sim_matrix,
        eval_results=eval_results,
        query_id=qid,
        k=diagnostic_k,
        lambda_values=lambda_values,
    )

    artifact = {
        'query_id': qkey,
        'query': {
            'query_id': qid,
            'query_text': query_row['query_text'],
            'query_type': 'mimic_structural',
            'condition_id': query_row['icd10_3char'],
            'condition_display': query_row['icd10_3char'],
        },
        'pool_scope': 'primary_condition_restricted',
        'candidate_chunk_ids': pool.chunk_ids,
        'candidate_vectors': pool.vectors,
        'sim_to_query': sim_to_query.astype(np.float32),
        'sim_matrix': sim_matrix.astype(np.float32),
        'coords': coords,
        'query_coord': query_coord,
        'labels': label_payload['labels'],
        'label_ids': label_payload['label_ids'],
        'roles': label_payload['roles'],
        'is_gold': label_payload['is_gold'],
        'cluster_labels': cluster_labels,
        'k': diagnostic_k,
        'lambda_values': lambda_values,
        'reduction_method': 'umap_2d',
        'selections': selections,
        'selection_group': group,
    }

    query_dir = out_root / group / f'{qkey}_{query_row["icd10_3char"]}'
    query_dir.mkdir(parents=True, exist_ok=True)
    plot_query_overview_4panel(artifact, query_dir)
    plot_strategy_overlay(artifact, query_dir)
    for k in k_values:
        plot_full_strategy_selection_overlay(artifact, query_dir, k=k)

    selected_sets = {
        'top_k': {
            int(idx) for idx in selections.get('top_k', {}).get('local_indices', np.array([], dtype=int))
        },
        'mmr': {
            int(idx) for idx in selections.get('mmr', {}).get('local_indices', np.array([], dtype=int))
        },
        'fac_loc': {
            int(idx)
            for idx in selections.get('fac_loc', {}).get('local_indices', np.array([], dtype=int))
        },
    }
    point_rows = _point_rows(
        artifact=artifact,
        pool=pool,
        points_df=points_df,
        selected_sets=selected_sets,
        group=group,
    )
    stat_row = _query_stat_row(
        artifact=artifact,
        stats_row=stats_row,
        facet_onehot=facet_onehot,
        group=group,
    )
    return point_rows, stat_row


def _pool_from_points(points_df: pl.DataFrame, builder: ChunkPoolBuilder) -> ChunkPool | None:
    chunk_ids = points_df['chunk_id'].to_list()
    if not chunk_ids:
        return None

    fetched = builder.get_by_chunk_ids(set(chunk_ids))
    if fetched.n == 0:
        return None
    idx_by_chunk_id = {chunk_id: idx for idx, chunk_id in enumerate(fetched.chunk_ids)}
    keep = [idx_by_chunk_id[chunk_id] for chunk_id in chunk_ids if chunk_id in idx_by_chunk_id]
    if not keep:
        return None
    return fetched.select_by_indices(np.array(keep, dtype=np.intp))


def _facet_onehot(
    *,
    pool: ChunkPool,
    icd10_3char: str,
    modifiers: list[QueryModifier],
    builder: ChunkPoolBuilder,
) -> NDArray[np.bool_]:
    n_chunks = pool.n
    n_modifiers = len(modifiers)
    onehot = np.zeros((n_chunks, n_modifiers), dtype=bool)
    hadm_ids = pool.hadm_ids.tolist()

    for modifier_idx, modifier in enumerate(modifiers):
        qualifying = builder.get_hadm_ids_by_condition_modifier(icd10_3char, modifier)
        for row_idx, hadm_id in enumerate(hadm_ids):
            if int(hadm_id) in qualifying:
                onehot[row_idx, modifier_idx] = True
    return onehot


def _label_payload(
    *,
    qkey: str,
    facet_onehot: NDArray[np.bool_],
    modifiers: list[QueryModifier],
    cluster_labels: NDArray[np.int32],
    lof: NDArray[np.float32] | None,
) -> dict[str, Any]:
    labels: list[str] = []
    label_ids: list[str] = []
    roles: list[str] = []
    is_gold: list[bool] = []
    facet_displays = [_modifier_display(modifier) for modifier in modifiers]
    lof_cutoff = float(np.quantile(lof, 0.9)) if lof is not None and len(lof) >= 10 else None

    for row_idx, flags in enumerate(facet_onehot):
        n_true = int(flags.sum())
        if n_true == 0:
            is_background = bool(cluster_labels[row_idx] == -1)
            if not is_background and lof_cutoff is not None and lof is not None:
                is_background = bool(lof[row_idx] >= lof_cutoff)
            if is_background:
                labels.append(_BACKGROUND_LABEL)
                label_ids.append('background_clinical_cluster')
                roles.append('background_outlier')
            else:
                labels.append(_SOFT_DISTRACTOR_LABEL)
                label_ids.append('same_condition_other_subgroup')
                roles.append('distractor')
            is_gold.append(False)
            continue

        gold_indices = [idx for idx, flag in enumerate(flags.tolist()) if flag]
        if len(gold_indices) == 1:
            facet_idx = gold_indices[0]
            labels.append(facet_displays[facet_idx])
            label_ids.append(f'{qkey}::{modifiers[facet_idx].label}')
        else:
            labels.append(
                'shared evidence: both modifiers'
                if len(gold_indices) == 2
                else 'shared evidence: multiple modifiers'
            )
            label_ids.append(f'{qkey}::shared')
        roles.append('facet')
        is_gold.append(True)

    return {
        'labels': labels,
        'label_ids': label_ids,
        'roles': roles,
        'is_gold': is_gold,
    }


def _choose_overlay_selections(
    *,
    sim_to_query: NDArray[np.float32],
    sim_matrix: NDArray[np.float32],
    eval_results: pl.DataFrame,
    query_id: int,
    k: int,
    lambda_values: list[float],
) -> dict[str, dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}

    selections['top_k'] = {
        'local_indices': select_indices(
            strategy='top_k',
            sim_to_query=sim_to_query,
            sim_matrix=sim_matrix,
            k=k,
            lam=None,
        ),
        'lam': None,
    }

    for strategy in ['mmr', 'fac_loc']:
        chosen_lam = _best_lambda_for_query(eval_results, query_id, strategy, k)
        if chosen_lam is None and lambda_values:
            chosen_lam = lambda_values[len(lambda_values) // 2]
        if chosen_lam is None:
            continue
        selections[strategy] = {
            'local_indices': select_indices(
                strategy=cast(Any, strategy),
                sim_to_query=sim_to_query,
                sim_matrix=sim_matrix,
                k=k,
                lam=float(chosen_lam),
            ),
            'lam': float(chosen_lam),
        }
    return selections


def _best_lambda_for_query(
    eval_results: pl.DataFrame,
    query_id: int,
    strategy: str,
    k: int,
) -> float | None:
    if eval_results.is_empty() or 'query_id' not in eval_results.columns:
        return None
    subset = eval_results.filter(
        (pl.col('query_id') == query_id) & (pl.col('strategy') == strategy) & (pl.col('k') == k)
    )
    if subset.is_empty() or 'lam' not in subset.columns:
        return None
    ranked = (
        subset.with_columns((1.0 - pl.col('gold_precision')).alias('distractor_rate'))
        .sort(
            [
                'aspect_recall',
                'gold_precision',
                'weighted_aspect_recall',
                'distractor_rate',
                'lam',
            ],
            descending=[True, True, True, False, False],
        )
    )
    lam = ranked['lam'][0]
    return None if lam is None else float(lam)


def _point_rows(
    *,
    artifact: dict[str, Any],
    pool: ChunkPool,
    points_df: pl.DataFrame,
    selected_sets: dict[str, set[int]],
    group: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qid = int(artifact['query']['query_id'])
    for idx, chunk_id in enumerate(pool.chunk_ids):
        rows.append({
            'query_id': qid,
            'query_key': artifact['query_id'],
            'selection_group': group,
            'point_kind': 'chunk',
            'chunk_id': chunk_id,
            'hadm_id': int(pool.hadm_ids[idx]),
            'section_name': pool.section_names[idx],
            'rank': idx + 1,
            'x': float(artifact['coords'][idx, 0]),
            'y': float(artifact['coords'][idx, 1]),
            'reduction_method': artifact['reduction_method'],
            'sim_to_query': float(artifact['sim_to_query'][idx]),
            'plot_label': artifact['labels'][idx],
            'label_id': artifact['label_ids'][idx],
            'cluster_role': artifact['roles'][idx],
            'is_gold': bool(artifact['is_gold'][idx]),
            'facet_combined': points_df['facet_combined'][idx],
            'hdbscan_label': int(artifact['cluster_labels'][idx]),
            'selected_top_k': idx in selected_sets.get('top_k', set()),
            'selected_mmr': idx in selected_sets.get('mmr', set()),
            'selected_fac_loc': idx in selected_sets.get('fac_loc', set()),
        })

    rows.append({
        'query_id': qid,
        'query_key': artifact['query_id'],
        'selection_group': group,
        'point_kind': 'query',
        'chunk_id': None,
        'hadm_id': None,
        'section_name': None,
        'rank': 0,
        'x': float(artifact['query_coord'][0]),
        'y': float(artifact['query_coord'][1]),
        'reduction_method': artifact['reduction_method'],
        'sim_to_query': 1.0,
        'plot_label': 'query',
        'label_id': 'query',
        'cluster_role': 'query',
        'is_gold': False,
        'facet_combined': None,
        'hdbscan_label': None,
        'selected_top_k': False,
        'selected_mmr': False,
        'selected_fac_loc': False,
    })
    return rows


def _query_stat_row(
    *,
    artifact: dict[str, Any],
    stats_row: dict[str, Any],
    facet_onehot: NDArray[np.bool_],
    group: str,
) -> dict[str, Any]:
    hidden_codes = _string_codes(artifact['label_ids'])
    cluster_labels = artifact['cluster_labels']
    ari, nmi = _cluster_agreement(hidden_codes, cluster_labels)
    gold_silhouette = _gold_silhouette(artifact)
    mean_in, mean_cross = _in_cross_similarity(artifact)
    is_gold = np.array(artifact['is_gold'], dtype=bool)
    n_clusters = len({int(value) for value in cluster_labels.tolist() if int(value) != -1})

    row: dict[str, Any] = {
        'query_id': int(artifact['query']['query_id']),
        'query_key': artifact['query_id'],
        'selection_group': group,
        'icd10_3char': stats_row.get('icd10_3char'),
        'stratum': stats_row.get('stratum'),
        'pool_size': len(artifact['candidate_chunk_ids']),
        'plot_k': artifact['k'],
        'reduction_method': artifact['reduction_method'],
        'n_hidden_labels': len(set(artifact['label_ids'])),
        'n_gold_points': int(is_gold.sum()),
        'n_distractor_points': int((~is_gold).sum()),
        'gold_silhouette_cosine': gold_silhouette,
        'mean_in_facet_similarity': mean_in,
        'mean_cross_facet_similarity': mean_cross,
        'in_minus_cross_similarity': mean_in - mean_cross,
        'query_to_gold_mean': _masked_mean(artifact['sim_to_query'], is_gold),
        'query_to_distractor_mean': _masked_mean(artifact['sim_to_query'], ~is_gold),
        'hdbscan_n_clusters': n_clusters,
        'hdbscan_noise_rate': float(np.mean(cluster_labels == -1)),
        'hdbscan_ari_hidden': ari,
        'hdbscan_nmi_hidden': nmi,
        'facet_lda_acc_cv': stats_row.get('facet_lda_acc_cv'),
        'facet_logreg_acc_cv': stats_row.get('facet_logreg_acc_cv'),
        'nmi_cluster_facet_hdb': stats_row.get('nmi_cluster_facet_hdb'),
        'ari_cluster_facet_hdb': stats_row.get('ari_cluster_facet_hdb'),
        'effective_rank': stats_row.get('effective_rank'),
        'dom_cluster_frac_hdb': stats_row.get('dom_cluster_frac_hdb'),
        'frac_outliers_hdb': stats_row.get('frac_outliers_hdb'),
        'intra_minus_cross': stats_row.get('intra_minus_cross'),
    }
    for strategy, payload in artifact['selections'].items():
        summary = _selection_summary(
            facet_onehot=facet_onehot,
            is_gold=is_gold,
            label_ids=artifact['label_ids'],
            local_indices=payload['local_indices'],
        )
        row[f'{strategy}_n_facets_selected'] = summary['n_facets_selected']
        row[f'{strategy}_gold_precision'] = summary['gold_precision']
        row[f'{strategy}_distractor_rate'] = summary['distractor_rate']
        row[f'{strategy}_dominant_fraction'] = summary['dominant_fraction']
    return row


def _selection_summary(
    *,
    facet_onehot: NDArray[np.bool_],
    is_gold: NDArray[np.bool_],
    label_ids: list[str],
    local_indices: NDArray[np.intp],
) -> dict[str, float | int]:
    indices = [int(idx) for idx in local_indices]
    if not indices:
        return {
            'n_facets_selected': 0,
            'gold_precision': 0.0,
            'distractor_rate': 0.0,
            'dominant_fraction': 0.0,
        }

    selected_gold = is_gold[indices]
    selected_onehot = facet_onehot[indices]
    selected_labels = [label_ids[idx] for idx in indices if is_gold[idx]]
    label_counts = Counter(selected_labels)
    dominant = label_counts.most_common(1)[0][1] if label_counts else 0
    gold_precision = float(selected_gold.mean())
    return {
        'n_facets_selected': int(selected_onehot.any(axis=0).sum()),
        'gold_precision': gold_precision,
        'distractor_rate': 1.0 - gold_precision,
        'dominant_fraction': dominant / len(indices),
    }


def _gold_silhouette(artifact: dict[str, Any]) -> float | None:
    from sklearn.metrics import silhouette_score

    gold_idx = [idx for idx, flag in enumerate(artifact['is_gold']) if flag]
    labels = [artifact['label_ids'][idx] for idx in gold_idx]
    if len(gold_idx) <= len(set(labels)) or len(set(labels)) < 2:
        return None
    try:
        return float(silhouette_score(artifact['candidate_vectors'][gold_idx], labels, metric='cosine'))
    except Exception:
        return None


def _in_cross_similarity(artifact: dict[str, Any]) -> tuple[float, float]:
    gold_idx = [idx for idx, flag in enumerate(artifact['is_gold']) if flag]
    labels = np.array([artifact['label_ids'][idx] for idx in gold_idx])
    if len(gold_idx) < 2 or len(set(labels.tolist())) < 2:
        return 0.0, 0.0

    sim = artifact['sim_matrix'][np.ix_(gold_idx, gold_idx)]
    same = labels[:, None] == labels[None, :]
    not_self = ~np.eye(len(gold_idx), dtype=bool)
    in_values = sim[same & not_self]
    cross_values = sim[~same & not_self]
    return (
        float(in_values.mean()) if len(in_values) else 0.0,
        float(cross_values.mean()) if len(cross_values) else 0.0,
    )


def _cluster_agreement(
    hidden_labels: NDArray[np.int32],
    cluster_labels: NDArray[np.int32],
) -> tuple[float | None, float | None]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if len(hidden_labels) < 2 or len(set(hidden_labels.tolist())) < 2:
        return None, None
    try:
        return (
            float(adjusted_rand_score(hidden_labels, cluster_labels)),
            float(normalized_mutual_info_score(hidden_labels, cluster_labels)),
        )
    except Exception:
        return None, None


def _masked_mean(values: NDArray[np.float32], mask: NDArray[np.bool_]) -> float | None:
    if not bool(mask.any()):
        return None
    return float(values[mask].mean())


def _query_coord(coords: NDArray[np.float32], sim_to_query: NDArray[np.float32]) -> NDArray[np.float32]:
    if len(coords) == 0:
        return np.zeros(2, dtype=np.float32)
    weights = sim_to_query - float(sim_to_query.min())
    if float(weights.sum()) <= 0.0:
        return coords.mean(axis=0).astype(np.float32)
    normalized = weights / float(weights.sum())
    return (coords * normalized[:, None]).sum(axis=0).astype(np.float32)


def _query_groups(ranked: pl.DataFrame, n_queries: int) -> dict[str, list[int]]:
    if ranked.is_empty() or n_queries <= 0:
        return {}
    n_good, n_mid, n_bad = _mixed_group_sizes(min(n_queries, ranked.height))
    good = ranked['query_id'].head(n_good).to_list()
    bad = (
        ranked.sort(
            ['selection_score', 'fac_loc_ar_gain', 'facet_lda_acc_cv', 'query_id'],
            descending=[False, False, False, False],
        )['query_id']
        .head(n_bad)
        .to_list()
    )
    excluded = [*good, *bad]
    remaining = ranked.filter(~pl.col('query_id').is_in(excluded))
    mid_start = max(0, (remaining.height - n_mid) // 2)
    mid = remaining['query_id'].slice(mid_start, n_mid).to_list()
    return {
        key: value
        for key, value in [('good', good), ('mid', mid), ('bad', bad)]
        if value
    }


def _mixed_group_sizes(n_queries: int) -> tuple[int, int, int]:
    base = n_queries // 3
    remainder = n_queries % 3
    n_good = base + int(remainder >= 1)
    n_mid = base + int(remainder >= 2)
    n_bad = base
    return n_good, n_mid, n_bad


def _rank_queries_for_embedding_geometry(
    stats_df: pl.DataFrame,
    eval_results: pl.DataFrame,
    diagnostic_k: int,
) -> pl.DataFrame:
    base = stats_df.select(
        'query_id',
        'facet_lda_acc_cv',
        'facet_logreg_acc_cv',
        'intra_minus_cross',
        'frac_outliers_hdb',
        'dom_cluster_frac_hdb',
    ).with_columns(
        pl.col('facet_lda_acc_cv').fill_null(0.0),
        pl.col('facet_logreg_acc_cv').fill_null(0.0),
        pl.col('intra_minus_cross').fill_null(0.0),
        pl.col('frac_outliers_hdb').fill_null(0.0),
        pl.col('dom_cluster_frac_hdb').fill_null(0.0),
    )

    gains = _evaluation_gain_table(eval_results, diagnostic_k)
    if not gains.is_empty():
        base = base.join(gains, on='query_id', how='left')
    if 'fac_loc_ar_gain' not in base.columns:
        base = base.with_columns(
            pl.lit(0.0).alias('fac_loc_ar_gain'),
            pl.lit(0.0).alias('mmr_ar_gain'),
        )
    ranked = base.with_columns(
        pl.col('fac_loc_ar_gain').fill_null(0.0),
        pl.col('mmr_ar_gain').fill_null(0.0),
    ).with_columns(
        (
            pl.col('fac_loc_ar_gain') * 10.0
            + pl.col('facet_lda_acc_cv') * 2.0
            + pl.col('facet_logreg_acc_cv') * 1.0
            + pl.col('intra_minus_cross') * 2.0
            - pl.col('frac_outliers_hdb') * 1.5
            + pl.col('dom_cluster_frac_hdb') * 0.5
        ).alias('selection_score')
    )
    return ranked.sort(
        ['selection_score', 'fac_loc_ar_gain', 'facet_lda_acc_cv', 'query_id'],
        descending=[True, True, True, False],
    )


def _evaluation_gain_table(eval_results: pl.DataFrame, k: int) -> pl.DataFrame:
    if eval_results.is_empty():
        return pl.DataFrame()

    available_k = sorted(eval_results['k'].unique().to_list())
    if not available_k:
        return pl.DataFrame()
    if k not in available_k:
        k = available_k[len(available_k) // 2]

    topk = eval_results.filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k)).select(
        'query_id', pl.col('aspect_recall').alias('topk_ar')
    )
    if topk.is_empty():
        return pl.DataFrame()

    joined = topk
    for strategy in ['fac_loc', 'mmr']:
        subset = eval_results.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
        if subset.is_empty():
            continue
        best = (
            subset.with_columns((1.0 - pl.col('gold_precision')).alias('distractor_rate'))
            .sort(
                [
                    'query_id',
                    'aspect_recall',
                    'gold_precision',
                    'weighted_aspect_recall',
                    'distractor_rate',
                    'lam',
                ],
                descending=[False, True, True, True, False, False],
            )
            .group_by('query_id')
            .first()
            .select('query_id', pl.col('aspect_recall').alias(f'{strategy}_ar'))
        )
        joined = joined.join(best, on='query_id', how='left')

    if 'fac_loc_ar' not in joined.columns:
        joined = joined.with_columns(pl.lit(None).alias('fac_loc_ar'))
    if 'mmr_ar' not in joined.columns:
        joined = joined.with_columns(pl.lit(None).alias('mmr_ar'))
    return joined.with_columns(
        (pl.col('fac_loc_ar').fill_null(pl.col('topk_ar')) - pl.col('topk_ar')).alias(
            'fac_loc_ar_gain'
        ),
        (pl.col('mmr_ar').fill_null(pl.col('topk_ar')) - pl.col('topk_ar')).alias('mmr_ar_gain'),
    ).select('query_id', 'fac_loc_ar_gain', 'mmr_ar_gain')


def _available_k_values(eval_results: pl.DataFrame) -> list[int]:
    if eval_results.is_empty() or 'k' not in eval_results.columns:
        return [20]
    values = sorted(int(value) for value in eval_results['k'].drop_nulls().unique().to_list())
    return values or [20]


def _diagnostic_k(k_values: list[int], stats_df: pl.DataFrame) -> int:
    if k_values:
        return int(k_values[len(k_values) // 2])
    pool_size = int(stats_df['pool_size'].median()) if 'pool_size' in stats_df.columns else 20
    return max(1, min(20, pool_size))


def _available_lambda_values(eval_results: pl.DataFrame) -> list[float]:
    if eval_results.is_empty() or 'lam' not in eval_results.columns:
        return _DEFAULT_LAMBDA_VALUES
    values = sorted(float(value) for value in eval_results['lam'].drop_nulls().unique().to_list())
    return values or _DEFAULT_LAMBDA_VALUES


def _query_key(query_id: int) -> str:
    return f'q{query_id:04d}'


def _modifier_display(modifier: QueryModifier) -> str:
    return modifier.text.strip()


def _string_codes(labels: list[str]) -> NDArray[np.int32]:
    mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([mapping[label] for label in labels], dtype=np.int32)


def _maybe_read_eval_table(name: str) -> pl.DataFrame:
    path = MimicPaths.experiment_dir / f'{name}.parquet'
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame()
