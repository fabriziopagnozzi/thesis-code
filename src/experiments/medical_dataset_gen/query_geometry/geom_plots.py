from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from textwrap import fill
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.retrieval_utils import select_indices
from experiments.medical_dataset_gen.query_geometry.geom_plots_configs import (
    BACKGROUND_OUTLIER_COLOR,
    BACKGROUND_OUTLIER_LABEL_ID,
    BACKGROUND_OUTLIER_MAP_MARKER_SIZE,
    BACKGROUND_OUTLIER_RANK_MARKER_SIZE,
    BACKGROUND_OUTLIER_ROLE,
    BACKGROUND_OUTLIER_SELECTION_SELECTED_MARKER_SIZE,
    BACKGROUND_OUTLIER_SELECTION_UNSELECTED_MARKER_SIZE,
    BACKGROUND_OUTLIER_SIMILARITY_MAP_MARKER_SIZE,
    FULL_STRATEGY_BOTTOM_MARGIN_IN,
    FULL_STRATEGY_COLORBAR_GAP_IN,
    FULL_STRATEGY_COLORBAR_WIDTH_IN,
    FULL_STRATEGY_LEFT_MARGIN_IN,
    FULL_STRATEGY_LEGEND_GAP_IN,
    FULL_STRATEGY_LEGEND_WIDTH_IN,
    FULL_STRATEGY_PANEL_GAP_X_IN,
    FULL_STRATEGY_PANEL_GAP_Y_IN,
    FULL_STRATEGY_PANEL_SIZE_IN,
    FULL_STRATEGY_RIGHT_MARGIN_IN,
    FULL_STRATEGY_TOP_MARGIN_IN,
    QUERY_OVERVIEW_FIGURE_SIZE,
    QUERY_OVERVIEW_LEGEND_BBOX_TO_ANCHOR_X,
    QUERY_OVERVIEW_LEGEND_FONT_SIZE,
    QUERY_OVERVIEW_LEGEND_WRAP_WIDTH,
    SAME_CONDITION_WRONG_AXIS_LABEL_ID,
    SAME_CONDITION_WRONG_AXIS_ROLE,
    UNSELECTED_DATA_POINT_COLOR,
)
from experiments.medical_dataset_gen.schemas.query_geometry_schemas import (
    GeometryArtifact,
    GeometrySelection,
)
from experiments.medical_dataset_gen.schemas.retrieval_schemas import RetrievalStrategy

LEGEND_CHILD_INDENT = '    '
LEGEND_HEADER_LABEL = 'Primary Condition / Subgroup / Clinical Axis\n'
QUERY_TITLE_SUBGROUP_COLOR = '#7B2CBF'
QUERY_TITLE_AXIS_COLOR = '#2D6A4F'


def plot_strategy_overlay(artifact: GeometryArtifact, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    strategy_order: tuple[RetrievalStrategy, ...] = ('top_k', 'mmr', 'fac_loc')
    strategies = [strategy for strategy in strategy_order if strategy in artifact.selections]
    if not strategies:
        return

    fig, axes = plt.subplots(
        1, len(strategies), figsize=(5.3 * len(strategies), 5.2), sharex=True, sharey=True
    )
    palette = label_palette(artifact)
    axes_list = [axes] if len(strategies) == 1 else list(axes)
    for ax, strategy in zip(axes_list, strategies, strict=True):
        payload = artifact.selections[strategy]
        title = _selection_panel_title(
            artifact=artifact,
            strategy=strategy,
            lam=payload.lam,
            local_indices=payload.local_indices,
        )
        _plot_selection_panel(
            ax=ax,
            artifact=artifact,
            palette=palette,
            local_indices=payload.local_indices,
            title=title,
        )
    legend_handles = _selection_legend_handles(axes_list)
    legend_entries = _display_legend_entries(artifact, legend_handles)
    fig.legend(
        [handle for _, handle in legend_entries],
        [_format_legend_label(artifact, label) for label, _ in legend_entries],
        fontsize=7,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(0.985, 0.5),
        alignment='left',
    )
    _draw_artifact_figure_title(fig, artifact, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.92, 0.94))
    fig.savefig(out_dir / 'strategy_selection_overlay.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_full_strategy_selection_overlay(
    artifact: GeometryArtifact,
    out_dir: Path,
    *,
    k: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    lambda_values = [float(lam) for lam in artifact.lambda_values]
    if not lambda_values:
        return
    render_k = int(artifact.k if k is None else k)
    effective_k = min(render_k, len(artifact.sim_to_query))
    selection_variants = _selection_variants_for_k(artifact, effective_k)
    rows = ['mmr', 'fac_loc']
    row_variants = {
        strategy: _meaningful_lambda_selection_variants(
            selection_variants.get(strategy, []),
            lambda_values=lambda_values,
        )
        for strategy in rows
    }
    mmr_cols = max(1, len(row_variants['mmr']))
    facloc_cols = max(1, len(row_variants['fac_loc']))
    top_row_width = (
        2 * FULL_STRATEGY_PANEL_SIZE_IN
        + FULL_STRATEGY_PANEL_GAP_X_IN
        + FULL_STRATEGY_COLORBAR_GAP_IN
        + FULL_STRATEGY_COLORBAR_WIDTH_IN
        + FULL_STRATEGY_LEGEND_GAP_IN
        + FULL_STRATEGY_LEGEND_WIDTH_IN
    )
    widest_selection_row = (
        max(mmr_cols, facloc_cols) * FULL_STRATEGY_PANEL_SIZE_IN
        + max(0, max(mmr_cols, facloc_cols) - 1) * FULL_STRATEGY_PANEL_GAP_X_IN
    )
    fig_width = (
        FULL_STRATEGY_LEFT_MARGIN_IN
        + max(top_row_width, widest_selection_row)
        + FULL_STRATEGY_RIGHT_MARGIN_IN
    )
    fig_height = (
        FULL_STRATEGY_BOTTOM_MARGIN_IN
        + 3 * FULL_STRATEGY_PANEL_SIZE_IN
        + 2 * FULL_STRATEGY_PANEL_GAP_Y_IN
        + FULL_STRATEGY_TOP_MARGIN_IN
    )
    fig = plt.figure(figsize=(fig_width, fig_height))  # type: ignore

    palette = label_palette(artifact)
    topk_variants = selection_variants.get('top_k', [])
    top_row_bottom = FULL_STRATEGY_BOTTOM_MARGIN_IN + 2 * (
        FULL_STRATEGY_PANEL_SIZE_IN + FULL_STRATEGY_PANEL_GAP_Y_IN
    )
    middle_row_bottom = FULL_STRATEGY_BOTTOM_MARGIN_IN + (
        FULL_STRATEGY_PANEL_SIZE_IN + FULL_STRATEGY_PANEL_GAP_Y_IN
    )
    bottom_row_bottom = FULL_STRATEGY_BOTTOM_MARGIN_IN

    ax_topk = _add_axes_in_inches(
        fig,
        fig_width=fig_width,
        fig_height=fig_height,
        left=FULL_STRATEGY_LEFT_MARGIN_IN,
        bottom=top_row_bottom,
        width=FULL_STRATEGY_PANEL_SIZE_IN,
        height=FULL_STRATEGY_PANEL_SIZE_IN,
    )
    if topk_variants:
        _plot_selection_panel(
            ax=ax_topk,
            artifact=artifact,
            palette=palette,
            local_indices=topk_variants[0].local_indices,
            title=_selection_panel_title(
                artifact=artifact,
                strategy='top_k',
                lam=None,
                local_indices=topk_variants[0].local_indices,
            ),
        )
    else:
        ax_topk.axis('off')
    ax_topk.set_box_aspect(1)

    similarity_left = (
        FULL_STRATEGY_LEFT_MARGIN_IN + FULL_STRATEGY_PANEL_SIZE_IN + FULL_STRATEGY_PANEL_GAP_X_IN
    )
    ax_similarity = _add_axes_in_inches(
        fig,
        fig_width=fig_width,
        fig_height=fig_height,
        left=similarity_left,
        bottom=top_row_bottom,
        width=FULL_STRATEGY_PANEL_SIZE_IN,
        height=FULL_STRATEGY_PANEL_SIZE_IN,
    )
    similarity_points = _plot_query_similarity_map_ax(
        ax_similarity, artifact, include_title_prefix=False
    )
    ax_similarity.set_box_aspect(1)
    ax_colorbar = _add_axes_in_inches(
        fig,
        fig_width=fig_width,
        fig_height=fig_height,
        left=similarity_left + FULL_STRATEGY_PANEL_SIZE_IN + FULL_STRATEGY_COLORBAR_GAP_IN,
        bottom=top_row_bottom,
        width=FULL_STRATEGY_COLORBAR_WIDTH_IN,
        height=FULL_STRATEGY_PANEL_SIZE_IN,
    )
    if similarity_points is not None:
        fig.colorbar(
            similarity_points,
            cax=ax_colorbar,
            label='query cosine similarity',
        )
    else:
        ax_colorbar.axis('off')

    ax_legend = _add_axes_in_inches(
        fig,
        fig_width=fig_width,
        fig_height=fig_height,
        left=(
            similarity_left
            + FULL_STRATEGY_PANEL_SIZE_IN
            + FULL_STRATEGY_COLORBAR_GAP_IN
            + FULL_STRATEGY_COLORBAR_WIDTH_IN
            + FULL_STRATEGY_LEGEND_GAP_IN
        ),
        bottom=top_row_bottom,
        width=FULL_STRATEGY_LEGEND_WIDTH_IN,
        height=FULL_STRATEGY_PANEL_SIZE_IN,
    )
    mmr_axes: list[Any] = []
    facloc_axes: list[Any] = []
    for strategy, row_bottom in [('mmr', middle_row_bottom), ('fac_loc', bottom_row_bottom)]:
        variants = row_variants[strategy]
        row_axes = mmr_axes if strategy == 'mmr' else facloc_axes
        for col_idx, payload in enumerate(variants):
            ax = _add_axes_in_inches(
                fig,
                fig_width=fig_width,
                fig_height=fig_height,
                left=FULL_STRATEGY_LEFT_MARGIN_IN
                + col_idx * (FULL_STRATEGY_PANEL_SIZE_IN + FULL_STRATEGY_PANEL_GAP_X_IN),
                bottom=row_bottom,
                width=FULL_STRATEGY_PANEL_SIZE_IN,
                height=FULL_STRATEGY_PANEL_SIZE_IN,
            )
            _plot_selection_panel(
                ax=ax,
                artifact=artifact,
                palette=palette,
                local_indices=payload.local_indices,
                title=_selection_panel_title(
                    artifact=artifact,
                    strategy=strategy,
                    lam=float(payload.lam) if payload.lam else None,
                    local_indices=payload.local_indices,
                ),
            )
            ax.set_box_aspect(1)
            row_axes.append(ax)

    legend_axes = [ax_topk, *mmr_axes, *facloc_axes]
    legend_handles = _selection_legend_handles(legend_axes)
    _draw_selection_legend_panel(ax_legend, artifact, legend_handles)

    _draw_artifact_figure_title(fig, artifact, fontsize=12, y=1 - 0.16 / fig_height)
    fig.savefig(
        out_dir / f'full_strategy_selection_overlay_k{render_k}.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_query_overview_4panel(
    artifact: GeometryArtifact,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=QUERY_OVERVIEW_FIGURE_SIZE)  # type: ignore
    ax_map, ax_cos, ax_rank, ax_cluster = axes.ravel()

    _plot_query_map_ax(ax_map, artifact, include_title_prefix=False)
    _draw_facet_family_side_legend(ax_map, artifact)

    similarity_points = _plot_query_similarity_map_ax(ax_cos, artifact, include_title_prefix=False)
    if similarity_points is not None:
        fig.colorbar(
            similarity_points,
            ax=ax_cos,
            fraction=0.05,
            pad=0.02,
            label='query cosine',
        )

    _plot_query_rank_ax(ax_rank, artifact, include_title_prefix=False)
    _draw_facet_family_side_legend(ax_rank, artifact)

    _plot_discovered_clusters_ax(ax_cluster, artifact, include_title_prefix=False)
    _draw_wrapped_side_legend(ax_cluster)
    for ax in axes.ravel():
        ax.set_box_aspect(1)
    fig.subplots_adjust(left=0.055, right=0.94, top=0.88, bottom=0.06, wspace=0.78, hspace=0.30)
    _draw_artifact_figure_title(fig, artifact, fontsize=14)
    fig.savefig(out_dir / 'query_overview_4panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_cluster_quality_overview(stats: pl.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cols = [
        ('gold_silhouette_cosine', 'Gold-facet silhouette'),
        ('mean_in_facet_similarity', 'Mean in-facet cosine'),
        ('mean_cross_facet_similarity', 'Mean cross-facet cosine'),
        ('hdbscan_ari_hidden', 'HDBSCAN ARI vs hidden labels'),
        ('hdbscan_noise_rate', 'HDBSCAN noise rate'),
    ]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.1 * len(cols), 4.2))
    for ax, (col, title) in zip(axes, cols, strict=True):
        values = stats[col].drop_nulls().to_list() if col in stats.columns else []
        if values:
            ax.boxplot(values, tick_labels=['queries'])
        else:
            ax.text(0.5, 0.5, 'n/a', ha='center')
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Embedding geometry diagnostics across visualized queries', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'cluster_quality_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_selection_panel(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    palette: dict[str, Any],
    local_indices: Any,
    title: str,
) -> None:
    coords = artifact.coords
    ax.scatter(coords[:, 0], coords[:, 1], color='#dddddd', s=24, alpha=0.55, edgecolors='none')
    selected = {int(i) for i in local_indices}
    label_groups = _label_groups(artifact)
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if _is_background_outlier_point(artifact, first_idx):
            continue
        idx = [i for i in indices if i in selected]
        if not idx:
            continue
        marker = _label_marker(artifact, first_idx)
        scatter_kwargs: dict[str, Any] = {
            's': 74,
            'color': palette[label],
            'marker': marker,
            'zorder': 4,
        }
        if marker == 'x':
            scatter_kwargs['linewidths'] = 1.35
        else:
            scatter_kwargs['edgecolors'] = 'black'
            scatter_kwargs['linewidths'] = 0.65
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=label,
            **scatter_kwargs,
        )
    _plot_background_outliers_for_selection(
        ax,
        artifact,
        selected=selected,
        palette=palette,
        label_groups=label_groups,
    )
    draw_query_marker(ax, artifact)
    ax.set_title(title)
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def _selection_variants_for_k(
    artifact: GeometryArtifact,
    k: int,
) -> dict[str, list[GeometrySelection]]:
    sim_to_query = np.asarray(artifact.sim_to_query)
    sim_matrix = np.asarray(artifact.sim_matrix)
    mmr_window = artifact.mmr_window
    lambda_values = [float(lam) for lam in artifact.lambda_values]

    variants: dict[str, list[GeometrySelection]] = {
        'top_k': [
            GeometrySelection(
                local_indices=select_indices(
                    strategy='top_k',
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=None,
                    mmr_window=mmr_window,
                ),
                lam=None,
            )
        ]
    }
    for strategy in cast(list[RetrievalStrategy], ['mmr', 'fac_loc']):
        variants[strategy] = [
            GeometrySelection(
                local_indices=select_indices(
                    strategy=strategy,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                    k=k,
                    lam=float(lam),
                    mmr_window=mmr_window,
                ),
                lam=float(lam),
            )
            for lam in lambda_values
        ]
    return variants


def _meaningful_lambda_selection_variants(
    variants: list[GeometrySelection],
    *,
    lambda_values: list[float],
) -> list[GeometrySelection]:
    ordered_variants: list[GeometrySelection] = []
    for lam in sorted(lambda_values, reverse=True):
        payload = next(
            (
                variant
                for variant in variants
                if variant.lam is not None and abs(float(variant.lam) - lam) < 1e-12
            ),
            None,
        )
        if payload is not None:
            ordered_variants.append(payload)

    meaningful: list[GeometrySelection] = []
    previous_signature: frozenset[int] | None = None
    for payload in ordered_variants:
        signature = frozenset(int(index) for index in payload.local_indices)
        if previous_signature is None or signature != previous_signature:
            meaningful.append(payload)
        previous_signature = signature
    return meaningful


def _selection_panel_title(
    artifact: GeometryArtifact,
    strategy: str,
    lam: float | None,
    local_indices: Any,
    *,
    lambda_label: float | None = None,
) -> str:
    selected = {int(i) for i in local_indices}
    selected_labels = [artifact.label_ids[i] for i in selected]
    selected_gold = sum(1 for i in selected if artifact.is_gold[i])
    selected_facets = len({x for x in selected_labels if x.startswith(artifact.query_id)})
    parts = []
    if strategy == 'top_k' and lambda_label is not None:
        parts.append(f'lambda={lambda_label:.2f}')
    parts.append(strategy_title(strategy, lam))
    parts.append(f'facets={selected_facets}, gold={selected_gold}/{len(selected)}')
    return '\n'.join(parts)


def _selection_legend_handles(axes: list[Any]) -> dict[str, Any]:
    from matplotlib.lines import Line2D

    legend_handles: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            legend_handles.setdefault(label, handle)
    legend_handles.setdefault(
        'query',
        Line2D(
            [0],
            [0],
            marker='*',
            color='none',
            markerfacecolor='black',
            markeredgecolor='white',
            markersize=12,
            label='query',
        ),
    )
    return legend_handles


def _draw_selection_legend_panel(
    ax: Any,
    artifact: GeometryArtifact,
    legend_handles: dict[str, Any],
) -> None:
    ax.axis('off')
    ax.set_title('Legend')
    legend_entries = _display_legend_entries(artifact, legend_handles)
    ax.legend(
        [handle for _, handle in legend_entries],
        [_format_legend_label(artifact, label) for label, _ in legend_entries],
        fontsize=7,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(0.0, 0.5),
        alignment='left',
    )


def _draw_wrapped_side_legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        [_wrap_legend_label(label) for label in labels],
        fontsize=QUERY_OVERVIEW_LEGEND_FONT_SIZE,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(QUERY_OVERVIEW_LEGEND_BBOX_TO_ANCHOR_X, 0.5),
        alignment='left',
    )


def _draw_facet_family_side_legend(ax: Any, artifact: GeometryArtifact) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend_handles = {label: handle for handle, label in zip(handles, labels, strict=True)}
    legend_entries = _display_legend_entries(artifact, legend_handles)
    ax.legend(
        [handle for _, handle in legend_entries],
        [_format_legend_label(artifact, label) for label, _ in legend_entries],
        fontsize=QUERY_OVERVIEW_LEGEND_FONT_SIZE,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(QUERY_OVERVIEW_LEGEND_BBOX_TO_ANCHOR_X, 0.5),
        alignment='left',
    )


def _format_legend_label(artifact: GeometryArtifact, label: str) -> str:
    if label == LEGEND_HEADER_LABEL:
        return label
    indent = LEGEND_CHILD_INDENT if _is_child_legend_label(artifact, label) else ''
    return _wrap_legend_label(label, indent=indent)


def _wrap_legend_label(
    label: str,
    *,
    width: int = QUERY_OVERVIEW_LEGEND_WRAP_WIDTH,
    indent: str = '',
) -> str:
    # Preserve explicit line breaks if any label already carries semantic grouping.
    return '\n'.join(
        fill(
            segment.strip(),
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for segment in label.splitlines()
    )


def _ordered_facet_family_legend_entries(
    artifact: GeometryArtifact,
    legend_handles: dict[str, Any],
) -> list[tuple[str, Any]]:
    label_groups = _label_groups(artifact)
    ordered_facet_ids = _ordered_facet_ids(artifact)
    ordered_facet_id_set = set(ordered_facet_ids)
    gold_label_by_facet_id: dict[str, str] = {}
    distractor_labels_by_facet_id: dict[str, list[str]] = defaultdict(list)
    background_labels: list[str] = []
    extra_labels: list[str] = []

    for label, indices in label_groups.items():
        if label not in legend_handles:
            continue
        first_idx = indices[0]
        if artifact.is_gold[first_idx]:
            gold_label_by_facet_id[artifact.label_ids[first_idx]] = label
            continue
        if _is_background_outlier_point(artifact, first_idx):
            background_labels.append(label)
            continue
        target_facet_id = _target_facet_id_for_index(artifact, first_idx)
        if target_facet_id is not None:
            distractor_labels_by_facet_id[target_facet_id].append(label)
        else:
            extra_labels.append(label)

    entries: list[tuple[str, Any]] = []
    added_labels: set[str] = set()
    for facet_id in ordered_facet_ids:
        facet_label = gold_label_by_facet_id.get(facet_id)
        if facet_label is not None:
            entries.append((facet_label, legend_handles[facet_label]))
            added_labels.add(facet_label)
        for label in _sorted_distractor_labels(
            artifact,
            distractor_labels_by_facet_id.get(facet_id, []),
        ):
            entries.append((label, legend_handles[label]))
            added_labels.add(label)

    for label in sorted(gold_label_by_facet_id.values(), key=str.lower):
        if label in added_labels:
            continue
        entries.append((label, legend_handles[label]))
        added_labels.add(label)

    for facet_id in sorted(distractor_labels_by_facet_id):
        if facet_id in ordered_facet_id_set:
            continue
        for label in _sorted_distractor_labels(artifact, distractor_labels_by_facet_id[facet_id]):
            if label in added_labels:
                continue
            entries.append((label, legend_handles[label]))
            added_labels.add(label)

    for label in sorted(extra_labels, key=str.lower):
        if label in added_labels:
            continue
        entries.append((label, legend_handles[label]))
        added_labels.add(label)

    for label in _sorted_background_labels(artifact, background_labels):
        if label in added_labels:
            continue
        entries.append((label, legend_handles[label]))
        added_labels.add(label)

    for label, handle in legend_handles.items():
        if label in added_labels:
            continue
        entries.append((label, handle))
        added_labels.add(label)

    return entries


def _display_legend_entries(
    artifact: GeometryArtifact,
    legend_handles: dict[str, Any],
) -> list[tuple[str, Any]]:
    return [
        (LEGEND_HEADER_LABEL, _legend_header_handle()),
        *_ordered_facet_family_legend_entries(artifact, legend_handles),
    ]


def _ordered_facet_ids(artifact: GeometryArtifact) -> list[str]:
    if artifact.facets_by_id:
        return list(artifact.facets_by_id.keys())
    label_groups = _label_groups(artifact)
    return [
        artifact.label_ids[indices[0]]
        for _, indices in label_groups.items()
        if artifact.is_gold[indices[0]]
    ]


def _legend_header_handle() -> Any:
    from matplotlib.lines import Line2D

    return Line2D([], [], linestyle='none', linewidth=0, marker=None, color='none')  # type: ignore


def _is_child_legend_label(artifact: GeometryArtifact, label: str) -> bool:
    idx = _first_index_for_label(artifact, label)
    if idx is None:
        return False
    return not artifact.is_gold[idx] and not _is_background_outlier_point(artifact, idx)


def _sorted_distractor_labels(artifact: GeometryArtifact, labels: list[str]) -> list[str]:
    return sorted(
        labels,
        key=lambda label: (
            _distractor_type_sort_key(_distractor_type_for_label(artifact, label)),
            label.lower(),
        ),
    )


def _sorted_background_labels(artifact: GeometryArtifact, labels: list[str]) -> list[str]:
    return sorted(
        labels,
        key=lambda label: ((_cluster_id_for_label(artifact, label) or label), label.lower()),
    )


def _plot_query_map_ax(
    ax: Any, artifact: GeometryArtifact, *, include_title_prefix: bool = True
) -> None:
    palette = label_palette(artifact)
    label_groups = _label_groups(artifact)
    for label, idx in label_groups.items():
        first_idx = idx[0]
        if _is_background_outlier_point(artifact, first_idx):
            continue
        marker = _label_marker(artifact, first_idx)
        scatter_kwargs: dict[str, Any] = {
            's': 34,
            'alpha': 0.78,
            'color': palette[label],
            'marker': marker,
            'label': label,
        }
        if marker == 'x':
            scatter_kwargs['linewidths'] = 1.15
        else:
            scatter_kwargs['edgecolors'] = 'none'
        ax.scatter(
            artifact.coords[idx, 0],
            artifact.coords[idx, 1],
            **scatter_kwargs,
        )
    _plot_background_outliers(
        ax,
        artifact,
        palette=palette,
        label_groups=label_groups,
    )
    draw_query_marker(ax, artifact)
    title = (
        f'Candidate-pool map ({artifact.reduction_method}, n={len(artifact.candidate_chunk_ids)})'
    )
    ax.set_title(_axis_title(artifact, title, include_title_prefix=include_title_prefix))
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def _plot_query_similarity_map_ax(
    ax: Any, artifact: GeometryArtifact, *, include_title_prefix: bool = True
) -> Any:
    return _plot_query_similarity_points(
        ax=ax,
        artifact=artifact,
        coords=artifact.coords,
        query_coord=artifact.query_coord,
        sim_to_query=artifact.sim_to_query,
        title=f'Map colored by query cosine ({artifact.reduction_method}, n={len(artifact.coords)})',
        x_label='dim 1',
        y_label='dim 2',
        include_title_prefix=include_title_prefix,
    )


def _plot_query_similarity_points(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    coords: NDArray[np.float32],
    query_coord: NDArray[np.float32],
    sim_to_query: NDArray[np.float32],
    title: str,
    x_label: str,
    y_label: str,
    include_title_prefix: bool,
) -> Any:
    order = np.arange(len(artifact.candidate_chunk_ids))
    plot_coords = coords[order]
    plot_sims = sim_to_query[order]
    label_groups = _label_groups(artifact)

    vmin = float(plot_sims.min()) if len(plot_sims) else 0.0
    vmax = float(plot_sims.max()) if len(plot_sims) else 1.0
    points = None
    for label, indices in label_groups.items():
        idx = np.array(indices, dtype=int)
        first_idx = indices[0]
        marker = _label_marker(artifact, first_idx)
        if _is_background_outlier_point(artifact, first_idx):
            points = _scatter_circled_x(
                ax,
                plot_coords[idx, 0],
                plot_coords[idx, 1],
                color_values=plot_sims[idx],
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                s=BACKGROUND_OUTLIER_SIMILARITY_MAP_MARKER_SIZE,
                alpha=0.96,
                linewidth=0.9,
                zorder=5,
                label=label,
            )
            continue
        scatter_kwargs: dict[str, Any] = {
            'c': plot_sims[idx],
            'cmap': 'viridis',
            'vmin': vmin,
            'vmax': vmax,
            'alpha': 0.92,
            'marker': marker,
            'zorder': 4,
            'label': label,
        }
        if marker == 'o':
            scatter_kwargs.update({'s': 44, 'edgecolors': 'none'})
        elif marker == 'D':
            scatter_kwargs.update({'s': 52, 'edgecolors': 'black', 'linewidths': 0.6})
        else:
            scatter_kwargs.update({'s': 38, 'linewidths': 1.25})
        points = ax.scatter(
            plot_coords[idx, 0],
            plot_coords[idx, 1],
            **scatter_kwargs,
        )
    if points is None:
        points = ax.scatter([], [], c=[], cmap='viridis', vmin=vmin, vmax=vmax)
    _draw_query_marker_at(ax, query_coord)
    ax.set_title(_axis_title(artifact, title, include_title_prefix=include_title_prefix))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _apply_coord_limits(ax, coords, query_coord)
    ax.grid(alpha=0.18)
    return points


def _plot_query_rank_ax(
    ax: Any, artifact: GeometryArtifact, *, include_title_prefix: bool = True
) -> None:
    ranks = np.arange(1, len(artifact.sim_to_query) + 1)
    palette = label_palette(artifact)
    label_groups = _label_groups(artifact)
    for label, idx in label_groups.items():
        first_idx = idx[0]
        if _is_background_outlier_point(artifact, first_idx):
            continue
        marker = _label_marker(artifact, first_idx)
        scatter_kwargs: dict[str, Any] = {
            'color': palette[label],
            's': 28,
            'alpha': 0.78,
            'marker': marker,
            'label': label,
        }
        if marker == 'x':
            scatter_kwargs['linewidths'] = 1.15
        else:
            scatter_kwargs['edgecolors'] = 'none'
        ax.scatter(
            ranks[idx],
            artifact.sim_to_query[idx],
            **scatter_kwargs,
        )
    _plot_background_outliers_rank(
        ax,
        artifact,
        ranks=ranks,
        palette=palette,
        label_groups=label_groups,
    )
    ax.axvline(artifact.k, color='black', lw=1.0, ls='--', alpha=0.7, label=f'k={artifact.k}')
    ax.set_xlabel('rank by query cosine')
    ax.set_ylabel('query cosine similarity')
    ax.set_title(
        _axis_title(artifact, 'Query-similarity ranking', include_title_prefix=include_title_prefix)
    )
    ax.grid(axis='y', alpha=0.25)


def _plot_discovered_clusters_ax(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    include_title_prefix: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    coords = artifact.coords
    cluster_labels = artifact.cluster_labels
    unique = sorted(set(int(x) for x in cluster_labels))
    cmap = plt.get_cmap('tab20')  # type: ignore
    for idx, cluster_id in enumerate(unique):
        mask = cluster_labels == cluster_id
        color = '#bbbbbb' if cluster_id == -1 else cmap(idx % 20)
        label = 'noise' if cluster_id == -1 else f'cluster {cluster_id}'
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=34,
            alpha=0.78,
            color=color,
            label=label,
            edgecolors='none',
        )
    draw_query_marker(ax, artifact)
    ax.set_title(
        _axis_title(artifact, 'HDBSCAN clusters', include_title_prefix=include_title_prefix)
    )
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    _apply_embedding_limits(ax, artifact)
    ax.grid(alpha=0.18)


def label_palette(artifact: GeometryArtifact) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    label_groups = _label_groups(artifact)
    gold_labels = [label for label, indices in label_groups.items() if artifact.is_gold[indices[0]]]
    gold_label_by_facet_id = {
        artifact.label_ids[label_groups[label][0]]: label for label in gold_labels
    }
    gold_cmap = plt.get_cmap('tab20')  # type: ignore
    facet_palette = {
        facet_id: gold_cmap(index % 20)
        for index, facet_id in enumerate(_ordered_facet_ids(artifact))
        if facet_id in gold_label_by_facet_id
    }
    background_cluster_ids = sorted({
        _cluster_id_for_index(artifact, indices[0]) or label
        for label, indices in label_groups.items()
        if _is_background_outlier_point(artifact, indices[0])
    })
    background_cmap = plt.get_cmap('Dark2')  # type: ignore
    background_palette = {
        cluster_id: background_cmap(i % max(1, background_cmap.N))
        for i, cluster_id in enumerate(background_cluster_ids)
    }
    palette: dict[str, Any] = {}
    for label in gold_labels:
        facet_id = artifact.label_ids[label_groups[label][0]]
        palette[label] = facet_palette.get(facet_id, gold_cmap(len(palette) % 20))
    for label, indices in label_groups.items():
        if label in palette:
            continue
        first_idx = indices[0]
        if _is_background_outlier_point(artifact, first_idx):
            cluster_id = _cluster_id_for_index(artifact, first_idx) or label
            palette[label] = background_palette.get(cluster_id, BACKGROUND_OUTLIER_COLOR)
            continue
        target_facet_id = _target_facet_id_for_index(artifact, first_idx)
        if target_facet_id is not None and target_facet_id in facet_palette:
            palette[label] = facet_palette[target_facet_id]
            continue

        raise RuntimeError('Unexpected distractor without parent facet_id')
    return palette


def _label_groups(artifact: GeometryArtifact) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, label in enumerate(artifact.labels):
        grouped.setdefault(label, []).append(idx)
    return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))


def _label_marker(artifact: GeometryArtifact, idx: int) -> str:
    if _is_background_outlier_point(artifact, idx):
        return 'x'
    if _is_same_condition_wrong_axis_point(artifact, idx):
        return 'D'
    return 'x' if not artifact.is_gold[idx] else 'o'


def _is_background_outlier_point(artifact: GeometryArtifact, idx: int) -> bool:
    label_id = artifact.label_ids[idx]
    role = artifact.roles[idx]
    return label_id == BACKGROUND_OUTLIER_LABEL_ID or role == BACKGROUND_OUTLIER_ROLE


def _is_same_condition_wrong_axis_point(artifact: GeometryArtifact, idx: int) -> bool:
    label_id = artifact.label_ids[idx]
    role = artifact.roles[idx]
    return label_id == SAME_CONDITION_WRONG_AXIS_LABEL_ID or role == SAME_CONDITION_WRONG_AXIS_ROLE


def _distractor_type_for_index(artifact: GeometryArtifact, idx: int) -> str:
    chunk_id = artifact.candidate_chunk_ids[idx]
    qrel = artifact.qrel_by_chunk_id.get(chunk_id)
    return qrel.distractor_type if qrel is not None and qrel.distractor_type else 'hard_distractor'


def _cluster_id_for_index(artifact: GeometryArtifact, idx: int) -> str | None:
    chunk_id = artifact.candidate_chunk_ids[idx]
    qrel = artifact.qrel_by_chunk_id.get(chunk_id)
    return qrel.cluster_id if qrel is not None else None


def _target_facet_id_for_index(artifact: GeometryArtifact, idx: int) -> str | None:
    chunk_id = artifact.candidate_chunk_ids[idx]
    qrel = artifact.qrel_by_chunk_id.get(chunk_id)
    return qrel.target_facet_id if qrel is not None else None


def _first_index_for_label(artifact: GeometryArtifact, label: str) -> int | None:
    for idx, candidate_label in enumerate(artifact.labels):
        if candidate_label == label:
            return idx
    return None


def _distractor_type_for_label(artifact: GeometryArtifact, label: str) -> str:
    idx = _first_index_for_label(artifact, label)
    if idx is None:
        return 'hard_distractor'
    return _distractor_type_for_index(artifact, idx)


def _cluster_id_for_label(artifact: GeometryArtifact, label: str) -> str | None:
    idx = _first_index_for_label(artifact, label)
    if idx is None:
        return None
    return _cluster_id_for_index(artifact, idx)


def _distractor_type_sort_key(distractor_type: str) -> tuple[int, str]:
    order = {
        'same_axis_wrong_condition': 0,
        'same_condition_wrong_axis': 1,
        'same_condition_wrong_subgroup': 2,
        'same_subgroup_wrong_condition': 3,
        'hard_distractor': 4,
    }
    return (order.get(distractor_type, len(order)), distractor_type)


def _background_outlier_indices(artifact: GeometryArtifact) -> list[int]:
    return [
        idx
        for idx in range(len(artifact.candidate_chunk_ids))
        if _is_background_outlier_point(artifact, idx)
    ]


def _plot_background_outliers(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    palette: dict[str, Any] | None,
    label_groups: dict[str, list[int]],
    fallback_color: str | None = None,
) -> None:
    coords = artifact.coords
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if not _is_background_outlier_point(artifact, first_idx):
            continue
        _scatter_circled_x(
            ax,
            coords[indices, 0],
            coords[indices, 1],
            color=(palette or {}).get(label, fallback_color or BACKGROUND_OUTLIER_COLOR),
            s=BACKGROUND_OUTLIER_MAP_MARKER_SIZE,
            alpha=0.95,
            label=label,
            linewidth=0.75,
            zorder=5,
        )


def _plot_background_outliers_for_selection(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    selected: set[int],
    palette: dict[str, Any],
    label_groups: dict[str, list[int]],
) -> None:
    coords = artifact.coords
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if not _is_background_outlier_point(artifact, first_idx):
            continue
        selected_idx = [point_idx for point_idx in indices if point_idx in selected]
        unselected_idx = [point_idx for point_idx in indices if point_idx not in selected]
        if unselected_idx:
            _scatter_circled_x(
                ax,
                coords[unselected_idx, 0],
                coords[unselected_idx, 1],
                color=UNSELECTED_DATA_POINT_COLOR,
                s=BACKGROUND_OUTLIER_SELECTION_UNSELECTED_MARKER_SIZE,
                alpha=0.58,
                linewidth=0.55,
                zorder=3,
            )
        if selected_idx:
            _scatter_circled_x(
                ax,
                coords[selected_idx, 0],
                coords[selected_idx, 1],
                color=palette[label],
                s=BACKGROUND_OUTLIER_SELECTION_SELECTED_MARKER_SIZE,
                alpha=0.95,
                label=label,
                linewidth=0.75,
                zorder=5,
            )


def _plot_background_outliers_rank(
    ax: Any,
    artifact: GeometryArtifact,
    *,
    ranks: NDArray[np.int_],
    palette: dict[str, Any],
    label_groups: dict[str, list[int]],
) -> None:
    for label, indices in label_groups.items():
        first_idx = indices[0]
        if not _is_background_outlier_point(artifact, first_idx):
            continue
        _scatter_circled_x(
            ax,
            ranks[indices],
            artifact.sim_to_query[indices],
            color=palette[label],
            s=BACKGROUND_OUTLIER_RANK_MARKER_SIZE,
            alpha=0.94,
            label=label,
            linewidth=0.75,
            zorder=5,
        )


def _scatter_circled_x(
    ax: Any,
    x: Any,
    y: Any,
    *,
    color: str | None = None,
    color_values: Any = None,
    circle_edgecolors: Any = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    s: Any = 54,
    alpha: float = 0.95,
    label: str | None = None,
    linewidth: float = 0.75,
    zorder: int = 5,
) -> Any:
    if color_values is not None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap or 'viridis')  # type: ignore
        circle_kwargs = {
            'edgecolors': (
                circle_edgecolors if circle_edgecolors is not None else cmap_obj(norm(color_values))
            )
        }
        x_kwargs = {'c': color_values, 'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
    else:
        circle_kwargs = {'edgecolors': color}
        x_kwargs = {'color': color}
    circle = ax.scatter(
        x,
        y,
        s=s,
        alpha=alpha,
        marker='o',
        facecolors='none',
        linewidths=linewidth,
        label=label,
        zorder=zorder,
        **circle_kwargs,
    )
    x_points = ax.scatter(
        x,
        y,
        s=np.asarray(s) * 0.46,
        alpha=alpha,
        marker='x',
        linewidths=linewidth,
        zorder=zorder + 0.1,
        **x_kwargs,
    )
    return x_points if color_values is not None else circle


def draw_query_marker(ax: Any, artifact: GeometryArtifact) -> None:
    _draw_query_marker_at(ax, artifact.query_coord)


def _add_axes_in_inches(
    fig: Any,
    *,
    fig_width: float,
    fig_height: float,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> Any:
    return fig.add_axes([
        left / fig_width,
        bottom / fig_height,
        width / fig_width,
        height / fig_height,
    ])


def _draw_query_marker_at(ax: Any, query_coord: NDArray[np.float32]) -> None:
    ax.scatter(
        [query_coord[0]],
        [query_coord[1]],
        marker='*',
        s=220,
        color='black',
        edgecolors='white',
        linewidths=0.8,
        label='query',
        zorder=6,
    )


def artifact_title_prefix(artifact: GeometryArtifact) -> str:
    condition = artifact.query.condition_display or artifact.query.condition_id
    return f'{artifact.query_id} - {condition}'


def _artifact_figure_title_parts(artifact: GeometryArtifact) -> tuple[str, str, str]:
    condition = artifact.query.condition_display or artifact.query.condition_id or 'unknown condition'
    subgroup_labels = _artifact_subgroup_labels(artifact)
    axis_labels = (
        str(artifact.query.primary_axis).replace('_', ' '),
        str(artifact.query.secondary_axis).replace('_', ' '),
    )
    subgroup_text = ' vs. '.join(subgroup_labels)
    axis_text = ' & '.join(axis_labels)
    prefix = f'{artifact.query_id} - {condition} / '
    return prefix, subgroup_text, f' / {axis_text}'


def _artifact_subgroup_labels(artifact: GeometryArtifact) -> tuple[str, str]:
    facets_json = artifact.query.facets_json
    if facets_json:
        subgroup_labels: list[str] = []
        seen_labels: set[str] = set()
        for facet in json.loads(facets_json):
            subgroup_label = str(facet.get('subgroup_label') or '').strip()
            if subgroup_label and subgroup_label not in seen_labels:
                subgroup_labels.append(subgroup_label)
                seen_labels.add(subgroup_label)
            if len(subgroup_labels) == 2:
                return subgroup_labels[0], subgroup_labels[1]
    return 'unknown subgroup', 'unknown subgroup'


def _draw_artifact_figure_title(
    fig: Any,
    artifact: GeometryArtifact,
    *,
    fontsize: int,
    y: float = 0.985,
) -> None:
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    prefix, subgroup_text, axis_suffix = _artifact_figure_title_parts(artifact)
    title_box = HPacker(
        children=[
            TextArea(prefix, textprops={'fontsize': fontsize, 'color': 'black'}),
            TextArea(
                subgroup_text,
                textprops={'fontsize': fontsize, 'color': QUERY_TITLE_SUBGROUP_COLOR},
            ),
            TextArea(
                axis_suffix,
                textprops={'fontsize': fontsize, 'color': QUERY_TITLE_AXIS_COLOR},
            ),
        ],
        align='center',
        pad=0,
        sep=0,
    )
    anchored = AnchoredOffsetbox(
        loc='upper center',
        child=title_box,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        bbox_transform=fig.transFigure,
        borderpad=0.0,
    )
    fig.add_artist(anchored)


def _axis_title(artifact: GeometryArtifact, title: str, *, include_title_prefix: bool) -> str:
    if include_title_prefix:
        return f'{artifact_title_prefix(artifact)}\n{title}'
    return title


def _apply_embedding_limits(ax: Any, artifact: GeometryArtifact) -> None:
    _apply_coord_limits(ax, artifact.coords, artifact.query_coord)


def _apply_coord_limits(
    ax: Any,
    coords: NDArray[np.float32],
    query_coord: NDArray[np.float32],
) -> None:
    x_left, x_right, y_bottom, y_top = _coord_limits(coords, query_coord)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)


def _coord_limits(
    coords: NDArray[np.float32],
    query_coord: NDArray[np.float32],
) -> tuple[float, float, float, float]:
    coords_all = np.vstack([coords, query_coord[None, :]])
    x_min, y_min = coords_all.min(axis=0)
    x_max, y_max = coords_all.max(axis=0)
    x_pad = max(float(x_max - x_min) * 0.05, 0.5)
    y_pad = max(float(y_max - y_min) * 0.05, 0.5)
    return (
        float(coords_all[:, 0].min()) - x_pad,
        float(coords_all[:, 0].max()) + x_pad,
        float(coords_all[:, 1].min()) - y_pad,
        float(coords_all[:, 1].max()) + y_pad,
    )


def strategy_title(strategy: str, lam: float | None) -> str:
    if strategy == 'top_k':
        return 'top-k'
    label = 'MMR' if strategy == 'mmr' else 'Coverage'
    return f'{label} lambda={lam:.2f}' if lam is not None else label
