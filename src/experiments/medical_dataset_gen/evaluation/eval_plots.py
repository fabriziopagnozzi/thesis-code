from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from matplotlib.figure import Figure
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.eval_plots_configs import (
    DEFAULT_PLOT_GRID_LAYOUTS,
    LAMBDA_POLICY_NOTE,
    PLOT_METRIC_TITLES,
    PLOTTED_ANSWER_ROUGE_METRIC_NAMES,
    PLOTTED_DIAGNOSTIC_METRIC_NAMES,
    PLOTTED_MAIN_METRIC_NAMES,
    PRIMARY_DESC,
    PRIMARY_SORT,
    STRATEGY_STYLE,
    EvalPlotFileName,
    NamedPlotMetric,
)
from experiments.medical_dataset_gen.evaluation.retrieval_utils import ci_half_width
from experiments.medical_dataset_gen.schemas.metrics_schemas import METRIC_NAME_TO_FIELD


def plot_strategy_comparison(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Metric-vs-k lines.

    Top-k is shown as the baseline. Diversity methods show only the selected
    lambda* path for each k, with the exact lambda annotated at each marker.
    """
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)
    metrics = _available_metrics(stats_df, results_df)
    fig, axes = _grid_figure('strategy_comparison', sharex=True)

    for ax, (stats_col, result_col, title, higher_is_better) in zip(
        axes.flatten(),
        metrics,
        strict=False,
    ):
        for strategy in strategies:
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            if strategy == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [_cell_value(sub, k, None, stats_col) for k in xs]
                ci = [
                    ci_half_width(_query_vals(results_df, strategy, k, None, result_col))
                    for k in xs
                ]
                _plot_ci_line(ax, xs, ys, ci, style, lw=2.0, label=style['label'], zorder=3)
                continue

            best_df = _best_lam_rows(stats_df, strategy, k_values)
            if best_df.height == 0:
                continue
            xs = best_df['k'].to_list()
            ys = [float(v) for v in best_df[stats_col].to_list()]
            lams = [float(v) for v in best_df['lam'].to_list()]
            k_to_lam = dict(zip(xs, lams, strict=True))
            ci = [
                ci_half_width(_query_vals(results_df, strategy, k, k_to_lam.get(k), result_col))
                for k in xs
            ]
            ax.plot(
                xs,
                ys,
                color=style['color'],
                ls=style['ls'],
                lw=2.0,
                label=style['label'],
                zorder=3,
            )
            ax.scatter(
                xs,
                ys,
                s=42,
                facecolors='white',
                edgecolors=style['color'],
                linewidths=1.4,
                zorder=4,
            )
            _plot_error_caps(ax, xs, ys, ci, style, zorder=2)
            _annotate_lambda_points(
                ax,
                xs,
                ys,
                lams,
                color=style['color'],
                placement='above' if strategy == 'mmr' else 'below',
            )

        ax.set_title(_panel_title(title, higher_is_better), fontsize=10)
        ax.set_ylabel(stats_col, fontsize=9)
        _set_k_tick_labels(ax, k_values)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelbottom=True)

    for ax in axes[-1]:
        ax.set_xlabel('k', fontsize=9)
    for ax in axes.flatten()[len(metrics) :]:
        ax.set_visible(False)

    _figure_legend(fig, axes.flatten())
    fig.suptitle(
        'Strategy comparison - lambda* path per strategy with exact lambda labels at each k',
        fontsize=12,
    )
    _figure_note(
        fig,
        f'{LAMBDA_POLICY_NOTE}; see lambda_sensitivity_metrics.png for the full sweep',
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'strategy_comparison.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_lambda_sensitivity_metrics(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Main benchmark metrics as lambda changes."""
    _plot_lambda_sensitivity(
        stats_df,
        out_dir,
        plot_name='lambda_sensitivity_metrics',
        metric_names=PLOTTED_MAIN_METRIC_NAMES,
        figure_title='Lambda sensitivity (main metrics) - each row is a metric, each column is a strategy',
        output_filename='lambda_sensitivity_metrics.png',
    )


def plot_lambda_sensitivity_diagnostics(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Diagnostic metrics as lambda changes."""
    _plot_lambda_sensitivity(
        stats_df,
        out_dir,
        plot_name='lambda_sensitivity_diagnostics',
        metric_names=PLOTTED_DIAGNOSTIC_METRIC_NAMES,
        figure_title='Lambda sensitivity (diagnostics) - each row is a metric, each column is a strategy',
        output_filename='lambda_sensitivity_diagnostics.png',
    )


def _plot_lambda_sensitivity(
    stats_df: pl.DataFrame,
    out_dir: Path,
    *,
    plot_name: EvalPlotFileName,
    metric_names: list[str],
    figure_title: str,
    output_filename: str,
) -> None:
    """Shared lambda-sensitivity renderer for fixed metric groups."""
    import matplotlib.pyplot as plt

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    lambda_values = _lambda_values(stats_df)
    sampled_lambda_values = _sample_tick_values(lambda_values)
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    metric_cols = [
        (metric.stats_col, metric.title)
        for metric in _available_metric_specs(
            stats_df,
            results_df=None,
            metric_names=metric_names,
            require_result_col=False,
        )
    ]
    if not metric_cols:
        return
    cmap = plt.get_cmap('viridis')  # type: ignore[attr-defined]
    k_colors = {k: cmap(i / max(len(k_values) - 1, 1)) for i, k in enumerate(k_values)}

    fig, axes = _grid_figure(
        plot_name,
        rows=len(metric_cols),
        cols=len(diversity_strategies),
        sharex=True,
    )

    for row_idx, (metric, metric_title) in enumerate(metric_cols):
        row_ylim = _shared_lambda_sensitivity_ylim(
            stats_df,
            metric=metric,
            strategies=diversity_strategies,
            k_values=k_values,
        )
        for col_idx, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            ax = axes[row_idx][col_idx]
            for k in k_values:
                ksub = sub.filter(pl.col('k') == k).sort('lam')
                if ksub.height == 0:
                    continue
                ax.plot(
                    ksub['lam'].to_list(),
                    ksub[metric].to_list(),
                    color=k_colors[k],
                    ls=style['ls'],
                    lw=1.8,
                    marker='o',
                    ms=4,
                    label=f'k={k}',
                )
                ref = topk_df.filter(pl.col('k') == k)
                if ref.height > 0:
                    ax.axhline(float(ref[metric][0]), color=k_colors[k], ls='--', lw=1.0, alpha=0.5)

            if row_idx == 0:
                ax.set_title(style['label'], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(metric_title, fontsize=10)
            if row_idx == len(metric_cols) - 1:
                ax.set_xlabel('lambda', fontsize=9)
            _set_lambda_tick_labels(ax, sampled_lambda_values)
            ax.set_ylim(*row_ylim)
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=len(k_values),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.suptitle(figure_title, fontsize=12)
    _figure_note(
        fig, 'Dashed horizontal lines are the top-k reference at each k; line colors identify k'
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / output_filename, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_strategy_comparison_heatmap(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Static heatmaps across k x lambda for FacLoc, MMR, and FacLoc advantage."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize, TwoSlopeNorm

    heatmap_data = _build_strategy_lambda_heatmap_data(stats_df, results_df)
    if heatmap_data is None:
        return
    metric_matrices, k_values, lambda_values = heatmap_data
    metrics = _available_metrics(stats_df, results_df)
    fig = plt.figure(figsize=(15.5, 2.7 * len(metrics) + 1.8))
    grid = fig.add_gridspec(
        nrows=len(metrics),
        ncols=7,
        # Dedicated spacer columns prevent repeated axis labels and colorbar
        # labels from encroaching on the neighboring heatmaps.
        width_ratios=[1.0, 0.08, 1.0, 0.035, 0.16, 1.0, 0.035],
        wspace=0.08,
        hspace=0.48,
    )

    panel_titles = ['FacLoc', 'MMR', 'FacLoc Advantage']

    annotate_cells = len(k_values) * len(lambda_values) <= 42
    sampled_lambda_tick_positions = _sample_tick_indices(len(lambda_values), max_ticks=6)
    for row_idx, (stats_col, _result_col, title, higher_is_better) in enumerate(metrics):
        matrices = metric_matrices[stats_col]
        raw_min = float(
            np.nanmin(np.concatenate([matrices['fac_loc'].ravel(), matrices['mmr'].ravel()]))
        )
        raw_max = float(
            np.nanmax(np.concatenate([matrices['fac_loc'].ravel(), matrices['mmr'].ravel()]))
        )
        adv_abs_max = float(np.nanmax(np.abs(matrices['advantage'])))
        raw_norm = Normalize(vmin=raw_min, vmax=raw_max)
        adv_norm = TwoSlopeNorm(vmin=-adv_abs_max, vcenter=0.0, vmax=adv_abs_max)

        fac_ax = fig.add_subplot(grid[row_idx, 0])
        mmr_ax = fig.add_subplot(grid[row_idx, 2])
        raw_cax = fig.add_subplot(grid[row_idx, 3])
        adv_ax = fig.add_subplot(grid[row_idx, 5])
        adv_cax = fig.add_subplot(grid[row_idx, 6])
        row_axes = [fac_ax, mmr_ax, adv_ax]
        if row_idx == 0:
            for ax, panel_title in zip(row_axes, panel_titles, strict=True):
                ax.set_title(panel_title, fontsize=10)

        raw_image = None
        adv_image = None
        for col_idx, (ax, key) in enumerate(
            zip(row_axes, ['fac_loc', 'mmr', 'advantage'], strict=True)
        ):
            matrix = matrices[key]
            image = ax.imshow(
                matrix,
                origin='lower',
                aspect='auto',
                cmap='viridis' if key != 'advantage' else 'RdBu_r',
                norm=raw_norm if key != 'advantage' else adv_norm,
                interpolation='nearest',
            )
            if key != 'advantage' and raw_image is None:
                raw_image = image
            if key == 'advantage':
                adv_image = image
            if col_idx == 0:
                ax.set_ylabel(_panel_title(title, higher_is_better), fontsize=9)
            else:
                ax.set_ylabel('')
            _set_heatmap_axis_labels(
                ax,
                k_values,
                lambda_values,
                sampled_lambda_tick_positions=sampled_lambda_tick_positions,
            )
            if col_idx > 0:
                ax.tick_params(axis='y', labelleft=False, left=False)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel('lambda', fontsize=9)
            ax.tick_params(axis='x', labelbottom=True)
            _draw_heatmap_grid(ax, len(k_values), len(lambda_values))
            if annotate_cells:
                _annotate_heatmap_cells(ax, matrix, diverging=(key == 'advantage'))

        if raw_image is None or adv_image is None:
            continue
        raw_cbar = fig.colorbar(raw_image, cax=raw_cax)
        raw_cbar.ax.tick_params(labelsize=7)
        raw_cbar.set_label('metric value', fontsize=8)
        adv_cbar = fig.colorbar(adv_image, cax=adv_cax)
        adv_cbar.ax.tick_params(labelsize=7)
        adv_cbar.set_label('positive = FacLoc better', fontsize=8)

    fig.suptitle(
        'Strategy comparison heatmap - k x lambda',
        fontsize=12,
    )
    _figure_note(
        fig,
        (
            'Rows are metrics; columns are FacLoc, MMR, and FacLoc advantage. '
            'For lower-better metrics, advantage is computed so positive still means FacLoc better.'
        ),
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.92)
    fig.savefig(out_dir / 'strategy_comparison_heatmap.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_strategy_comparison_heatmap_html(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Interactive HTML explorer for strategy-vs-lambda heatmaps."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    heatmap_data = _build_strategy_lambda_heatmap_data(stats_df, results_df)
    if heatmap_data is None:
        return

    metric_matrices, k_values, lambda_values = heatmap_data
    metrics = _available_metrics(stats_df, results_df)
    subplot_titles = ['FacLoc', 'MMR', 'FacLoc Advantage']
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.09,
    )

    button_defs: list[dict[str, object]] = []
    total_traces = len(metrics) * 3
    first_metric_title = metrics[0][2]
    first_metric_higher = metrics[0][3]
    for metric_idx, (stats_col, _result_col, title, higher_is_better) in enumerate(metrics):
        matrices = metric_matrices[stats_col]
        raw_min = min(
            float(matrices['fac_loc'].min()),
            float(matrices['mmr'].min()),
        )
        raw_max = max(
            float(matrices['fac_loc'].max()),
            float(matrices['mmr'].max()),
        )
        advantage_abs_max = max(
            abs(float(matrices['advantage'].min())), abs(float(matrices['advantage'].max()))
        )
        for col_idx, key in enumerate(['fac_loc', 'mmr', 'advantage'], start=1):
            matrix = matrices[key]
            hovertemplate = _plotly_heatmap_hovertemplate(
                title=title,
                panel_title=subplot_titles[col_idx - 1],
                higher_is_better=higher_is_better,
                advantage=(key == 'advantage'),
            )
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=[f'{lam:.2f}' for lam in lambda_values],
                    y=[f'k={k}' for k in k_values],
                    colorscale='Viridis' if key != 'advantage' else 'RdBu_r',
                    zmin=raw_min if key != 'advantage' else -advantage_abs_max,
                    zmax=raw_max if key != 'advantage' else advantage_abs_max,
                    zmid=0.0 if key == 'advantage' else None,
                    showscale=(col_idx == 1) or (key == 'advantage'),
                    colorbar=(
                        {'title': 'metric value', 'x': 0.63, 'len': 0.78}
                        if col_idx == 1
                        else (
                            {'title': 'positive = FacLoc better', 'x': 1.02, 'len': 0.78}
                            if key == 'advantage'
                            else None
                        )
                    ),
                    visible=metric_idx == 0,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=col_idx,
            )

        visibility = [False] * total_traces
        base_idx = metric_idx * 3
        visibility[base_idx : base_idx + 3] = [True, True, True]
        button_defs.append(
            {
                'label': title,
                'method': 'update',
                'args': [
                    {'visible': visibility},
                    {
                        'title': (
                            f'Strategy comparison heatmap - {_panel_title(title, higher_is_better)}'
                        )
                    },
                ],
            }
        )

    fig.update_layout(
        title=f'Strategy comparison heatmap - {_panel_title(first_metric_title, first_metric_higher)}',
        updatemenus=[
            {
                'buttons': button_defs,
                'direction': 'down',
                'showactive': True,
                'x': 0.0,
                'xanchor': 'left',
                'y': 1.18,
                'yanchor': 'top',
            }
        ],
        margin={'l': 65, 'r': 90, 't': 110, 'b': 70},
    )
    for col_idx in range(1, 4):
        fig.update_xaxes(title_text='lambda', row=1, col=col_idx)
        fig.update_yaxes(title_text='k', row=1, col=col_idx)

    fig.write_html(
        out_dir / 'strategy_comparison_heatmap.html',
        include_plotlyjs=True,
        full_html=True,
    )


def plot_per_query_distributions(results_df: pl.DataFrame, out_dir: Path) -> None:
    """Violin plots at the top-k best k and each strategy's best primary lambda."""
    import matplotlib.pyplot as plt
    import numpy as np

    best_k = _best_topk_k(results_df)
    strategies = _ordered_strategies(results_df)
    metric_cols = [
        ('facet_coverage', 'MeanFacetHitRate@k'),
        ('weighted_facet_coverage', 'MeanFacetRecall@k'),
        ('distractor_rate', 'DistractorRate@k'),
        ('gold_recall', 'Recall@k'),
        ('alpha_ndcg', 'alpha-nDCG@k'),
        ('answer_rouge2_recall', 'Answer ROUGE-2 Recall@k'),
    ]
    metric_cols = [(col, title) for col, title in metric_cols if col in results_df.columns]

    per_strategy_data: dict[str, dict[str, list[float]]] = {}
    for strategy in strategies:
        slice_df = _best_result_slice(results_df, strategy, best_k)
        per_strategy_data[strategy] = {
            col: [float(v) for v in slice_df[col].drop_nulls().to_list()] for col, _ in metric_cols
        }

    labels = [get_style(s)['label'] for s in strategies]
    colors = [get_style(s)['color'] for s in strategies]
    fig, axes = _grid_figure('per_query_distributions')

    for ax, (col, title) in zip(axes.flatten(), metric_cols, strict=False):
        data = [per_strategy_data[s][col] for s in strategies]
        positions = [i for i, values in enumerate(data) if len(values) > 1]
        violin_data = [data[i] for i in positions]
        if violin_data:
            parts = ax.violinplot(
                violin_data,
                positions=positions,
                showmedians=False,
                showextrema=False,
            )
            for body, pos in zip(parts['bodies'], positions, strict=True):
                body.set_facecolor(colors[pos])
                body.set_alpha(0.65)
        medians = [float(np.median(values)) if values else float('nan') for values in data]
        ax.scatter(
            range(len(strategies)), medians, color='white', edgecolors='black', s=42, zorder=4
        )
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_title(f'{title} (k={best_k}, coverage-first lambda*)', fontsize=10)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    for ax in axes.flatten()[len(metric_cols) :]:
        ax.set_visible(False)

    fig.suptitle('Per-query score distributions', fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / 'per_query_distributions.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_gain_over_topk(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Paired delta bars relative to top-k.

    For lower-is-better diagnostics, negative bars are favorable.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    lambda_values = _lambda_values(stats_df)
    if not lambda_values:
        return
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    metrics = _available_metrics(stats_df, results_df)
    n_groups = len(diversity_strategies) * len(lambda_values)
    geometry = _gain_over_topk_geometry(len(k_values), n_groups)
    footer_layout = _gain_over_topk_footer_layout(lambda_values)
    x = np.asarray(geometry.k_positions, dtype=float)
    width_per_k = max(1.45, 0.08 * n_groups + 0.45)
    fig_width = len(k_values) * width_per_k + 1.4

    fig, axes = _grid_figure(
        'gain_over_topk',
        rows=len(metrics),
        cols=1,
        sharex=True,
        width_per_col=fig_width,
        footer_height=footer_layout.footer_height,
    )
    flat_axes = axes.flatten()
    for row_idx, (ax, (stats_col, result_col, title, higher_is_better)) in enumerate(
        zip(flat_axes, metrics, strict=False)
    ):
        extrema_by_strategy_k: dict[tuple[str, int], list[tuple[Any, float]]] = {}
        slot_centers: list[float] = []
        for i, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            for j, lam in enumerate(lambda_values):
                deltas: list[float] = []
                cis: list[float] = []
                for k in k_values:
                    ref_row = topk_df.filter(pl.col('k') == k)
                    ref_val = float(ref_row[stats_col][0]) if ref_row.height > 0 else 0.0
                    sub = stats_df.filter(
                        (pl.col('strategy') == strategy)
                        & (pl.col('k') == k)
                        & (pl.col('lam') == lam)
                    )
                    strat_val = float(sub[stats_col][0]) if sub.height > 0 else ref_val
                    deltas.append(strat_val - ref_val)
                    cis.append(_paired_delta_ci(results_df, strategy, k, lam, result_col))

                ci_display = [0.0 if np.isnan(c) else c for c in cis]
                group_idx = i * len(lambda_values) + j
                offset = _gain_over_topk_offset(
                    group_idx=group_idx,
                    n_groups=n_groups,
                    slot_width=geometry.slot_width,
                )
                color = _lambda_shade(style['color'], lam, lambda_values)
                bars = ax.bar(
                    x + offset,
                    deltas,
                    width=geometry.bar_width,
                    color=color,
                    alpha=0.9,
                    edgecolor='#222222',
                    linewidth=0.35,
                    yerr=ci_display,
                    capsize=2.5,
                    error_kw={'linewidth': 0.75, 'ecolor': color},
                )
                for bar, k, delta in zip(bars, k_values, deltas, strict=True):
                    extrema_by_strategy_k.setdefault((strategy, int(k)), []).append((bar, delta))
                    slot_centers.append(float(bar.get_x() + bar.get_width() / 2))

        ax.axhline(0, color='black', lw=0.8)
        _draw_bar_slot_guides(ax, slot_centers, geometry.bar_width)
        _annotate_extreme_delta_bars(ax, extrema_by_strategy_k)
        ax.set_title(
            _panel_title(f'Delta {title} vs top-k', higher_is_better, delta=True),
            fontsize=10,
        )
        ax.set_ylabel(f'Delta {stats_col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(geometry.x_min, geometry.x_max)
        ax.margins(x=0.0, y=0.18)
        _set_k_tick_labels(ax, k_values, positions=geometry.k_positions)
        ax.tick_params(labelbottom=True)
        if row_idx == len(metrics) - 1:
            ax.set_xlabel('k', fontsize=9)

    for ax in flat_axes[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(
        'Gain over top-k by strategy and lambda, paired 95% CI',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, footer_layout.tight_layout_bottom, 1, 1))
    _add_lambda_shade_legend(
        fig,
        diversity_strategies,
        lambda_values,
        bottom=footer_layout.legend_bottom,
        values_per_line=footer_layout.values_per_line,
    )
    fig.savefig(out_dir / 'gain_over_topk.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_gain_over_topk_html(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Interactive HTML companion for the full gain-over-top-k lambda sweep."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(int(k) for k in stats_df['k'].unique().to_list())
    lambda_values = _lambda_values(stats_df)
    if not lambda_values:
        return
    metrics = _available_metrics(stats_df, results_df)
    if not metrics:
        return

    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    geometry = _gain_over_topk_geometry(
        len(k_values), len(diversity_strategies) * len(lambda_values)
    )
    subplot_titles = [
        _panel_title(f'Delta {title} vs top-k', higher_is_better, delta=True)
        for _stats_col, _result_col, title, higher_is_better in metrics
    ]
    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=subplot_titles,
    )

    for row_idx, (stats_col, result_col, _title, higher_is_better) in enumerate(metrics, start=1):
        direction_note = (
            'Positive delta is favorable' if higher_is_better else 'Negative delta is favorable'
        )
        for strategy_idx, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            x_positions: list[float] = []
            deltas: list[float] = []
            ci_values: list[float] = []
            colors: list[str] = []
            customdata: list[list[float | int | str]] = []

            for lambda_idx, lam in enumerate(lambda_values):
                group_idx = strategy_idx * len(lambda_values) + lambda_idx
                offset = _gain_over_topk_offset(
                    group_idx=group_idx,
                    n_groups=len(diversity_strategies) * len(lambda_values),
                    slot_width=geometry.slot_width,
                )
                color_rgb = _lambda_shade(style['color'], lam, lambda_values)
                color_hex = _rgb_to_plotly_color(color_rgb)

                for k_position, k in zip(geometry.k_positions, k_values, strict=True):
                    ref_row = topk_df.filter(pl.col('k') == k)
                    ref_val = float(ref_row[stats_col][0]) if ref_row.height > 0 else 0.0
                    sub = stats_df.filter(
                        (pl.col('strategy') == strategy)
                        & (pl.col('k') == k)
                        & (pl.col('lam') == lam)
                    )
                    strat_val = float(sub[stats_col][0]) if sub.height > 0 else ref_val
                    delta = strat_val - ref_val
                    ci_value = _paired_delta_ci(results_df, strategy, k, lam, result_col)

                    x_positions.append(k_position + offset)
                    deltas.append(delta)
                    ci_values.append(0.0 if np.isnan(ci_value) else ci_value)
                    colors.append(color_hex)
                    customdata.append(
                        [
                            get_style(strategy)['label'],
                            lam,
                            k,
                            strat_val,
                            ref_val,
                            0.0 if np.isnan(ci_value) else ci_value,
                            direction_note,
                        ]
                    )

            fig.add_trace(
                go.Bar(
                    x=x_positions,
                    y=deltas,
                    width=[geometry.bar_width] * len(x_positions),
                    marker={
                        'color': colors,
                        'line': {'color': '#222222', 'width': 0.45},
                    },
                    error_y={
                        'type': 'data',
                        'array': ci_values,
                        'visible': True,
                        'thickness': 0.8,
                        'width': 2,
                        'color': style['color'],
                    },
                    customdata=customdata,
                    hovertemplate=(
                        '%{customdata[0]}<br>'
                        'k=%{customdata[2]}<br>'
                        'lambda=%{customdata[1]:.2f}<br>'
                        'delta=%{y:+.4f}<br>'
                        'strategy value=%{customdata[3]:.4f}<br>'
                        'top-k value=%{customdata[4]:.4f}<br>'
                        'paired 95% CI half-width=%{customdata[5]:.4f}<br>'
                        '%{customdata[6]}'
                        '<extra></extra>'
                    ),
                    name=style['label'],
                    legendgroup=strategy,
                    showlegend=row_idx == 1,
                    opacity=0.92,
                ),
                row=row_idx,
                col=1,
            )
        fig.add_hline(y=0.0, line_width=1.0, line_color='black', row=row_idx, col=1)  # type: ignore
        fig.update_yaxes(title_text=f'Delta {stats_col}', row=row_idx, col=1)

    for row_idx in range(1, len(metrics) + 1):
        fig.update_xaxes(
            tickmode='array',
            tickvals=geometry.k_positions,
            ticktext=[f'k={k}' for k in k_values],
            range=[geometry.x_min, geometry.x_max],
            row=row_idx,
            col=1,
        )
    fig.update_xaxes(title_text='k', row=len(metrics), col=1)
    fig.update_layout(
        title='Gain over top-k by strategy and lambda, paired 95% CI',
        barmode='overlay',
        height=250 * len(metrics) + 120,
        margin={'l': 80, 'r': 40, 't': 120, 'b': 55},
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.01,
            'xanchor': 'right',
            'x': 1.0,
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=[
            *list(fig.layout.annotations),  # type: ignore
            {
                'text': 'Bar shade encodes lambda within each strategy; hover a bar for lambda, values, and CI.',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.0,
                'y': 1.08,
                'showarrow': False,
                'font': {'size': 12, 'color': '#444444'},
                'align': 'left',
            },
        ],
    )

    fig.write_html(
        out_dir / 'gain_over_topk.html',
        include_plotlyjs=True,
        full_html=True,
    )


def plot_gain_over_topk_best_facet_hit_lambda(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Simpler paired delta bars using lambda* per strategy and k."""
    import matplotlib.pyplot as plt
    import numpy as np

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    metrics = _available_metrics(stats_df, results_df)
    x = np.arange(len(k_values))
    width = 0.8 / max(len(diversity_strategies), 1)

    fig, axes = _grid_figure('gain_over_topk_best_facet_hit_lambda')
    for ax, (stats_col, result_col, title, higher_is_better) in zip(
        axes.flatten(),
        metrics,
        strict=False,
    ):
        for i, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            deltas: list[float] = []
            cis: list[float] = []
            for k in k_values:
                ref_row = topk_df.filter(pl.col('k') == k)
                ref_val = float(ref_row[stats_col][0]) if ref_row.height > 0 else 0.0
                best_row = _best_lam_rows(stats_df, strategy, [k])
                if best_row.height > 0:
                    strat_val = float(best_row[stats_col][0])
                    lam = float(best_row['lam'][0])
                else:
                    strat_val = ref_val
                    lam = None
                deltas.append(strat_val - ref_val)
                cis.append(_paired_delta_ci(results_df, strategy, k, lam, result_col))

            ci_display = [0.0 if np.isnan(c) else c for c in cis]
            offset = (i - len(diversity_strategies) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                deltas,
                width=width * 0.9,
                color=style['color'],
                label=style['label'],
                alpha=0.85,
                yerr=ci_display,
                capsize=3,
                error_kw={'linewidth': 0.8, 'ecolor': style['color']},
            )
            _annotate_delta_bars(ax, bars, deltas)

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=9)
        ax.set_title(
            _panel_title(f'Delta {title} vs top-k', higher_is_better, delta=True),
            fontsize=10,
        )
        ax.set_ylabel(f'Delta {stats_col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.margins(y=0.18)

    _figure_legend(fig, axes.flatten())
    for ax in axes.flatten()[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(
        'Gain over top-k - coverage-first lambda* within strategy x k, paired 95% CI',
        fontsize=12,
    )
    _figure_note(
        fig,
        _best_lambda_note(stats_df, diversity_strategies, k_values),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'gain_over_topk_best_facet_hit_lambda.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_lambda_agreement(agreement_df: pl.DataFrame, out_dir: Path) -> None:
    """Heatmaps of FacLoc-vs-MMR lambda pairs ranked by mean absolute metric disagreement."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    if agreement_df.is_empty():
        return

    k_values = sorted(int(v) for v in agreement_df['k'].unique().to_list())
    fac_lams = sorted(float(v) for v in agreement_df['fac_loc_lam'].unique().to_list())
    mmr_lams = sorted(float(v) for v in agreement_df['mmr_lam'].unique().to_list())
    if not k_values or not fac_lams or not mmr_lams:
        return

    metric_diff_cols = sorted(
        col
        for col in agreement_df.columns
        if col.startswith('abs_diff__') and col != 'mean_abs_diff'
    )
    score_col = (
        'weighted_mean_abs_diff'
        if 'weighted_mean_abs_diff' in agreement_df.columns
        else 'mean_abs_diff'
    )
    rank_col = (
        'weighted_rank_within_k'
        if 'weighted_rank_within_k' in agreement_df.columns
        else 'rank_within_k'
    )
    uses_kernel_weighting = score_col == 'weighted_mean_abs_diff'
    score_label = 'weighted mean abs diff' if uses_kernel_weighting else 'mean abs diff'
    lambda_cutoff_note = (
        ' Lambda pairs above 0.80 are excluded.'
        if fac_lams and mmr_lams and max(fac_lams) <= 0.80 and max(mmr_lams) <= 0.80
        else ''
    )
    vmin = float(agreement_df[score_col].min())  # type:ignore
    vmax = float(agreement_df[score_col].max())  # type:ignore
    color_norm = None
    if uses_kernel_weighting and vmax > 0:
        positive_scores = agreement_df.filter(pl.col(score_col) > 0)[score_col]
        if not positive_scores.is_empty():
            color_norm = LogNorm(vmin=float(positive_scores.min()), vmax=vmax)  # type:ignore
    fig_width = max(6.4, len(k_values) * max(5.8, 0.30 * len(mmr_lams))) + 1.6
    fig_height = max(6.2, 0.24 * len(fac_lams)) + 1.7
    fig, axes = plt.subplots(
        1,
        len(k_values),
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.flatten()
    fac_idx = {lam: idx for idx, lam in enumerate(fac_lams)}
    mmr_idx = {lam: idx for idx, lam in enumerate(mmr_lams)}
    image = None

    for ax, k in zip(flat_axes, k_values, strict=False):
        sub = agreement_df.filter(pl.col('k') == k).sort(
            [rank_col, 'fac_loc_lam', 'mmr_lam'],
            descending=[False, False, False],
        )
        matrix = np.full((len(fac_lams), len(mmr_lams)), np.nan, dtype=float)
        for row in sub.iter_rows(named=True):
            matrix[fac_idx[float(row['fac_loc_lam'])], mmr_idx[float(row['mmr_lam'])]] = float(
                row[score_col]
            )

        image_kwargs = (
            {'norm': color_norm} if color_norm is not None else {'vmin': vmin, 'vmax': vmax}
        )
        image = ax.imshow(
            matrix,
            origin='lower',
            aspect='equal',
            cmap='viridis_r',
            interpolation='nearest',
            **image_kwargs,
        )
        best_row = sub.row(0, named=True)
        best_fac_idx = fac_idx[float(best_row['fac_loc_lam'])]
        best_mmr_idx = mmr_idx[float(best_row['mmr_lam'])]
        ax.add_patch(
            Rectangle(
                (best_mmr_idx - 0.5, best_fac_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor='white',
                linewidth=1.6,
                zorder=4,
            )
        )
        ax.scatter(
            best_mmr_idx,
            best_fac_idx,
            marker='*',
            s=120,
            facecolor='white',
            edgecolor='black',
            linewidth=0.9,
            zorder=5,
        )
        ax.set_title(
            '\n'.join(
                [
                    f'k={k}',
                    (
                        f'best FacLoc λ={float(best_row["fac_loc_lam"]):.2f} | '
                        f'MMR λ={float(best_row["mmr_lam"]):.2f}'
                    ),
                    f'{score_label}={float(best_row[score_col]):.3f}',
                ]
            ),
            fontsize=8.5,
        )
        ax.set_xticks(range(len(mmr_lams)))
        ax.set_xticklabels([f'{lam:.2f}' for lam in mmr_lams], rotation=90, fontsize=6)
        ax.set_xlabel('MMR λ', fontsize=9)
        ax.set_yticks(range(len(fac_lams)))
        ax.set_yticklabels([f'{lam:.2f}' for lam in fac_lams], fontsize=6)
        ax.tick_params(labelleft=True)
        if ax is flat_axes[0]:
            ax.set_ylabel('FacLoc λ', fontsize=9)

    if image is not None:
        cbar_ax = fig.add_axes([0.976, 0.17, 0.012, 0.7])  # type: ignore[attr-defined]
        cbar = fig.colorbar(
            image,
            cax=cbar_ax,
        )
        cbar.set_label(
            'Kernel-weighted mean absolute metric difference (log scale)'
            if uses_kernel_weighting
            else 'Mean absolute metric difference',
            fontsize=9,
        )
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        'Lambda agreement FacLoc vs MMR',
        fontsize=12,
    )
    _figure_note(
        fig,
        (
            'Cell values are kernel-weighted mean absolute differences across '
            f'{len(metric_diff_cols)} summary metrics; lower is better.{lambda_cutoff_note}'
        )
        if uses_kernel_weighting
        else (
            'Cell values are mean absolute differences across '
            f'{len(metric_diff_cols)} summary metrics; lower is better.{lambda_cutoff_note}'
        ),
    )
    fig.subplots_adjust(left=0.05, right=0.955, bottom=0.11, top=0.90, wspace=0.20)
    fig.savefig(out_dir / 'lambda_agreement_facloc_mmr.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_selection_diagnostics(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Diagnostic metrics that explain why a strategy wins or fails."""
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)
    metric_cols = [
        (metric.stats_col, metric.title, metric.higher_is_better)
        for metric in _available_metric_specs(
            stats_df,
            results_df=None,
            metric_names=PLOTTED_DIAGNOSTIC_METRIC_NAMES,
            require_result_col=False,
        )
    ]

    fig, axes = _grid_figure('selection_diagnostics', sharex=True)
    for ax, (col, title, higher_is_better) in zip(axes.flatten(), metric_cols, strict=False):
        for strategy in strategies:
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            if strategy == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [_cell_value(sub, k, None, col) for k in xs]
                ax.plot(xs, ys, color=style['color'], ls=style['ls'], lw=2.0, label=style['label'])
                continue
            best_df = _best_lam_rows(stats_df, strategy, k_values)
            if best_df.height > 0:
                ax.plot(
                    best_df['k'].to_list(),
                    best_df[col].to_list(),
                    color=style['color'],
                    ls=style['ls'],
                    lw=2.0,
                    marker='o',
                    ms=4,
                    label=style['label'],
                )
        ax.set_title(_panel_title(title, higher_is_better), fontsize=10)
        ax.set_ylabel(col, fontsize=9)
        _set_k_tick_labels(ax, k_values)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelbottom=True)

    for ax in axes[-1]:
        ax.set_xlabel('k', fontsize=9)
    for ax in axes.flatten()[len(metric_cols) :]:
        ax.set_visible(False)

    _figure_legend(fig, axes.flatten())
    fig.suptitle(
        'Selection diagnostics - coverage-first lambda* within strategy x k',
        fontsize=12,
    )
    _figure_note(
        fig, _best_lambda_note(stats_df, [s for s in strategies if s != 'top_k'], k_values)
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'selection_diagnostics.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_answer_rouge_comparison(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Auxiliary answer-token overlap diagnostics."""
    import matplotlib.pyplot as plt

    metrics = _available_answer_rouge_metrics(stats_df, results_df)
    if not metrics:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)
    fig, axes = _grid_figure('answer_rouge_comparison', sharex=True)

    for ax, (stats_col, result_col, title, higher_is_better) in zip(
        axes.flatten(),
        metrics,
        strict=False,
    ):
        for strategy in strategies:
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            if strategy == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [_cell_value(sub, k, None, stats_col) for k in xs]
                ci = [
                    ci_half_width(_query_vals(results_df, strategy, k, None, result_col))
                    for k in xs
                ]
                _plot_ci_line(ax, xs, ys, ci, style, lw=2.0, label=style['label'], zorder=3)
                continue

            best_df = _best_lam_rows(stats_df, strategy, k_values)
            if best_df.height == 0:
                continue
            xs = best_df['k'].to_list()
            ys = [float(v) for v in best_df[stats_col].to_list()]
            lams = [float(v) for v in best_df['lam'].to_list()]
            k_to_lam = dict(zip(xs, lams, strict=True))
            ci = [
                ci_half_width(_query_vals(results_df, strategy, k, k_to_lam.get(k), result_col))
                for k in xs
            ]
            ax.plot(
                xs,
                ys,
                color=style['color'],
                ls=style['ls'],
                lw=2.0,
                label=style['label'],
                zorder=3,
            )
            ax.scatter(
                xs,
                ys,
                s=42,
                facecolors='white',
                edgecolors=style['color'],
                linewidths=1.4,
                zorder=4,
            )
            _plot_error_caps(ax, xs, ys, ci, style, zorder=2)
            _annotate_lambda_points(
                ax,
                xs,
                ys,
                lams,
                color=style['color'],
                placement='above' if strategy == 'mmr' else 'below',
            )

        ax.set_title(_panel_title(title, higher_is_better), fontsize=10)
        ax.set_ylabel(stats_col, fontsize=9)
        ax.set_xticks(k_values)
        ax.grid(axis='y', alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('k', fontsize=9)
    for ax in axes.flatten()[len(metrics) :]:
        ax.set_visible(False)

    _figure_legend(fig, axes.flatten())
    fig.suptitle(
        'Auxiliary answer-token ROUGE diagnostics - lambda* path per strategy',
        fontsize=12,
    )
    _figure_note(
        fig,
        f'ROUGE is diagnostic only; lambda* is still selected by coverage metrics. {LAMBDA_POLICY_NOTE}',
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'answer_rouge_comparison.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_answer_rouge_lambda_sensitivity(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Auxiliary ROUGE metrics as lambda changes."""
    import matplotlib.pyplot as plt

    metric_cols = [
        (metric.stats_col, metric.title)
        for metric in _available_metric_specs(
            stats_df,
            results_df=None,
            metric_names=PLOTTED_ANSWER_ROUGE_METRIC_NAMES,
            require_result_col=False,
        )
    ]
    if not metric_cols:
        return

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    lambda_values = _lambda_values(stats_df)
    sampled_lambda_values = _sample_tick_values(lambda_values)
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    cmap = plt.get_cmap('viridis')  # type: ignore[attr-defined]
    k_colors = {k: cmap(i / max(len(k_values) - 1, 1)) for i, k in enumerate(k_values)}

    fig, axes = _grid_figure(
        'answer_rouge_lambda_sensitivity',
        rows=len(metric_cols),
        cols=len(diversity_strategies),
        sharex=True,
    )

    for row_idx, (metric, title) in enumerate(metric_cols):
        for col_idx, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            ax = axes[row_idx][col_idx]
            for k in k_values:
                ksub = sub.filter(pl.col('k') == k).sort('lam')
                if ksub.height == 0:
                    continue
                ax.plot(
                    ksub['lam'].to_list(),
                    ksub[metric].to_list(),
                    color=k_colors[k],
                    ls=style['ls'],
                    lw=1.8,
                    marker='o',
                    ms=4,
                    label=f'k={k}',
                )
                ref = topk_df.filter(pl.col('k') == k)
                if ref.height > 0:
                    ax.axhline(
                        float(ref[metric][0]),
                        color=k_colors[k],
                        ls='--',
                        lw=1.0,
                        alpha=0.5,
                    )

            if row_idx == 0:
                ax.set_title(style['label'], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=10)
            if row_idx == len(metric_cols) - 1:
                ax.set_xlabel('lambda', fontsize=9)
            _set_lambda_tick_labels(ax, sampled_lambda_values)
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=len(k_values),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.suptitle(
        'Auxiliary answer-token ROUGE lambda sensitivity',
        fontsize=12,
    )
    _figure_note(
        fig, 'Dashed horizontal lines are the top-k reference at each k; line colors identify k'
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'answer_rouge_lambda_sensitivity.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def get_style(strategy: str) -> dict[str, str]:
    return STRATEGY_STYLE.get(strategy, {'color': '#aaaaaa', 'ls': '-', 'label': strategy})


def _ordered_strategies(df: pl.DataFrame) -> list[str]:
    preferred = ['top_k', 'fac_loc', 'mmr']
    present = set(df['strategy'].unique().to_list())
    ordered = [s for s in preferred if s in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _available_metrics(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
) -> list[tuple[str, str, str, bool]]:
    return [
        metric.as_tuple()
        for metric in _available_metric_specs(
            stats_df,
            results_df=results_df,
            metric_names=PLOTTED_MAIN_METRIC_NAMES,
        )
    ]


def _main_metric_grid_figure(n_metrics: int) -> tuple[Figure, NDArray[Any]]:
    """Compatibility wrapper for the standard 3x2 main-metric grid."""
    del n_metrics
    return _grid_figure('strategy_comparison', sharex=True)


def _available_answer_rouge_metrics(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
) -> list[tuple[str, str, str, bool]]:
    return [
        metric.as_tuple()
        for metric in _available_metric_specs(
            stats_df,
            results_df=results_df,
            metric_names=PLOTTED_ANSWER_ROUGE_METRIC_NAMES,
        )
    ]


def _available_metric_specs(
    stats_df: pl.DataFrame,
    *,
    results_df: pl.DataFrame | None,
    metric_names: list[str],
    require_result_col: bool = True,
) -> list[NamedPlotMetric]:
    metrics: list[NamedPlotMetric] = []
    result_columns = set(results_df.columns) if results_df is not None else set[str]()

    for metric_name in metric_names:
        metric_spec = METRIC_NAME_TO_FIELD.get(metric_name)
        if metric_spec is None or metric_name not in stats_df.columns:
            continue
        if require_result_col and metric_spec.result_col not in result_columns:
            continue
        metrics.append(
            NamedPlotMetric(
                stats_col=metric_name,
                result_col=metric_spec.result_col,
                title=PLOT_METRIC_TITLES.get(metric_name, metric_name),
                higher_is_better=metric_spec.higher_is_better,
            )
        )

    return metrics


def _available_sort(df: pl.DataFrame) -> tuple[list[str], list[bool]]:
    pairs = [
        (col, desc)
        for col, desc in zip(PRIMARY_SORT, PRIMARY_DESC, strict=True)
        if col in df.columns
    ]
    if not pairs:
        return ['lam'], [False]
    cols, desc = zip(*pairs, strict=True)
    return list(cols), list(desc)


def _grid_figure(
    layout_name: EvalPlotFileName,
    *,
    sharex: bool = False,
    rows: int | None = None,
    cols: int | None = None,
    width_per_col: float | None = None,
    height_per_row: float | None = None,
    footer_height: float | None = None,
) -> tuple[Figure, NDArray[Any]]:
    layout = DEFAULT_PLOT_GRID_LAYOUTS[layout_name]
    return _custom_grid_figure(
        rows=layout.rows if rows is None else rows,
        cols=layout.cols if cols is None else cols,
        sharex=sharex,
        width_per_col=layout.width_per_col if width_per_col is None else width_per_col,
        height_per_row=layout.height_per_row if height_per_row is None else height_per_row,
        footer_height=layout.footer_height if footer_height is None else footer_height,
    )


def _custom_grid_figure(
    *,
    rows: int,
    cols: int,
    sharex: bool = False,
    width_per_col: float = 4.0,
    height_per_row: float = 3.0,
    footer_height: float = 1.2,
) -> tuple[Figure, NDArray[Any]]:
    import matplotlib.pyplot as plt

    width = width_per_col * cols
    height = height_per_row * rows + footer_height
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharex=sharex, squeeze=False)
    return fig, axes


def _shared_lambda_sensitivity_ylim(
    stats_df: pl.DataFrame,
    *,
    metric: str,
    strategies: list[str],
    k_values: list[int],
) -> tuple[float, float]:
    values: list[float] = []
    for strategy in strategies:
        strategy_df = stats_df.filter(pl.col('strategy') == strategy)
        for value in strategy_df[metric].drop_nulls().to_list():
            values.append(float(value))

    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    for k in k_values:
        ref_df = topk_df.filter(pl.col('k') == k)
        if ref_df.height == 0:
            continue
        values.append(float(ref_df[metric][0]))

    if not values:
        return (0.0, 1.0)

    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = abs(lower) * 0.05 if lower != 0.0 else 0.05
        return (lower - padding, upper + padding)

    padding = (upper - lower) * 0.05
    return (lower - padding, upper + padding)


def _best_lambda_note(
    stats_df: pl.DataFrame,
    strategies: list[str],
    k_values: list[int],
) -> str:
    parts = [LAMBDA_POLICY_NOTE]
    for strategy in strategies:
        best_df = _best_lam_rows(stats_df, strategy, k_values)
        if best_df.height == 0 or 'lam' not in best_df.columns:
            continue
        mapping = ', '.join(
            f'k={int(k)} -> {float(lam):.2f}'
            for k, lam in zip(best_df['k'].to_list(), best_df['lam'].to_list(), strict=True)
        )
        parts.append(f'{get_style(strategy)["label"]}: {mapping}')
    return ' | '.join(parts)


def _figure_note(fig: Figure, text: str, *, y: float = 0.025) -> None:
    fig.text(0.5, y, text, ha='center', va='bottom', fontsize=8, color='#444444')


def _panel_title(title: str, higher_is_better: bool, *, delta: bool = False) -> str:
    if higher_is_better:
        return title
    suffix = 'negative better' if delta else 'lower better'
    return f'{title} [{suffix}]'


def _build_strategy_lambda_heatmap_data(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
) -> tuple[dict[str, dict[str, NDArray[Any]]], list[int], list[float]] | None:
    import numpy as np

    lambda_values = _lambda_values(stats_df)
    if not lambda_values:
        return None

    present_strategies = set(stats_df['strategy'].unique().to_list())
    if 'fac_loc' not in present_strategies or 'mmr' not in present_strategies:
        return None

    k_values = sorted(int(k) for k in stats_df['k'].unique().to_list())
    metrics = _available_metrics(stats_df, results_df)
    if not metrics:
        return None

    metric_matrices: dict[str, dict[str, NDArray[Any]]] = {}
    for stats_col, _result_col, _title, higher_is_better in metrics:
        fac_matrix = np.full((len(k_values), len(lambda_values)), np.nan, dtype=float)
        mmr_matrix = np.full((len(k_values), len(lambda_values)), np.nan, dtype=float)
        adv_matrix = np.full((len(k_values), len(lambda_values)), np.nan, dtype=float)
        for k_idx, k in enumerate(k_values):
            for lam_idx, lam in enumerate(lambda_values):
                fac_val = _cell_value_for_strategy(stats_df, 'fac_loc', k, lam, stats_col)
                mmr_val = _cell_value_for_strategy(stats_df, 'mmr', k, lam, stats_col)
                fac_matrix[k_idx, lam_idx] = fac_val
                mmr_matrix[k_idx, lam_idx] = mmr_val
                if np.isnan(fac_val) or np.isnan(mmr_val):
                    continue
                adv_matrix[k_idx, lam_idx] = (
                    fac_val - mmr_val if higher_is_better else mmr_val - fac_val
                )
        metric_matrices[stats_col] = {
            'fac_loc': fac_matrix,
            'mmr': mmr_matrix,
            'advantage': adv_matrix,
        }
    return metric_matrices, k_values, lambda_values


def _cell_value_for_strategy(
    stats_df: pl.DataFrame,
    strategy: str,
    k: int,
    lam: float,
    col: str,
) -> float:
    sub = stats_df.filter(
        (pl.col('strategy') == strategy) & (pl.col('k') == k) & (pl.col('lam') == lam)
    )
    return float(sub[col][0]) if sub.height > 0 else float('nan')


def _set_heatmap_axis_labels(
    ax: Any,
    k_values: list[int],
    lambda_values: list[float],
    *,
    sampled_lambda_tick_positions: list[int] | None = None,
) -> None:
    tick_positions = (
        sampled_lambda_tick_positions
        if sampled_lambda_tick_positions is not None
        else list(range(len(lambda_values)))
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f'{lambda_values[idx]:.2f}' for idx in tick_positions],
        rotation=45,
        ha='right',
        fontsize=8,
    )
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([f'k={k}' for k in k_values], fontsize=8)


def _draw_heatmap_grid(ax: Any, n_rows: int, n_cols: int) -> None:
    ax.set_xticks([idx - 0.5 for idx in range(1, n_cols)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, n_rows)], minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8, alpha=0.9)
    ax.tick_params(which='minor', bottom=False, left=False)


def _annotate_heatmap_cells(ax: Any, matrix: NDArray[Any], *, diverging: bool) -> None:
    import numpy as np

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        return
    threshold = float(np.nanmean(finite_values))
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix[row_idx, col_idx])
            if np.isnan(value):
                continue
            if diverging:
                text_color = (
                    'white'
                    if abs(value)
                    > max(abs(float(finite_values.min())), abs(float(finite_values.max()))) * 0.45
                    else '#222222'
                )
            else:
                text_color = 'white' if value >= threshold else '#222222'
            ax.text(
                col_idx,
                row_idx,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=6.5,
                color=text_color,
            )


def _plotly_heatmap_hovertemplate(
    *,
    title: str,
    panel_title: str,
    higher_is_better: bool,
    advantage: bool,
) -> str:
    direction_note = (
        'Positive values mean FacLoc is better for this metric'
        if advantage
        else ('Higher is better' if higher_is_better else 'Lower is better')
    )
    return (
        f'{panel_title}<br>'
        f'metric={title}<br>'
        'lambda=%{x}<br>'
        'k=%{y}<br>'
        'value=%{z:.4f}<br>'
        f'{direction_note}'
        '<extra></extra>'
    )


def _set_k_tick_labels(
    ax: Any,
    k_values: list[int],
    *,
    positions: list[float] | None = None,
    fontsize: float = 9,
) -> None:
    tick_positions = positions if positions is not None else k_values
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=fontsize)


def _sample_tick_values(values: list[float], max_ticks: int = 6) -> list[float]:
    if len(values) <= max_ticks:
        return values
    if max_ticks <= 1:
        return [values[0]]

    indices = {round(idx * (len(values) - 1) / (max_ticks - 1)) for idx in range(max_ticks)}
    return [values[idx] for idx in sorted(indices)]


def _sample_tick_indices(n_values: int, max_ticks: int = 6) -> list[int]:
    if n_values <= max_ticks:
        return list(range(n_values))
    if max_ticks <= 1:
        return [0]

    return sorted({round(idx * (n_values - 1) / (max_ticks - 1)) for idx in range(max_ticks)})


def _set_lambda_tick_labels(
    ax: Any,
    lambda_values: list[float],
    *,
    fontsize: float = 8,
) -> None:
    ax.set_xticks(lambda_values)
    ax.set_xticklabels([f'{lam:.2f}' for lam in lambda_values], fontsize=fontsize)


def _add_lambda_shade_legend(
    fig: Figure,
    strategies: list[str],
    lambda_values: list[float],
    *,
    bottom: float = 0.072,
    values_per_line: int = 20,
) -> None:
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if not strategies or not lambda_values:
        return

    lo = min(lambda_values)
    hi = max(lambda_values)
    bar_width = 0.16
    bar_height = 0.015
    gap = 0.03
    total_width = len(strategies) * bar_width + max(len(strategies) - 1, 0) * gap
    left = max(0.08, 0.5 - total_width / 2)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    for idx, strategy in enumerate(strategies):
        style = get_style(strategy)
        light = _lambda_shade(style['color'], lo, lambda_values)
        dark = _lambda_shade(style['color'], hi, lambda_values)
        cmap = LinearSegmentedColormap.from_list(
            f'lambda_{strategy}',
            [light, dark],
        )
        bar_left = left + idx * (bar_width + gap)
        ax = fig.add_axes([bar_left, bottom, bar_width, bar_height])  # type: ignore[attr-defined]
        ax.imshow(gradient, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xticks([0, gradient.shape[1] - 1])
        ax.set_xticklabels([f'{lo:.2f}', f'{hi:.2f}'], fontsize=7)
        ax.set_yticks([])
        ax.tick_params(axis='x', length=0, pad=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(  # type: ignore[attr-defined]
            bar_left + bar_width / 2,
            bottom + bar_height + 0.006,
            f'{style["label"]} λ',
            ha='center',
            va='bottom',
            fontsize=7.5,
            color=style['color'],
        )
    lambda_lines = _format_lambda_value_lines(lambda_values, values_per_line=values_per_line)
    fig.text(  # type: ignore[attr-defined]
        0.5,
        bottom - 0.012,
        'λ values:\n' + '\n'.join(lambda_lines),
        ha='center',
        va='top',
        fontsize=7.5,
        color='#444444',
    )


def _format_lambda_value_lines(
    lambda_values: list[float],
    *,
    values_per_line: int,
) -> list[str]:
    if values_per_line <= 0:
        return [', '.join(f'{lam:.2f}' for lam in lambda_values)]

    formatted_values = [f'{lam:.2f}' for lam in lambda_values]
    return [
        ', '.join(formatted_values[idx : idx + values_per_line])
        for idx in range(0, len(formatted_values), values_per_line)
    ]


def _lambda_values(df: pl.DataFrame) -> list[float]:
    if 'lam' not in df.columns:
        return []
    return sorted(float(lam) for lam in df['lam'].drop_nulls().unique().to_list())


@dataclass(frozen=True)
class GainOverTopkGeometry:
    k_positions: list[float]
    slot_width: float
    bar_width: float
    x_min: float
    x_max: float


@dataclass(frozen=True)
class GainOverTopkFooterLayout:
    footer_height: float
    tight_layout_bottom: float
    legend_bottom: float
    values_per_line: int


def _gain_over_topk_geometry(n_k_values: int, n_groups: int) -> GainOverTopkGeometry:
    import numpy as np

    min_group_span = 2.8
    min_bar_slot_width = 0.12
    slot_width = max(min_group_span / max(n_groups, 1), min_bar_slot_width)
    group_span = slot_width * n_groups
    k_gap = max(0.9, min(2.2, group_span * 0.16))
    k_positions = (np.arange(n_k_values, dtype=float) * (group_span + k_gap)).tolist()
    bar_width = slot_width * 0.96
    first_center = k_positions[0] + _gain_over_topk_offset(
        group_idx=0,
        n_groups=n_groups,
        slot_width=slot_width,
    )
    last_center = k_positions[-1] + _gain_over_topk_offset(
        group_idx=n_groups - 1,
        n_groups=n_groups,
        slot_width=slot_width,
    )
    padding = max(slot_width * 0.45, 0.10)
    return GainOverTopkGeometry(
        k_positions=k_positions,
        slot_width=slot_width,
        bar_width=bar_width,
        x_min=first_center - bar_width / 2 - padding,
        x_max=last_center + bar_width / 2 + padding,
    )


def _gain_over_topk_offset(*, group_idx: int, n_groups: int, slot_width: float) -> float:
    return (group_idx - n_groups / 2 + 0.5) * slot_width


def _gain_over_topk_footer_layout(lambda_values: list[float]) -> GainOverTopkFooterLayout:
    values_per_line = 35
    lambda_line_count = len(
        _format_lambda_value_lines(lambda_values, values_per_line=values_per_line)
    )
    extra_lines = max(lambda_line_count - 1, 0)
    legend_bottom = 0.040 + 0.018 * extra_lines
    return GainOverTopkFooterLayout(
        footer_height=1.15 + 0.30 * extra_lines,
        tight_layout_bottom=legend_bottom + 0.048,
        legend_bottom=legend_bottom,
        values_per_line=values_per_line,
    )


def _lambda_shade(
    base_color: str, lam: float, lambda_values: list[float]
) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    base_rgb = to_rgb(base_color)
    if len(lambda_values) <= 1:
        return base_rgb
    lo = min(lambda_values)
    hi = max(lambda_values)
    if hi <= lo:
        return base_rgb
    strength = (lam - lo) / (hi - lo)
    mix = 0.82 - 0.72 * strength
    mix = min(max(mix, 0.1), 0.82)
    return tuple(channel * (1.0 - mix) + mix for channel in base_rgb)  # type:ignore


def _rgb_to_plotly_color(rgb: tuple[float, float, float]) -> str:
    r, g, b = (round(channel * 255) for channel in rgb)
    return f'rgb({r}, {g}, {b})'


def _annotate_lambda_points(
    ax: Any,
    xs: list[int],
    ys: list[float],
    lambda_values: list[float],
    *,
    color: str,
    placement: str,
) -> None:
    offsets = {
        'above': (0, 7, 'bottom'),
        'below': (0, -9, 'top'),
    }
    dx, dy, va = offsets.get(placement, offsets['above'])
    for x, y, lam in zip(xs, ys, lambda_values, strict=True):
        ax.annotate(
            f'λ={lam:.2f}',
            xy=(x, y),
            xytext=(dx, dy),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=6.5,
            color=color,
            bbox={
                'boxstyle': 'round,pad=0.18',
                'facecolor': 'white',
                'edgecolor': color,
                'linewidth': 0.45,
                'alpha': 0.92,
            },
            zorder=5,
        )


def _annotate_delta_bars(ax: object, bars: Iterable, values: list[float]) -> None:
    ymin, ymax = ax.get_ylim()  # type: ignore[attr-defined]
    span = max(ymax - ymin, 1e-6)
    offset = span * 0.025
    for bar, value in zip(bars, values, strict=True):
        if abs(value) < 0.005:
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = value + offset if value >= 0 else value - offset
        va = 'bottom' if value >= 0 else 'top'
        ax.text(  # type: ignore[attr-defined]
            x,
            y,
            f'{value:+.3f}',
            ha='center',
            va=va,
            fontsize=6.5,
            clip_on=True,
        )


def _draw_bar_slot_guides(ax: Any, centers: list[float], width: float) -> None:
    if width <= 0:
        return
    for center in centers:
        ax.axvspan(
            center - width / 2,
            center + width / 2,
            facecolor='none',
            edgecolor='#cfcfcf',
            linewidth=0.4,
            zorder=0.1,
        )


def _annotate_extreme_delta_bars(
    ax: Any,
    bars_by_strategy_k: dict[tuple[str, int], list[tuple[Any, float]]],
) -> None:
    for (strategy, _k), bar_values in bars_by_strategy_k.items():
        if not bar_values:
            continue
        max_idx = max(range(len(bar_values)), key=lambda idx: bar_values[idx][1])
        min_idx = min(range(len(bar_values)), key=lambda idx: bar_values[idx][1])
        label_indices = sorted({min_idx, max_idx})
        for idx in label_indices:
            bar, value = bar_values[idx]
            _annotate_single_delta_bar(ax, bar, value, color=get_style(strategy)['color'])


def _annotate_single_delta_bar(ax: Any, bar: Any, value: float, *, color: str) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    offset = span * 0.025
    x = bar.get_x() + bar.get_width() / 2
    y = value + offset if value >= 0 else value - offset
    va = 'bottom' if value >= 0 else 'top'
    ax.text(
        x,
        y,
        f'{value:+.3f}',
        ha='center',
        va=va,
        fontsize=6.3,
        color=color,
        bbox={
            'boxstyle': 'round,pad=0.14',
            'facecolor': 'white',
            'edgecolor': color,
            'linewidth': 0.35,
            'alpha': 0.9,
        },
        clip_on=True,
        zorder=5,
    )


def _best_lam_rows(stats_df: pl.DataFrame, strategy: str, k_values: list[int]) -> pl.DataFrame:
    sub = stats_df.filter(pl.col('strategy') == strategy)
    rows = []
    for k in k_values:
        ksub = sub.filter(pl.col('k') == k)
        if ksub.height > 0:
            sort_cols, desc = _available_sort(ksub)
            rows.append(ksub.sort(sort_cols, descending=desc).head(1))
    return pl.concat(rows).sort('k') if rows else pl.DataFrame()


def _best_topk_k(results_df: pl.DataFrame) -> int:
    topk = results_df.filter(pl.col('strategy') == 'top_k')
    if topk.height == 0:
        return int(results_df['k'].max())  # type: ignore[arg-type]
    ranked = (
        topk.group_by('k')
        .agg(
            pl.col('facet_coverage').median().alias('med_fc'),
        )
        .sort(['med_fc', 'k'], descending=[True, False])
    )
    return int(ranked['k'][0])


def _best_result_slice(results_df: pl.DataFrame, strategy: str, k: int) -> pl.DataFrame:
    strat_df = results_df.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
    if strategy == 'top_k' or strat_df.height == 0:
        return strat_df
    ranked = strat_df.group_by('lam').agg(
        pl.col('facet_coverage').mean().alias('MeanFacetHitRate@k'),
        pl.col('alpha_ndcg').mean().alias('alpha-nDCG@k')
        if 'alpha_ndcg' in strat_df.columns
        else pl.col('facet_coverage').mean().alias('alpha-nDCG@k'),
        pl.col('gold_precision').mean().alias('Precision@k'),
        pl.col('distractor_rate').mean().alias('DistractorRate'),
    )

    sort_cols, desc = _available_sort(ranked)
    ranked = ranked.sort(sort_cols, descending=desc)
    return strat_df.filter(pl.col('lam') == ranked['lam'][0]) if ranked.height > 0 else strat_df


def _cell_value(df: pl.DataFrame, k: int, lam: float | None, col: str) -> float:
    mask = pl.col('k') == k
    if lam is not None:
        mask = mask & (pl.col('lam') == lam)
    sub = df.filter(mask)
    return float(sub[col][0]) if sub.height > 0 else float('nan')


def _query_vals(
    results_df: pl.DataFrame,
    strategy: str,
    k: int,
    lam: float | None,
    col: str,
) -> list[float]:
    if col not in results_df.columns:
        return []
    mask = (pl.col('strategy') == strategy) & (pl.col('k') == k)
    if lam is not None:
        mask = mask & (pl.col('lam') == lam)
    return [float(v) for v in results_df.filter(mask)[col].drop_nulls().to_list()]


def _paired_delta_ci(
    results_df: pl.DataFrame,
    strategy: str,
    k: int,
    lam: float | None,
    col: str,
) -> float:
    import numpy as np

    if col not in results_df.columns or 'query_id' not in results_df.columns:
        return float('nan')
    topk_sub = results_df.filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k)).select(
        'query_id',
        pl.col(col).alias('topk_val'),
    )
    strategy_mask = (pl.col('strategy') == strategy) & (pl.col('k') == k)
    if lam is not None:
        strategy_mask = strategy_mask & (pl.col('lam') == lam)
    strat_sub = results_df.filter(strategy_mask).select(
        'query_id', pl.col(col).alias('strategy_val')
    )
    joined = topk_sub.join(strat_sub, on='query_id', how='inner')
    if joined.height < 2:
        return float('nan')
    deltas = (joined['strategy_val'] - joined['topk_val']).to_numpy()
    return ci_half_width([float(v) for v in np.asarray(deltas, dtype=float)])


def _plot_ci_line(
    ax: object,
    xs: list[int],
    ys: list[float],
    ci: list[float],
    style: dict[str, str],
    lw: float,
    label: str,
    zorder: float,
) -> None:
    import numpy as np

    ci_display = [0.0 if np.isnan(c) else c for c in ci]
    ax.fill_between(  # type: ignore[attr-defined]
        xs,
        [y - c for y, c in zip(ys, ci_display, strict=True)],
        [y + c for y, c in zip(ys, ci_display, strict=True)],
        color=style['color'],
        alpha=0.12,
        zorder=zorder - 0.2,
    )
    ax.plot(  # type: ignore[attr-defined]
        xs,
        ys,
        color=style['color'],
        ls=style['ls'],
        lw=lw,
        label=label,
        zorder=zorder,
    )


def _plot_error_caps(
    ax: object,
    xs: list[int],
    ys: list[float],
    ci: list[float],
    style: dict[str, str],
    zorder: int,
) -> None:
    import numpy as np

    ci_arr = np.asarray(ci, dtype=float)
    yerr = np.where(np.isnan(ci_arr), 0.0, ci_arr)
    ax.errorbar(  # type: ignore[attr-defined]
        xs,
        ys,
        yerr=yerr,
        fmt='none',
        ecolor=style['color'],
        elinewidth=0.9,
        capsize=3,
        alpha=0.7,
        zorder=zorder,
    )


def _figure_legend(fig: Figure, axes: NDArray[Any]) -> None:
    handles, labels, seen = [], [], set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels, strict=False):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    if handles:
        fig.legend(  # type: ignore[attr-defined]
            handles,
            labels,
            loc='lower center',
            ncol=min(len(seen), 5),
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
        )
