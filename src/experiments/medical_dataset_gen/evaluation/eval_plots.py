from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
from matplotlib.figure import Figure
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.eval_plots_configs import (
    DEFAULT_EVAL_PLOT_THEME,
    DEFAULT_PLOT_GRID_LAYOUTS,
    EVAL_PLOT_DIVERGING_CMAP,
    EVAL_PLOT_HEATMAP_CMAP,
    EVAL_PLOT_K_COLORMAP,
    EVAL_PLOT_STRATEGY_STYLES,
    EVAL_PLOT_THEMES,
    FOR_LAMBDA_K_CURVE_BEST_MARKER_SIZE,
    FOR_LAMBDA_K_CURVE_MARKER_SIZE,
    PLOT_METRIC_TITLES,
    PLOTTED_ANSWER_ROUGE_METRIC_NAMES,
    PLOTTED_DIAGNOSTIC_METRIC_NAMES,
    PLOTTED_MAIN_METRIC_NAMES,
    EvalPlotFileName,
    NamedPlotMetric,
)
from experiments.medical_dataset_gen.evaluation.lambda_selection import (
    lambda_selection_policy_note,
    lambda_selection_short_label,
    select_best_lambda_row,
    select_best_lambda_rows,
)
from experiments.medical_dataset_gen.evaluation.retrieval_utils import ci_half_width
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
    EvalPlotTheme,
    LambdaSelectionCfg,
)
from experiments.medical_dataset_gen.schemas.metrics_schemas import METRIC_NAME_TO_FIELD


def plot_metrics_at_best_lambda_for_k(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
    lambda_selection: LambdaSelectionCfg,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Metric-vs-k lines.

    Top-k is shown as the baseline. Diversity methods show only the selected
    lambda* path for each k, with the exact lambda annotated at each marker.
    """
    _set_active_plot_theme(plot_theme)
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)
    metrics = _available_metrics(stats_df, results_df)
    fig, axes = _grid_figure('metrics_at_best_lambda_for_k', sharex=True)

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

            best_df = _best_lam_rows(stats_df, strategy, k_values, lambda_selection)
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
                facecolors=_theme_color('marker_facecolor'),
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
        f'{lambda_selection_policy_note(lambda_selection)}; see metrics_k_curves_for_lambda.png for the full sweep',
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'metrics_at_best_lambda_for_k.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_k_curves_for_lambda(
    stats_df: pl.DataFrame,
    out_dir: Path,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Main benchmark metrics as lambda changes."""
    _set_active_plot_theme(plot_theme)
    _plot_lambda_sensitivity(
        stats_df,
        out_dir,
        plot_name='metrics_k_curves_for_lambda',
        metric_names=PLOTTED_MAIN_METRIC_NAMES,
        figure_title=' ',
        output_filename='metrics_k_curves_for_lambda.png',
    )


def plot_diagnostics_k_curves_for_lambda(
    stats_df: pl.DataFrame,
    out_dir: Path,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Diagnostic metrics as lambda changes."""
    _set_active_plot_theme(plot_theme)
    _plot_lambda_sensitivity(
        stats_df,
        out_dir,
        plot_name='diagnostics_k_curves_for_lambda',
        metric_names=PLOTTED_DIAGNOSTIC_METRIC_NAMES,
        figure_title=' ',
        output_filename='diagnostics_k_curves_for_lambda.png',
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
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    metric_cols = [
        (metric.stats_col, metric.title, metric.higher_is_better)
        for metric in _available_metric_specs(
            stats_df,
            results_df=None,
            metric_names=metric_names,
            require_result_col=False,
        )
    ]
    if not metric_cols:
        return
    k_colors = _k_colors(plt, k_values)

    fig, axes = _grid_figure(
        plot_name,
        rows=len(metric_cols),
        cols=len(diversity_strategies),
        sharex=False,
    )

    for row_idx, (metric, metric_title, higher_is_better) in enumerate(metric_cols):
        row_ylim = _shared_lambda_sensitivity_ylim(
            stats_df,
            metric=metric,
            strategies=diversity_strategies,
            k_values=k_values,
        )
        for col_idx, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            sampled_lambda_values = _sample_tick_values(
                _lambda_values_for_strategy(stats_df, strategy)
            )
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
                    ms=FOR_LAMBDA_K_CURVE_MARKER_SIZE,
                    label=f'k={k}',
                )
                _mark_best_lambda_curve_point(
                    ax,
                    ksub['lam'].to_list(),
                    ksub[metric].to_list(),
                    color=k_colors[k],
                    higher_is_better=higher_is_better,
                )
                ref = topk_df.filter(pl.col('k') == k)
                if ref.height > 0:
                    ax.axhline(float(ref[metric][0]), color=k_colors[k], ls='--', lw=1.0, alpha=0.5)

            if row_idx == 0:
                ax.set_title(style['label'], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(metric_title, fontsize=10)
            if row_idx == len(metric_cols) - 1:
                ax.set_xlabel(f'{style["label"]} lambda', fontsize=9)
            _set_lambda_tick_labels(ax, sampled_lambda_values)
            ax.set_ylim(*row_ylim)
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=len(k_values),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        _style_legend(legend)
    fig.suptitle(figure_title, fontsize=12)
    _figure_note(
        fig, 'Dashed horizontal lines are the top-k reference at each k; line colors identify k'
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / output_filename, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_heatmap_k_lambda_grid(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Static heatmaps across k x lambda for FacLoc, MMR, and FacLoc advantage."""
    _set_active_plot_theme(plot_theme)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize, TwoSlopeNorm

    _activate_plot_theme()
    heatmap_data = _build_strategy_lambda_heatmap_data(stats_df, results_df)
    if heatmap_data is None:
        return
    metric_matrices, k_values, lambda_axes = heatmap_data
    metrics = _available_metrics(stats_df, results_df)
    fig = plt.figure(
        figsize=(15.5, 2.7 * len(metrics) + 1.8), facecolor=_theme_color('figure_facecolor')
    )
    grid = fig.add_gridspec(
        nrows=len(metrics),
        ncols=7,
        # Dedicated spacer columns prevent repeated axis labels and colorbar
        # labels from encroaching on the neighboring heatmaps.
        width_ratios=[1.0, 0.08, 1.0, 0.035, 0.16, 1.0, 0.035],
        wspace=0.08,
        hspace=0.48,
    )

    panel_titles = ['FacLoc', 'MMR', 'Normalized FacLoc Advantage']

    for row_idx, (stats_col, _result_col, title, higher_is_better) in enumerate(metrics):
        matrices = metric_matrices[stats_col]
        raw_min = float(
            np.nanmin(np.concatenate([matrices['fac_loc'].ravel(), matrices['mmr'].ravel()]))
        )
        raw_max = float(
            np.nanmax(np.concatenate([matrices['fac_loc'].ravel(), matrices['mmr'].ravel()]))
        )
        adv_abs_max = float(np.nanmax(np.abs(matrices['advantage'])))
        if not np.isfinite(adv_abs_max) or adv_abs_max == 0.0:
            adv_abs_max = 1e-9
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
            x_values = lambda_axes[key]
            sampled_lambda_tick_positions = _sample_tick_indices(len(x_values), max_ticks=6)
            annotate_cells = len(k_values) * len(x_values) <= 42
            image = ax.imshow(
                matrix,
                origin='lower',
                aspect='auto',
                cmap=EVAL_PLOT_HEATMAP_CMAP if key != 'advantage' else EVAL_PLOT_DIVERGING_CMAP,
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
                x_values,
                sampled_lambda_tick_positions=sampled_lambda_tick_positions,
            )
            if col_idx > 0:
                ax.tick_params(axis='y', labelleft=False, left=False)
            if row_idx == len(metrics) - 1:
                if key == 'advantage':
                    ax.set_xlabel('normalized grid position', fontsize=9)
                else:
                    ax.set_xlabel(f'{get_style(key)["label"]} lambda', fontsize=9)
            ax.tick_params(axis='x', labelbottom=True)
            _draw_heatmap_grid(ax, len(k_values), len(x_values))
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
            'Advantage is matched by normalized grid position, not raw lambda equality; '
            'for lower-better metrics, positive still means FacLoc better.'
        ),
    )
    _style_existing_axes(fig)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.92)
    fig.savefig(out_dir / 'metrics_heatmap_k_lambda_grid.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_heatmap_k_lambda_grid_html(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Interactive HTML explorer for strategy-vs-lambda heatmaps."""
    _set_active_plot_theme(plot_theme)
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    heatmap_data = _build_strategy_lambda_heatmap_data(stats_df, results_df)
    if heatmap_data is None:
        return

    metric_matrices, k_values, lambda_axes = heatmap_data
    metrics = _available_metrics(stats_df, results_df)
    subplot_titles = ['FacLoc', 'MMR', 'Normalized FacLoc Advantage']
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
        raw_min = float(
            np.nanmin(np.concatenate([matrices['fac_loc'].ravel(), matrices['mmr'].ravel()]))
        )
        raw_max = float(
            np.nanmax(np.concatenate([matrices['fac_loc'].ravel(), matrices['mmr'].ravel()]))
        )
        advantage_abs_max = float(np.nanmax(np.abs(matrices['advantage'])))
        if not np.isfinite(advantage_abs_max) or advantage_abs_max == 0.0:
            advantage_abs_max = 1e-9
        for col_idx, key in enumerate(['fac_loc', 'mmr', 'advantage'], start=1):
            matrix = matrices[key]
            x_values = lambda_axes[key]
            x_label = (
                'normalized grid position'
                if key == 'advantage'
                else f'{get_style(key)["label"]} lambda'
            )
            hovertemplate = _plotly_heatmap_hovertemplate(
                title=title,
                panel_title=subplot_titles[col_idx - 1],
                higher_is_better=higher_is_better,
                advantage=(key == 'advantage'),
                x_label=x_label,
            )
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=[f'{value:.2f}' for value in x_values],
                    y=[f'k={k}' for k in k_values],
                    colorscale=EVAL_PLOT_HEATMAP_CMAP
                    if key != 'advantage'
                    else EVAL_PLOT_DIVERGING_CMAP,
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
        button_defs.append({
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
        })

    fig.update_layout(
        title=f'Strategy comparison heatmap - {_panel_title(first_metric_title, first_metric_higher)}',
        template=_active_theme()['plotly_template'],
        paper_bgcolor=_theme_color('figure_facecolor'),
        plot_bgcolor=_theme_color('axes_facecolor'),
        font={'color': _theme_color('text_color')},
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
    for col_idx, x_title in enumerate(
        ['FacLoc lambda', 'MMR lambda', 'normalized grid position'],
        start=1,
    ):
        fig.update_xaxes(
            title_text=x_title,
            row=1,
            col=col_idx,
            gridcolor=_theme_color('grid_color'),
            zerolinecolor=_theme_color('zero_line_color'),
            linecolor=_theme_color('spine_color'),
            tickfont={'color': _theme_color('tick_color')},
            title_font={'color': _theme_color('text_color')},
        )
        fig.update_yaxes(
            title_text='k',
            row=1,
            col=col_idx,
            gridcolor=_theme_color('grid_color'),
            zerolinecolor=_theme_color('zero_line_color'),
            linecolor=_theme_color('spine_color'),
            tickfont={'color': _theme_color('tick_color')},
            title_font={'color': _theme_color('text_color')},
        )

    fig.write_html(
        out_dir / 'metrics_heatmap_k_lambda_grid.html',
        include_plotlyjs=True,
        full_html=True,
    )


def plot_metrics_distributions(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
    lambda_selection: LambdaSelectionCfg,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Violin plots at the top-k best k and each strategy's configured lambda*."""
    _set_active_plot_theme(plot_theme)
    import matplotlib.pyplot as plt
    import numpy as np

    best_k = _best_topk_k(results_df)
    strategies = _ordered_strategies(results_df)
    metric_cols = [
        (metric.result_col, metric.title)
        for metric in _available_metric_specs(
            stats_df,
            results_df=results_df,
            metric_names=PLOTTED_MAIN_METRIC_NAMES,
        )
    ]
    if not metric_cols:
        return

    per_strategy_data: dict[str, dict[str, list[float]]] = {}
    for strategy in strategies:
        slice_df = _best_result_slice(
            stats_df,
            results_df,
            strategy,
            best_k,
            lambda_selection,
        )
        per_strategy_data[strategy] = {
            col: [float(v) for v in slice_df[col].drop_nulls().to_list()] for col, _ in metric_cols
        }

    labels = [get_style(s)['label'] for s in strategies]
    colors = [get_style(s)['color'] for s in strategies]
    fig, axes = _grid_figure('metrics_distributions')

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
            range(len(strategies)),
            medians,
            color=_theme_color('marker_facecolor'),
            edgecolors=_theme_color('figure_facecolor'),
            s=42,
            zorder=4,
        )
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_title(f'{title} (k={best_k}, configured lambda*)', fontsize=10)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    for ax in axes.flatten()[len(metric_cols) :]:
        ax.set_visible(False)

    fig.suptitle('Per-query score distributions', fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / 'metrics_distributions.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_delta_vs_topk_k_curves_for_lambda(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Paired delta curves relative to top-k on each strategy's native lambda grid.

    For lower-is-better diagnostics, negative bars are favorable.
    """
    _set_active_plot_theme(plot_theme)
    import matplotlib.pyplot as plt
    import numpy as np

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    metrics = _available_metrics(stats_df, results_df)
    if not metrics:
        return
    k_colors = _k_colors(plt, k_values)

    fig, axes = _grid_figure(
        'metrics_delta_vs_topk_k_curves_for_lambda',
        rows=len(metrics),
        cols=len(diversity_strategies),
        sharex=False,
        width_per_col=4.4,
        footer_height=1.0,
    )
    for row_idx, (stats_col, result_col, title, higher_is_better) in enumerate(metrics):
        row_ylim = _shared_delta_vs_topk_ylim(
            stats_df,
            results_df,
            metric=stats_col,
            result_col=result_col,
            strategies=diversity_strategies,
            k_values=k_values,
        )
        for col_idx, strategy in enumerate(diversity_strategies):
            ax = axes[row_idx][col_idx]
            style = get_style(strategy)
            lambda_values = _lambda_values_for_strategy(stats_df, strategy)
            if not lambda_values:
                ax.set_visible(False)
                continue
            x = np.asarray(lambda_values, dtype=float)

            for k in k_values:
                ref_row = topk_df.filter(pl.col('k') == k)
                ref_val = float(ref_row[stats_col][0]) if ref_row.height > 0 else 0.0
                deltas: list[float] = []
                cis: list[float] = []
                for lam in lambda_values:
                    sub = stats_df.filter(
                        (pl.col('strategy') == strategy)
                        & (pl.col('k') == k)
                        & (pl.col('lam') == lam)
                    )
                    strat_val = float(sub[stats_col][0]) if sub.height > 0 else ref_val
                    deltas.append(strat_val - ref_val)
                    cis.append(_paired_delta_ci(results_df, strategy, k, lam, result_col))

                y = np.asarray(deltas, dtype=float)
                ci = np.asarray([0.0 if np.isnan(c) else c for c in cis], dtype=float)
                ax.plot(
                    x,
                    y,
                    color=k_colors[k],
                    ls=style['ls'],
                    lw=1.8,
                    marker='o',
                    ms=FOR_LAMBDA_K_CURVE_MARKER_SIZE,
                    label=f'k={k}',
                )
                _mark_best_lambda_curve_point(
                    ax,
                    x.tolist(),
                    y.tolist(),
                    color=k_colors[k],
                    higher_is_better=higher_is_better,
                )
                if np.any(ci > 0):
                    ax.fill_between(x, y - ci, y + ci, color=k_colors[k], alpha=0.12, linewidth=0)

            ax.axhline(0, color=_theme_color('zero_line_color'), lw=0.8)
            if row_idx == 0:
                ax.set_title(style['label'], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(
                    _panel_title(f'Delta {title} vs top-k', higher_is_better, delta=True),
                    fontsize=9,
                )
            if row_idx == len(metrics) - 1:
                ax.set_xlabel(f'{style["label"]} lambda', fontsize=9)
            _set_lambda_tick_labels(ax, _sample_tick_values(lambda_values))
            ax.set_ylim(*row_ylim)
            ax.grid(alpha=0.3)
            ax.tick_params(labelbottom=True)
            _bold_zero_ytick_label(ax)

    handles, labels = _first_legend_handles(axes)
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=len(k_values),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
        _style_legend(legend)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(
        out_dir / 'metrics_delta_vs_topk_k_curves_for_lambda.png',
        dpi=140,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_metrics_delta_vs_topk_at_best_lambda_for_k(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
    lambda_selection: LambdaSelectionCfg,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Paired delta bars using lambda* per strategy and k, with per-bar lambda labels."""
    _set_active_plot_theme(plot_theme)
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

    fig, axes = _grid_figure('metrics_delta_vs_topk_at_best_lambda_for_k')
    for ax, (stats_col, result_col, title, higher_is_better) in zip(
        axes.flatten(),
        metrics,
        strict=False,
    ):
        for i, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            deltas: list[float] = []
            cis: list[float] = []
            selected_lambdas: list[float | None] = []
            for k in k_values:
                ref_row = topk_df.filter(pl.col('k') == k)
                ref_val = float(ref_row[stats_col][0]) if ref_row.height > 0 else 0.0
                best_row = _best_lam_rows(stats_df, strategy, [k], lambda_selection)
                if best_row.height > 0:
                    strat_val = float(best_row[stats_col][0])
                    lam = float(best_row['lam'][0])
                else:
                    strat_val = ref_val
                    lam = None
                deltas.append(strat_val - ref_val)
                cis.append(_paired_delta_ci(results_df, strategy, k, lam, result_col))
                selected_lambdas.append(lam)

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
            _annotate_delta_bars(
                ax,
                bars,
                deltas,
                lambda_values=selected_lambdas,
                color=style['color'],
            )

        ax.axhline(0, color=_theme_color('zero_line_color'), lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=9)
        ax.set_title(
            _panel_title(f'Delta {title} vs top-k', higher_is_better, delta=True),
            fontsize=10,
        )
        ax.set_ylabel(f'Delta {stats_col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.margins(y=0.28)

    _figure_legend(fig, axes.flatten())
    for ax in axes.flatten()[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(
        'Gain over top-k - configured lambda* within strategy x k, paired 95% CI',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(
        out_dir / 'metrics_delta_vs_topk_at_best_lambda_for_k.png', dpi=140, bbox_inches='tight'
    )
    plt.close(fig)


def plot_profiles_metrics_by_k_at_best_lambda(
    stats_df: pl.DataFrame,
    out_dir: Path,
    lambda_selection: LambdaSelectionCfg,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Control-only evaluation metrics using each strategy's configured lambda*."""
    _set_active_plot_theme(plot_theme)
    _plot_best_lambda_metric_group(
        stats_df,
        out_dir,
        plot_name='profiles_metrics_by_k_at_best_lambda',
        row_title='Evaluation control',
        metric_names=PLOTTED_MAIN_METRIC_NAMES,
        figure_title='Configured lambda* evaluation control metrics by k',
        output_filename='profiles_metrics_by_k_at_best_lambda.png',
        lambda_selection=lambda_selection,
    )


def plot_profiles_diagnostics_by_k_at_best_lambda(
    stats_df: pl.DataFrame,
    out_dir: Path,
    lambda_selection: LambdaSelectionCfg,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Control-only diagnostic metrics using each strategy's configured lambda*."""
    _set_active_plot_theme(plot_theme)
    _plot_best_lambda_metric_group(
        stats_df,
        metric_names=PLOTTED_DIAGNOSTIC_METRIC_NAMES,
        out_dir=out_dir,
        plot_name='profiles_diagnostics_by_k_at_best_lambda',
        row_title='Diagnostic control',
        figure_title='Configured lambda* diagnostic control metrics by k',
        output_filename='profiles_diagnostics_by_k_at_best_lambda.png',
        lambda_selection=lambda_selection,
    )


def _plot_best_lambda_metric_group(
    stats_df: pl.DataFrame,
    out_dir: Path,
    *,
    plot_name: EvalPlotFileName,
    row_title: str,
    metric_names: list[str],
    figure_title: str,
    output_filename: str,
    lambda_selection: LambdaSelectionCfg,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    if stats_df.is_empty():
        return

    metrics = _available_metric_specs(
        stats_df,
        results_df=None,
        metric_names=metric_names,
        require_result_col=False,
    )
    if not metrics:
        return

    k_values = sorted(int(v) for v in stats_df['k'].unique().to_list())
    control_pairs = _best_lambda_control_rows(stats_df, k_values, lambda_selection)
    if control_pairs.is_empty():
        return

    fig_width = max(6.4, 0.72 * len(metrics) + 2.0)
    layout = DEFAULT_PLOT_GRID_LAYOUTS[plot_name]
    fig, axes = _custom_grid_figure(
        rows=len(k_values),
        cols=1,
        width_per_col=fig_width,
        height_per_row=layout.height_per_row,
        footer_height=layout.footer_height,
    )

    styles = {
        'mmr': get_style('mmr'),
        'fac_loc': get_style('fac_loc'),
    }
    control_facecolor = _theme_color('panel_highlight_facecolor')
    row_ylim = _lambda_agreeing_metrics_ylim(
        stats_df,
        control_pairs,
        [m.stats_col for m in metrics],
        include_topk=True,
    )

    for row_idx, k in enumerate(k_values):
        ax = axes[row_idx][0]
        control_row = _k_row(control_pairs, k)
        if control_row is None:
            ax.set_visible(False)
            continue

        fac_loc_lam = float(control_row['fac_loc_lam'])  # type:ignore
        mmr_lam = float(control_row['mmr_lam'])  # type:ignore
        topk_row = _topk_k_row(stats_df, k)
        fac_loc_row = _strategy_k_lambda_row(stats_df, strategy='fac_loc', k=k, lam=fac_loc_lam)
        mmr_row = _strategy_k_lambda_row(stats_df, strategy='mmr', k=k, lam=mmr_lam)
        if topk_row is None or fac_loc_row is None or mmr_row is None:
            ax.set_visible(False)
            continue

        x_positions = np.arange(len(metrics))
        labels = [metric.title for metric in metrics]
        topk_values = [float(topk_row[metric.stats_col]) for metric in metrics]  # type:ignore
        mmr_values = [float(mmr_row[metric.stats_col]) for metric in metrics]  # type:ignore
        fac_loc_values = [float(fac_loc_row[metric.stats_col]) for metric in metrics]  # type:ignore

        ax.set_facecolor(control_facecolor)
        ax.plot(
            x_positions,
            topk_values,
            color=get_style('top_k')['color'],
            lw=2.0,
            marker='o',
            ms=4.5,
            label='top-k',
        )
        ax.plot(
            x_positions,
            mmr_values,
            color=styles['mmr']['color'],
            lw=2.0,
            marker='o',
            ms=4.5,
            label='MMR',
        )
        ax.plot(
            x_positions,
            fac_loc_values,
            color=styles['fac_loc']['color'],
            lw=2.0,
            marker='o',
            ms=4.5,
            label='FacLoc',
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(*row_ylim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, min_n_ticks=7))
        ax.yaxis.set_major_formatter(
            FormatStrFormatter(_best_facet_hit_rate_tick_format(*row_ylim))
        )
        _annotate_pair_metric_deltas(
            ax,
            x_positions=x_positions,
            fac_loc_values=fac_loc_values,
            mmr_values=mmr_values,
            metrics=metrics,
        )
        ax.set_ylabel(row_title, fontsize=9)
        ax.set_title(
            '\n'.join([
                f'k={k}',
                f'FacLoc λ={fac_loc_lam:.2f} | MMR λ={mmr_lam:.2f}',
                lambda_selection_short_label(lambda_selection),
            ]),
            fontsize=8.8,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=3,
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 0.05),
        )
        _style_legend(legend)

    fig.suptitle(figure_title, fontsize=12)
    _figure_note(
        fig,
        f'Each row shows bare top-k for that k plus each strategy {lambda_selection_policy_note(lambda_selection)}.',
    )
    fig.tight_layout(rect=(0.02, 0.12, 1, 0.95))
    fig.savefig(out_dir / output_filename, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_diagnostics_at_best_lambda_for_k(
    stats_df: pl.DataFrame,
    out_dir: Path,
    lambda_selection: LambdaSelectionCfg,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Diagnostic metrics that explain why a strategy wins or fails."""
    _set_active_plot_theme(plot_theme)
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

    fig, axes = _grid_figure('diagnostics_at_best_lambda_for_k', sharex=True)
    for ax, (col, title, higher_is_better) in zip(axes.flatten(), metric_cols, strict=False):
        for strategy in strategies:
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            if strategy == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [_cell_value(sub, k, None, col) for k in xs]
                ax.plot(xs, ys, color=style['color'], ls=style['ls'], lw=2.0, label=style['label'])
                continue
            best_df = _best_lam_rows(stats_df, strategy, k_values, lambda_selection)
            if best_df.height > 0:
                ax.plot(
                    best_df['k'].to_list(),
                    best_df[col].to_list(),
                    color=style['color'],
                    ls=style['ls'],
                    lw=2.0,
                    label=style['label'],
                )
                xs = best_df['k'].to_list()
                ys = [float(v) for v in best_df[col].to_list()]
                lams = [float(v) for v in best_df['lam'].to_list()]
                ax.scatter(
                    xs,
                    ys,
                    s=42,
                    facecolors=_theme_color('marker_facecolor'),
                    edgecolors=style['color'],
                    linewidths=1.4,
                    zorder=4,
                )
                _annotate_lambda_points(
                    ax,
                    xs,
                    ys,
                    lams,
                    color=style['color'],
                    placement='above' if strategy == 'mmr' else 'below',
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
        'Selection diagnostics - configured lambda* within strategy x k',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / 'diagnostics_at_best_lambda_for_k.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_answer_metrics_at_best_lambda_for_k(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
    lambda_selection: LambdaSelectionCfg,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Auxiliary answer-token overlap diagnostics."""
    _set_active_plot_theme(plot_theme)
    import matplotlib.pyplot as plt

    metrics = _available_answer_rouge_metrics(stats_df, results_df)
    if not metrics:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)
    fig, axes = _grid_figure('answer_metrics_at_best_lambda_for_k', sharex=True)

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

            best_df = _best_lam_rows(stats_df, strategy, k_values, lambda_selection)
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
                facecolors=_theme_color('marker_facecolor'),
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

    handles, labels = _first_legend_handles(axes)
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=min(len(labels), 5),
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
        _style_legend(legend)
    fig.suptitle(
        'Auxiliary answer-token ROUGE diagnostics - lambda* path per strategy',
        fontsize=12,
    )
    _figure_note(
        fig,
        lambda_selection_policy_note(lambda_selection),
        y=0.065,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out_dir / 'answer_metrics_at_best_lambda_for_k.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_answer_metrics_k_curves_for_lambda(
    stats_df: pl.DataFrame,
    out_dir: Path,
    plot_theme: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME,
) -> None:
    """Auxiliary ROUGE metrics as lambda changes."""
    _set_active_plot_theme(plot_theme)
    import matplotlib.pyplot as plt

    metric_cols = [
        (metric.stats_col, metric.title, metric.higher_is_better)
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
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    k_colors = _k_colors(plt, k_values)

    fig, axes = _grid_figure(
        'answer_metrics_k_curves_for_lambda',
        rows=len(metric_cols),
        cols=len(diversity_strategies),
        sharex=False,
    )

    for row_idx, (metric, title, higher_is_better) in enumerate(metric_cols):
        row_ylim = _shared_lambda_sensitivity_ylim(
            stats_df,
            metric=metric,
            strategies=diversity_strategies,
            k_values=k_values,
        )
        for col_idx, strategy in enumerate(diversity_strategies):
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            sampled_lambda_values = _sample_tick_values(
                _lambda_values_for_strategy(stats_df, strategy)
            )
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
                    ms=FOR_LAMBDA_K_CURVE_MARKER_SIZE,
                    label=f'k={k}',
                )
                _mark_best_lambda_curve_point(
                    ax,
                    ksub['lam'].to_list(),
                    ksub[metric].to_list(),
                    color=k_colors[k],
                    higher_is_better=higher_is_better,
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
                ax.set_xlabel(f'{style["label"]} lambda', fontsize=9)
            _set_lambda_tick_labels(ax, sampled_lambda_values)
            ax.set_ylim(*row_ylim)
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=len(k_values),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        _style_legend(legend)
    fig.suptitle(
        'Auxiliary answer-token ROUGE lambda sensitivity',
        fontsize=12,
    )
    _figure_note(
        fig, 'Dashed horizontal lines are the top-k reference at each k; line colors identify k'
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'answer_metrics_k_curves_for_lambda.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def get_style(strategy: str) -> dict[str, str]:
    styles = EVAL_PLOT_STRATEGY_STYLES[_ACTIVE_EVAL_PLOT_THEME]
    return styles.get(
        strategy,
        {'color': _theme_color('fallback_color'), 'ls': '-', 'label': strategy},
    )


_ACTIVE_EVAL_PLOT_THEME: EvalPlotTheme = DEFAULT_EVAL_PLOT_THEME
_ACTIVE_MATPLOTLIB_THEME: EvalPlotTheme | None = None


def _normalize_plot_theme(plot_theme: EvalPlotTheme | str | None = None) -> EvalPlotTheme:
    if plot_theme is None:
        return DEFAULT_EVAL_PLOT_THEME
    if plot_theme in EVAL_PLOT_THEMES:
        return cast(EvalPlotTheme, plot_theme)
    valid = ', '.join(repr(name) for name in EVAL_PLOT_THEMES)
    raise ValueError(f'Unknown evaluation plot theme {plot_theme!r}; expected one of: {valid}')


def _set_active_plot_theme(plot_theme: EvalPlotTheme | str | None = None) -> EvalPlotTheme:
    global _ACTIVE_EVAL_PLOT_THEME
    _ACTIVE_EVAL_PLOT_THEME = _normalize_plot_theme(plot_theme)
    return _ACTIVE_EVAL_PLOT_THEME


def _active_theme() -> dict[str, str]:
    return EVAL_PLOT_THEMES[_ACTIVE_EVAL_PLOT_THEME]


def _theme_color(key: str) -> str:
    return str(_active_theme()[key])


def _activate_plot_theme() -> None:
    """Apply the selected evaluation theme to Matplotlib defaults."""
    import matplotlib as mpl

    global _ACTIVE_MATPLOTLIB_THEME
    if _ACTIVE_MATPLOTLIB_THEME == _ACTIVE_EVAL_PLOT_THEME:
        return

    mpl.rcParams.update({
        'figure.facecolor': _theme_color('figure_facecolor'),
        'figure.edgecolor': _theme_color('figure_facecolor'),
        'savefig.facecolor': _theme_color('figure_facecolor'),
        'savefig.edgecolor': _theme_color('figure_facecolor'),
        'axes.facecolor': _theme_color('axes_facecolor'),
        'axes.edgecolor': _theme_color('spine_color'),
        'axes.labelcolor': _theme_color('text_color'),
        'axes.titlecolor': _theme_color('text_color'),
        'text.color': _theme_color('text_color'),
        'xtick.color': _theme_color('tick_color'),
        'ytick.color': _theme_color('tick_color'),
        'grid.color': _theme_color('grid_color'),
        'grid.alpha': 0.28,
        'legend.facecolor': _theme_color('figure_facecolor'),
        'legend.edgecolor': _theme_color('spine_color'),
        'legend.labelcolor': _theme_color('text_color'),
        'patch.edgecolor': _theme_color('spine_color'),
    })
    _ACTIVE_MATPLOTLIB_THEME = _ACTIVE_EVAL_PLOT_THEME


def _style_figure(fig: Figure) -> None:
    fig.patch.set_facecolor(_theme_color('figure_facecolor'))  # type:ignore
    fig.patch.set_edgecolor(_theme_color('figure_facecolor'))  # type:ignore


def _style_axis(ax: Any, *, facecolor: str | None = None) -> None:
    ax.set_facecolor(facecolor or _theme_color('axes_facecolor'))
    ax.tick_params(colors=_theme_color('tick_color'), which='both')
    ax.xaxis.label.set_color(_theme_color('text_color'))
    ax.yaxis.label.set_color(_theme_color('text_color'))
    ax.title.set_color(_theme_color('text_color'))
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_color(_theme_color('tick_color'))
    for spine in ax.spines.values():
        spine.set_color(_theme_color('spine_color'))


def _style_existing_axes(fig: Figure) -> None:
    _style_figure(fig)
    for ax in fig.axes:
        _style_axis(ax)


def _style_legend(legend: Any | None) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(_theme_color('figure_facecolor'))
    frame.set_edgecolor(_theme_color('spine_color'))
    for text in legend.get_texts():
        text.set_color(_theme_color('text_color'))


def _k_colors(plt: Any, k_values: list[int]) -> dict[int, Any]:
    """Sample a colormap away from its darkest end so every k line is visible."""
    cmap = plt.get_cmap(EVAL_PLOT_K_COLORMAP)  # type: ignore[attr-defined]
    denom = max(len(k_values) - 1, 1)
    return {k: cmap(0.20 + 0.75 * i / denom) for i, k in enumerate(k_values)}


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
    """Compatibility wrapper for the standard 4x3 main-metric grid."""
    del n_metrics
    return _grid_figure('metrics_at_best_lambda_for_k', sharex=True)


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

    _activate_plot_theme()
    width = width_per_col * cols
    height = height_per_row * rows + footer_height
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharex=sharex, squeeze=False)
    _style_figure(fig)
    for ax in axes.flatten():
        _style_axis(ax)
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

    return _padded_ylim(values)


def _shared_delta_vs_topk_ylim(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    *,
    metric: str,
    result_col: str,
    strategies: list[str],
    k_values: list[int],
) -> tuple[float, float]:
    values: list[float] = [0.0]
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')

    for strategy in strategies:
        lambda_values = _lambda_values_for_strategy(stats_df, strategy)
        for k in k_values:
            ref_row = topk_df.filter(pl.col('k') == k)
            ref_val = float(ref_row[metric][0]) if ref_row.height > 0 else 0.0
            for lam in lambda_values:
                sub = stats_df.filter(
                    (pl.col('strategy') == strategy) & (pl.col('k') == k) & (pl.col('lam') == lam)
                )
                delta = float(sub[metric][0]) - ref_val if sub.height > 0 else 0.0
                ci = _paired_delta_ci(results_df, strategy, k, lam, result_col)
                ci_display = 0.0 if _is_nonfinite_float(ci) else ci
                values.extend([delta - ci_display, delta, delta + ci_display])

    return _padded_ylim(values)


def _padded_ylim(values: Iterable[float]) -> tuple[float, float]:
    finite_values = [value for value in values if not _is_nonfinite_float(value)]
    if not finite_values:
        return (0.0, 1.0)

    lower = min(finite_values)
    upper = max(finite_values)
    if lower == upper:
        padding = abs(lower) * 0.05 if lower != 0.0 else 0.05
        return (lower - padding, upper + padding)

    padding = (upper - lower) * 0.05
    return (lower - padding, upper + padding)


def _mark_best_lambda_curve_point(
    ax: Any,
    xs: Sequence[object],
    ys: Sequence[object],
    *,
    color: Any,
    higher_is_better: bool,
) -> None:
    valid_points: list[tuple[int, float]] = []
    for idx, y in enumerate(ys):
        value = _finite_float_or_none(y)
        if value is not None:
            valid_points.append((idx, value))
    if not valid_points:
        return

    best_idx, _best_value = (
        max(valid_points, key=lambda item: item[1])
        if higher_is_better
        else min(valid_points, key=lambda item: item[1])
    )
    best_x = _finite_float_or_none(xs[best_idx])
    best_y = _finite_float_or_none(ys[best_idx])
    if best_x is None or best_y is None:
        return

    ax.scatter(
        [best_x],
        [best_y],
        s=FOR_LAMBDA_K_CURVE_BEST_MARKER_SIZE,
        facecolors=_theme_color('marker_facecolor'),
        edgecolors=color,
        linewidths=2.0,
        zorder=5,
    )


def _is_nonfinite_float(value: float) -> bool:
    import math

    return not math.isfinite(value)


def _finite_float_or_none(value: object) -> float | None:
    try:
        numeric_value = float(value)  # type:ignore
    except TypeError, ValueError:
        return None
    return None if _is_nonfinite_float(numeric_value) else numeric_value


def _best_lambda_note(
    stats_df: pl.DataFrame,
    strategies: list[str],
    k_values: list[int],
    lambda_selection: LambdaSelectionCfg,
) -> str:
    parts = [lambda_selection_policy_note(lambda_selection)]
    for strategy in strategies:
        best_df = _best_lam_rows(stats_df, strategy, k_values, lambda_selection)
        if best_df.height == 0 or 'lam' not in best_df.columns:
            continue
        mapping = ', '.join(
            f'k={int(k)} -> {float(lam):.2f}'
            for k, lam in zip(best_df['k'].to_list(), best_df['lam'].to_list(), strict=True)
        )
        parts.append(f'{get_style(strategy)["label"]}: {mapping}')
    return ' | '.join(parts)


def _figure_note(fig: Figure, text: str, *, y: float = 0.025) -> None:
    fig.text(
        0.5, y, text, ha='center', va='bottom', fontsize=8, color=_theme_color('muted_text_color')
    )


def _add_centered_row_labels(
    fig: Figure,
    axes: NDArray[Any],
    labels: list[str],
    *,
    fontsize: float,
    top_pad: float,
) -> None:
    n_rows = min(len(labels), axes.shape[0])
    first_row_gap = None
    if n_rows > 1:
        first_row_bottom = min(ax.get_position().y0 for ax in axes[0])
        second_row_top = max(ax.get_position().y1 for ax in axes[1])
        first_row_gap = first_row_bottom - second_row_top
    for row_idx in range(n_rows):
        row_axes = [axes[row_idx][col_idx] for col_idx in range(axes.shape[1])]
        left = min(ax.get_position().x0 for ax in row_axes)
        right = max(ax.get_position().x1 for ax in row_axes)
        top = max(ax.get_position().y1 for ax in row_axes)
        if row_idx == 0:
            y = top + (first_row_gap * 0.5 if first_row_gap is not None else top_pad)
        else:
            prev_row_axes = [axes[row_idx - 1][col_idx] for col_idx in range(axes.shape[1])]
            prev_bottom = min(ax.get_position().y0 for ax in prev_row_axes)
            y = top + (prev_bottom - top) * 0.5
        fig.text(
            (left + right) / 2,
            y,
            labels[row_idx],
            ha='center',
            va='center',
            fontsize=fontsize,
            color=_theme_color('text_color'),
        )


def _panel_title(title: str, higher_is_better: bool, *, delta: bool = False) -> str:
    del delta
    if higher_is_better:
        return title
    return f'{title} [lower better]'


def _build_strategy_lambda_heatmap_data(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
) -> tuple[dict[str, dict[str, NDArray[Any]]], list[int], dict[str, list[float]]] | None:
    import numpy as np

    present_strategies = set(stats_df['strategy'].unique().to_list())
    if 'fac_loc' not in present_strategies or 'mmr' not in present_strategies:
        return None

    fac_lambdas = _lambda_values_for_strategy(stats_df, 'fac_loc')
    mmr_lambdas = _lambda_values_for_strategy(stats_df, 'mmr')
    if not fac_lambdas or not mmr_lambdas:
        return None
    normalized_positions = _normalized_grid_positions(max(len(fac_lambdas), len(mmr_lambdas)))

    k_values = sorted(int(k) for k in stats_df['k'].unique().to_list())
    metrics = _available_metrics(stats_df, results_df)
    if not metrics:
        return None

    metric_matrices: dict[str, dict[str, NDArray[Any]]] = {}
    for stats_col, _result_col, _title, higher_is_better in metrics:
        fac_matrix = np.full((len(k_values), len(fac_lambdas)), np.nan, dtype=float)
        mmr_matrix = np.full((len(k_values), len(mmr_lambdas)), np.nan, dtype=float)
        adv_matrix = np.full((len(k_values), len(normalized_positions)), np.nan, dtype=float)
        for k_idx, k in enumerate(k_values):
            for lam_idx, lam in enumerate(fac_lambdas):
                fac_val = _cell_value_for_strategy(stats_df, 'fac_loc', k, lam, stats_col)
                fac_matrix[k_idx, lam_idx] = fac_val
            for lam_idx, lam in enumerate(mmr_lambdas):
                mmr_val = _cell_value_for_strategy(stats_df, 'mmr', k, lam, stats_col)
                mmr_matrix[k_idx, lam_idx] = mmr_val
            for pos_idx, position in enumerate(normalized_positions):
                fac_lam = _lambda_at_normalized_position(fac_lambdas, position)
                mmr_lam = _lambda_at_normalized_position(mmr_lambdas, position)
                fac_val = _cell_value_for_strategy(stats_df, 'fac_loc', k, fac_lam, stats_col)
                mmr_val = _cell_value_for_strategy(stats_df, 'mmr', k, mmr_lam, stats_col)
                if np.isnan(fac_val) or np.isnan(mmr_val):
                    continue
                adv_matrix[k_idx, pos_idx] = (
                    fac_val - mmr_val if higher_is_better else mmr_val - fac_val
                )
        metric_matrices[stats_col] = {
            'fac_loc': fac_matrix,
            'mmr': mmr_matrix,
            'advantage': adv_matrix,
        }
    return (
        metric_matrices,
        k_values,
        {
            'fac_loc': fac_lambdas,
            'mmr': mmr_lambdas,
            'advantage': normalized_positions,
        },
    )


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
    ax.grid(which='minor', color=_theme_color('heatmap_grid_color'), linewidth=0.8, alpha=0.9)
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
                    _theme_color('heatmap_text_light')
                    if abs(value)
                    > max(abs(float(finite_values.min())), abs(float(finite_values.max()))) * 0.45
                    else _theme_color('heatmap_text_dark')
                )
            else:
                text_color = (
                    _theme_color('heatmap_text_light')
                    if value >= threshold
                    else _theme_color('heatmap_text_dark')
                )
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
    x_label: str,
) -> str:
    direction_note = (
        'Positive values mean FacLoc is better for this metric'
        if advantage
        else ('Higher is better' if higher_is_better else 'Lower is better')
    )
    return (
        f'{panel_title}<br>'
        f'metric={title}<br>'
        f'{x_label}=%{{x}}<br>'
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


def _bold_zero_ytick_label(ax: Any) -> None:
    for tick_value, label in zip(ax.get_yticks(), ax.get_yticklabels(), strict=False):
        if abs(float(tick_value)) < 1e-12:
            label.set_fontweight('bold')
            label.set_color(_theme_color('zero_line_color'))


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
        color=_theme_color('muted_text_color'),
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


def _lambda_values_for_strategy(df: pl.DataFrame, strategy: str) -> list[float]:
    if 'lam' not in df.columns or 'strategy' not in df.columns:
        return []
    strategy_df = df.filter(pl.col('strategy') == strategy)
    return sorted(float(lam) for lam in strategy_df['lam'].drop_nulls().unique().to_list())


def _normalized_grid_positions(n_values: int) -> list[float]:
    if n_values <= 0:
        return []
    if n_values == 1:
        return [0.0]
    return [idx / (n_values - 1) for idx in range(n_values)]


def _lambda_at_normalized_position(lambda_values: list[float], position: float) -> float:
    if not lambda_values:
        raise ValueError('cannot map a normalized position onto an empty lambda grid')
    if len(lambda_values) == 1:
        return lambda_values[0]
    bounded_position = min(max(float(position), 0.0), 1.0)
    index = int(bounded_position * (len(lambda_values) - 1) + 0.5)
    return lambda_values[min(index, len(lambda_values) - 1)]


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


def _metrics_delta_vs_topk_k_curves_for_lambda_geometry(
    n_k_values: int, n_groups: int
) -> GainOverTopkGeometry:
    import numpy as np

    min_group_span = 2.8
    min_bar_slot_width = 0.12
    slot_width = max(min_group_span / max(n_groups, 1), min_bar_slot_width)
    group_span = slot_width * n_groups
    k_gap = max(0.9, min(2.2, group_span * 0.16))
    k_positions = (np.arange(n_k_values, dtype=float) * (group_span + k_gap)).tolist()
    bar_width = slot_width * 0.96
    first_center = k_positions[0] + _metrics_delta_vs_topk_k_curves_for_lambda_offset(
        group_idx=0,
        n_groups=n_groups,
        slot_width=slot_width,
    )
    last_center = k_positions[-1] + _metrics_delta_vs_topk_k_curves_for_lambda_offset(
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


def _metrics_delta_vs_topk_k_curves_for_lambda_offset(
    *, group_idx: int, n_groups: int, slot_width: float
) -> float:
    return (group_idx - n_groups / 2 + 0.5) * slot_width


def _metrics_delta_vs_topk_k_curves_for_lambda_footer_layout(
    lambda_values: list[float],
) -> GainOverTopkFooterLayout:
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
                'facecolor': _theme_color('annotation_facecolor'),
                'edgecolor': color,
                'linewidth': 0.45,
                'alpha': 0.92,
            },
            zorder=5,
        )


def _annotate_delta_bars(
    ax: Any,
    bars: Iterable[Any],
    values: list[float],
    *,
    lambda_values: list[float | None] | None = None,
    color: str | None = None,
) -> None:
    color = _theme_color('text_color') if color is None else color
    ymin, ymax = ax.get_ylim()  # type: ignore[attr-defined]
    span = max(ymax - ymin, 1e-6)
    value_offset = span * 0.025
    lambda_step_points = 11
    lambda_labels = lambda_values if lambda_values is not None else [None for _ in values]
    for bar, value, lam in zip(bars, values, lambda_labels, strict=True):
        x = bar.get_x() + bar.get_width() / 2
        y = value + value_offset if value >= 0 else value - value_offset
        va = 'bottom' if value >= 0 else 'top'
        if abs(value) >= 0.005:
            ax.text(  # type: ignore[attr-defined]
                x,
                y,
                f'{value:+.3f}',
                ha='center',
                va=va,
                fontsize=6.5,
                clip_on=True,
            )
        if lam is None:
            continue
        ax.annotate(
            f'λ={lam:.2f}',
            xy=(x, y),
            xytext=(0, lambda_step_points if value >= 0 else -lambda_step_points),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=6.5,
            color=color,
            bbox={
                'boxstyle': 'round,pad=0.18',
                'facecolor': _theme_color('annotation_facecolor'),
                'edgecolor': color,
                'linewidth': 0.45,
                'alpha': 0.92,
            },
            clip_on=True,
            zorder=5,
        )


def _draw_bar_slot_guides(ax: Any, centers: list[float], width: float) -> None:
    if width <= 0:
        return
    for center in centers:
        ax.axvspan(
            center - width / 2,
            center + width / 2,
            facecolor='none',
            edgecolor=_theme_color('spine_color'),
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
            'facecolor': _theme_color('annotation_facecolor'),
            'edgecolor': color,
            'linewidth': 0.35,
            'alpha': 0.9,
        },
        clip_on=True,
        zorder=5,
    )


def _best_lam_rows(
    stats_df: pl.DataFrame,
    strategy: str,
    k_values: list[int],
    lambda_selection: LambdaSelectionCfg,
) -> pl.DataFrame:
    return select_best_lambda_rows(
        stats_df,
        strategy=strategy,
        k_values=k_values,
        cfg=lambda_selection,
    )


def _best_lambda_control_rows(
    stats_df: pl.DataFrame,
    k_values: list[int],
    lambda_selection: LambdaSelectionCfg,
) -> pl.DataFrame:
    rows = []
    for k in [int(v) for v in k_values]:
        fac_loc_best = select_best_lambda_row(
            stats_df,
            strategy='fac_loc',
            k=k,
            cfg=lambda_selection,
        )
        mmr_best = select_best_lambda_row(
            stats_df,
            strategy='mmr',
            k=k,
            cfg=lambda_selection,
        )
        if fac_loc_best is None or mmr_best is None:
            continue
        rows.append(
            pl.DataFrame({
                'k': [k],
                'fac_loc_lam': [float(fac_loc_best['lam'])],  # type:ignore
                'mmr_lam': [float(mmr_best['lam'])],  # type:ignore
            })
        )
    return pl.concat(rows).sort('k') if rows else pl.DataFrame()


def _strategy_k_lambda_row(
    stats_df: pl.DataFrame,
    *,
    strategy: str,
    k: int,
    lam: float,
) -> dict[str, object] | None:
    sub = stats_df.filter(
        (pl.col('strategy') == strategy) & (pl.col('k') == k) & (pl.col('lam') == lam)
    )
    return sub.row(0, named=True) if sub.height > 0 else None


def _k_row(df: pl.DataFrame, k: int) -> dict[str, object] | None:
    sub = df.filter(pl.col('k') == k)
    return sub.row(0, named=True) if sub.height > 0 else None


def _topk_k_row(stats_df: pl.DataFrame, k: int) -> dict[str, object] | None:
    sub = stats_df.filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k))
    return sub.row(0, named=True) if sub.height > 0 else None


def _best_facet_hit_rate_tick_format(ymin: float, ymax: float) -> str:
    span = abs(ymax - ymin)
    if span <= 0.1:
        return '%.3f'
    if span <= 1.0:
        return '%.2f'
    return '%.1f'


def _annotate_pair_metric_deltas(
    ax: Any,
    *,
    x_positions: Any,
    fac_loc_values: list[float],
    mmr_values: list[float],
    metrics: list[NamedPlotMetric],
) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    offset = span * 0.04
    for x, fac_val, mmr_val, metric in zip(
        x_positions,
        fac_loc_values,
        mmr_values,
        metrics,
        strict=True,
    ):
        delta = fac_val - mmr_val
        if abs(delta) < 1e-12:
            color = _theme_color('muted_text_color')
        else:
            fac_loc_better = delta > 0.0 if metric.higher_is_better else delta < 0.0
            color = get_style('fac_loc')['color'] if fac_loc_better else get_style('mmr')['color']
        y = max(fac_val, mmr_val) + offset
        ax.text(
            x,
            y,
            f'{delta:+.3f}',
            ha='center',
            va='bottom',
            fontsize=6.3,
            color=color,
            bbox={
                'boxstyle': 'round,pad=0.14',
                'facecolor': _theme_color('annotation_facecolor'),
                'edgecolor': color,
                'linewidth': 0.35,
                'alpha': 0.9,
            },
            clip_on=False,
            zorder=6,
        )


def _lambda_agreeing_metrics_ylim(
    stats_df: pl.DataFrame,
    best_pairs: pl.DataFrame,
    metric_cols: list[str],
    *,
    include_topk: bool = False,
) -> tuple[float, float]:
    values: list[float] = []
    for row in best_pairs.iter_rows(named=True):
        if include_topk:
            topk_row = _topk_k_row(stats_df, int(row['k']))
            if topk_row is not None:
                for metric_col in metric_cols:
                    value = topk_row.get(metric_col)
                    if value is None:
                        continue
                    values.append(float(value))  # type:ignore
        for strategy, lam_key in [('fac_loc', 'fac_loc_lam'), ('mmr', 'mmr_lam')]:
            metric_row = _strategy_k_lambda_row(
                stats_df,
                strategy=strategy,
                k=int(row['k']),
                lam=float(row[lam_key]),
            )
            if metric_row is None:
                continue
            for metric_col in metric_cols:
                value = metric_row.get(metric_col)
                if value is None:
                    continue
                values.append(float(value))  # type:ignore

    if not values:
        return (0.0, 1.0)

    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = abs(lower) * 0.05 if lower != 0.0 else 0.05
        return (lower - padding, upper + padding)

    padding = (upper - lower) * 0.14
    return (lower - padding, upper + padding)


def _best_topk_k(results_df: pl.DataFrame) -> int:
    topk = results_df.filter(pl.col('strategy') == 'top_k')
    if topk.height == 0:
        return int(results_df['k'].max())  # type: ignore[arg-type]
    ranked = (
        topk
        .group_by('k')
        .agg(
            pl.col('facet_coverage').median().alias('med_fc'),
        )
        .sort(['med_fc', 'k'], descending=[True, False])
    )
    return int(ranked['k'][0])


def _best_result_slice(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    strategy: str,
    k: int,
    lambda_selection: LambdaSelectionCfg,
) -> pl.DataFrame:
    strat_df = results_df.filter((pl.col('strategy') == strategy) & (pl.col('k') == k))
    if strategy == 'top_k' or strat_df.height == 0:
        return strat_df
    best_row = select_best_lambda_row(
        stats_df,
        strategy=strategy,
        k=k,
        cfg=lambda_selection,
    )
    return (
        strat_df.filter(pl.col('lam') == float(best_row['lam']))
        if best_row is not None and 'lam' in best_row
        else strat_df
    )


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
        legend = fig.legend(  # type: ignore[attr-defined]
            handles,
            labels,
            loc='lower center',
            ncol=min(len(seen), 5),
            fontsize=9,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
        )
        _style_legend(legend)


def _first_legend_handles(axes: NDArray[Any]) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes.flatten():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels, strict=True):
            if label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    return handles, labels
