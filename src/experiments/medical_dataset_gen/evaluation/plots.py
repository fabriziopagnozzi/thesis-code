from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from matplotlib.figure import Figure
from numpy.typing import NDArray

from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
    read_parquet,
)

STRATEGY_STYLE: dict[str, dict[str, str]] = {
    'top_k': {'color': '#333333', 'ls': '--', 'label': 'top-k'},
    'mmr': {'color': '#1f77b4', 'ls': '-', 'label': 'MMR'},
    'fac_loc': {'color': '#d62728', 'ls': '-', 'label': 'FacLoc'},
}

_METRICS = [
    ('FacetCoverage@k', 'facet_coverage', 'FacetCoverage@k', True),
    ('Precision@k', 'gold_precision', 'Precision@k', True),
    ('Recall@k', 'gold_recall', 'Recall@k', True),
    ('FacetMRR@k', 'facet_mrr_at_k', 'FacetMRR@k', True),
    ('alpha-nDCG@k', 'alpha_ndcg', 'alpha-nDCG@k', True),
    ('DistractorRate', 'distractor_rate', 'DistractorRate', False),
]

_DIAGNOSTIC_METRICS = [
    ('DistractorRate', 'distractor_rate', 'DistractorRate', False),
    ('NearMissDistractorRate', 'near_miss_distractor_rate', 'NearMissDistractorRate', False),
    ('BackgroundOutlierRate', 'background_outlier_rate', 'BackgroundOutlierRate', False),
    ('DominantFacetRate', 'dominant_facet_rate', 'DominantFacetRate', False),
    ('RedundantGoldRate', 'redundant_gold_rate', 'RedundantGoldRate', False),
    ('fac', 'fac_cov_score', 'Facility-Location Objective', True),
    ('avg_cos', 'avg_cos', 'Average Query Cosine', True),
    ('jac', 'jaccard_vs_topk', 'Jaccard vs top-k', True),
]

_ANSWER_ROUGE_METRICS = [
    ('AnswerROUGE1Recall@k', 'answer_rouge1_recall', 'Answer ROUGE-1 Recall@k', True),
    ('AnswerROUGE1Precision@k', 'answer_rouge1_precision', 'Answer ROUGE-1 Precision@k', True),
    ('AnswerROUGE2Recall@k', 'answer_rouge2_recall', 'Answer ROUGE-2 Recall@k', True),
    (
        'MacroFacetAnswerROUGE1Recall@k',
        'macro_facet_answer_rouge1_recall',
        'Macro Facet Answer ROUGE-1 Recall@k',
        True,
    ),
]

_PRIMARY_SORT = ['FacetCoverage@k', 'Precision@k', 'DistractorRate', 'alpha-nDCG@k']
_PRIMARY_DESC = [True, True, False, True]
_LAMBDA_POLICY_NOTE = (
    'lambda*: max mean FacetCoverage@k within strategy x k; ties prefer higher '
    'Precision@k, then lower DistractorRate'
)


def store_eval_figures(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> None:
    _ = cfg
    stats_path = paths.table_path('evaluation_stats')
    results_path = paths.table_path('evaluation_results')
    if not stats_path.exists() or not results_path.exists():
        print('Skipping eval figures: evaluation_stats or evaluation_results not found')
        return

    stats_df = read_parquet(paths, 'evaluation_stats')
    results_df = read_parquet(paths, 'evaluation_results')
    if stats_df.is_empty() or results_df.is_empty():
        print('Skipping eval figures: evaluation tables are empty')
        return

    out_dir = paths.figures_dir / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_strategy_comparison(stats_df, results_df, out_dir)
    plot_strategy_comparison_all_lambdas(stats_df, results_df, out_dir)
    plot_lambda_sensitivity(stats_df, out_dir)
    plot_per_query_distributions(results_df, out_dir)
    plot_gain_over_topk(stats_df, results_df, out_dir)
    plot_gain_over_topk_simple(stats_df, results_df, out_dir)
    plot_selection_diagnostics(stats_df, out_dir)
    plot_answer_rouge_comparison(stats_df, results_df, out_dir)
    plot_answer_rouge_lambda_sensitivity(stats_df, out_dir)

    print(f'[plots] saved evaluation figures to {out_dir}')


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
    fig, axes = _metric_grid_figure(len(metrics), sharex=True)

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
                    _ci_half_width(_query_vals(results_df, strategy, k, None, result_col))
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
                _ci_half_width(_query_vals(results_df, strategy, k, k_to_lam.get(k), result_col))
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

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(stats_col, fontsize=9)
        ax.set_xticks(k_values)
        ax.grid(axis='y', alpha=0.3)
        if not higher_is_better:
            ax.text(
                0.98,
                0.04,
                'lower is better',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=7,
                color='#555555',
            )

    for ax in axes[-1]:
        ax.set_xlabel('k', fontsize=9)
    for ax in axes.flatten()[len(metrics) :]:
        ax.set_visible(False)

    _figure_legend(fig, axes.flatten())
    fig.suptitle(
        'Strategy comparison - lambda* path per strategy with exact lambda labels at each k',
        fontsize=12,
    )
    _figure_note(fig, f'{_LAMBDA_POLICY_NOTE}; see lambda_sensitivity.png for the full sweep')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'strategy_comparison.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_lambda_sensitivity(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """All strategy metrics as lambda changes."""
    import matplotlib.pyplot as plt

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    lambda_values = _lambda_values(stats_df)
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    metric_cols = [
        (stats_col, title) for stats_col, _, title, _ in _METRICS if stats_col in stats_df.columns
    ]
    cmap = plt.get_cmap('viridis')  # type: ignore[attr-defined]
    k_colors = {k: cmap(i / max(len(k_values) - 1, 1)) for i, k in enumerate(k_values)}

    fig, axes = plt.subplots(
        len(metric_cols),
        len(diversity_strategies),
        figsize=(4.0 * len(diversity_strategies), 2.0 * len(metric_cols) + 2.0),
        sharex=True,
        squeeze=False,
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
                    ax.axhline(float(ref[metric][0]), color=k_colors[k], ls='--', lw=1.0, alpha=0.5)

            if row_idx == 0:
                ax.set_title(style['label'], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=10)
            if row_idx == len(metric_cols) - 1:
                ax.set_xlabel('lambda', fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            ax.set_xticks(lambda_values)
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
        'Lambda sensitivity - each row is a metric, each column is a strategy', fontsize=12
    )
    _figure_note(
        fig, 'Dashed horizontal lines are the top-k reference at each k; line colors identify k'
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'lambda_sensitivity.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_strategy_comparison_all_lambdas(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Metric-vs-k lines, fac-loc vs MMR, split into one column per lambda."""
    import matplotlib.pyplot as plt

    lambda_values = _lambda_values(stats_df)
    if not lambda_values:
        return

    present_strategies = set(stats_df['strategy'].unique().to_list())
    strategies = [strategy for strategy in ['fac_loc', 'mmr'] if strategy in present_strategies]
    if len(strategies) < 2:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    metrics = _available_metrics(stats_df, results_df)
    if not metrics:
        return

    fig, axes = _variable_grid_figure(
        rows=len(metrics),
        cols=len(lambda_values),
        sharex=True,
        width_per_col=3.8,
        height_per_row=2.4,
        footer_height=1.4,
    )

    for row_idx, (stats_col, result_col, title, higher_is_better) in enumerate(metrics):
        for col_idx, lam in enumerate(lambda_values):
            ax = axes[row_idx][col_idx]
            for strategy in strategies:
                style = get_style(strategy)
                sub = stats_df.filter(
                    (pl.col('strategy') == strategy) & (pl.col('lam') == lam)
                ).sort('k')
                if sub.height == 0:
                    continue
                xs = [int(k) for k in sub['k'].to_list()]
                ys = [float(v) for v in sub[stats_col].to_list()]
                ci = [
                    _ci_half_width(_query_vals(results_df, strategy, k, lam, result_col))
                    for k in xs
                ]
                _plot_ci_line(ax, xs, ys, ci, style, lw=2.0, label=style['label'], zorder=3)
                ax.scatter(
                    xs,
                    ys,
                    s=28,
                    facecolors='white',
                    edgecolors=style['color'],
                    linewidths=1.2,
                    zorder=4,
                )
                _plot_error_caps(ax, xs, ys, ci, style, zorder=2)

            if row_idx == 0:
                ax.set_title(f'lambda={lam:.2f}', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.set_xticks(k_values)
            ax.grid(axis='y', alpha=0.3)
            if not higher_is_better:
                ax.text(
                    0.98,
                    0.04,
                    'lower is better',
                    transform=ax.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=7,
                    color='#555555',
                )
            if row_idx == len(metrics) - 1:
                ax.set_xlabel('k', fontsize=9)
            else:
                ax.tick_params(labelbottom=False)

    _figure_legend(fig, axes.flatten())
    fig.suptitle(
        'Strategy comparison across all lambdas - each row is a metric, each column is a lambda',
        fontsize=12,
    )
    _figure_note(
        fig, 'Each panel compares FacLoc and MMR at a fixed lambda; shaded bands are 95% CI'
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'strategy_comparison_all_lambdas.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_per_query_distributions(results_df: pl.DataFrame, out_dir: Path) -> None:
    """Violin plots at the top-k best k and each strategy's best primary lambda."""
    import matplotlib.pyplot as plt
    import numpy as np

    best_k = _best_topk_k(results_df)
    strategies = _ordered_strategies(results_df)
    metric_cols = [
        ('facet_coverage', 'FacetCoverage@k'),
        ('distractor_rate', 'DistractorRate'),
        ('gold_precision', 'Precision@k'),
        ('gold_recall', 'Recall@k'),
        ('facet_mrr_at_k', 'FacetMRR@k'),
        ('alpha_ndcg', 'alpha-nDCG@k'),
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
    fig, axes = _metric_grid_figure(len(metric_cols))

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
    x = np.arange(len(k_values)) * 3.6
    n_groups = len(diversity_strategies) * len(lambda_values)
    group_span = 2.8
    width = group_span / max(n_groups, 1)

    fig, axes = _metric_grid_figure(len(metrics), width_scale=2)
    for ax, (stats_col, result_col, title, higher_is_better) in zip(
        axes.flatten(),
        metrics,
        strict=False,
    ):
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
                offset = (group_idx - n_groups / 2 + 0.5) * width
                color = _lambda_shade(style['color'], lam, lambda_values)
                bars = ax.bar(
                    x + offset,
                    deltas,
                    width=width * 0.96,
                    color=color,
                    alpha=0.9,
                    edgecolor='#222222',
                    linewidth=0.35,
                    yerr=ci_display,
                    capsize=2.5,
                    error_kw={'linewidth': 0.75, 'ecolor': color},
                )
                _annotate_delta_bars(ax, bars, deltas)

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=9)
        ax.set_title(f'Delta {title} vs top-k', fontsize=10)
        ax.set_ylabel(f'Delta {stats_col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.margins(y=0.18)
        if not higher_is_better:
            ax.text(
                0.98,
                0.04,
                'negative is better',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=7,
                color='#555555',
            )

    for ax in axes.flatten()[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(
        'Gain over top-k by strategy and lambda, paired 95% CI',
        fontsize=12,
    )
    _add_lambda_shade_legend(fig, diversity_strategies, lambda_values)
    _figure_note(
        fig,
        'Each k bin is subdivided into per-lambda bars; top-k is the zero baseline',
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out_dir / 'gain_over_topk.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_gain_over_topk_simple(
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

    fig, axes = _metric_grid_figure(len(metrics), width_scale=1.15)
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
        ax.set_title(f'Delta {title} vs top-k', fontsize=10)
        ax.set_ylabel(f'Delta {stats_col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.margins(y=0.18)
        if not higher_is_better:
            ax.text(
                0.98,
                0.04,
                'negative is better',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=7,
                color='#555555',
            )

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
    fig.savefig(out_dir / 'gain_over_topk_simple.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_selection_diagnostics(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Diagnostic metrics that explain why a strategy wins or fails."""
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)
    metric_cols = [
        (stats_col, title, higher_is_better)
        for stats_col, _, title, higher_is_better in _DIAGNOSTIC_METRICS
        if stats_col in stats_df.columns
    ]

    fig, axes = _metric_grid_figure(len(metric_cols), sharex=True)
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
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(col, fontsize=9)
        ax.set_xticks(k_values)
        ax.grid(axis='y', alpha=0.3)
        if not higher_is_better:
            ax.text(
                0.98,
                0.04,
                'lower is better',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=7,
                color='#555555',
            )

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
    fig, axes = _metric_grid_figure(len(metrics), sharex=True)

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
                    _ci_half_width(_query_vals(results_df, strategy, k, None, result_col))
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
                _ci_half_width(_query_vals(results_df, strategy, k, k_to_lam.get(k), result_col))
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

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(stats_col, fontsize=9)
        ax.set_xticks(k_values)
        ax.grid(axis='y', alpha=0.3)
        if not higher_is_better:
            ax.text(
                0.98,
                0.04,
                'lower is better',
                transform=ax.transAxes,
                ha='right',
                va='bottom',
                fontsize=7,
                color='#555555',
            )

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
        f'ROUGE is diagnostic only; lambda* is still selected by coverage metrics. {_LAMBDA_POLICY_NOTE}',
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / 'answer_rouge_comparison.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_answer_rouge_lambda_sensitivity(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Auxiliary ROUGE metrics as lambda changes."""
    import matplotlib.pyplot as plt

    metric_cols = [
        (stats_col, title)
        for stats_col, _, title, _ in _ANSWER_ROUGE_METRICS
        if stats_col in stats_df.columns
    ]
    if not metric_cols:
        return

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    lambda_values = _lambda_values(stats_df)
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    cmap = plt.get_cmap('viridis')  # type: ignore[attr-defined]
    k_colors = {k: cmap(i / max(len(k_values) - 1, 1)) for i, k in enumerate(k_values)}

    fig, axes = plt.subplots(
        len(metric_cols),
        len(diversity_strategies),
        figsize=(4.0 * len(diversity_strategies), 2.0 * len(metric_cols) + 2.0),
        sharex=True,
        squeeze=False,
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
            else:
                ax.tick_params(labelbottom=False)
            ax.set_xticks(lambda_values)
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
        metric
        for metric in _METRICS
        if metric[0] in stats_df.columns and metric[1] in results_df.columns
    ]


def _available_answer_rouge_metrics(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
) -> list[tuple[str, str, str, bool]]:
    return [
        metric
        for metric in _ANSWER_ROUGE_METRICS
        if metric[0] in stats_df.columns and metric[1] in results_df.columns
    ]


def _available_sort(df: pl.DataFrame) -> tuple[list[str], list[bool]]:
    pairs = [
        (col, desc)
        for col, desc in zip(_PRIMARY_SORT, _PRIMARY_DESC, strict=True)
        if col in df.columns
    ]
    if not pairs:
        return ['lam'], [False]
    cols, desc = zip(*pairs, strict=True)
    return list(cols), list(desc)


def _metric_grid_figure(
    n_metrics: int,
    sharex: bool = False,
    width_scale: float = 1.0,
) -> tuple[Figure, NDArray[Any]]:
    import matplotlib.pyplot as plt

    _ = n_metrics
    rows = 2
    cols = 3
    width = 4.2 * cols * width_scale
    height = 3.4 * rows + 1.4
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharex=sharex, squeeze=False)
    return fig, axes


def _variable_grid_figure(
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


def _best_lambda_note(
    stats_df: pl.DataFrame,
    strategies: list[str],
    k_values: list[int],
) -> str:
    parts = [_LAMBDA_POLICY_NOTE]
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


def _figure_note(fig: Figure, text: str) -> None:
    fig.text(0.5, 0.025, text, ha='center', va='bottom', fontsize=8, color='#444444')


def _add_lambda_shade_legend(
    fig: Figure,
    strategies: list[str],
    lambda_values: list[float],
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
    bottom = 0.072
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
    fig.text(  # type: ignore[attr-defined]
        min(left + total_width + 0.025, 0.98),
        bottom + bar_height / 2,
        'λ values: ' + ', '.join(f'{lam:.2f}' for lam in lambda_values),
        ha='left',
        va='center',
        fontsize=7.5,
        color='#444444',
    )


def _lambda_values(df: pl.DataFrame) -> list[float]:
    if 'lam' not in df.columns:
        return []
    return sorted(float(lam) for lam in df['lam'].drop_nulls().unique().to_list())


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
    return tuple(channel * (1.0 - mix) + mix for channel in base_rgb)


def _annotate_lambda_points(
    ax: object,
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


def _annotate_delta_bars(ax: object, bars: object, values: list[float]) -> None:
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
        topk
        .group_by('k')
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
    ranked = (
        strat_df
        .group_by('lam')
        .agg(
            pl.col('facet_coverage').mean().alias('FacetCoverage@k'),
            pl.col('alpha_ndcg').mean().alias('alpha-nDCG@k')
            if 'alpha_ndcg' in strat_df.columns
            else pl.col('facet_coverage').mean().alias('alpha-nDCG@k'),
            pl.col('gold_precision').mean().alias('Precision@k'),
            pl.col('distractor_rate').mean().alias('DistractorRate'),
        )
        .sort(_PRIMARY_SORT, descending=_PRIMARY_DESC)
    )
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
    return _ci_half_width([float(v) for v in np.asarray(deltas, dtype=float)])


def _ci_half_width(values: list[float], z: float = 1.96) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float('nan')
    return z * float(arr.std(ddof=1)) / float(np.sqrt(len(arr)))


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


if __name__ == '__main__':
    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    store_eval_figures(cfg, paths)
