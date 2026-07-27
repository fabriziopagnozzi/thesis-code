from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.evaluation import plots as synthetic_eval_plots
from experiments.mimic.global_configs import MimicPaths, get_table_path, setup_logging

from .plot_adapters import adapt_results_for_synthetic_plots, adapt_stats_for_synthetic_plots
from .schemas_evaluation import EvaluateCfg

STRATEGY_STYLE: dict[str, dict] = {
    'top_k': {'color': '#333333', 'ls': '--', 'label': 'top-k'},
    'mmr': {'color': '#1f77b4', 'ls': '-', 'label': 'MMR'},
    'gmmr': {'color': '#2ca02c', 'ls': '-', 'label': 'gMMR'},
    'fac_loc': {'color': '#d62728', 'ls': '-', 'label': 'FacLoc'},
    'fps': {'color': '#9467bd', 'ls': '-', 'label': 'FPS'},
}

_METRICS = [
    ('AR', 'Aspect Recall (AR)'),
    ('WAR', 'Weighted AR'),
    ('GP', 'Gold Precision'),
    ('GR', 'Gold Recall'),
]

# Columns names differ between stats_df (aggregated) and results_df (per-query)
_ANSWER_METRICS_STATS = [
    ('ans_rouge1_rec', 'ROUGE-1 Recall'),
    ('ans_rouge1_prec', 'ROUGE-1 Precision'),
    ('ans_tfidf', 'TF-IDF Cosine'),
    ('ans_rouge1_f1', 'ROUGE-1 F1'),
]
_ANSWER_METRICS_RESULTS = [
    ('answer_rouge1_recall', 'ROUGE-1 Recall'),
    ('answer_rouge1_precision', 'ROUGE-1 Precision'),
    ('answer_tfidf_cosine', 'TF-IDF Cosine'),
    ('answer_rouge1_f1', 'ROUGE-1 F1'),
]

# Maps aggregated stats column names to per-query results column names
_STATS_TO_RESULTS_COL: dict[str, str] = {
    'AR': 'aspect_recall',
    'WAR': 'weighted_aspect_recall',
    'GP': 'gold_precision',
    'GR': 'gold_recall',
    'ans_rouge1_rec': 'answer_rouge1_recall',
    'ans_rouge1_prec': 'answer_rouge1_precision',
    'ans_tfidf': 'answer_tfidf_cosine',
    'ans_rouge1_f1': 'answer_rouge1_f1',
}


def get_style(strategy: str) -> dict:
    return STRATEGY_STYLE.get(strategy, {'color': '#aaaaaa', 'ls': '-', 'label': strategy})


def store_eval_figures(cfg: EvaluateCfg) -> None:
    out_dir = _resolve_eval_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = get_table_path('evaluation_stats')
    results_path = get_table_path('evaluation_results')
    if not stats_path.exists() or not results_path.exists():
        print('Skipping eval figures: evaluation_stats or evaluation_results not found')
        return

    stats_df = pl.read_parquet(stats_path)
    results_df = pl.read_parquet(results_path)

    synth_stats = adapt_stats_for_synthetic_plots(stats_df)
    synth_results = adapt_results_for_synthetic_plots(results_df)

    synthetic_eval_plots.plot_strategy_comparison(synth_stats, synth_results, out_dir)
    synthetic_eval_plots.plot_strategy_comparison_heatmap(synth_stats, synth_results, out_dir)
    synthetic_eval_plots.plot_lambda_sensitivity(synth_stats, out_dir)
    synthetic_eval_plots.plot_per_query_distributions(synth_results, out_dir)
    synthetic_eval_plots.plot_gain_over_topk(synth_stats, synth_results, out_dir)
    synthetic_eval_plots.plot_gain_over_topk_simple(synth_stats, synth_results, out_dir)
    synthetic_eval_plots.plot_selection_diagnostics(synth_stats, out_dir)
    synthetic_eval_plots.plot_answer_rouge_comparison(synth_stats, synth_results, out_dir)
    synthetic_eval_plots.plot_answer_rouge_lambda_sensitivity(synth_stats, out_dir)

    stratum_path = get_table_path('evaluation_stats_by_stratum')
    if stratum_path.exists():
        stratum_df = pl.read_parquet(stratum_path)
        plot_stratum_breakdown(stratum_df, out_dir)

        strata = sorted(stratum_df['stratum'].drop_nulls().unique().to_list())
        for s in strata:
            s_dir = out_dir / f'stratum_{s}'
            s_dir.mkdir(exist_ok=True)
            s_stats = adapt_stats_for_synthetic_plots(
                stratum_df.filter(pl.col('stratum') == s).drop('stratum')
            )
            s_results = adapt_results_for_synthetic_plots(
                results_df.filter(pl.col('stratum') == s)
                if 'stratum' in results_df.columns
                else results_df
            )
            synthetic_eval_plots.plot_strategy_comparison(s_stats, s_results, s_dir)
            synthetic_eval_plots.plot_strategy_comparison_heatmap(s_stats, s_results, s_dir)
            synthetic_eval_plots.plot_lambda_sensitivity(s_stats, s_dir)
            synthetic_eval_plots.plot_per_query_distributions(s_results, s_dir)
            synthetic_eval_plots.plot_gain_over_topk(s_stats, s_results, s_dir)
            synthetic_eval_plots.plot_gain_over_topk_simple(s_stats, s_results, s_dir)
            synthetic_eval_plots.plot_selection_diagnostics(s_stats, s_dir)
            synthetic_eval_plots.plot_answer_rouge_comparison(s_stats, s_results, s_dir)
            synthetic_eval_plots.plot_answer_rouge_lambda_sensitivity(s_stats, s_dir)

    print(f'Saved eval figures to {out_dir}')


def _resolve_eval_output_dir(cfg: EvaluateCfg) -> Path:
    if cfg.gold_mode == 'llm':
        return MimicPaths.figures_dir / 'eval_gold_annotations' / cfg.figures_subdir

    candidates = [
        MimicPaths.figures_dir / 'eval_structural' / cfg.figures_subdir,
        MimicPaths.figures_dir / 'eval' / cfg.figures_subdir,
    ]
    for candidate in candidates:
        if candidate.parent.exists():
            return candidate
    return candidates[0]


def plot_strategy_comparison(
    stats_df: pl.DataFrame, results_df: pl.DataFrame, out_dir: Path
) -> None:
    """2x2 grid: AR/WAR/GP/GR vs k, one line per strategy.

    Diversity strategies: thin faded lines per λ + bold line at best-AR λ.
    top_k: single dashed black baseline.
    Shaded bands = 95% CI across queries (n=66).
    """
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = sorted(stats_df['strategy'].unique().to_list())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for ax, (col, title) in zip(axes.flatten(), _METRICS, strict=True):
        res_col = _STATS_TO_RESULTS_COL.get(col, col)
        for strat in strategies:
            s = get_style(strat)
            sub = stats_df.filter(pl.col('strategy') == strat)

            if strat == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [float(sub.filter(pl.col('k') == k)[col][0]) for k in xs]
                ci = [
                    _ci_half_width(_query_vals(results_df, 'top_k', k, None, res_col)) for k in xs
                ]
                ax.fill_between(
                    xs,
                    [y - c for y, c in zip(ys, ci, strict=True)],
                    [y + c for y, c in zip(ys, ci, strict=True)],
                    color=s['color'],
                    alpha=0.12,
                    zorder=2,
                )
                ax.plot(xs, ys, color=s['color'], ls=s['ls'], lw=2.0, label=s['label'], zorder=3)
            else:
                lam_values = sorted(sub['lam'].drop_nulls().unique().to_list())
                for lam in lam_values:
                    lsub = sub.filter(pl.col('lam') == lam).sort('k')
                    ax.plot(
                        lsub['k'].to_list(),
                        lsub[col].to_list(),
                        color=s['color'],
                        ls=s['ls'],
                        lw=0.6,
                        alpha=0.3,
                        zorder=1,
                    )
                best_df = _best_lam_rows(stats_df, strat, k_values)
                if best_df.height > 0:
                    xs = best_df['k'].to_list()
                    ys = [float(v) for v in best_df[col].to_list()]
                    k_to_lam = dict(
                        zip(best_df['k'].to_list(), best_df['lam'].to_list(), strict=True)
                    )
                    ci = [
                        _ci_half_width(_query_vals(results_df, strat, k, k_to_lam.get(k), res_col))
                        for k in xs
                    ]
                    ax.fill_between(
                        xs,
                        [y - c for y, c in zip(ys, ci, strict=True)],
                        [y + c for y, c in zip(ys, ci, strict=True)],
                        color=s['color'],
                        alpha=0.12,
                        zorder=1.5,
                    )
                    ax.plot(
                        xs, ys, color=s['color'], ls=s['ls'], lw=2.2, label=s['label'], zorder=2
                    )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(col, fontsize=9)
        ax.set_xticks(k_values)
        ax.grid(axis='y', alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel('k', fontsize=9)

    handles, labels, seen = [], [], set()
    for ax in axes.flatten():
        for h, lbl in zip(*ax.get_legend_handles_labels(), strict=False):
            if lbl not in seen:
                handles.append(h)
                labels.append(lbl)
                seen.add(lbl)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(seen),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle('Strategy comparison - bold = best AR-λ, shaded = 95% CI', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / 'strategy_comparison.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_lambda_sensitivity(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """One subplot per diversity strategy: AR vs λ, one line per k.

    Dashed horizontal lines show the top_k reference AR at each k.
    Reveals whether fac_loc is more robust (flatter) than MMR.
    """
    import matplotlib.pyplot as plt

    diversity_strategies = [
        s for s in sorted(stats_df['strategy'].unique().to_list()) if s != 'top_k'
    ]
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')

    n = len(diversity_strategies)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    cmap = plt.get_cmap('viridis')  # type:ignore
    k_colors = {k: cmap(i / max(len(k_values) - 1, 1)) for i, k in enumerate(k_values)}

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, strat in enumerate(diversity_strategies):
        ax = axes[idx // ncols][idx % ncols]
        s = get_style(strat)
        sub = stats_df.filter(pl.col('strategy') == strat)

        for k in k_values:
            ksub = sub.filter(pl.col('k') == k).sort('lam')
            if ksub.height == 0:
                continue
            ax.plot(
                ksub['lam'].to_list(),
                ksub['AR'].to_list(),
                color=k_colors[k],
                ls=s['ls'],
                lw=1.8,
                marker='o',
                ms=4,
                label=f'k={k}',
            )
            ref = topk_df.filter(pl.col('k') == k)
            if ref.height > 0:
                ax.axhline(float(ref['AR'][0]), color=k_colors[k], ls='--', lw=1.0, alpha=0.5)

        ax.set_title(f'{s["label"]} - AR vs λ', fontsize=10)
        ax.set_xlabel('λ', fontsize=9)
        ax.set_ylabel('AR (mean)', fontsize=9)
        lam_ticks = sorted(sub['lam'].drop_nulls().unique().to_list())
        ax.set_xticks(lam_ticks)
        ax.grid(alpha=0.3)

    for idx in range(len(diversity_strategies), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if axes[0][0].get_legend_handles_labels()[0]:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=len(k_values),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle('λ sensitivity - AR (solid=strategy, dashed=top-k baseline)', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'lambda_sensitivity.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_per_query_distributions(results_df: pl.DataFrame, out_dir: Path) -> None:
    """2x2 violin plots: AR/WAR/GP/GR per strategy at best-k and best-λ.
    Uses per-query granularity (evaluation_results.parquet) to show spread,
    not just means.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    topk_df = results_df.filter(pl.col('strategy') == 'top_k')
    k_ar = (
        topk_df.group_by('k')
        .agg(pl.col('aspect_recall').median().alias('med_ar'))
        .sort('med_ar', descending=True)
    )
    best_k = int(k_ar['k'][0]) if k_ar.height > 0 else int(results_df['k'].max())  # type: ignore

    strategies = sorted(results_df['strategy'].unique().to_list())
    metric_cols = [
        ('aspect_recall', 'AR'),
        ('weighted_aspect_recall', 'WAR'),
        ('gold_precision', 'GP'),
        ('gold_recall', 'GR'),
    ]

    per_strategy_data: dict[str, dict[str, list[float]]] = {}
    for strat in strategies:
        strat_df = results_df.filter((pl.col('strategy') == strat) & (pl.col('k') == best_k))
        if strat == 'top_k':
            slice_df = strat_df
        else:
            lam_ar = (
                strat_df.group_by('lam')
                .agg(pl.col('aspect_recall').mean().alias('mean_ar'))
                .sort('mean_ar', descending=True)
            )
            if lam_ar.height > 0:
                slice_df = strat_df.filter(pl.col('lam') == lam_ar['lam'][0])
            else:
                slice_df = strat_df
        per_strategy_data[strat] = {
            col: slice_df[col].drop_nulls().to_list() for col, _ in metric_cols
        }

    strat_labels = [get_style(s)['label'] for s in strategies]
    colors = [get_style(s)['color'] for s in strategies]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, title) in zip(axes.flatten(), metric_cols, strict=True):
        data = [per_strategy_data[s][col] for s in strategies]
        positions = [i for i, d in enumerate(data) if len(d) > 1]
        violin_data = [data[i] for i in positions]
        if violin_data:
            parts = ax.violinplot(
                violin_data, positions=positions, showmedians=False, showextrema=False
            )
            for pc, pos in zip(parts['bodies'], positions, strict=True):
                pc.set_facecolor(colors[pos])
                pc.set_alpha(0.65)
        medians = [float(np.median(d)) if d else float('nan') for d in data]
        ax.scatter(
            range(len(strategies)), medians, color='white', edgecolors='black', s=45, zorder=4
        )
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strat_labels, fontsize=9, rotation=15)
        ax.set_title(f'{title}  (k={best_k}, best AR-λ)', fontsize=11)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-query score distributions by strategy', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'per_query_distributions.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_gain_over_topk(stats_df: pl.DataFrame, results_df: pl.DataFrame, out_dir: Path) -> None:
    """2x2 grouped bar chart: Δ(metric) = strategy(best AR-λ) - top_k at each k.

    Positive bars = strategy beats top-k. Error bars = 95% CI of paired per-query delta.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k_values = sorted(stats_df['k'].unique().to_list())
    diversity_strategies = [
        s for s in sorted(stats_df['strategy'].unique().to_list()) if s != 'top_k'
    ]
    if not diversity_strategies:
        return

    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    best_strat: dict[str, dict[int, dict]] = {}
    for strat in diversity_strategies:
        best_strat[strat] = {}
        for k in k_values:
            rows = _best_lam_rows(stats_df, strat, [k])
            if rows.height > 0:
                best_strat[strat][k] = rows.row(0, named=True)

    x = np.arange(len(k_values))
    width = 0.8 / max(len(diversity_strategies), 1)

    fig, axes = plt.subplots(
        len(_METRICS),
        1,
        figsize=(7.4, 2.7 * len(_METRICS) + 1.6),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()
    for row_idx, (ax, (col, _)) in enumerate(zip(flat_axes, _METRICS, strict=True)):
        res_col = _STATS_TO_RESULTS_COL.get(col, col)
        for i, strat in enumerate(diversity_strategies):
            s = get_style(strat)
            deltas, ci_vals = [], []
            for k in k_values:
                ref_row = topk_df.filter(pl.col('k') == k)
                ref_val = float(ref_row[col][0]) if ref_row.height > 0 else 0.0
                strat_val = float(best_strat[strat].get(k, {}).get(col, ref_val))
                deltas.append(strat_val - ref_val)
                best_lam = best_strat[strat].get(k, {}).get('lam')
                ci_vals.append(_paired_delta_ci(results_df, strat, k, best_lam, res_col))
            ci_display = [0.0 if c != c else c for c in ci_vals]  # nan -> 0
            offset = (i - len(diversity_strategies) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                deltas,
                width=width * 0.9,
                color=s['color'],
                label=s['label'],
                alpha=0.85,
                yerr=ci_display,
                capsize=3,
                error_kw={'linewidth': 0.8, 'ecolor': s['color']},
            )
            for bar, d in zip(bars, deltas, strict=True):
                if abs(d) >= 0.01:
                    ypos = bar.get_height() + (0.003 if d >= 0 else -0.015)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        ypos,
                        f'{d:+.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=6.5,
                    )

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_title(f'Δ{col} vs top-k (best AR-λ)', fontsize=11)
        ax.set_ylabel(f'Δ{col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        if row_idx == len(_METRICS) - 1:
            ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=9)
            ax.set_xlabel('k', fontsize=9)
        else:
            ax.tick_params(labelbottom=False)

    handles, labels, seen = [], [], set()
    for ax in flat_axes:
        for h, lbl in zip(*ax.get_legend_handles_labels(), strict=True):
            if lbl not in seen:
                handles.append(h)
                labels.append(lbl)
                seen.add(lbl)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(seen),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle('Gain over top-k baseline - error bars = 95% CI (paired)', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / 'gain_over_topk.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_stratum_breakdown(stratum_df: pl.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: best mean facet recall per strategy for each query stratum."""
    import matplotlib.pyplot as plt
    import numpy as np

    stratum_df = adapt_stats_for_synthetic_plots(stratum_df)

    strata = sorted(stratum_df['stratum'].drop_nulls().unique().to_list())
    strategies = sorted(stratum_df['strategy'].unique().to_list())
    if not strata or not strategies:
        return

    metric_col = 'MeanFacetRecall@k' if 'MeanFacetRecall@k' in stratum_df.columns else 'WAR'
    x = np.arange(len(strata))
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(strata) * 2), 5))
    for i, strat in enumerate(strategies):
        s = get_style(strat)
        metric_vals = []
        for stratum in strata:
            sub = stratum_df.filter((pl.col('strategy') == strat) & (pl.col('stratum') == stratum))
            if sub.height == 0:
                metric_vals.append(0.0)
                continue

            sort_cols, descending = _stratum_best_sort(sub)
            best_row = sub.sort(sort_cols, descending=descending).head(1)
            metric_vals.append(float(best_row[metric_col][0]))
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            metric_vals,
            width=width * 0.9,
            color=s['color'],
            label=s['label'],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f'stratum {s}' for s in strata], fontsize=9)
    ylabel = (
        'MeanFacetRecall@k (coverage-first best row)'
        if metric_col == 'MeanFacetRecall@k'
        else 'WAR (best row)'
    )
    title = (
        'MeanFacetRecall@k by stratum'
        if metric_col == 'MeanFacetRecall@k'
        else 'Weighted AR by stratum'
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(
        title,
        fontsize=11,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'stratum_breakdown.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_answer_strategy_comparison(
    stats_df: pl.DataFrame, results_df: pl.DataFrame, out_dir: Path
) -> None:
    """2x2 grid: ROUGE-1 Recall/Precision/F1 + TF-IDF Cosine vs k, one line per strategy.

    Shaded bands = 95% CI across queries.
    """
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = sorted(stats_df['strategy'].unique().to_list())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for ax, (col, title) in zip(axes.flatten(), _ANSWER_METRICS_STATS, strict=True):
        res_col = _STATS_TO_RESULTS_COL.get(col, col)
        for strat in strategies:
            s = get_style(strat)
            sub = stats_df.filter(pl.col('strategy') == strat)

            if strat == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [float(sub.filter(pl.col('k') == k)[col][0]) for k in xs]
                ci = [
                    _ci_half_width(_query_vals(results_df, 'top_k', k, None, res_col)) for k in xs
                ]
                ax.fill_between(
                    xs,
                    [y - c for y, c in zip(ys, ci, strict=True)],
                    [y + c for y, c in zip(ys, ci, strict=True)],
                    color=s['color'],
                    alpha=0.12,
                    zorder=2,
                )
                ax.plot(xs, ys, color=s['color'], ls=s['ls'], lw=2.0, label=s['label'], zorder=3)
            else:
                lam_values = sorted(sub['lam'].drop_nulls().unique().to_list())
                for lam in lam_values:
                    lsub = sub.filter(pl.col('lam') == lam).sort('k')
                    ax.plot(
                        lsub['k'].to_list(),
                        lsub[col].to_list(),
                        color=s['color'],
                        ls=s['ls'],
                        lw=0.6,
                        alpha=0.3,
                        zorder=1,
                    )
                best_df = _best_lam_rows(stats_df, strat, k_values)
                if best_df.height > 0:
                    xs = best_df['k'].to_list()
                    ys = [float(v) for v in best_df[col].to_list()]
                    k_to_lam = dict(
                        zip(best_df['k'].to_list(), best_df['lam'].to_list(), strict=True)
                    )
                    ci = [
                        _ci_half_width(_query_vals(results_df, strat, k, k_to_lam.get(k), res_col))
                        for k in xs
                    ]
                    ax.fill_between(
                        xs,
                        [y - c for y, c in zip(ys, ci, strict=True)],
                        [y + c for y, c in zip(ys, ci, strict=True)],
                        color=s['color'],
                        alpha=0.12,
                        zorder=1.5,
                    )
                    ax.plot(
                        xs, ys, color=s['color'], ls=s['ls'], lw=2.2, label=s['label'], zorder=2
                    )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(col, fontsize=9)
        ax.set_xticks(k_values)
        ax.grid(axis='y', alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel('k', fontsize=9)

    handles, labels, seen = [], [], set()
    for ax in axes.flatten():
        for h, lbl in zip(*ax.get_legend_handles_labels(), strict=False):
            if lbl not in seen:
                handles.append(h)
                labels.append(lbl)
                seen.add(lbl)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(seen),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle('Answer support metrics - bold = best AR-λ, shaded = 95% CI', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / 'answer_strategy_comparison.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_answer_gain_over_topk(
    stats_df: pl.DataFrame, results_df: pl.DataFrame, out_dir: Path
) -> None:
    """2x2 bar chart: Δ(answer metric) = strategy(best AR-λ) - top_k at each k.

    Error bars = 95% CI of paired per-query delta.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    k_values = sorted(stats_df['k'].unique().to_list())
    diversity_strategies = [
        s for s in sorted(stats_df['strategy'].unique().to_list()) if s != 'top_k'
    ]
    if not diversity_strategies:
        return

    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    best_strat: dict[str, dict[int, dict]] = {}
    for strat in diversity_strategies:
        best_strat[strat] = {}
        for k in k_values:
            rows = _best_lam_rows(stats_df, strat, [k])
            if rows.height > 0:
                best_strat[strat][k] = rows.row(0, named=True)

    x = np.arange(len(k_values))
    width = 0.8 / max(len(diversity_strategies), 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, _) in zip(axes.flatten(), _ANSWER_METRICS_STATS, strict=True):
        res_col = _STATS_TO_RESULTS_COL.get(col, col)
        for i, strat in enumerate(diversity_strategies):
            s = get_style(strat)
            deltas, ci_vals = [], []
            for k in k_values:
                ref_row = topk_df.filter(pl.col('k') == k)
                ref_val = float(ref_row[col][0]) if ref_row.height > 0 else 0.0
                strat_val = float(best_strat[strat].get(k, {}).get(col, ref_val))
                deltas.append(strat_val - ref_val)
                best_lam = best_strat[strat].get(k, {}).get('lam')
                ci_vals.append(_paired_delta_ci(results_df, strat, k, best_lam, res_col))
            ci_display = [0.0 if c != c else c for c in ci_vals]  # nan -> 0
            offset = (i - len(diversity_strategies) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                deltas,
                width=width * 0.9,
                color=s['color'],
                label=s['label'],
                alpha=0.85,
                yerr=ci_display,
                capsize=3,
                error_kw={'linewidth': 0.8, 'ecolor': s['color']},
            )
            for bar, d in zip(bars, deltas, strict=True):
                if abs(d) >= 0.005:
                    ypos = bar.get_height() + (0.002 if d >= 0 else -0.012)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        ypos,
                        f'{d:+.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=6.5,
                    )
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=9)
        ax.set_title(f'Δ{col} vs top-k (best AR-λ)', fontsize=11)
        ax.set_ylabel(f'Δ{col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    handles, labels, seen = [], [], set()
    for ax in axes.flatten():
        for h, lbl in zip(*ax.get_legend_handles_labels(), strict=True):
            if lbl not in seen:
                handles.append(h)
                labels.append(lbl)
                seen.add(lbl)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(seen),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle('Answer support gain over top-k - error bars = 95% CI (paired)', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / 'answer_gain_over_topk.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_answer_distributions(results_df: pl.DataFrame, out_dir: Path) -> None:
    """1x3 violin plots: answer support metrics per strategy at best-k and best-λ."""
    import matplotlib.pyplot as plt
    import numpy as np

    topk_df = results_df.filter(pl.col('strategy') == 'top_k')
    k_ar = (
        topk_df.group_by('k')
        .agg(pl.col('aspect_recall').median().alias('med_ar'))
        .sort('med_ar', descending=True)
    )
    best_k = int(k_ar['k'][0]) if k_ar.height > 0 else int(results_df['k'].max())  # type: ignore

    strategies = sorted(results_df['strategy'].unique().to_list())
    per_strategy_data: dict[str, dict[str, list[float]]] = {}
    for strat in strategies:
        strat_df = results_df.filter((pl.col('strategy') == strat) & (pl.col('k') == best_k))
        if strat == 'top_k':
            slice_df = strat_df
        else:
            lam_ar = (
                strat_df.group_by('lam')
                .agg(pl.col('aspect_recall').mean().alias('mean_ar'))
                .sort('mean_ar', descending=True)
            )
            slice_df = (
                strat_df.filter(pl.col('lam') == lam_ar['lam'][0])
                if lam_ar.height > 0
                else strat_df
            )
        per_strategy_data[strat] = {
            col: slice_df[col].drop_nulls().to_list() for col, _ in _ANSWER_METRICS_RESULTS
        }

    strat_labels = [get_style(s)['label'] for s in strategies]
    colors = [get_style(s)['color'] for s in strategies]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, title) in zip(axes.flatten(), _ANSWER_METRICS_RESULTS, strict=True):
        data = [per_strategy_data[s][col] for s in strategies]
        positions = [i for i, d in enumerate(data) if len(d) > 1]
        violin_data = [data[i] for i in positions]
        if violin_data:
            parts = ax.violinplot(
                violin_data, positions=positions, showmedians=False, showextrema=False
            )
            for pc, pos in zip(parts['bodies'], positions, strict=True):
                pc.set_facecolor(colors[pos])
                pc.set_alpha(0.65)
        medians = [float(np.median(d)) if d else float('nan') for d in data]
        ax.scatter(
            range(len(strategies)), medians, color='white', edgecolors='black', s=45, zorder=4
        )
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strat_labels, fontsize=9, rotation=15)
        ax.set_title(f'{title}  (k={best_k}, best AR-λ)', fontsize=11)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Answer support distributions by strategy', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'answer_distributions.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def _ci_half_width(values: list[float], z: float = 1.96) -> float:
    """95% CI half-width: z * std(values, ddof=1) / sqrt(n)."""
    import numpy as np

    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        return float('nan')
    return z * float(arr.std(ddof=1)) / float(np.sqrt(n))


def _query_vals(
    results_df: pl.DataFrame,
    strategy: str,
    k: int,
    lam: float | None,
    col: str,
) -> list[float]:
    """Per-query values for a given (strategy, k, lam) cell."""
    mask = (pl.col('strategy') == strategy) & (pl.col('k') == k)
    if lam is not None:
        mask = mask & (pl.col('lam') == lam)
    sub = results_df.filter(mask)
    if col not in sub.columns:
        return []
    return sub[col].drop_nulls().to_list()


def _paired_delta_ci(
    results_df: pl.DataFrame,
    strat: str,
    k: int,
    lam: float | None,
    col: str,
) -> float:
    """95% CI half-width of the paired per-query delta (strategy - top_k)."""
    if col not in results_df.columns or 'query_id' not in results_df.columns:
        return float('nan')
    topk_sub = results_df.filter((pl.col('strategy') == 'top_k') & (pl.col('k') == k)).select(
        'query_id', pl.col(col).alias('topk_val')
    )
    strat_mask = (pl.col('strategy') == strat) & (pl.col('k') == k)
    if lam is not None:
        strat_mask = strat_mask & (pl.col('lam') == lam)
    strat_sub = results_df.filter(strat_mask).select('query_id', pl.col(col).alias('strat_val'))
    joined = topk_sub.join(strat_sub, on='query_id', how='inner')
    if joined.height < 2:
        return float('nan')
    deltas = (joined['strat_val'] - joined['topk_val']).to_list()
    return _ci_half_width(deltas)


def _best_lam_rows(stats_df: pl.DataFrame, strategy: str, k_values: list[int]) -> pl.DataFrame:
    """For a diversity strategy, return one row per k with the best-AR λ."""
    sub = stats_df.filter(pl.col('strategy') == strategy)
    rows = []
    for k in k_values:
        ksub = sub.filter(pl.col('k') == k).sort('AR', descending=True)
        if ksub.height > 0:
            rows.append(ksub.head(1))
    return pl.concat(rows).sort('k') if rows else pl.DataFrame()


def _stratum_best_sort(df: pl.DataFrame) -> tuple[list[str], list[bool]]:
    candidates = [
        ('FacetCoverage@k', True),
        ('Precision@k', True),
        ('DistractorRate', False),
        ('alpha-nDCG@k', True),
        ('MeanFacetRecall@k', True),
        ('WAR', True),
    ]
    pairs = [(col, desc) for col, desc in candidates if col in df.columns]
    if not pairs:
        return ['strategy'], [False]
    cols, desc = zip(*pairs, strict=True)
    return list(cols), list(desc)


if __name__ == '__main__':
    setup_logging()
    store_eval_figures(cfg=EvaluateCfg.load())
