from pathlib import Path

import polars as pl

from experiments.mimic.global_configs import MimicPaths, get_table_path, setup_logging

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


def get_style(strategy: str) -> dict:
    return STRATEGY_STYLE.get(strategy, {'color': '#aaaaaa', 'ls': '-', 'label': strategy})


def store_eval_figures(cfg: EvaluateCfg) -> None:
    from .plots import (
        plot_gain_over_topk,
        plot_lambda_sensitivity,
        plot_per_query_distributions,
        plot_strategy_comparison,
        plot_stratum_breakdown,
    )

    out_dir = (
        MimicPaths.experiment_dir
        / 'figures'
        / ('eval_structural' if cfg.gold_mode == 'structural' else 'eval_gold_annotations')
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = get_table_path('evaluation_stats')
    results_path = get_table_path('evaluation_results')
    if not stats_path.exists() or not results_path.exists():
        print('Skipping eval figures: evaluation_stats or evaluation_results not found')
        return

    stats_df = pl.read_parquet(stats_path)
    results_df = pl.read_parquet(results_path)

    plot_strategy_comparison(stats_df, out_dir)
    plot_lambda_sensitivity(stats_df, out_dir)
    plot_per_query_distributions(results_df, out_dir)
    plot_gain_over_topk(stats_df, out_dir)

    stratum_path = get_table_path('evaluation_stats_by_stratum')
    if stratum_path.exists():
        stratum_df = pl.read_parquet(stratum_path)
        plot_stratum_breakdown(stratum_df, out_dir)

        strata = sorted(stratum_df['stratum'].drop_nulls().unique().to_list())
        for s in strata:
            s_dir = out_dir / f'stratum_{s}'
            s_dir.mkdir(exist_ok=True)
            s_stats = stratum_df.filter(pl.col('stratum') == s).drop('stratum')
            s_results = (
                results_df.filter(pl.col('stratum') == s)
                if 'stratum' in results_df.columns
                else results_df
            )
            plot_strategy_comparison(s_stats, s_dir)
            plot_lambda_sensitivity(s_stats, s_dir)
            plot_per_query_distributions(s_results, s_dir)
            plot_gain_over_topk(s_stats, s_dir)

    print(f'Saved eval figures to {out_dir}')


def plot_strategy_comparison(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """2x2 grid: AR/WAR/GP/GR vs k, one line per strategy.

    Diversity strategies: thin faded lines per λ + bold line at best-AR λ.
    top_k: single dashed black baseline.
    """
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = sorted(stats_df['strategy'].unique().to_list())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    for ax, (col, title) in zip(axes.flatten(), _METRICS, strict=True):
        for strat in strategies:
            s = get_style(strat)
            sub = stats_df.filter(pl.col('strategy') == strat)

            if strat == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [sub.filter(pl.col('k') == k)[col][0] for k in xs]
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
                    ax.plot(
                        best_df['k'].to_list(),
                        best_df[col].to_list(),
                        color=s['color'],
                        ls=s['ls'],
                        lw=2.2,
                        label=s['label'],
                        zorder=2,
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
    fig.suptitle('Strategy comparison (bold line = best λ per k)', fontsize=12)
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
        topk_df
        .group_by('k')
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
                strat_df
                .group_by('lam')
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
        ax.set_title(f'{title}  (k={best_k}, best λ)', fontsize=11)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-query score distributions by strategy', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'per_query_distributions.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_gain_over_topk(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """2x2 grouped bar chart: Δ(metric) = strategy(best λ) - top_k at each k.

    Positive bars = strategy beats top-k. Directly quantifies the thesis claim.
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
    for ax, (col, _) in zip(axes.flatten(), _METRICS, strict=True):
        for i, strat in enumerate(diversity_strategies):
            s = get_style(strat)
            deltas = []
            for k in k_values:
                ref_row = topk_df.filter(pl.col('k') == k)
                ref_val = float(ref_row[col][0]) if ref_row.height > 0 else 0.0
                strat_val = float(best_strat[strat].get(k, {}).get(col, ref_val))
                deltas.append(strat_val - ref_val)
            offset = (i - len(diversity_strategies) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                deltas,
                width=width * 0.9,
                color=s['color'],
                label=s['label'],
                alpha=0.85,
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
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=9)
        ax.set_title(f'Δ{col} vs top-k (best λ)', fontsize=11)
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
    fig.suptitle('Gain over top-k baseline', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / 'gain_over_topk.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_stratum_breakdown(stratum_df: pl.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: best WAR per strategy for each query stratum."""
    import matplotlib.pyplot as plt
    import numpy as np

    strata = sorted(stratum_df['stratum'].drop_nulls().unique().to_list())
    strategies = sorted(stratum_df['strategy'].unique().to_list())
    if not strata or not strategies:
        return

    x = np.arange(len(strata))
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(strata) * 2), 5))
    for i, strat in enumerate(strategies):
        s = get_style(strat)
        wars = []
        for stratum in strata:
            sub = stratum_df.filter((pl.col('strategy') == strat) & (pl.col('stratum') == stratum))
            wars.append(float(sub['WAR'].max()) if sub.height > 0 else 0.0)  # type: ignore
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, wars, width=width * 0.9, color=s['color'], label=s['label'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'stratum {s}' for s in strata], fontsize=9)
    ax.set_ylabel('WAR (best λ, k)', fontsize=9)
    ax.set_title(
        'Weighted AR by stratum',
        fontsize=11,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'stratum_breakdown.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def _best_lam_rows(stats_df: pl.DataFrame, strategy: str, k_values: list[int]) -> pl.DataFrame:
    """For a diversity strategy, return one row per k with the best-AR λ."""
    sub = stats_df.filter(pl.col('strategy') == strategy)
    rows = []
    for k in k_values:
        ksub = sub.filter(pl.col('k') == k).sort('AR', descending=True)
        if ksub.height > 0:
            rows.append(ksub.head(1))
    return pl.concat(rows).sort('k') if rows else pl.DataFrame()


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='queries')
    store_eval_figures(cfg=EvaluateCfg(**raw['evaluate']))
