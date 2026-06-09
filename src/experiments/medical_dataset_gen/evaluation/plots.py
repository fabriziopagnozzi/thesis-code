from pathlib import Path

import polars as pl

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
    ('FC', 'facet_coverage', 'Facet Coverage', True),
    ('WFC', 'weighted_facet_coverage', 'Weighted Facet Coverage', True),
    ('GP', 'gold_precision', 'Gold Precision', True),
    ('GR', 'gold_recall', 'Gold Recall', True),
    ('DR', 'distractor_rate', 'Distractor Rate', False),
    ('DCC', 'dominant_cluster_concentration', 'Dominant Cluster Concentration', False),
]

_PRIMARY_SORT = ['FC', 'WFC', 'GP', 'DR']
_PRIMARY_DESC = [True, True, True, False]


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
    plot_lambda_sensitivity(stats_df, out_dir)
    plot_per_query_distributions(results_df, out_dir)
    plot_gain_over_topk(stats_df, results_df, out_dir)
    plot_coverage_precision_tradeoff(stats_df, out_dir)
    plot_selection_diagnostics(stats_df, out_dir)

    print(f'[plots] saved evaluation figures to {out_dir}')


def plot_strategy_comparison(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Metric-vs-k lines.

    Diversity methods show thin per-lambda traces plus a bold line using the
    best lambda at each k. Shaded bands are 95% CIs across queries.
    """
    import matplotlib.pyplot as plt

    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)

    for ax, (stats_col, result_col, title, higher_is_better) in zip(
        axes.flatten(),
        _METRICS,
        strict=True,
    ):
        for strategy in strategies:
            style = get_style(strategy)
            sub = stats_df.filter(pl.col('strategy') == strategy)
            if strategy == 'top_k':
                xs = sorted(sub['k'].unique().to_list())
                ys = [_cell_value(sub, k, None, stats_col) for k in xs]
                ci = [_ci_half_width(_query_vals(results_df, strategy, k, None, result_col)) for k in xs]
                _plot_ci_line(ax, xs, ys, ci, style, lw=2.0, label=style['label'], zorder=3)
                continue

            for lam in sorted(sub['lam'].drop_nulls().unique().to_list()):
                lsub = sub.filter(pl.col('lam') == lam).sort('k')
                ax.plot(
                    lsub['k'].to_list(),
                    lsub[stats_col].to_list(),
                    color=style['color'],
                    ls=style['ls'],
                    lw=0.7,
                    alpha=0.28,
                    zorder=1,
                )

            best_df = _best_lam_rows(stats_df, strategy, k_values)
            if best_df.height == 0:
                continue
            xs = best_df['k'].to_list()
            ys = [float(v) for v in best_df[stats_col].to_list()]
            k_to_lam = dict(zip(best_df['k'].to_list(), best_df['lam'].to_list(), strict=True))
            ci = [
                _ci_half_width(_query_vals(results_df, strategy, k, k_to_lam.get(k), result_col))
                for k in xs
            ]
            _plot_ci_line(ax, xs, ys, ci, style, lw=2.2, label=style['label'], zorder=2)

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

    for ax in axes[1]:
        ax.set_xlabel('k', fontsize=9)

    _figure_legend(fig, axes.flatten())
    fig.suptitle('Strategy comparison - bold = best FC lambda, shaded = 95% CI', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'strategy_comparison.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_lambda_sensitivity(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Facet coverage and distractor rate as lambda changes."""
    import matplotlib.pyplot as plt

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    cmap = plt.get_cmap('viridis')  # type: ignore[attr-defined]
    k_colors = {k: cmap(i / max(len(k_values) - 1, 1)) for i, k in enumerate(k_values)}

    fig, axes = plt.subplots(
        len(diversity_strategies),
        2,
        figsize=(12, 4.2 * len(diversity_strategies)),
        squeeze=False,
    )

    for row_idx, strategy in enumerate(diversity_strategies):
        style = get_style(strategy)
        sub = stats_df.filter(pl.col('strategy') == strategy)
        for col_idx, (metric, title) in enumerate([('FC', 'Facet Coverage'), ('DR', 'Distractor Rate')]):
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

            ax.set_title(f'{style["label"]}: {title} vs lambda', fontsize=10)
            ax.set_xlabel('lambda', fontsize=9)
            ax.set_ylabel(metric, fontsize=9)
            ax.set_xticks(sorted(sub['lam'].drop_nulls().unique().to_list()))
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
    fig.suptitle('Lambda sensitivity - solid=strategy, dashed=top-k reference', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'lambda_sensitivity.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_per_query_distributions(results_df: pl.DataFrame, out_dir: Path) -> None:
    """Violin plots at the top-k best k and each strategy's best-FC lambda."""
    import matplotlib.pyplot as plt
    import numpy as np

    best_k = _best_topk_k(results_df)
    strategies = _ordered_strategies(results_df)
    metric_cols = [
        ('facet_coverage', 'FC'),
        ('weighted_facet_coverage', 'WFC'),
        ('gold_precision', 'GP'),
        ('distractor_rate', 'DR'),
        ('dominant_cluster_concentration', 'DCC'),
        ('gold_recall', 'GR'),
    ]

    per_strategy_data: dict[str, dict[str, list[float]]] = {}
    for strategy in strategies:
        slice_df = _best_result_slice(results_df, strategy, best_k)
        per_strategy_data[strategy] = {
            col: [float(v) for v in slice_df[col].drop_nulls().to_list()]
            for col, _ in metric_cols
        }

    labels = [get_style(s)['label'] for s in strategies]
    colors = [get_style(s)['color'] for s in strategies]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    for ax, (col, title) in zip(axes.flatten(), metric_cols, strict=True):
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
        ax.scatter(range(len(strategies)), medians, color='white', edgecolors='black', s=42, zorder=4)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.set_title(f'{title} (k={best_k}, best FC lambda)', fontsize=10)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Per-query score distributions', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'per_query_distributions.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_gain_over_topk(
    stats_df: pl.DataFrame,
    results_df: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Paired delta bars relative to top-k.

    For DR and DCC, negative bars are favorable because lower is better.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    diversity_strategies = [s for s in _ordered_strategies(stats_df) if s != 'top_k']
    if not diversity_strategies:
        return

    k_values = sorted(stats_df['k'].unique().to_list())
    topk_df = stats_df.filter(pl.col('strategy') == 'top_k')
    x = np.arange(len(k_values))
    width = 0.8 / max(len(diversity_strategies), 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (stats_col, result_col, title, higher_is_better) in zip(
        axes.flatten(),
        _METRICS,
        strict=True,
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
            for bar, delta in zip(bars, deltas, strict=True):
                if abs(delta) >= 0.005:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        delta + (0.003 if delta >= 0 else -0.018),
                        f'{delta:+.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=6.5,
                    )

        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=9)
        ax.set_title(f'Delta {title} vs top-k', fontsize=10)
        ax.set_ylabel(f'Delta {stats_col}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
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
    fig.suptitle('Gain over top-k - best FC lambda, paired 95% CI', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'gain_over_topk.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_coverage_precision_tradeoff(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """FC vs distractor rate for each strategy/lambda/k cell."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 6))
    k_values = sorted(stats_df['k'].unique().to_list())
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    k_to_marker = {k: markers[i % len(markers)] for i, k in enumerate(k_values)}
    k_to_line_style = {k: line_styles[i % len(line_styles)] for i, k in enumerate(k_values)}

    for strategy in _ordered_strategies(stats_df):
        style = get_style(strategy)
        sub = stats_df.filter(pl.col('strategy') == strategy)
        if strategy == 'top_k':
            for row in sub.iter_rows(named=True):
                k = int(row['k'])
                ax.scatter(
                    float(row['DR']),
                    float(row['FC']),
                    color=style['color'],
                    marker=k_to_marker[k],
                    s=90,
                    label=style['label'] if k == k_values[0] else None,
                    zorder=4,
                )
            continue

        for k in sorted(sub['k'].unique().to_list()):
            ksub = sub.filter(pl.col('k') == k).sort('lam')
            ax.plot(
                ksub['DR'].to_list(),
                ksub['FC'].to_list(),
                color=style['color'],
                ls=k_to_line_style[k],
                lw=1.7,
                marker=k_to_marker[k],
                markerfacecolor='white',
                markeredgewidth=1.1,
                ms=5,
                alpha=0.85,
                label=f'{style["label"]} k={k}',
            )
            for row in ksub.iter_rows(named=True):
                ax.text(float(row['DR']), float(row['FC']), f' {row["lam"]:.1f}', fontsize=7)

    ax.set_xlabel('Distractor Rate (lower is better)', fontsize=9)
    ax.set_ylabel('Facet Coverage (higher is better)', fontsize=9)
    ax.set_title('Coverage vs distractor tradeoff', fontsize=11)
    ax.grid(alpha=0.3)

    strategy_handles = [
        Line2D(
            [0],
            [0],
            color=get_style(strategy)['color'],
            lw=2.0,
            label=get_style(strategy)['label'],
        )
        for strategy in _ordered_strategies(stats_df)
    ]
    k_handles = [
        Line2D(
            [0],
            [0],
            color='#555555',
            ls=k_to_line_style[k],
            marker=k_to_marker[k],
            markerfacecolor='white',
            markeredgewidth=1.1,
            lw=1.7,
            label=f'k={k}',
        )
        for k in k_values
    ]
    first_legend = ax.legend(
        handles=strategy_handles,
        title='Strategy',
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        loc='lower right',
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=k_handles,
        title='k',
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        loc='upper left',
    )
    fig.tight_layout()
    fig.savefig(out_dir / 'coverage_precision_tradeoff.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

def plot_selection_diagnostics(stats_df: pl.DataFrame, out_dir: Path) -> None:
    """Diagnostic metrics that explain why a strategy wins or fails."""
    import matplotlib.pyplot as plt

    metric_cols = [
        ('DCC', 'Dominant Cluster Concentration'),
        ('DR', 'Distractor Rate'),
        ('fac', 'Facility-Location Objective'),
        ('avg_cos', 'Average Query Cosine'),
        ('jac', 'Jaccard vs top-k'),
    ]
    k_values = sorted(stats_df['k'].unique().to_list())
    strategies = _ordered_strategies(stats_df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for ax, (col, title) in zip(axes.flatten(), metric_cols, strict=False):
        if col not in stats_df.columns:
            ax.set_visible(False)
            continue
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

    for ax in axes[1]:
        ax.set_xlabel('k', fontsize=9)
    if len(metric_cols) < len(axes.flatten()):
        axes.flatten()[-1].set_visible(False)

    _figure_legend(fig, axes.flatten())
    fig.suptitle('Selection diagnostics - best FC lambda', fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / 'selection_diagnostics.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def get_style(strategy: str) -> dict[str, str]:
    return STRATEGY_STYLE.get(strategy, {'color': '#aaaaaa', 'ls': '-', 'label': strategy})


def _ordered_strategies(df: pl.DataFrame) -> list[str]:
    preferred = ['top_k', 'fac_loc', 'mmr']
    present = set(df['strategy'].unique().to_list())
    ordered = [s for s in preferred if s in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _best_lam_rows(stats_df: pl.DataFrame, strategy: str, k_values: list[int]) -> pl.DataFrame:
    sub = stats_df.filter(pl.col('strategy') == strategy)
    rows = []
    for k in k_values:
        ksub = sub.filter(pl.col('k') == k)
        if ksub.height > 0:
            rows.append(ksub.sort(_PRIMARY_SORT, descending=_PRIMARY_DESC).head(1))
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
            pl.col('weighted_facet_coverage').median().alias('med_wfc'),
        )
        .sort(['med_fc', 'med_wfc', 'k'], descending=[True, True, False])
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
            pl.col('facet_coverage').mean().alias('FC'),
            pl.col('weighted_facet_coverage').mean().alias('WFC'),
            pl.col('gold_precision').mean().alias('GP'),
            pl.col('distractor_rate').mean().alias('DR'),
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
    strat_sub = results_df.filter(strategy_mask).select('query_id', pl.col(col).alias('strategy_val'))
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


def _figure_legend(fig: object, axes: object) -> None:
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
