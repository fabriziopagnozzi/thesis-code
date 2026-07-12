"""Figures for paired, profile-clustered statistical summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.medical_dataset_gen.analysis.helpers import float_or_none, short_experiment_id
from experiments.medical_dataset_gen.analysis.models import PlotFormat
from experiments.medical_dataset_gen.analysis.statistical import CORE_EMBEDDING_MODELS


def plot_paired_fcp_forest(
    *,
    plt: object,
    cell_rows: Sequence[Mapping[str, object]],
    suite_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_format: PlotFormat,
) -> list[Path]:
    """Plot low-budget core-suite FCP paired effects with profile bootstrap CIs."""
    rows = [
        row
        for row in cell_rows
        if row.get('BudgetCategory') == 'low_budget'
        and row.get('EmbeddingModel') in CORE_EMBEDDING_MODELS
        and float_or_none(row.get('MeanDeltaFacLocMMR')) is not None
        and float_or_none(row.get('CI95Low')) is not None
        and float_or_none(row.get('CI95High')) is not None
    ]
    if not rows:
        return []
    rows.sort(
        key=lambda row: (
            str(row.get('ExperimentFamilyLabel') or ''),
            float_or_none(row.get('MeanDeltaFacLocMMR')) or 0.0,
            str(row.get('Experiment') or ''),
        )
    )
    labels = [
        f'{short_experiment_id(str(row.get("Experiment") or ""))} | '
        f'{str(row.get("EmbeddingModel") or "").rsplit("/", 1)[-1]}'
        for row in rows
    ]
    means = [float_or_none(row.get('MeanDeltaFacLocMMR')) or 0.0 for row in rows]
    lows = [float_or_none(row.get('CI95Low')) or 0.0 for row in rows]
    highs = [float_or_none(row.get('CI95High')) or 0.0 for row in rows]
    positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.22 * len(rows) + 2.2)))  # type: ignore[attr-defined]
    try:
        ax.errorbar(
            means,
            positions,
            xerr=[[mean - low for mean, low in zip(means, lows, strict=True)], [
                high - mean for mean, high in zip(means, highs, strict=True)
            ]],
            fmt='o',
            color='#287C8E',
            ecolor='#6B7280',
            capsize=2.5,
            markersize=4,
        )
        ax.axvline(0.0, color='#111827', linewidth=0.9)
        threshold = float_or_none(rows[0].get('PracticalThreshold'))
        if threshold is not None:
            ax.axvspan(-threshold, threshold, color='#E5E7EB', alpha=0.65, zorder=0)
        core = next(
            (
                row
                for row in suite_rows
                if row.get('Scope') == 'Core suite' and row.get('BudgetCategory') == 'low_budget'
            ),
            None,
        )
        if core is not None:
            mean = float_or_none(core.get('MeanDeltaFacLocMMR'))
            low = float_or_none(core.get('CI95Low'))
            high = float_or_none(core.get('CI95High'))
            if mean is not None and low is not None and high is not None:
                summary_position = len(rows) + 1
                ax.errorbar(
                    [mean],
                    [summary_position],
                    xerr=[[mean - low], [high - mean]],
                    fmt='D',
                    color='#C44E52',
                    ecolor='#C44E52',
                    capsize=3,
                    markersize=6,
                )
                positions.append(summary_position)
                labels.append('Core suite (equal-family weighted)')
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Held-out FacLoc - MMR FCP effect (95% profile-bootstrap CI)')
        ax.set_title('Low-budget paired FCP effects in the fully crossed core suite')
        ax.grid(axis='x', alpha=0.22)
        fig.tight_layout()
        path = output_dir / f'paired_fcp_low_budget_forest.{plot_format}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        return [path]
    finally:
        plt.close(fig)  # type: ignore[attr-defined]
