"""Regenerate the thesis geometry audit from frozen per-query diagnostics.

This selective entrypoint shares the same audit definition as the main report
pipeline and reads only ``geometry_stats.parquet`` artifacts.

The legacy ``passes_filter`` value remains untouched in the frozen artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from experiments.medical_dataset_gen.reports.analysis_constants import (
    EXPERIMENT_FAMILY_COLORS,
    EXPERIMENT_FAMILY_LABELS,
    ExperimentFamilyId,
)
from experiments.medical_dataset_gen.reports.geometry_coverage import (
    COVERAGE_STRESS_DEFINITION,
    geometry_coverage_columns,
    summarize_geometry_coverage,
)
from experiments.medical_dataset_gen.reports.helpers import (
    experiment_plot_label,
    short_experiment_id,
)
from experiments.medical_dataset_gen.reports.latex_tables import (
    _embedding_model_result_token,
)
from experiments.medical_dataset_gen.reports.plot_diagnostics import family_color_for_row
from experiments.medical_dataset_gen.reports.report_config import (
    EMBEDDING_MODEL_REPORT_SPECS,
    embedding_model_display_label,
)
from experiments.medical_dataset_gen.reports.suite_analysis import (
    report_eligible_manifest,
)
from experiments.medical_dataset_gen.suites.core import SuiteManifestCell, load_suite_manifest
from experiments.medical_dataset_gen.suites.runtime import load_cell_config

_MACRO_BLOCK_START = '% BEGIN selectively recomputed geometry coverage-stress macros'
_MACRO_BLOCK_END = '% END selectively recomputed geometry coverage-stress macros'
_THESIS_MODEL_TOKEN_ORDER = tuple(spec.macro_token for spec in EMBEDDING_MODEL_REPORT_SPECS)


@dataclass(frozen=True)
class CellInput:
    suite_id: str
    cell: SuiteManifestCell
    geometry_path: Path
    embedding_model: str


@dataclass(frozen=True)
class CellGeometrySummary:
    suite_id: str
    experiment: str
    short_experiment: str
    distribution_id: str
    family_id: str
    family_label: str
    analysis_tier: str
    embedding_model: str
    query_structure: str
    document_surface: str
    query_count: int
    coverage_stress_pass_count: int
    coverage_stress_rate: float
    gold_near_miss_margin: float | None
    gold_background_margin: float | None

    def plot_row(self) -> dict[str, object]:
        return {
            'Experiment': self.experiment,
            'ShortExperiment': self.short_experiment,
            'Distribution': self.distribution_id,
            'ExperimentFamily': self.family_id,
            'ExperimentFamilyLabel': self.family_label,
            'AnalysisTier': self.analysis_tier,
            'EmbeddingModel': self.embedding_model,
            'WordingConfigLabel': (f'{self.query_structure}, {self.document_surface}'),
            'CoverageStressRate': self.coverage_stress_rate,
        }


@dataclass(frozen=True)
class ModelGeometrySummary:
    embedding_model: str
    model_token: str
    display_label: str
    coverage_stress_mean: float
    coverage_stress_median: float
    gold_near_miss_margin_mean: float
    gold_background_margin_mean: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Selectively regenerate geometry coverage-stress macros and figures from frozen '
            'geometry_stats.parquet artifacts.'
        )
    )
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--suite', action='append', required=True, dest='suite_ids')
    parser.add_argument(
        '--workers',
        type=int,
        default=min(8, os.cpu_count() or 1),
        help='Number of concurrent Parquet readers.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Calculate and validate values without writing macros or figures.',
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def _cell_inputs(results_dir: Path, suite_ids: Sequence[str]) -> list[CellInput]:
    inputs: list[CellInput] = []
    for suite_id in suite_ids:
        manifest, excluded = report_eligible_manifest(load_suite_manifest(results_dir, suite_id))
        if excluded:
            print(f'[{suite_id}] excluding report-ineligible distributions: {sorted(excluded)}')
        suite_root = results_dir / 'v5' / 'suites' / suite_id
        for cell in manifest.cells:
            cfg = load_cell_config(suite_root, cell)
            geometry_path = suite_root / cell.attempt_root / 'geometry_stats.parquet'
            if not geometry_path.is_file():
                raise FileNotFoundError(geometry_path)
            inputs.append(
                CellInput(
                    suite_id=suite_id,
                    cell=cell,
                    geometry_path=geometry_path,
                    embedding_model=cfg.embeddings.model_name,
                )
            )
    return inputs


def _summarize_cell(cell_input: CellInput) -> CellGeometrySummary:
    frame = pl.read_parquet(
        cell_input.geometry_path,
        columns=list(geometry_coverage_columns()),
    )
    audit = summarize_geometry_coverage(frame, source=str(cell_input.geometry_path))

    cell = cell_input.cell
    return CellGeometrySummary(
        suite_id=cell_input.suite_id,
        experiment=cell.name,
        short_experiment=short_experiment_id(cell.name),
        distribution_id=cell.distribution_id,
        family_id=cell.family_id,
        family_label=cell.family_label,
        analysis_tier=cell.analysis_tier,
        embedding_model=cell_input.embedding_model,
        query_structure=str(cell.run_profile_factors.get('query_structure') or 'unknown'),
        document_surface=str(cell.run_profile_factors.get('document_surface') or 'unknown'),
        query_count=audit.query_count,
        coverage_stress_pass_count=audit.coverage_stress_pass_count,
        coverage_stress_rate=audit.coverage_stress_rate,
        gold_near_miss_margin=audit.gold_near_miss_margin,
        gold_background_margin=audit.gold_background_margin,
    )


def _family_balanced_mean(
    rows: Sequence[CellGeometrySummary],
    value_for_row: Callable[[CellGeometrySummary], float | None],
) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = value_for_row(row)
        if value is not None:
            by_family[row.family_label].append(value)
    family_means = [statistics.fmean(values) for values in by_family.values() if values]
    if not family_means:
        raise ValueError('Cannot compute a family-balanced mean from empty values')
    return statistics.fmean(family_means)


def _model_display_label(token: str, model: str) -> str:
    del token
    return embedding_model_display_label(model)


def _model_summaries(rows: Sequence[CellGeometrySummary]) -> list[ModelGeometrySummary]:
    by_model: dict[str, list[CellGeometrySummary]] = defaultdict(list)
    for row in rows:
        # The thesis embedding panel summarizes the five distribution families;
        # crossed interaction cells are reported in their own figure.
        if row.analysis_tier == 'interaction':
            continue
        by_model[row.embedding_model].append(row)

    summaries: list[ModelGeometrySummary] = []
    for model, model_rows in by_model.items():
        token = _embedding_model_result_token(model)
        if token is None:
            raise ValueError(f'No thesis macro token is registered for embedding model {model!r}')
        summaries.append(
            ModelGeometrySummary(
                embedding_model=model,
                model_token=token,
                display_label=_model_display_label(token, model),
                coverage_stress_mean=_family_balanced_mean(
                    model_rows, lambda row: row.coverage_stress_rate
                ),
                coverage_stress_median=statistics.median(
                    row.coverage_stress_rate for row in model_rows
                ),
                gold_near_miss_margin_mean=_family_balanced_mean(
                    model_rows, lambda row: row.gold_near_miss_margin
                ),
                gold_background_margin_mean=_family_balanced_mean(
                    model_rows, lambda row: row.gold_background_margin
                ),
            )
        )
    order = {token: index for index, token in enumerate(_THESIS_MODEL_TOKEN_ORDER)}
    return sorted(
        summaries, key=lambda row: (order.get(row.model_token, len(order)), row.model_token)
    )


def _fixed(value: float) -> str:
    return f'{value:.3f}'


def _coverage_macros(model_rows: Sequence[ModelGeometrySummary]) -> dict[str, str]:
    macros: dict[str, str] = {}
    model_means = [row.coverage_stress_mean for row in model_rows]
    for row in model_rows:
        prefix = f'ResultEmbedding{row.model_token}GeometryCoverageStress'
        macros[f'{prefix}Mean'] = _fixed(row.coverage_stress_mean)
        macros[f'{prefix}Median'] = _fixed(row.coverage_stress_median)
    macros.update(
        {
            'ResultGeometryCoverageStressMean': _fixed(statistics.fmean(model_means)),
            'ResultGeometryCoverageStressMedian': _fixed(statistics.median(model_means)),
            'ResultGeometryCoverageStressMinModel': _fixed(min(model_means)),
            'ResultGeometryCoverageStressMaxModel': _fixed(max(model_means)),
        }
    )
    return macros


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def _write_macro_block(path: Path, macros: Mapping[str, str]) -> None:
    original = path.read_text()
    definitions = '\n'.join(
        f'\\newcommand{{\\{name}}}{{{value}}}' for name, value in sorted(macros.items())
    )
    block = f'{_MACRO_BLOCK_START}\n{definitions}\n{_MACRO_BLOCK_END}'
    block_pattern = re.compile(
        rf'{re.escape(_MACRO_BLOCK_START)}.*?{re.escape(_MACRO_BLOCK_END)}', re.DOTALL
    )
    if block_pattern.search(original):
        updated = block_pattern.sub(lambda _match: block, original)
    else:
        updated = original.rstrip() + '\n\n' + block + '\n'
    _atomic_write_text(path, updated)


def _plot_cell_rates(*, rows: Sequence[CellGeometrySummary], output_path: Path, title: str) -> None:
    plot_rows = [row.plot_row() for row in rows]
    labels = [
        f'{experiment_plot_label(plot_row)}/'
        f'{_model_display_label(_embedding_model_result_token(summary.embedding_model) or "", summary.embedding_model)}'
        for summary, plot_row in zip(rows, plot_rows, strict=True)
    ]
    values = [row.coverage_stress_rate for row in rows]
    colors = [family_color_for_row(row) for row in plot_rows]
    figure_height = max(5.0, 0.28 * len(labels) + 1.6)
    figure, axis = plt.subplots(figsize=(8.5, figure_height))
    try:
        axis.barh(range(len(labels)), values, color=colors)
        axis.set_title(title)
        axis.set_xlabel('Coverage-stress rate')
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 1.08)
        axis.grid(axis='x', alpha=0.25)
        for position, value in enumerate(values):
            axis.text(value + 0.008, position, f'{value:.3f}', va='center', fontsize=7)
        present_families: list[ExperimentFamilyId] = [
            cast(ExperimentFamilyId, family_id)
            for family_id in EXPERIMENT_FAMILY_LABELS
            if any(row.family_id == family_id for row in rows)
        ]
        if present_families:
            from matplotlib.patches import Patch

            figure.legend(
                handles=[
                    Patch(
                        facecolor=EXPERIMENT_FAMILY_COLORS[family_id],
                        label=EXPERIMENT_FAMILY_LABELS[family_id],
                    )
                    for family_id in present_families
                ],
                loc='lower center',
                ncol=min(4, len(present_families)),
                frameon=False,
                fontsize=8,
            )
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180)
    finally:
        plt.close(figure)


def _plot_model_overview(*, model_rows: Sequence[ModelGeometrySummary], output_path: Path) -> None:
    labels = [row.display_label for row in model_rows]
    positions = np.arange(len(labels), dtype=np.float64)
    figure, (categorical_axis, margin_axis) = plt.subplots(1, 2, figsize=(13.8, 5.64))
    try:
        coverage_values = [row.coverage_stress_mean for row in model_rows]
        categorical_axis.bar(positions, coverage_values, width=0.55, color='#287C8E')
        categorical_axis.set_title('Early-ranking coverage stress')
        categorical_axis.set_ylabel('Family-balanced rate')
        categorical_axis.set_xticks(positions, labels, rotation=20, ha='right')
        categorical_axis.set_ylim(0, 1.05)
        categorical_axis.grid(axis='y', alpha=0.25)
        for position, value in zip(positions, coverage_values, strict=True):
            categorical_axis.text(position, value + 0.015, f'{value:.3f}', ha='center')

        margin_series = (
            (
                'Gold - near miss',
                [row.gold_near_miss_margin_mean for row in model_rows],
                '#D08A2E',
            ),
            (
                'Gold - background',
                [row.gold_background_margin_mean for row in model_rows],
                '#3D71B7',
            ),
        )
        width = 0.34
        for index, (label, values, color) in enumerate(margin_series):
            offset = (index - 0.5) * width
            margin_axis.bar(positions + offset, values, width, label=label, color=color)
        margin_axis.set_title('Gold separation from distractors')
        margin_axis.set_ylabel('Within-model cosine-similarity margin')
        margin_axis.set_xticks(positions, labels, rotation=20, ha='right')
        margin_axis.grid(axis='y', alpha=0.25)
        margin_axis.legend(frameon=False)

        figure.suptitle('Representation-space audit by embedding model', fontsize=16)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)


def _plot_family_heatmap(
    *,
    rows: Sequence[CellGeometrySummary],
    model_rows: Sequence[ModelGeometrySummary],
    output_path: Path,
) -> None:
    primary_rows = [row for row in rows if row.analysis_tier != 'interaction']
    family_labels = sorted({row.family_label for row in primary_rows})
    model_labels = [row.display_label for row in model_rows]
    values = np.zeros((len(model_rows), len(family_labels)), dtype=np.float64)
    for model_index, model in enumerate(model_rows):
        for family_index, family in enumerate(family_labels):
            family_values = [
                row.coverage_stress_rate
                for row in primary_rows
                if row.embedding_model == model.embedding_model and row.family_label == family
            ]
            if not family_values:
                raise ValueError(f'No rows for model={model.embedding_model}, family={family}')
            values[model_index, family_index] = statistics.fmean(family_values)

    figure, axis = plt.subplots(figsize=(10.5, 5.3))
    try:
        image = axis.imshow(values, vmin=0.5, vmax=1.0, cmap='viridis', aspect='auto')
        axis.set_title('Coverage-stress rate by embedding model and distribution family')
        axis.set_xticks(range(len(family_labels)), family_labels, rotation=27, ha='right')
        axis.set_yticks(range(len(model_labels)), model_labels)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                text_color = 'white' if value < 0.74 else 'black'
                axis.text(
                    column_index,
                    row_index,
                    f'{value:.3f}',
                    ha='center',
                    va='center',
                    color=text_color,
                )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label('Mean coverage-stress rate')
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)


def _write_audit_manifest(
    *,
    path: Path,
    suite_ids: Sequence[str],
    rows: Sequence[CellGeometrySummary],
    models: Sequence[ModelGeometrySummary],
) -> None:
    payload = {
        'kind': 'selective_geometry_coverage_stress_report',
        'definition': COVERAGE_STRESS_DEFINITION,
        'legacy_passes_filter_modified': False,
        'suite_ids': list(suite_ids),
        'cells': len(rows),
        'query_rows': sum(row.query_count for row in rows),
        'models': [asdict(model) for model in models],
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + '\n')


def main() -> None:
    args = _parse_args()
    inputs = _cell_inputs(args.results_dir, args.suite_ids)
    print(f'Reading {len(inputs)} frozen geometry tables with {args.workers} workers...')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_summarize_cell, inputs))
    model_rows = _model_summaries(rows)

    print(f'Loaded {sum(row.query_count for row in rows):,} query rows across {len(rows)} cells.')
    for model in model_rows:
        print(f'{model.display_label}: coverage-stress={model.coverage_stress_mean:.3f}')
    if args.dry_run:
        return

    macro_path = args.report_dir / 'latex' / 'exp_results_macros.tex'
    _write_macro_block(macro_path, _coverage_macros(model_rows))
    figures_dir = args.report_dir / 'figures'
    primary_rows = [row for row in rows if row.analysis_tier != 'interaction']
    interaction_rows = [row for row in rows if row.analysis_tier == 'interaction']
    _plot_cell_rates(
        rows=primary_rows,
        output_path=figures_dir / 'geometry_pass_rate_by_embedding.png',
        title='Coverage-stress rate by experiment and embedding model',
    )
    _plot_cell_rates(
        rows=interaction_rows,
        output_path=figures_dir / 'interactions' / 'geometry_pass_rate_by_embedding.png',
        title='Coverage-stress rate for interaction experiments',
    )
    aggregate_dir = figures_dir / 'aggregates' / 'embedding_models'
    _plot_model_overview(
        model_rows=model_rows,
        output_path=aggregate_dir / 'embedding_geometry_overview.png',
    )
    _plot_family_heatmap(
        rows=rows,
        model_rows=model_rows,
        output_path=aggregate_dir / 'embedding_geometry_family_heatmap.png',
    )
    _write_audit_manifest(
        path=args.report_dir / 'geometry_coverage_stress_update.json',
        suite_ids=args.suite_ids,
        rows=rows,
        models=model_rows,
    )
    print(f'Updated selective geometry macros and plots under {args.report_dir}.')


if __name__ == '__main__':
    main()
