"""Paired, profile-clustered inference for the experiment-comparison report.

The experiment grid is a fixed, deliberately constructed benchmark.  These
procedures therefore quantify sensitivity to the held-out evidence-profile
composition; they do not treat the hand-authored distributions as a random
sample of all possible clinical retrieval settings.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from experiments.medical_dataset_gen.reports.analysis_constants import (
    DeltaMetricLabel,
    StrategyName,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import (
    float_or_none,
    int_or_none,
    wording_config_metadata,
)
from experiments.medical_dataset_gen.reports.models import BudgetCategory, ExperimentRecord
from experiments.medical_dataset_gen.reports.report_config import BUDGET_CATEGORIES

type CellKey = tuple[str, int]

CORE_EMBEDDING_MODELS: tuple[str, str] = (
    'BAAI/bge-m3',
    'Qwen/Qwen3-Embedding-0.6B',
)
CORE_RUN_LABELS: tuple[str, str] = ('bge_m3', 'qwen3_06')
THESIS_STATISTICAL_TABLE_PATH = Path(
    '/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/docs/thesis/paired_statistical_tables.tex'
)


@dataclass(frozen=True)
class PairedMetricSpec:
    label: DeltaMetricLabel
    metric_name: str
    result_column: str


PAIRED_METRIC_SPECS: tuple[PairedMetricSpec, ...] = (
    PairedMetricSpec('FCP', 'FacetCoveragePurity@k', 'facet_coverage_purity'),
    PairedMetricSpec('FacetCoverage', 'FacetCoverage@k', 'facet_coverage'),
    PairedMetricSpec(
        'AllFacetCoverageRate',
        'AllFacetCoverageRate@k',
        'all_facet_coverage',
    ),
    PairedMetricSpec('AllFacetCleanRate', 'AllFacetCleanRate@k', 'all_facet_clean'),
    PairedMetricSpec('Precision', 'Precision@k', 'gold_precision'),
    PairedMetricSpec('alpha_nDCG', 'alpha-nDCG@k', 'alpha_ndcg'),
)
_PAIRED_STRATEGY_LABELS: tuple[tuple[StrategyName, str], ...] = (
    ('top_k', 'TopK'),
    ('mmr', 'MMR'),
    ('fac_loc', 'FacLoc'),
)


def write_paired_effect_datasets(
    *,
    records: Sequence[ExperimentRecord],
    strategy_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    warnings: list[str],
) -> pl.DataFrame:
    """Write partitioned paired effects and return only profile-level FCP rows.

    Keeping every query-level row from the complete suite in one in-memory
    frame is unnecessarily costly. Per-run partitions remain directly
    inspectable while the compact FCP subset supports the primary bootstrap.
    """
    selections = _selected_lambda_map(strategy_rows)
    query_dir = output_dir / 'paired_query_effects'
    profile_dir = output_dir / 'paired_profile_effects'
    _reset_dataset_dir(query_dir)
    _reset_dataset_dir(profile_dir)
    partitions_written = 0
    for record_index, record in enumerate(records):
        record_selections = {
            key: value for key, value in selections.items() if key[0] == record.name
        }
        if not record_selections:
            warnings.append(f'{record.name}: no selected lambda rows for paired inference')
            continue
        frame = _paired_query_effects_for_record(
            record=record,
            selections=record_selections,
            warnings=warnings,
        )
        if not frame.is_empty():
            profile_effects = (
                frame
                .group_by(
                    'Experiment',
                    'Distribution',
                    'ExperimentFamily',
                    'ExperimentFamilyLabel',
                    'RunLabel',
                    'QueryMode',
                    'FocusMode',
                    'ChunkTextMode',
                    'QueryStructure',
                    'ChunkTextStyle',
                    'WordingConfig',
                    'WordingConfigLabel',
                    'EmbeddingModel',
                    'QueryScope',
                    'k',
                    'evidence_profile_id',
                    'MetricLabel',
                    'MetricName',
                )
                .agg(
                    pl.len().alias('QueryOrientations'),
                    pl.col('FacLocValue').mean().alias('FacLocValue'),
                    pl.col('MMRValue').mean().alias('MMRValue'),
                    pl.col('TopKValue').mean().alias('TopKValue'),
                    pl.col('DeltaFacLocMMR').mean().alias('DeltaFacLocMMR'),
                    pl.col('DeltaFacLocTopK').mean().alias('DeltaFacLocTopK'),
                )
                .sort('Experiment', 'k', 'evidence_profile_id', 'MetricLabel')
            )
            partition_name = f'{record_index:03d}_{_safe_partition_name(record.name)}.parquet'
            frame.write_parquet(query_dir / partition_name)
            profile_effects.write_parquet(profile_dir / partition_name)
            partitions_written += 1
            # Retain neither large frame in memory once its artifacts exist.
            del profile_effects
        del frame

    if not partitions_written:
        return pl.DataFrame()
    return (
        pl
        .scan_parquet(profile_dir / '*.parquet')
        .filter(pl.col('MetricLabel') == 'FCP')
        .collect(engine='streaming')
    )


def cell_effect_summary_rows(
    *,
    profile_effects: pl.DataFrame,
    budget_rows: Sequence[Mapping[str, object]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    """Summarize primary FCP FacLoc--MMR effects at configured budget points."""
    if profile_effects.is_empty():
        return []
    budget_by_cell = _budget_by_cell(budget_rows)
    fcp = profile_effects.filter(pl.col('MetricLabel') == 'FCP')
    rows: list[dict[str, object]] = []
    seed_sequence = np.random.SeedSequence(bootstrap_seed)
    cell_groups = list(fcp.group_by('Experiment', 'k', maintain_order=True))
    child_sequences = seed_sequence.spawn(len(cell_groups))
    for ((experiment, k), group), child_seed in zip(cell_groups, child_sequences, strict=True):
        budget = budget_by_cell.get((str(experiment), int(k)))
        if budget is None:
            continue
        values = group['DeltaFacLocMMR'].to_numpy().astype(np.float64, copy=False)
        estimate = _bootstrap_mean_summary(
            values=values,
            bootstrap_replicates=bootstrap_replicates,
            seed=child_seed,
            threshold=practical_effect_threshold('FCP'),
        )
        first = group.row(0, named=True)
        rows.append({
            'Experiment': str(experiment),
            'Distribution': first['Distribution'],
            'ExperimentFamily': first['ExperimentFamily'],
            'ExperimentFamilyLabel': first['ExperimentFamilyLabel'],
            'EmbeddingModel': first['EmbeddingModel'],
            'QueryScope': first['QueryScope'],
            'k': int(k),
            'BudgetCategory': budget,
            'MetricLabel': 'FCP',
            'MetricName': 'FacetCoveragePurity@k',
            'Profiles': len(values),
            'QueryOrientations': int(group['QueryOrientations'].sum()),
            'PracticalThreshold': practical_effect_threshold('FCP'),
            **estimate,
        })
    return rows


def suite_effect_summary_rows(
    *,
    profile_effects: pl.DataFrame,
    budget_rows: Sequence[Mapping[str, object]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    """Equal-family weighted FCP summaries for the fully crossed core suite."""
    if profile_effects.is_empty():
        return []
    budget_by_cell = _budget_by_cell(budget_rows)
    fcp = profile_effects.filter(
        (pl.col('MetricLabel') == 'FCP') & pl.col('EmbeddingModel').is_in(CORE_EMBEDDING_MODELS)
    )
    if fcp.is_empty():
        return []
    fcp = _with_budget_category(fcp, budget_by_cell)
    rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(BUDGET_CATEGORIES):
        budget_frame = fcp.filter(pl.col('BudgetCategory') == budget)
        core_frame = _fully_crossed_core_frame(budget_frame)
        if core_frame.is_empty():
            continue
        rows.append(
            _suite_summary_row(
                frame=core_frame,
                budget=budget,
                scope='Core suite',
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + budget_index,
            )
        )
        for family_index, (family_key, family_frame) in enumerate(
            core_frame.group_by('ExperimentFamilyLabel', maintain_order=True)
        ):
            family = family_key[0] if isinstance(family_key, tuple) else family_key
            rows.append(
                _suite_summary_row(
                    frame=family_frame,
                    budget=budget,
                    scope=str(family),
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed + 100 + budget_index * 10 + family_index,
                )
            )
    return rows


def configuration_suite_effect_summary_rows(
    *,
    profile_effects: pl.DataFrame,
    budget_rows: Sequence[Mapping[str, object]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    """Profile-bootstrap FCP summaries grouped by wording configuration.

    Rows are intentionally equal-family weighted, matching the core suite
    summary. Configuration-level rows pool the two core embedding models;
    configuration-by-model rows expose embedding sensitivity without letting
    sparsely populated auxiliary models affect the comparison.
    """
    if profile_effects.is_empty() or 'WordingConfig' not in profile_effects.columns:
        return []
    budget_by_cell = _budget_by_cell(budget_rows)
    fcp = profile_effects.filter(
        (pl.col('MetricLabel') == 'FCP') & pl.col('EmbeddingModel').is_in(CORE_EMBEDDING_MODELS)
    )
    if fcp.is_empty():
        return []
    fcp = _with_budget_category(fcp, budget_by_cell)
    rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(BUDGET_CATEGORIES):
        budget_frame = fcp.filter(pl.col('BudgetCategory') == budget)
        for config_index, (config_key, config_frame) in enumerate(
            budget_frame.group_by('WordingConfig', maintain_order=True)
        ):
            config = str(config_key[0] if isinstance(config_key, tuple) else config_key)
            if not config:
                continue
            core_frame = _fully_crossed_core_frame(config_frame)
            if core_frame.is_empty():
                continue
            rows.append(
                _configuration_summary_row(
                    frame=core_frame,
                    budget=budget,
                    scope='Configuration',
                    embedding_model='core embeddings',
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed + 1_000 + budget_index * 100 + config_index,
                )
            )
            for model_index, model in enumerate(CORE_EMBEDDING_MODELS):
                model_frame = config_frame.filter(pl.col('EmbeddingModel') == model)
                if model_frame.is_empty():
                    continue
                rows.append(
                    _configuration_summary_row(
                        frame=model_frame,
                        budget=budget,
                        scope='Configuration x embedding',
                        embedding_model=model,
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_seed=(
                            bootstrap_seed
                            + 2_000
                            + budget_index * 200
                            + config_index * 10
                            + model_index
                        ),
                    )
                )
            for family_index, (family_key, family_frame) in enumerate(
                core_frame.group_by('ExperimentFamilyLabel', maintain_order=True)
            ):
                family = str(family_key[0] if isinstance(family_key, tuple) else family_key)
                rows.append(
                    _configuration_summary_row(
                        frame=family_frame,
                        budget=budget,
                        scope='Configuration x family',
                        embedding_model='core embeddings',
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_seed=(
                            bootstrap_seed
                            + 3_000
                            + budget_index * 500
                            + config_index * 20
                            + family_index
                        ),
                        family_label=family,
                    )
                )
    return rows


def leave_one_out_sensitivity_rows(
    *,
    profile_effects: pl.DataFrame,
    budget_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Point-estimate sensitivity for excluding one core distribution or family."""
    if profile_effects.is_empty():
        return []
    budget_by_cell = _budget_by_cell(budget_rows)
    fcp = _with_budget_category(
        profile_effects.filter(
            (pl.col('MetricLabel') == 'FCP') & pl.col('EmbeddingModel').is_in(CORE_EMBEDDING_MODELS)
        ),
        budget_by_cell,
    )
    rows: list[dict[str, object]] = []
    for budget in BUDGET_CATEGORIES:
        core_frame = _fully_crossed_core_frame(fcp.filter(pl.col('BudgetCategory') == budget))
        if core_frame.is_empty():
            continue
        for distribution in core_frame['Distribution'].unique().sort().to_list():
            subset = core_frame.filter(pl.col('Distribution') != distribution)
            rows.append(_sensitivity_row(subset, budget, 'distribution', str(distribution)))
        for family in core_frame['ExperimentFamilyLabel'].unique().sort().to_list():
            subset = core_frame.filter(pl.col('ExperimentFamilyLabel') != family)
            rows.append(_sensitivity_row(subset, budget, 'family', str(family)))
    return rows


def render_statistical_latex_table(rows: Sequence[Mapping[str, object]]) -> str:
    """Render the compact paired-inference table imported by the thesis."""
    lines = [
        '% Auto-generated by experiments.medical_dataset_gen.reports.',
        '% Do not edit this file directly; rerun the report instead.',
        '',
        r'\subsection{Paired Statistical Estimates}',
        r'{\small',
        r'\begin{longtable}{@{}p{0.22\linewidth}lrrrlp{0.22\linewidth}@{}}',
        r'\caption{Profile-cluster bootstrap estimates for the held-out FacLoc--MMR FCP difference in the fully crossed core suite. Families receive equal weight in the Core suite row.}',
        r'\label{tab:paired-fcp-estimates} \\',
        r'\toprule',
        r'\textbf{Scope} & \textbf{Budget} & \textbf{Dist.} & \textbf{Runs} & \textbf{Effect} & \textbf{95\% CI} & \textbf{Interpretation} \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        r'\textbf{Scope} & \textbf{Budget} & \textbf{Dist.} & \textbf{Runs} & \textbf{Effect} & \textbf{95\% CI} & \textbf{Interpretation} \\',
        r'\midrule',
        r'\endhead',
    ]
    budget_labels = {
        'low_budget': 'Low',
        'medium_budget': 'Medium',
        'high_budget': 'High',
    }
    for row in rows:
        scope = _latex_escape(str(row.get('Scope') or ''))
        budget = budget_labels.get(str(row.get('BudgetCategory') or ''), '')
        effect = _format_signed(row.get('MeanDeltaFacLocMMR'))
        ci = f'[{_format_signed(row.get("CI95Low"))}, {_format_signed(row.get("CI95High"))}]'
        interpretation = _latex_escape(str(row.get('PracticalConclusion') or ''))
        distributions = int_or_none(row.get('Distributions')) or 0
        runs = int_or_none(row.get('Runs')) or 0
        lines.append(
            f'{scope} & {budget} & {distributions} & '
            f'{runs} & {effect} & {ci} & {interpretation} \\\\'
        )
    lines.extend([r'\bottomrule', r'\end{longtable}', r'}', ''])
    return '\n'.join(lines)


def _selected_lambda_map(
    strategy_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int, StrategyName], float | None]:
    selections: dict[tuple[str, int, StrategyName], float | None] = {}
    for row in strategy_rows:
        experiment = row.get('Experiment')
        k = int_or_none(row.get('k'))
        strategy = row.get('strategy')
        if (
            not isinstance(experiment, str)
            or k is None
            or strategy not in {'top_k', 'mmr', 'fac_loc'}
        ):
            continue
        selections[(experiment, k, cast(StrategyName, strategy))] = float_or_none(row.get('lam'))
    return selections


def _paired_query_effects_for_record(
    *,
    record: ExperimentRecord,
    selections: Mapping[tuple[str, int, StrategyName], float | None],
    warnings: list[str],
) -> pl.DataFrame:
    path = record.paths.table_path('evaluation_results')
    if not path.is_file():
        warnings.append(f'{record.name}: evaluation_results missing for paired inference')
        return pl.DataFrame()
    metric_columns = [spec.result_column for spec in PAIRED_METRIC_SPECS]
    try:
        schema = pl.read_parquet_schema(path)
    except Exception as exc:
        warnings.append(f'{record.name}: could not inspect evaluation_results ({exc})')
        return pl.DataFrame()
    schema_columns = set(schema)
    source_metric_columns = set(metric_columns)
    derive_all_facet_coverage = (
        'all_facet_coverage' in source_metric_columns
        and 'all_facet_coverage' not in schema_columns
        and 'facet_coverage' in schema_columns
    )
    if derive_all_facet_coverage:
        source_metric_columns.remove('all_facet_coverage')
    required_columns = {
        'query_id',
        'evidence_profile_id',
        'split',
        'strategy',
        'k',
        'lam',
        *source_metric_columns,
    }
    missing = sorted(required_columns - schema_columns)
    if missing:
        warnings.append(f'{record.name}: paired inference missing columns {", ".join(missing)}')
        return pl.DataFrame()

    selection_masks: list[pl.Expr] = []
    for (_experiment, k, strategy), lam in selections.items():
        mask = (pl.col('strategy') == strategy) & (pl.col('k') == k)
        if strategy != 'top_k':
            if lam is None:
                warnings.append(f'{record.name}: selected {strategy} lambda missing at k={k}')
                continue
            mask = mask & (pl.col('lam') == lam)
        selection_masks.append(mask)
    if not selection_masks:
        return pl.DataFrame()
    selected = (
        pl
        .scan_parquet(path)
        .select(sorted(required_columns))
        .filter(pl.col('split') == 'test')
        .filter(pl.any_horizontal(selection_masks))
        .collect()
    )
    if derive_all_facet_coverage:
        selected = selected.with_columns(
            (pl.col('facet_coverage') == 1.0).cast(pl.Float64).alias('all_facet_coverage')
        )
    if selected.is_empty():
        warnings.append(f'{record.name}: no selected held-out rows for paired inference')
        return pl.DataFrame()

    keys = ['query_id', 'evidence_profile_id', 'k']
    strategy_frames: dict[StrategyName, pl.DataFrame] = {}
    for strategy, label in _PAIRED_STRATEGY_LABELS:
        sub = selected.filter(pl.col('strategy') == strategy)
        duplicate_count = sub.group_by(*keys).len().filter(pl.col('len') > 1).height
        if duplicate_count:
            warnings.append(
                f'{record.name}: {duplicate_count} duplicate selected {strategy} query rows'
            )
            return pl.DataFrame()
        strategy_frames[strategy] = sub.select(
            *keys,
            *(pl.col(column).alias(f'{label}_{column}') for column in metric_columns),
        )
    joined = (
        strategy_frames['top_k']
        .join(strategy_frames['mmr'], on=keys, how='inner', validate='1:1')
        .join(strategy_frames['fac_loc'], on=keys, how='inner', validate='1:1')
    )
    if joined.is_empty():
        warnings.append(f'{record.name}: strategy query sets do not overlap for paired inference')
        return pl.DataFrame()

    metadata = {
        'Experiment': record.name,
        'Distribution': record.distribution_id,
        'ExperimentFamily': record.family_id,
        'ExperimentFamilyLabel': record.family_label,
        'RunLabel': record.run_label,
        'EmbeddingModel': record.embedding_model,
        'QueryScope': 'all-query',
        **wording_config_metadata(record),
    }
    frames: list[pl.DataFrame] = []
    for spec in PAIRED_METRIC_SPECS:
        frames.append(
            joined.select(
                *(pl.lit(value).alias(column) for column, value in metadata.items()),
                *keys,
                pl.lit(spec.label).alias('MetricLabel'),
                pl.lit(spec.metric_name).alias('MetricName'),
                pl.col(f'FacLoc_{spec.result_column}').alias('FacLocValue'),
                pl.col(f'MMR_{spec.result_column}').alias('MMRValue'),
                pl.col(f'TopK_{spec.result_column}').alias('TopKValue'),
                (
                    pl.col(f'FacLoc_{spec.result_column}') - pl.col(f'MMR_{spec.result_column}')
                ).alias('DeltaFacLocMMR'),
                (
                    pl.col(f'FacLoc_{spec.result_column}') - pl.col(f'TopK_{spec.result_column}')
                ).alias('DeltaFacLocTopK'),
            )
        )
    return pl.concat(frames, how='vertical_relaxed')


def _budget_by_cell(
    budget_rows: Sequence[Mapping[str, object]],
) -> dict[CellKey, BudgetCategory]:
    out: dict[CellKey, BudgetCategory] = {}
    for row in budget_rows:
        experiment = row.get('Experiment')
        k = int_or_none(row.get('k'))
        category = row.get('BudgetCategory')
        if not isinstance(experiment, str) or k is None or category not in BUDGET_CATEGORIES:
            continue
        out[(experiment, k)] = cast(BudgetCategory, category)
    return out


def _with_budget_category(
    frame: pl.DataFrame,
    budget_by_cell: Mapping[CellKey, BudgetCategory],
) -> pl.DataFrame:
    categories = [
        budget_by_cell.get((str(experiment), int(k)))
        for experiment, k in frame.select('Experiment', 'k').iter_rows(named=False)
    ]
    return frame.with_columns(pl.Series('BudgetCategory', categories, dtype=pl.String)).drop_nulls(
        'BudgetCategory'
    )


def _fully_crossed_core_frame(frame: pl.DataFrame) -> pl.DataFrame:
    core_frame = frame.filter(pl.col('EmbeddingModel').is_in(CORE_EMBEDDING_MODELS))
    required_models = set(CORE_EMBEDDING_MODELS)
    valid_distributions = [
        str(distribution)
        for distribution, models in (
            core_frame
            .group_by('Distribution')
            .agg(pl.col('EmbeddingModel').unique().alias('EmbeddingModels'))
            .iter_rows(named=False)
        )
        if set(cast(list[str], models)) == required_models
    ]
    return core_frame.filter(pl.col('Distribution').is_in(valid_distributions))


def _suite_summary_row(
    *,
    frame: pl.DataFrame,
    budget: BudgetCategory,
    scope: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    cell_values = _cell_profile_values(frame)
    estimate, ci_low, ci_high, ci90_low, ci90_high = _synchronized_suite_bootstrap(
        cell_values=cell_values,
        bootstrap_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    threshold = practical_effect_threshold('FCP')
    return {
        'Scope': scope,
        'BudgetCategory': budget,
        'MetricLabel': 'FCP',
        'MetricName': 'FacetCoveragePurity@k',
        'Distributions': frame['Distribution'].n_unique(),
        'Runs': frame.select('Distribution', 'EmbeddingModel').unique().height,
        'Profiles': frame['evidence_profile_id'].n_unique(),
        'PracticalThreshold': threshold,
        'MeanDeltaFacLocMMR': estimate,
        'CI95Low': ci_low,
        'CI95High': ci_high,
        'CI90Low': ci90_low,
        'CI90High': ci90_high,
        'PracticalConclusion': _practical_conclusion(
            ci95_low=ci_low,
            ci95_high=ci_high,
            ci90_low=ci90_low,
            ci90_high=ci90_high,
            threshold=threshold,
        ),
    }


def _configuration_summary_row(
    *,
    frame: pl.DataFrame,
    budget: BudgetCategory,
    scope: str,
    embedding_model: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    family_label: str | None = None,
) -> dict[str, object]:
    row = _suite_summary_row(
        frame=frame,
        budget=budget,
        scope=scope,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    first = frame.row(0, named=True)
    row.update({
        'WordingConfig': first.get('WordingConfig'),
        'WordingConfigLabel': first.get('WordingConfigLabel'),
        'QueryMode': first.get('QueryMode'),
        'FocusMode': first.get('FocusMode'),
        'ChunkTextMode': first.get('ChunkTextMode'),
        'EmbeddingModel': embedding_model,
        'ExperimentFamilyLabel': family_label,
    })
    return row


def _sensitivity_row(
    frame: pl.DataFrame,
    budget: BudgetCategory,
    omitted_kind: str,
    omitted_value: str,
) -> dict[str, object]:
    estimate = _equal_family_weighted_mean(_cell_profile_values(frame))
    return {
        'BudgetCategory': budget,
        'OmittedKind': omitted_kind,
        'OmittedValue': omitted_value,
        'MeanDeltaFacLocMMR': estimate,
        'Distributions': frame['Distribution'].n_unique(),
        'Runs': frame.select('Distribution', 'EmbeddingModel').unique().height,
    }


def _cell_profile_values(
    frame: pl.DataFrame,
) -> dict[tuple[str, str, str], tuple[NDArray[np.str_], NDArray[np.float64]]]:
    cells: dict[tuple[str, str, str], tuple[NDArray[np.str_], NDArray[np.float64]]] = {}
    for (family, distribution, embedding), group in frame.group_by(
        'ExperimentFamilyLabel', 'Distribution', 'EmbeddingModel', maintain_order=True
    ):
        values = group['DeltaFacLocMMR'].to_numpy().astype(np.float64, copy=False)
        profiles = group['evidence_profile_id'].to_numpy().astype(str, copy=False)
        cells[(str(family), str(distribution), str(embedding))] = (profiles, values)
    return cells


def _equal_family_weighted_mean(
    cells: Mapping[tuple[str, str, str], tuple[NDArray[np.str_], NDArray[np.float64]]],
) -> float:
    by_family: dict[str, dict[str, list[float]]] = {}
    for (family, distribution, _embedding), (_profiles, values) in cells.items():
        by_family.setdefault(family, {}).setdefault(distribution, []).append(float(np.mean(values)))
    family_means = [
        float(
            np.mean([float(np.mean(embedding_means)) for embedding_means in distributions.values()])
        )
        for distributions in by_family.values()
    ]
    return float(np.mean(family_means)) if family_means else float('nan')


def _synchronized_suite_bootstrap(
    *,
    cell_values: Mapping[tuple[str, str, str], tuple[NDArray[np.str_], NDArray[np.float64]]],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    estimate = _equal_family_weighted_mean(cell_values)
    all_profiles = sorted({
        profile for profiles, _values in cell_values.values() for profile in profiles
    })
    if len(all_profiles) < 2:
        return estimate, float('nan'), float('nan'), float('nan'), float('nan')
    profile_to_idx = {profile: index for index, profile in enumerate(all_profiles)}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(all_profiles),
        np.full(len(all_profiles), 1 / len(all_profiles), dtype=np.float64),
        size=bootstrap_replicates,
    ).astype(np.float64, copy=False)
    cell_bootstrap: dict[tuple[str, str, str], NDArray[np.float64]] = {}
    for key, (profiles, values) in cell_values.items():
        indices = np.fromiter((profile_to_idx[profile] for profile in profiles), dtype=np.intp)
        local_weights = weights[:, indices]
        denominators = local_weights.sum(axis=1)
        cell_bootstrap[key] = (local_weights @ values) / denominators
    replicates = _equal_family_weighted_replicates(cell_bootstrap)
    return (
        estimate,
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
        float(np.quantile(replicates, 0.05)),
        float(np.quantile(replicates, 0.95)),
    )


def _equal_family_weighted_replicates(
    cells: Mapping[tuple[str, str, str], NDArray[np.float64]],
) -> NDArray[np.float64]:
    by_family: dict[str, dict[str, list[NDArray[np.float64]]]] = {}
    for (family, distribution, _embedding), values in cells.items():
        by_family.setdefault(family, {}).setdefault(distribution, []).append(values)
    family_means = [
        np.mean(np.stack([np.mean(values, axis=0) for values in distributions.values()]), axis=0)
        for distributions in by_family.values()
    ]
    return np.mean(np.stack(family_means), axis=0)


def _bootstrap_mean_summary(
    *,
    values: NDArray[np.float64],
    bootstrap_replicates: int,
    seed: np.random.SeedSequence,
    threshold: float,
) -> dict[str, object]:
    mean = float(np.mean(values)) if len(values) else float('nan')
    if len(values) < 2:
        return {
            'MeanDeltaFacLocMMR': mean,
            'CI95Low': float('nan'),
            'CI95High': float('nan'),
            'CI90Low': float('nan'),
            'CI90High': float('nan'),
            'PracticalConclusion': 'insufficient profiles',
        }
    rng = np.random.default_rng(seed)
    replicates = np.empty(bootstrap_replicates, dtype=np.float64)
    batch_size = min(250, bootstrap_replicates)
    for start in range(0, bootstrap_replicates, batch_size):
        stop = min(start + batch_size, bootstrap_replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        replicates[start:stop] = values[indices].mean(axis=1)
    ci95_low = float(np.quantile(replicates, 0.025))
    ci95_high = float(np.quantile(replicates, 0.975))
    ci90_low = float(np.quantile(replicates, 0.05))
    ci90_high = float(np.quantile(replicates, 0.95))
    return {
        'MeanDeltaFacLocMMR': mean,
        'CI95Low': ci95_low,
        'CI95High': ci95_high,
        'CI90Low': ci90_low,
        'CI90High': ci90_high,
        'PracticalConclusion': _practical_conclusion(
            ci95_low=ci95_low,
            ci95_high=ci95_high,
            ci90_low=ci90_low,
            ci90_high=ci90_high,
            threshold=threshold,
        ),
    }


def _practical_conclusion(
    *,
    ci95_low: float,
    ci95_high: float,
    ci90_low: float,
    ci90_high: float,
    threshold: float,
) -> str:
    if ci95_low > threshold:
        return 'meaningful FacLoc advantage'
    if ci95_low > 0.0:
        return 'small FacLoc advantage'
    if ci90_low >= -threshold and ci90_high <= threshold:
        return 'practically equivalent'
    if ci95_high < -threshold:
        return 'meaningful MMR advantage'
    if ci95_high < 0.0:
        return 'small MMR advantage'
    return 'inconclusive'


def _format_signed(value: object) -> str:
    numeric = float_or_none(value)
    return '--' if numeric is None or not np.isfinite(numeric) else f'{numeric:+.3f}'


def _latex_escape(value: str) -> str:
    return value.replace('&', r'\&').replace('_', r'\_')


def _reset_dataset_dir(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _safe_partition_name(experiment: str) -> str:
    return experiment.replace('/', '__').replace(' ', '_')
