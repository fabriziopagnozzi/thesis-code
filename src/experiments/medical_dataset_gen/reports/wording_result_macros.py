from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from experiments.medical_dataset_gen.reports.analysis_constants import (
    DISTRIBUTION_EXPERIMENT_FAMILIES,
    practical_effect_threshold,
)
from experiments.medical_dataset_gen.reports.helpers import ordered_embedding_models
from experiments.medical_dataset_gen.reports.report_config import (
    PREFERRED_EMBEDDING_MODEL_ORDER,
    REPORT_METRIC_SPECS,
)

type ReportRow = Mapping[str, object]
type WordingKey = tuple[str, str, str]
type VariantKey = tuple[str, str]


@dataclass(frozen=True)
class FactorContrast:
    column: str
    source: str
    target: str


FACTOR_CONTRASTS: tuple[FactorContrast, ...] = (
    FactorContrast('QueryMode', 'biased', 'unbiased'),
    FactorContrast('FocusMode', 'list', 'natural'),
    FactorContrast('ChunkTextMode', 'simple', 'hardened'),
)
_PREFERRED_MODEL_TOKENS = dict(zip(PREFERRED_EMBEDDING_MODEL_ORDER, ('Bge', 'Qwen'), strict=True))


def _effective_embedding_models(
    *,
    rows: Sequence[ReportRow],
    embedding_models: Sequence[str],
) -> tuple[str, ...]:
    if embedding_models:
        return tuple(embedding_models)
    return tuple(
        ordered_embedding_models(
            str(row.get('EmbeddingModel') or '') for row in rows if row.get('EmbeddingModel')
        )
    )


def _model_token(model: str) -> str:
    token = _PREFERRED_MODEL_TOKENS.get(model)
    if token is not None:
        return token
    return _macro_token(model.rsplit('/', 1)[-1] or model)


def render_wording_result_macros(
    *,
    budget_rows: Sequence[ReportRow],
    geometry_rows: Sequence[ReportRow],
    embedding_models: Sequence[str] = (),
    require_complete_grid: bool = False,
) -> dict[str, str]:
    """Calculate low-budget wording-sweep scalars used by the results preview."""
    low_rows = [row for row in budget_rows if row.get('BudgetCategory') == 'low_budget']
    effective_embedding_models = _effective_embedding_models(
        rows=low_rows,
        embedding_models=embedding_models,
    )
    core_rows = [
        row
        for row in low_rows
        if row.get('EmbeddingModel') in effective_embedding_models
        and row.get('ExperimentFamily') in DISTRIBUTION_EXPERIMENT_FAMILIES
        and _wording_key(row) is not None
    ]
    if not core_rows:
        if require_complete_grid:
            raise ValueError('No fully typed low-budget wording rows were found in the report.')
        return {}

    grid_error = _wording_grid_error(core_rows, embedding_models=effective_embedding_models)
    if grid_error is not None:
        if require_complete_grid:
            raise ValueError(grid_error)
        return {}

    macros = _grid_metadata_macros(
        core_rows=core_rows,
        all_low_rows=low_rows,
        embedding_models=effective_embedding_models,
    )
    macros.update(_fcp_summary_macros('ResultWordingLowOverall', core_rows))
    macros.update(_grouped_fcp_macros(core_rows))
    macros.update(_factor_macros(core_rows))
    macros.update(_family_chunk_macros(core_rows))
    macros.update(_metric_decomposition_macros(core_rows))
    macros.update(_embedding_contrast_macros(core_rows, embedding_models=effective_embedding_models))
    macros.update(
        _geometry_macros(
            core_rows=core_rows,
            geometry_rows=geometry_rows,
            embedding_models=effective_embedding_models,
        )
    )
    macros['ResultWordingLowOverallPrecisionPositiveCells'] = _integer(
        sum(_numeric(row, 'Delta_FacLoc_MMR_Precision') > 0.0 for row in core_rows)
    )
    return macros


def _wording_grid_error(
    rows: Sequence[ReportRow],
    *,
    embedding_models: Sequence[str],
) -> str | None:
    configurations = {_required_wording_key(row) for row in rows}
    chunk_modes = sorted({key[2] for key in configurations})
    standard_configurations = {
        key for key in configurations if key[0] != 'label_only' and key[1] != 'label_only'
    }
    malformed_label_only = {
        key
        for key in configurations
        if (key[0] == 'label_only') != (key[1] == 'label_only')
    }
    standard_query_modes = sorted({key[0] for key in standard_configurations})
    standard_focus_modes = sorted({key[1] for key in standard_configurations})
    expected_configurations = {
        (query_mode, focus_mode, chunk_mode)
        for query_mode in standard_query_modes
        for focus_mode in standard_focus_modes
        for chunk_mode in chunk_modes
    }
    if any(key[0] == 'label_only' for key in configurations):
        expected_configurations.update(
            ('label_only', 'label_only', chunk_mode) for chunk_mode in chunk_modes
        )
    if malformed_label_only or configurations != expected_configurations:
        return (
            'The wording macro grid is ragged: balanced/unbalanced modes must contain their '
            'represented query/focus cross-product, while label_only must have exactly one '
            'query form for every represented chunk mode.'
        )

    expected_models = set(embedding_models)
    models = {str(row.get('EmbeddingModel') or '') for row in rows}
    if models != expected_models:
        return f'The wording macro grid has embedding models {sorted(models)}, expected {sorted(expected_models)}.'

    variants_by_configuration_model: dict[tuple[WordingKey, str], set[VariantKey]] = {}
    row_keys: set[tuple[WordingKey, str, VariantKey]] = set()
    for row in rows:
        wording_key = _required_wording_key(row)
        model = str(row.get('EmbeddingModel') or '')
        variant = (
            str(row.get('ExperimentFamily') or ''),
            str(row.get('Distribution') or ''),
        )
        key = (wording_key, model, variant)
        if key in row_keys:
            return f'The wording macro grid contains a duplicate cell: {key!r}.'
        row_keys.add(key)
        variants_by_configuration_model.setdefault((wording_key, model), set()).add(variant)

    variant_sets = list(variants_by_configuration_model.values())
    if not variant_sets or any(variants != variant_sets[0] for variants in variant_sets[1:]):
        return 'The wording macro grid does not contain the same distribution variants in every triplet/model slice.'
    represented_families = {family for family, _ in variant_sets[0]}
    if represented_families != set(DISTRIBUTION_EXPERIMENT_FAMILIES):
        return (
            f'The wording macro grid has families {sorted(represented_families)}, '
            f'expected {sorted(DISTRIBUTION_EXPERIMENT_FAMILIES)}.'
        )
    return None


def _grid_metadata_macros(
    *,
    core_rows: Sequence[ReportRow],
    all_low_rows: Sequence[ReportRow],
    embedding_models: Sequence[str],
) -> dict[str, str]:
    configurations = {_required_wording_key(row) for row in core_rows}
    variants = {
        (str(row.get('ExperimentFamily') or ''), str(row.get('Distribution') or ''))
        for row in core_rows
    }
    families = {str(row.get('ExperimentFamily') or '') for row in core_rows}
    budgets = {_integer(_numeric(row, 'k')) for row in core_rows}
    test_query_counts = {_integer(_numeric(row, 'TopK_n_queries')) for row in core_rows}
    if len(budgets) != 1 or len(test_query_counts) != 1:
        raise ValueError('The core wording grid must use one low budget and one test-query count.')
    auxiliary_rows = [
        row
        for row in all_low_rows
        if row.get('EmbeddingModel') not in embedding_models
        and row.get('ExperimentFamily') in DISTRIBUTION_EXPERIMENT_FAMILIES
        and _wording_key(row) is not None
    ]
    return {
        'ResultWordingLowBudgetK': next(iter(budgets)),
        'ResultWordingDistributionFamilies': _integer(len(families)),
        'ResultWordingDistributionVariants': _integer(len(variants)),
        'ResultWordingConfigurations': _integer(len(configurations)),
        'ResultWordingCoreEmbeddingModels': _integer(len(embedding_models)),
        'ResultWordingCoreCells': _integer(len(core_rows)),
        'ResultWordingCellsPerConfiguration': _integer(len(core_rows) // len(configurations)),
        'ResultWordingAuxiliaryCells': _integer(len(auxiliary_rows)),
        'ResultWordingEvaluationQueriesPerCell': next(iter(test_query_counts)),
        'ResultWordingFcpPracticalMargin': _fixed(practical_effect_threshold('FCP'), digits=2),
    }


def _grouped_fcp_macros(rows: Sequence[ReportRow]) -> dict[str, str]:
    macros: dict[str, str] = {}
    grouping_specs = (
        ('Config', lambda row: _macro_token('_'.join(_required_wording_key(row)))),
        ('Family', lambda row: _macro_token(str(row.get('ExperimentFamily') or ''))),
        ('Embedding', lambda row: _model_token(str(row.get('EmbeddingModel') or ''))),
        ('Distribution', lambda row: _macro_token(str(row.get('ShortDistribution') or ''))),
    )
    for group_name, token_for_row in grouping_specs:
        grouped: dict[str, list[ReportRow]] = {}
        for row in rows:
            grouped.setdefault(token_for_row(row), []).append(row)
        for token, grouped_rows in grouped.items():
            macros.update(_fcp_summary_macros(f'ResultWordingLow{group_name}{token}', grouped_rows))
    return macros


def _factor_macros(rows: Sequence[ReportRow]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for contrast in FACTOR_CONTRASTS:
        source_rows = [row for row in rows if row.get(contrast.column) == contrast.source]
        target_rows = [row for row in rows if row.get(contrast.column) == contrast.target]
        for level, level_rows in ((contrast.source, source_rows), (contrast.target, target_rows)):
            if level_rows:
                macros.update(
                    _fcp_summary_macros(
                        f'ResultWordingLowFactor{_macro_token(level)}',
                        level_rows,
                    )
                )
        if not source_rows or not target_rows:
            continue
        contrast_prefix = (
            'ResultWordingLowContrast'
            f'{_macro_token(contrast.source)}To{_macro_token(contrast.target)}'
        )
        for column, token in (
            ('TopK_FCP', 'TopKFcpMeanChange'),
            ('MMR_FCP', 'MmrFcpMeanChange'),
            ('FacLoc_FCP', 'FacLocFcpMeanChange'),
            ('Delta_FacLoc_MMR_FCP', 'FacLocMmrFcpMeanDeltaChange'),
        ):
            macros[f'{contrast_prefix}{token}'] = _signed(
                _column_mean(target_rows, column) - _column_mean(source_rows, column),
                digits=3,
            )
    return macros


def _family_chunk_macros(rows: Sequence[ReportRow]) -> dict[str, str]:
    macros: dict[str, str] = {}
    for family in DISTRIBUTION_EXPERIMENT_FAMILIES:
        family_rows = [row for row in rows if row.get('ExperimentFamily') == family]
        if not family_rows:
            continue
        chunk_rows = {
            chunk_mode: [row for row in family_rows if row.get('ChunkTextMode') == chunk_mode]
            for chunk_mode in ('simple', 'hardened')
        }
        prefix = f'ResultWordingLowFamily{_macro_token(family)}'
        for chunk_mode, matching_rows in chunk_rows.items():
            if matching_rows:
                macros[f'{prefix}Chunk{_macro_token(chunk_mode)}FacLocMmrFcpMeanDelta'] = _signed(
                    _column_mean(matching_rows, 'Delta_FacLoc_MMR_FCP'), digits=3
                )
        if chunk_rows['hardened'] and chunk_rows['simple']:
            macros[f'{prefix}ChunkHardenedMinusSimpleFacLocMmrFcpMeanDelta'] = _signed(
                _column_mean(chunk_rows['hardened'], 'Delta_FacLoc_MMR_FCP')
                - _column_mean(chunk_rows['simple'], 'Delta_FacLoc_MMR_FCP'),
                digits=3,
            )
    return macros


def _metric_decomposition_macros(rows: Sequence[ReportRow]) -> dict[str, str]:
    macros: dict[str, str] = {}
    scopes: dict[str, Sequence[ReportRow]] = {'Overall': rows}
    scopes.update(
        {
            f'Family{_macro_token(family)}': [
                row for row in rows if row.get('ExperimentFamily') == family
            ]
            for family in DISTRIBUTION_EXPERIMENT_FAMILIES
        }
    )
    for scope_token, scope_rows in scopes.items():
        for spec in REPORT_METRIC_SPECS:
            metric = spec.metric_label
            metric_token = _metric_macro_token(spec.metric_label)
            macros[f'ResultWordingLow{scope_token}{metric_token}FacLocMmrMeanDelta'] = _signed(
                _column_mean(scope_rows, f'Delta_FacLoc_MMR_{metric}'),
                digits=3,
            )
    return macros


def _embedding_contrast_macros(
    rows: Sequence[ReportRow],
    *,
    embedding_models: Sequence[str],
) -> dict[str, str]:
    macros: dict[str, str] = {}
    bge_model = 'BAAI/bge-m3'
    qwen_model = 'Qwen/Qwen3-Embedding-0.6B'
    if bge_model in embedding_models and qwen_model in embedding_models:
        bge_rows = [row for row in rows if row.get('EmbeddingModel') == bge_model]
        qwen_rows = [row for row in rows if row.get('EmbeddingModel') == qwen_model]
        for column, token in (
            ('TopK_FCP', 'TopKFcpMeanChange'),
            ('MMR_FCP', 'MmrFcpMeanChange'),
            ('FacLoc_FCP', 'FacLocFcpMeanChange'),
            ('Delta_FacLoc_MMR_FCP', 'FacLocMmrFcpMeanDeltaChange'),
        ):
            macros[f'ResultWordingLowQwenMinusBge{token}'] = _signed(
                _column_mean(qwen_rows, column) - _column_mean(bge_rows, column), digits=3
            )
    grouped_rows: dict[tuple[str, str], list[ReportRow]] = {}
    for row in rows:
        config_token = _macro_token('_'.join(_required_wording_key(row)))
        model_token = _model_token(str(row.get('EmbeddingModel') or ''))
        grouped_rows.setdefault((config_token, model_token), []).append(row)
    for (config_token, model_token), matching_rows in grouped_rows.items():
        macro_name = (
            f'ResultWordingLowConfig{config_token}Embedding{model_token}FacLocMmrFcpMeanDelta'
        )
        macros[macro_name] = _signed(_column_mean(matching_rows, 'Delta_FacLoc_MMR_FCP'), digits=3)
    return macros


def _geometry_macros(
    *,
    core_rows: Sequence[ReportRow],
    geometry_rows: Sequence[ReportRow],
    embedding_models: Sequence[str],
) -> dict[str, str]:
    core_experiments = {str(row.get('Experiment') or '') for row in core_rows}
    matching_rows = [
        row
        for row in geometry_rows
        if str(row.get('Experiment') or '') in core_experiments
        and row.get('EmbeddingModel') in embedding_models
    ]
    if len(matching_rows) != len(core_experiments):
        raise ValueError(
            'Geometry rows do not match the complete core wording grid: '
            f'expected {len(core_experiments)}, found {len(matching_rows)}.'
        )
    total_queries = {_integer(_numeric(row, 'GeometryQueries')) for row in matching_rows}
    if len(total_queries) != 1:
        raise ValueError('The core wording grid has inconsistent total query counts.')
    macros = {'ResultWordingDatasetQueries': next(iter(total_queries))}
    for model in embedding_models:
        token = _model_token(model)
        model_rows = [row for row in matching_rows if row.get('EmbeddingModel') == model]
        macros[f'ResultWordingGeometry{token}PassRateMean'] = _fixed(
            _column_mean(model_rows, 'GeometryPassRate'), digits=3
        )
    return macros


def _fcp_summary_macros(prefix: str, rows: Sequence[ReportRow]) -> dict[str, str]:
    if not rows:
        raise ValueError(f'Cannot calculate {prefix} from an empty row set.')
    threshold = practical_effect_threshold('FCP')
    deltas = [_numeric(row, 'Delta_FacLoc_MMR_FCP') for row in rows]
    return {
        f'{prefix}Cells': _integer(len(rows)),
        f'{prefix}TopKFcpMean': _fixed(_column_mean(rows, 'TopK_FCP'), digits=3),
        f'{prefix}MmrFcpMean': _fixed(_column_mean(rows, 'MMR_FCP'), digits=3),
        f'{prefix}FacLocFcpMean': _fixed(_column_mean(rows, 'FacLoc_FCP'), digits=3),
        f'{prefix}FacLocMmrFcpMeanDelta': _signed(
            _column_mean(rows, 'Delta_FacLoc_MMR_FCP'), digits=3
        ),
        f'{prefix}FacLocMmrFcpBetterCells': _integer(sum(delta > threshold for delta in deltas)),
        f'{prefix}FacLocMmrFcpTiedCells': _integer(
            sum(abs(delta) <= threshold for delta in deltas)
        ),
        f'{prefix}FacLocMmrFcpWorseCells': _integer(sum(delta < -threshold for delta in deltas)),
    }


def _wording_key(row: ReportRow) -> WordingKey | None:
    values = (
        str(row.get('QueryMode') or ''),
        str(row.get('FocusMode') or ''),
        str(row.get('ChunkTextMode') or ''),
    )
    return values if all(values) and 'unknown' not in values else None


def _required_wording_key(row: ReportRow) -> WordingKey:
    key = _wording_key(row)
    if key is None:
        raise ValueError(f'Row has incomplete wording metadata: {row.get("Experiment")!r}.')
    return key


def _column_mean(rows: Sequence[ReportRow], column: str) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        family = str(row.get('ExperimentFamily') or row.get('ExperimentFamilyLabel') or 'Unknown')
        grouped.setdefault(family, []).append(_numeric(row, column))
    family_means = [statistics.fmean(values) for values in grouped.values() if values]
    if not family_means:
        raise ValueError(f'Cannot calculate {column} from an empty row set.')
    return statistics.fmean(family_means)


def _numeric(row: ReportRow, column: str) -> float:
    value = row.get(column)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f'Expected numeric {column}, got {value!r}.')
    return float(value)


def _integer(value: int | float) -> str:
    return f'{int(value)}'


def _fixed(value: float, *, digits: int) -> str:
    return f'{value:.{digits}f}'


def _signed(value: float, *, digits: int) -> str:
    return f'{value:+.{digits}f}'


def _macro_token(value: str) -> str:
    digit_words = {
        '0': 'Zero',
        '1': 'One',
        '2': 'Two',
        '3': 'Three',
        '4': 'Four',
        '5': 'Five',
        '6': 'Six',
        '7': 'Seven',
        '8': 'Eight',
        '9': 'Nine',
    }
    parts: list[str] = []
    current = ''
    for character in value:
        if character.isalpha():
            current += character
            continue
        if current:
            parts.append(current.title())
            current = ''
        if character in digit_words:
            parts.append(digit_words[character])
    if current:
        parts.append(current.title())
    return ''.join(parts)


def _metric_macro_token(value: str) -> str:
    parts = [
        ''.join(character for character in part if character.isalpha()) for part in value.split('_')
    ]
    return ''.join(part[:1].upper() + part[1:] for part in parts if part)
