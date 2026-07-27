from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.reports.helpers import short_token
from experiments.medical_dataset_gen.reports.models import ExperimentRecord
from experiments.medical_dataset_gen.utils.global_schemas import (
    BackgroundDistractorSpec,
    ExperimentCfg,
    LambdaGridCfg,
    LocalChunkPoolCfg,
)


def render_experiment_config_recap(records: Sequence[ExperimentRecord]) -> str:
    grouped: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.distribution_id, []).append(record)

    lines = [
        'The first 3 characters in the Instance name identify the Experiment Family (BAL, BG, DOM, MIS, NIC).\nL, M, S stand for Large, Medium, Small and refer to the overall size of the dataset.\nGlobally, Retrieval Budgets are: Low = 6, Medium = 10, High = 14 document chunks.\n',
    ]
    for distribution_id in sorted(grouped):
        group = sorted(grouped[distribution_id], key=lambda record: record.name)
        representative = next((record for record in group if record.cfg is not None), group[0])
        parent_label = short_token(distribution_id)
        lines.append(f'Instance "{parent_label}":')
        lines.append(f'    Family: {representative.family_label}')
        if representative.cfg is None:
            lines.append(
                '    Config: unavailable; inspect experiment_manifest.csv for load errors.'
            )
        else:
            cfg = representative.cfg
            lines.append(f'    Pool: {_chunk_pool_recap(cfg)};')
            lines.append(f'    Background Outliers: {_background_outlier_recap(cfg)}')
            # lines.append(f'    Retrieval Budgets: {_budget_category_recap(cfg)}')
        lines.append('')
    return '\n'.join(lines)


def _chunk_pool_recap(cfg: ExperimentCfg) -> str:
    pools = cfg.generation.chunk_pools
    return ', '.join(
        [
            _pool_role_recap('dominant primary', pools.dominant_primary),
            _pool_role_recap('other primary', pools.other_primary),
            _pool_role_recap('secondary', pools.secondary),
            _pool_role_recap(
                'niche',
                pools.niche,
                suffix=f'; {pools.niche.num_clusters_per_query} niche clusters/query',
            )
            if pools.niche.num_clusters_per_query > 0
            else 'no niche cluster',
        ]
    )


def _pool_role_recap(
    label: str,
    pool: LocalChunkPoolCfg,
    *,
    suffix: str = '',
) -> str:
    near_miss = pool.total_distractor_chunks()
    return f'{pool.size} {label} (+{near_miss} near-miss){suffix}'


def _background_outlier_recap(cfg: ExperimentCfg) -> str:
    specs = cfg.generation.chunk_pools.background_outliers
    if not specs:
        return 'none configured'
    return '; '.join(_background_spec_recap(spec) for spec in specs)


def _background_spec_recap(spec: BackgroundDistractorSpec) -> str:
    if spec.size == 1:
        return f'{spec.num_clusters} isolated single points.'
    else:
        return f'{spec.num_clusters} clusters, {spec.size} documents each'


def _k_values_recap(cfg: ExperimentCfg) -> str:
    k_values = sorted(int(k) for k in cfg.retrieval.k_values)
    if not k_values:
        return 'none configured'
    low = k_values[0]
    medium = k_values[len(k_values) // 2]
    high = k_values[-1]
    return (
        f'{", ".join(str(k) for k in k_values)} '
        f'(Low Budget={low}, Medium Budget={medium}, High Budget={high})'
    )


def _budget_category_recap(cfg: ExperimentCfg) -> str:
    k_values = sorted(int(k) for k in cfg.retrieval.k_values)
    if not k_values:
        return 'none configured'
    return f'Low = {k_values[0]}, Medium = {k_values[len(k_values) // 2]}, High = {k_values[-1]}'


def _lambda_grid_recap(cfg: ExperimentCfg) -> str:
    return ', '.join(
        [
            _single_lambda_grid_recap('MMR', cfg.retrieval.lambdas_mmr),
            _single_lambda_grid_recap('FacLoc', cfg.retrieval.lambdas_fac_loc),
        ]
    )


def _single_lambda_grid_recap(label: str, grid: LambdaGridCfg) -> str:
    return f'{label} [{grid.start:.3g}, {grid.stop:.3g}] with {grid.num_values} values'


def _geometry_filter_recap(cfg: ExperimentCfg) -> str:
    geometry = cfg.geometry_filter
    return (
        f'{geometry.stress_horizon_basis} horizon {geometry.stress_horizon_fraction:.0%} '
        f'clamped to [{geometry.stress_horizon_min_k}, {geometry.stress_horizon_max_k}], '
        f'max retrieved-facet fraction {geometry.max_retrieved_facet_fraction:.0%}, '
        f'min primary-axis fraction {geometry.min_primary_axis_fraction:.0%}'
    )


def _child_config_recap(record: ExperimentRecord) -> str:
    child_label = _recap_experiment_label(record.name)
    model = _recap_model_label(record.embedding_model)
    if record.cfg is None:
        return f'"{child_label}": model "{model}" (config unavailable)'
    return f'"{child_label}": model "{model}"'


def _recap_model_label(model_name: str) -> str:
    if model_name == 'unknown':
        return model_name
    if model_name.startswith('jinaai/'):
        return model_name
    return model_name.rsplit('/', 1)[-1]


def _recap_experiment_label(exp_name: str) -> str:
    parts = Path(exp_name).parts
    return '/'.join(short_token(part) for part in parts)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    df = pl.DataFrame([dict(row) for row in rows], infer_schema_length=None)
    df.write_csv(path)
