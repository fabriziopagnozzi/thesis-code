"""Pure suite configuration resolution and scientific-contract validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from experiments.medical_dataset_gen.suites.contracts import (
    ComparisonGroup,
    Distribution,
    ExpandedCell,
    ScaleSupport,
    SetBackgroundComponents,
    SetBackgroundTopology,
    SetGoldMassVector,
    SetNearMissMass,
    SetNearMissMix,
    SetNearMissTopology,
    SuiteSpec,
    SuiteTransform,
)
from experiments.medical_dataset_gen.suites.io import canonical_json, sha256_json
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg


@dataclass(frozen=True)
class ValidationResult:
    resolved_configs: dict[str, dict[str, Any]]
    resolved_distributions: dict[str, dict[str, Any]]


def validate_suite(spec: SuiteSpec) -> ValidationResult:
    if spec.origin == 'derived':
        return ValidationResult(resolved_configs={}, resolved_distributions={})
    resolved_distributions: dict[str, dict[str, Any]] = {}
    for distribution_id, distribution in spec.distributions.items():
        raw = resolve_distribution_config(distribution)
        _validate_single_region_gold_facets(raw)
        ExperimentCfg.model_validate(raw)
        resolved_distributions[distribution_id] = raw
    cells = spec.expanded_cells()
    cells_by_id = {cell.cell_id: cell for cell in cells}
    resolved_configs = {cell.cell_id: resolve_cell_config(spec, cell) for cell in cells}
    for cell_id, resolved in resolved_configs.items():
        ExperimentCfg.model_validate(resolved)
        cell = cells_by_id[cell_id]
        if cell.nested_from is not None:
            parent = cells_by_id[cell.nested_from]
            if parent.run_profile_id != cell.run_profile_id:
                raise ValueError(f'{cell_id}: nested support must keep its run profile')
    _validate_comparison_groups(spec, cells, resolved_configs)
    _validate_declared_composition_factors(spec, resolved_distributions)
    _validate_family_semantics(spec, resolved_configs)
    return ValidationResult(
        resolved_configs=resolved_configs,
        resolved_distributions=resolved_distributions,
    )


def resolve_distribution_config(distribution: Distribution) -> dict[str, Any]:
    raw = deepcopy(distribution.config)
    raw['dataset_schema_version'] = 5
    for transform in distribution.transforms:
        _apply_transform(raw, transform)
    return raw


def resolve_cell_config(spec: SuiteSpec, cell: ExpandedCell) -> dict[str, Any]:
    distribution = spec.distributions[cell.distribution_id]
    profile = spec.run_profiles[cell.run_profile_id]
    raw = deep_merge(resolve_distribution_config(distribution), profile.config)
    raw.setdefault('global', {})
    raw['global']['output_experiment'] = cell.distribution_id
    return raw


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        merged[key] = (
            deep_merge(cast(Mapping[str, Any], existing), cast(Mapping[str, Any], value))
            if isinstance(existing, Mapping) and isinstance(value, Mapping)
            else deepcopy(value)
        )
    return merged


def dataset_hash(raw: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            'dataset_schema_version': raw.get('dataset_schema_version'),
            'global': {
                key: value
                for key, value in cast(Mapping[str, Any], raw.get('global', {})).items()
                if key not in {'output_experiment', 'result_dir_overrides'}
            },
            'generation': raw.get('generation'),
        }
    )


def declared_composition(cfg: ExperimentCfg) -> dict[str, Any]:
    pools = cfg.generation.chunk_pools
    niche_count = pools.niche.num_clusters_per_query
    if niche_count == 0:
        vector = [
            int(pools.dominant_primary.size or 0),
            int(pools.other_primary.size or 0),
            int(pools.secondary.size or 0),
            int(pools.secondary.size or 0),
        ]
    elif niche_count == 1:
        vector = [
            int(pools.dominant_primary.size or 0),
            int(pools.other_primary.size or 0),
            int(pools.secondary.size or 0),
            int(pools.niche.size or 0),
        ]
    else:
        vector = [
            int(pools.dominant_primary.size or 0),
            int(pools.other_primary.size or 0),
            int(pools.niche.size or 0),
            int(pools.niche.size or 0),
        ]
    near_miss_topology = [
        {
            'changes': list(spec.changes),
            'num_clusters': int(spec.num_clusters),
            'chunks_per_cluster': int(spec.chunks_per_cluster or 0),
            'mass': int(spec.size or 0),
        }
        for spec in cfg.generation.near_miss_specs
    ]
    background_topology = [
        {
            'changes': list(spec.changes),
            'num_clusters': int(spec.num_clusters),
            'chunks_per_cluster': int(spec.chunks_per_cluster or 0),
            'mass': int(spec.size or 0) * int(spec.num_clusters),
        }
        for spec in pools.background_outliers
    ]
    near_miss = sum(row['mass'] for row in near_miss_topology)
    background = sum(row['mass'] for row in background_topology)
    return {
        'gold_mass_vector': vector,
        'gold_mass': sum(vector),
        'niche_count': niche_count,
        'near_miss_mass': near_miss,
        'near_miss_topology': near_miss_topology,
        'background_mass': background,
        'background_topology_components': background_topology,
        'pool_mass': sum(vector) + near_miss + background,
    }


def _apply_transform(raw: dict[str, Any], transform: SuiteTransform) -> None:
    pools = cast(dict[str, Any], raw['generation']['chunk_pools'])
    if isinstance(transform, SetGoldMassVector):
        dominant, other, secondary_a, secondary_b = transform.masses
        pools['dominant_primary'] = _with_mass(pools['dominant_primary'], dominant)
        pools['other_primary'] = _with_mass(pools['other_primary'], other)
        if transform.niche_count == 0:
            if secondary_a != secondary_b:
                raise ValueError('zero niche facets require equal final two gold masses')
            pools['secondary'] = _with_mass(pools['secondary'], secondary_a)
            pools['niche'] = _with_mass(pools['niche'], 1)
            pools['niche']['num_clusters_per_query'] = 0
        elif transform.niche_count == 1:
            pools['secondary'] = _with_mass(pools['secondary'], secondary_a)
            pools['niche'] = _with_mass(pools['niche'], secondary_b)
            pools['niche']['num_clusters_per_query'] = 1
        else:
            if secondary_a != secondary_b:
                raise ValueError('two niche facets require equal final two gold masses')
            pools['secondary'] = _with_mass(pools['secondary'], secondary_a + 1)
            pools['niche'] = _with_mass(pools['niche'], secondary_a)
            pools['niche']['num_clusters_per_query'] = 2
        return
    if isinstance(transform, SetNearMissMass):
        raw['generation']['near_miss_specs'] = _resize_near_miss_specs(
            raw['generation']['near_miss_specs'], transform.mass
        )
        return
    if isinstance(transform, SetNearMissMix):
        specs = raw['generation']['near_miss_specs']
        raw['generation']['near_miss_specs'] = _near_miss_mix_specs(
            _near_miss_mass(specs), transform.one_change_fraction
        )
        return
    if isinstance(transform, SetBackgroundTopology):
        if transform.mass == 0:
            pools['background_outliers'] = []
            return
        if len(pools['background_outliers']) != 1:
            raise ValueError('background topology requires exactly one background component')
        current = pools['background_outliers'][0]
        clusters = transform.num_clusters or int(current.get('num_clusters', 1))
        if transform.mass % clusters:
            raise ValueError('background mass must be divisible by num_clusters')
        pools['background_outliers'] = [
            {**current, 'num_clusters': clusters, 'chunks_per_cluster': transform.mass // clusters}
        ]
        return
    if isinstance(transform, SetBackgroundComponents):
        pools['background_outliers'] = deepcopy(transform.components)
        return
    if isinstance(transform, SetNearMissTopology):
        specs = raw['generation']['near_miss_specs']
        requested_total = transform.num_clusters * transform.chunks_per_cluster
        if _near_miss_mass(specs) != requested_total:
            raise ValueError('near-miss mass must equal requested topology mass')
        if transform.num_clusters % len(specs):
            raise ValueError(
                'near-miss topology must allocate clusters equally across active types'
            )
        clusters_per_spec = transform.num_clusters // len(specs)
        raw['generation']['near_miss_specs'] = [
            {
                **spec,
                'num_clusters': clusters_per_spec,
                'chunks_per_cluster': transform.chunks_per_cluster,
            }
            for spec in specs
        ]
        return
    if isinstance(transform, ScaleSupport):
        pool_names = ['dominant_primary', 'other_primary', 'secondary']
        if int(pools['niche'].get('num_clusters_per_query', 0)) > 0:
            pool_names.append('niche')
        for pool_name in pool_names:
            _scale_pool(pools[pool_name], transform.multiplier, total_size=True)
        for component in pools['background_outliers']:
            _scale_pool(component, transform.multiplier, total_size=False)
        for spec in raw['generation']['near_miss_specs']:
            _scale_pool(spec, transform.multiplier, total_size=True)
        return
    raise TypeError(f'unsupported transform {transform!r}')


def _with_mass(pool: Mapping[str, Any], mass: int) -> dict[str, Any]:
    clusters = int(pool.get('num_clusters', 1))
    if mass % clusters:
        clusters = 1
    return {**pool, 'num_clusters': clusters, 'chunks_per_cluster': mass // clusters, 'size': mass}


def _scale_pool(pool: dict[str, Any], multiplier: float, *, total_size: bool) -> None:
    fraction = Fraction(str(multiplier))
    clusters = int(pool.get('num_clusters', 1))
    per_cluster = int(pool.get('chunks_per_cluster', pool.get('size', 1)))
    scaled = Fraction(per_cluster) * fraction
    if scaled.denominator != 1 or scaled < 1:
        raise ValueError(
            f'exact scale_support requires integral chunks_per_cluster: {per_cluster} x {multiplier}'
        )
    pool['num_clusters'] = clusters
    pool['chunks_per_cluster'] = int(scaled)
    pool['size'] = (
        int(pool['num_clusters']) * int(pool['chunks_per_cluster'])
        if total_size
        else int(pool['chunks_per_cluster'])
    )


def _near_miss_mass(specs: Iterable[Mapping[str, Any]]) -> int:
    return sum(
        int(spec.get('num_clusters', 1)) * int(spec.get('chunks_per_cluster', spec.get('size', 1)))
        for spec in specs
    )


def _resize_near_miss_specs(specs: list[dict[str, Any]], mass: int) -> list[dict[str, Any]]:
    if mass == 0:
        return []
    total_clusters = sum(int(spec.get('num_clusters', 1)) for spec in specs)
    if not total_clusters or mass % total_clusters:
        raise ValueError(
            f'near-miss mass {mass} must divide evenly across {total_clusters} stable clusters'
        )
    per_cluster = mass // total_clusters
    return [
        {
            **spec,
            'chunks_per_cluster': per_cluster,
            'size': per_cluster * int(spec.get('num_clusters', 1)),
        }
        for spec in specs
    ]


def _near_miss_mix_specs(mass: int, one_change_fraction: float) -> list[dict[str, Any]]:
    if mass == 0:
        return []
    total_clusters = 4
    one_clusters = int(total_clusters * one_change_fraction)
    if one_clusters not in {0, 2, 4} or mass % total_clusters:
        raise ValueError(
            'near-miss mix needs 0%, 50%, or 100% one-change mass at four fixed clusters'
        )
    per_cluster = mass // total_clusters
    groups = [
        (['condition'], one_clusters // 2),
        (['subgroup'], one_clusters // 2),
        (['condition', 'axis_value_bin'], (total_clusters - one_clusters) // 2),
        (['subgroup', 'axis_value_bin'], (total_clusters - one_clusters) // 2),
    ]
    return [
        {'changes': changes, 'num_clusters': clusters, 'chunks_per_cluster': per_cluster}
        for changes, clusters in groups
        if clusters
    ]


def _validate_single_region_gold_facets(raw: Mapping[str, Any]) -> None:
    generation = cast(Mapping[str, Any], raw['generation'])
    pools = cast(Mapping[str, Mapping[str, Any]], generation['chunk_pools'])
    for pool_name in ('dominant_primary', 'other_primary', 'secondary', 'niche'):
        if int(pools[pool_name].get('num_clusters', 1)) != 1:
            raise ValueError(
                f'{pool_name}: v5 gold facets must have one materialized gold region; '
                'use set_gold_mass_vector to change support'
            )


def _validate_declared_composition_factors(
    spec: SuiteSpec,
    resolved_distributions: Mapping[str, Mapping[str, Any]],
) -> None:
    for distribution_id, distribution in spec.distributions.items():
        composition = declared_composition(
            ExperimentCfg.model_validate(resolved_distributions[distribution_id])
        )
        actual: dict[str, Any] = {
            key: composition[key]
            for key in (
                'gold_mass_vector',
                'gold_mass',
                'niche_count',
                'near_miss_mass',
                'background_mass',
                'pool_mass',
            )
        }
        masses = composition['gold_mass_vector']
        actual['dominance_share'] = max(masses) / sum(masses)
        actual['near_miss_load_ratio'] = composition['near_miss_mass'] / composition['gold_mass']
        actual['near_miss_topology'] = _topology_label(composition['near_miss_topology'])
        actual['background_topology'] = _topology_label(
            composition['background_topology_components']
        )
        actual['one_change_fraction'] = _one_change_fraction(composition['near_miss_topology'])
        actual['background_shell'] = _background_shell(
            composition['background_topology_components']
        )
        for key, expected in distribution.factors.items():
            if (
                key in actual
                and actual[key] is not None
                and canonical_json(expected) != canonical_json(actual[key])
            ):
                raise ValueError(
                    f'{distribution_id}: declared factor {key}={expected!r} disagrees with '
                    f'resolved composition {actual[key]!r}'
                )


def _topology_label(components: object) -> str | None:
    if not isinstance(components, list):
        return None
    if not components:
        return 'none'
    support_counts = {int(component['chunks_per_cluster']) for component in components}
    if len(support_counts) != 1:
        return None
    return (
        f'{sum(int(component["num_clusters"]) for component in components)}x{support_counts.pop()}'
    )


def _one_change_fraction(components: object) -> float | None:
    if not isinstance(components, list) or not components:
        return None
    total = sum(int(component['mass']) for component in components)
    if not total:
        return None
    return (
        sum(int(component['mass']) for component in components if len(component['changes']) == 1)
        / total
    )


def _background_shell(components: object) -> str | None:
    if not isinstance(components, list) or not components:
        return None
    shells = {tuple(component['changes']) for component in components}
    if len(shells) != 1:
        return None
    return {
        ('subgroup',): 'near',
        ('condition', 'subgroup'): 'intermediate',
        ('condition', 'subgroup', 'axis'): 'far',
    }.get(shells.pop())


def _validate_comparison_groups(
    spec: SuiteSpec,
    cells: Sequence[ExpandedCell],
    resolved: Mapping[str, Mapping[str, Any]],
) -> None:
    by_distribution_profile = {(cell.distribution_id, cell.run_profile_id): cell for cell in cells}
    for group in spec.comparison_groups:
        grouped_cells = [
            by_distribution_profile[(distribution_id, profile_id)]
            for profile_id in spec.run_profiles
            for distribution_id in group.distribution_ids
        ]
        for factor in group.varying_factors:
            values = {
                canonical_json(_comparison_factor_value(group, cell, factor))
                for cell in grouped_cells
            }
            if any(_comparison_factor_value(group, cell, factor) is None for cell in grouped_cells):
                raise ValueError(f'{group.comparison_id}: a cell lacks varying factor {factor!r}')
            if len(values) < 2:
                raise ValueError(
                    f'{group.comparison_id}: varying factor {factor!r} has fewer than two values'
                )
            declared_order = group.factor_levels.get(factor)
            if declared_order is not None and set(map(canonical_json, declared_order)) != values:
                raise ValueError(
                    f'{group.comparison_id}: declared factor_levels differ from selected cells'
                )
        for factor in group.matching_factors:
            values = {canonical_json(cell.factors.get(factor)) for cell in grouped_cells}
            if len(values) != 1:
                raise ValueError(f'{group.comparison_id}: matching factor {factor!r} is confounded')
        for profile_id in spec.run_profiles:
            profile_cells = [cell for cell in grouped_cells if cell.run_profile_id == profile_id]
            _validate_group_factor_cross(group, profile_cells)
            _validate_config_diffs(group, profile_cells, resolved)


def _validate_group_factor_cross(group: ComparisonGroup, cells: Sequence[ExpandedCell]) -> None:
    observed = [
        tuple(
            canonical_json(_comparison_factor_value(group, cell, factor))
            for factor in group.varying_factors
        )
        for cell in cells
    ]
    if len(observed) != len(set(observed)):
        raise ValueError(f'{group.comparison_id}: duplicate factor combination in one run profile')
    expected = [
        tuple(canonical_json(value) for value in values) for values in _factor_product(group)
    ]
    if set(observed) != set(expected):
        raise ValueError(f'{group.comparison_id}: incomplete declared factor cross')


def _factor_product(group: ComparisonGroup) -> list[tuple[Any, ...]]:
    if len(group.varying_factors) == 1:
        return [(value,) for value in group.factor_levels[group.varying_factors[0]]]
    first, second = group.varying_factors
    return [
        (first_value, second_value)
        for first_value in group.factor_levels[first]
        for second_value in group.factor_levels[second]
    ]


def _comparison_factor_value(group: ComparisonGroup, cell: ExpandedCell, factor: str) -> object:
    return cell.factors.get(factor, group.reference_levels.get(factor))


def _validate_config_diffs(
    group: ComparisonGroup,
    cells: Sequence[ExpandedCell],
    resolved: Mapping[str, Mapping[str, Any]],
) -> None:
    if len(cells) < 2:
        return
    paths = [path for factor in group.varying_factors for path in group.owned_paths.get(factor, [])]
    if not paths:
        raise ValueError(f'{group.comparison_id}: every varying factor needs owned_paths')
    reference = _flatten_for_diff(resolved[cells[0].cell_id])
    for cell in cells[1:]:
        candidate = _flatten_for_diff(resolved[cell.cell_id])
        illegal = sorted(
            path
            for path in set(reference) | set(candidate)
            if reference.get(path) != candidate.get(path)
            and not any(path == allowed or path.startswith(f'{allowed}.') for allowed in paths)
        )
        if illegal:
            raise ValueError(
                f'{group.comparison_id}: undeclared configuration drift for '
                f'{cell.cell_id}: {illegal[:5]}'
            )


def _flatten_for_diff(value: object, prefix: str = '') -> dict[str, object]:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if prefix == 'global' and key == 'output_experiment':
                continue
            result.update(_flatten_for_diff(child, f'{prefix}.{key}' if prefix else str(key)))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten_for_diff(child, f'{prefix}.{index}'))
        return result
    return {prefix: value}


def _validate_family_semantics(spec: SuiteSpec, configs: Mapping[str, Mapping[str, Any]]) -> None:
    allowed = {
        'balanced_clean',
        'dominance',
        'sparse_niche',
        'near_miss_heavy',
        'background_variant',
        'interaction',
    }
    for cell in spec.expanded_cells():
        distribution = spec.distributions[cell.distribution_id]
        if distribution.family_id not in allowed:
            raise ValueError(f'{cell.cell_id}: non-thesis family_id {distribution.family_id!r}')
        composition = declared_composition(ExperimentCfg.model_validate(configs[cell.cell_id]))
        if distribution.family_id == 'balanced_clean':
            masses = composition['gold_mass_vector']
            if max(masses) / min(masses) > 1.35:
                raise ValueError(
                    f'{cell.cell_id}: balanced_clean gold masses are too uneven: {masses}'
                )
            if composition['near_miss_mass'] > composition['gold_mass']:
                raise ValueError(f'{cell.cell_id}: balanced_clean cannot be near-miss dominated')
