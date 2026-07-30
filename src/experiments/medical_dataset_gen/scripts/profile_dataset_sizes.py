from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from experiments.medical_dataset_gen.utils.exp_naming import child_experiment_names
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_raw_experiment_config,
)

type DatasetSizeMarker = Literal['S', 'M', 'L']
type ThresholdSource = Literal['inferred_natural_breaks', 'manual_cli']
type LoadedParent = tuple[str, Path, ExperimentCfg, PoolMassProfile]

DEFAULT_OUTPUT_RELATIVE_PATH = Path('_reports') / 'experiment_size_mapping.json'


class SizeThresholds(TypedDict):
    small_max_total_chunks: int
    medium_max_total_chunks: int


class PoolMassProfile(TypedDict):
    gold_chunks_per_query: int
    near_miss_chunks_per_query: int
    background_outlier_chunks_per_query: int
    total_chunks_per_query: int


class RolePoolProfile(TypedDict):
    gold_chunks_per_query: int
    near_miss_chunks_per_query: int


class ChunkPoolProfile(TypedDict):
    dominant_primary: RolePoolProfile
    other_primary: RolePoolProfile
    secondary: RolePoolProfile
    niche: RolePoolProfile
    niche_clusters_per_query: int
    background_outlier_chunks_per_query: int


class ParentSizeProfile(TypedDict):
    size_marker: DatasetSizeMarker
    pool_mass: PoolMassProfile
    chunk_pools: ChunkPoolProfile
    config_path: str
    children: list[str]


class DatasetSizeProfile(TypedDict):
    schema_version: int
    basis: str
    threshold_source: ThresholdSource
    thresholds: SizeThresholds
    experiment_size_mapping: dict[str, DatasetSizeMarker]
    parent_profiles: dict[str, ParentSizeProfile]
    warnings: list[str]


class CliArgs(TypedDict):
    results_dir: Path
    output: Path | None
    include_scrapped: bool
    small_max_total_chunks: int | None
    medium_max_total_chunks: int | None
    indent: int | None


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    profile = build_dataset_size_profile(
        results_dir=args['results_dir'],
        include_scrapped=args['include_scrapped'],
        manual_thresholds=_manual_thresholds_from_args(args),
    )
    payload = json.dumps(profile, indent=args['indent'], sort_keys=True)

    output = args['output']
    if output is None:
        print(payload)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f'{payload}\n')
    print(output)


def build_dataset_size_profile(
    *,
    results_dir: Path,
    include_scrapped: bool,
    manual_thresholds: SizeThresholds | None,
) -> DatasetSizeProfile:
    old_results_dir = MedicalDatasetGenPaths.results_dir
    MedicalDatasetGenPaths.results_dir = results_dir
    warnings: list[str] = []
    try:
        loaded_parents: list[LoadedParent] = []
        for parent_name in _parent_experiment_names(results_dir, include_scrapped=include_scrapped):
            parent_path = results_dir / parent_name
            config_path = parent_path / '_config.yaml'
            try:
                cfg = ExperimentCfg.model_validate(
                    load_raw_experiment_config(MedicalDatasetGenPaths(parent_name))
                )
            except Exception as exc:
                warnings.append(
                    f'{parent_name}: skipped because config could not be loaded ({exc})'
                )
                continue
            loaded_parents.append((parent_name, config_path, cfg, _pool_mass_profile(cfg)))

        thresholds = (
            manual_thresholds
            if manual_thresholds is not None
            else infer_size_thresholds(
                [
                    parent_profile['total_chunks_per_query']
                    for _parent_name, _config_path, _cfg, parent_profile in loaded_parents
                ]
            )
        )
        threshold_source: ThresholdSource = (
            'manual_cli' if manual_thresholds is not None else 'inferred_natural_breaks'
        )

        parent_profiles: dict[str, ParentSizeProfile] = {}
        experiment_size_mapping: dict[str, DatasetSizeMarker] = {}
        for parent_name, config_path, cfg, pool_mass in loaded_parents:
            size_marker = classify_pool_mass(
                total_chunks_per_query=pool_mass['total_chunks_per_query'],
                thresholds=thresholds,
            )
            children = child_experiment_names(parent_name, results_dir=results_dir)
            parent_profiles[parent_name] = {
                'size_marker': size_marker,
                'pool_mass': pool_mass,
                'chunk_pools': _chunk_pool_profile(cfg),
                'config_path': str(config_path),
                'children': children,
            }
            experiment_size_mapping[parent_name] = size_marker
            for child_name in children:
                experiment_size_mapping[child_name] = size_marker
    finally:
        MedicalDatasetGenPaths.results_dir = old_results_dir

    return {
        'schema_version': 1,
        'basis': (
            'S/M/L is classified from per-query chunk-pool mass only: '
            'gold chunks + near-miss distractor chunks + background outlier chunks.'
        ),
        'threshold_source': threshold_source,
        'thresholds': thresholds,
        'experiment_size_mapping': dict(sorted(experiment_size_mapping.items())),
        'parent_profiles': dict(sorted(parent_profiles.items())),
        'warnings': warnings,
    }


def classify_pool_mass(
    *,
    total_chunks_per_query: int,
    thresholds: SizeThresholds,
) -> DatasetSizeMarker:
    if total_chunks_per_query <= thresholds['small_max_total_chunks']:
        return 'S'
    if total_chunks_per_query <= thresholds['medium_max_total_chunks']:
        return 'M'
    return 'L'


def infer_size_thresholds(total_chunks_per_query: Sequence[int]) -> SizeThresholds:
    totals = sorted(total_chunks_per_query)
    if len(totals) < 3:
        raise ValueError('at least three loaded parent configs are required to infer S/M/L breaks')

    best_sse: float | None = None
    best_breaks: tuple[int, int] | None = None
    for first_break in range(1, len(totals) - 1):
        for second_break in range(first_break + 1, len(totals)):
            groups = (
                totals[:first_break],
                totals[first_break:second_break],
                totals[second_break:],
            )
            sse = sum(_within_group_sse(group) for group in groups)
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_breaks = (first_break, second_break)

    if best_breaks is None:
        raise ValueError('could not infer S/M/L breaks from loaded parent configs')

    first_break, second_break = best_breaks
    return {
        'small_max_total_chunks': totals[first_break - 1],
        'medium_max_total_chunks': totals[second_break - 1],
    }


def _within_group_sse(values: Sequence[int]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values)


def _pool_mass_profile(cfg: ExperimentCfg) -> PoolMassProfile:
    chunk_pools = cfg.generation.chunk_pools
    gold_chunks = chunk_pools.gold_chunks_per_query()
    near_miss_chunks = chunk_pools.near_miss_distractors_per_query()
    background_chunks = chunk_pools.background_outliers_per_query()
    return {
        'gold_chunks_per_query': gold_chunks,
        'near_miss_chunks_per_query': near_miss_chunks,
        'background_outlier_chunks_per_query': background_chunks,
        'total_chunks_per_query': gold_chunks + near_miss_chunks + background_chunks,
    }


def _chunk_pool_profile(cfg: ExperimentCfg) -> ChunkPoolProfile:
    chunk_pools = cfg.generation.chunk_pools
    return {
        'dominant_primary': {
            'gold_chunks_per_query': chunk_pools.dominant_primary.size,
            'near_miss_chunks_per_query': chunk_pools.dominant_primary.total_distractor_chunks(),
        },
        'other_primary': {
            'gold_chunks_per_query': chunk_pools.other_primary.size,
            'near_miss_chunks_per_query': chunk_pools.other_primary.total_distractor_chunks(),
        },
        'secondary': {
            'gold_chunks_per_query': chunk_pools.secondary.size,
            'near_miss_chunks_per_query': chunk_pools.secondary.total_distractor_chunks(),
        },
        'niche': {
            'gold_chunks_per_query': chunk_pools.niche.size,
            'near_miss_chunks_per_query': chunk_pools.niche.total_distractor_chunks(),
        },
        'niche_clusters_per_query': chunk_pools.niche.num_clusters_per_query,
        'background_outlier_chunks_per_query': (chunk_pools.background_outliers_per_query()),
    }


def _parent_experiment_names(results_dir: Path, *, include_scrapped: bool) -> list[str]:
    names: list[str] = []
    for config_path in sorted(results_dir.glob('*/_config.yaml')):
        name = config_path.parent.relative_to(results_dir).as_posix()
        if name.startswith('_'):
            continue
        if name == '00_scrapped':
            continue
        names.append(name)

    if include_scrapped:
        names.extend(
            config_path.parent.relative_to(results_dir).as_posix()
            for config_path in sorted((results_dir / '00_scrapped').glob('*/_config.yaml'))
        )
    return names


def _parse_args(argv: Sequence[str] | None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description='Profile experiment dataset sizes from configured chunk-pool mass.'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=MedicalDatasetGenPaths.results_dir,
        help='Root _results directory to scan.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='JSON output path. Defaults to the sibling _reports/experiment_size_mapping.json. Use "-" to print to stdout.',
    )
    parser.add_argument(
        '--include-scrapped',
        action='store_true',
        help='Also scan archived experiments under 00_scrapped.',
    )
    parser.add_argument(
        '--small-max-total-chunks',
        type=int,
        default=None,
        help='Manual largest per-query chunk-pool total classified as S. If omitted, thresholds are inferred from the profile.',
    )
    parser.add_argument(
        '--medium-max-total-chunks',
        type=int,
        default=None,
        help='Manual largest per-query chunk-pool total classified as M. Larger totals are L. If omitted, thresholds are inferred from the profile.',
    )
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Write compact JSON instead of pretty-printed JSON.',
    )
    namespace = parser.parse_args(argv)
    results_dir = cast(Path, namespace.results_dir)
    raw_output = cast(Path | None, namespace.output)
    output = results_dir.parent / DEFAULT_OUTPUT_RELATIVE_PATH if raw_output is None else raw_output
    return {
        'results_dir': results_dir,
        'output': None if str(output) == '-' else output,
        'include_scrapped': bool(namespace.include_scrapped),
        'small_max_total_chunks': cast(int | None, namespace.small_max_total_chunks),
        'medium_max_total_chunks': cast(int | None, namespace.medium_max_total_chunks),
        'indent': None if bool(namespace.compact) else 2,
    }


def _manual_thresholds_from_args(args: CliArgs) -> SizeThresholds | None:
    small_max = args['small_max_total_chunks']
    medium_max = args['medium_max_total_chunks']
    if small_max is None and medium_max is None:
        return None
    if small_max is None or medium_max is None:
        raise ValueError(
            '--small-max-total-chunks and --medium-max-total-chunks must be provided together'
        )
    if small_max >= medium_max:
        raise ValueError('small max threshold must be lower than medium max threshold')
    return {
        'small_max_total_chunks': small_max,
        'medium_max_total_chunks': medium_max,
    }


if __name__ == '__main__':
    main()
