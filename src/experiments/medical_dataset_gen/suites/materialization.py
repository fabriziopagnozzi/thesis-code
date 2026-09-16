"""Create-or-verify materialization for native and derived v5 suites."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from experiments.medical_dataset_gen.suites.contracts import (
    MANIFEST_VERSION,
    SUITE_LAYOUT_VERSION,
    SuiteManifest,
    SuiteManifestCell,
    SuiteManifestDistribution,
    SuiteManifestRunProfile,
    SuiteSpec,
)
from experiments.medical_dataset_gen.suites.io import (
    read_yaml_mapping,
    safe_relative,
    sha256_json,
    write_json,
)
from experiments.medical_dataset_gen.suites.manifests import (
    load_pinned_source_manifest,
    load_suite_manifest,
    suite_root,
    write_suite_manifest,
)
from experiments.medical_dataset_gen.suites.resolution import (
    ValidationResult,
    dataset_hash,
    declared_composition,
    validate_suite,
)
from experiments.medical_dataset_gen.utils.exp_naming import embedding_child_token
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import SHARED_GENERATION_TABLES


def materialize_suite(spec: SuiteSpec, *, results_dir: Path) -> SuiteManifest:
    """Create a suite, or verify that an existing suite has the same contract."""
    root = suite_root(results_dir, spec.suite_id)
    if (root / 'suite_manifest.json').exists():
        existing = load_suite_manifest(results_dir, spec.suite_id)
        if spec.origin == 'native':
            _verify_native(existing, spec, validate_suite(spec), root)
        else:
            _verify_derived(existing, spec, results_dir)
        return existing
    return (
        _materialize_native(spec, root, validate_suite(spec))
        if spec.origin == 'native'
        else _materialize_derived(spec, results_dir, root)
    )


def _verify_native(
    manifest: SuiteManifest,
    spec: SuiteSpec,
    validation: ValidationResult,
    root: Path,
) -> None:
    if manifest.suite_id != spec.suite_id or manifest.origin != 'native':
        raise ValueError(f'{spec.suite_id}: existing suite is not native')
    distributions = {item.distribution_id: item for item in manifest.distributions}
    profiles = {item.run_profile_id: item for item in manifest.run_profiles}
    cells = {item.cell_id: item for item in manifest.cells}
    if set(distributions) != set(spec.distributions):
        raise ValueError(f'{spec.suite_id}: existing distributions differ from the spec')
    if set(profiles) != set(spec.run_profiles):
        raise ValueError(f'{spec.suite_id}: existing run profiles differ from the spec')
    expected_cells = {cell.cell_id: cell for cell in spec.expanded_cells()}
    if set(cells) != set(expected_cells):
        raise ValueError(f'{spec.suite_id}: existing cell IDs differ from the spec')
    if manifest.comparison_groups != spec.comparison_groups:
        raise ValueError(f'{spec.suite_id}: existing comparison groups differ from the spec')
    for distribution_id, raw in validation.resolved_distributions.items():
        distribution = spec.distributions[distribution_id]
        item = distributions[distribution_id]
        factors = {
            **declared_composition(ExperimentCfg.model_validate(raw)),
            **distribution.factors,
        }
        expected = SuiteManifestDistribution(
            distribution_id=distribution_id,
            family_id=distribution.family_id,
            family_label=distribution.family_label,
            factors=factors,
            tags=distribution.tags,
            analysis_blocks=distribution.analysis_blocks,
            analysis_tier=distribution.analysis_tier,
            include_in_family_summary=distribution.include_in_family_summary,
            nested_from=distribution.nested_from,
            resolved_distribution_path=str(
                Path('distributions') / distribution_id / 'resolved_distribution.yaml'
            ),
            distribution_sha256=sha256_json(raw),
            dataset_sha256=dataset_hash(raw),
        )
        if (
            item.resolved_distribution_path != expected.resolved_distribution_path
            or item.distribution_sha256 != expected.distribution_sha256
            or item.dataset_sha256 != expected.dataset_sha256
            or item.nested_from != expected.nested_from
            or _resolved_yaml_hash(root, item.resolved_distribution_path)
            != expected.distribution_sha256
        ):
            raise ValueError(f'{distribution_id}: existing scientific configuration differs')
    for profile_id, profile in spec.run_profiles.items():
        item = profiles[profile_id]
        expected = SuiteManifestRunProfile(
            run_profile_id=profile_id,
            factors=profile.factors,
            resolved_run_profile_path=str(
                Path('run_profiles') / profile_id / 'resolved_run_profile.yaml'
            ),
            run_profile_sha256=sha256_json(profile.config),
        )
        if (
            item.resolved_run_profile_path != expected.resolved_run_profile_path
            or item.run_profile_sha256 != expected.run_profile_sha256
            or _resolved_yaml_hash(root, item.resolved_run_profile_path)
            != expected.run_profile_sha256
        ):
            raise ValueError(f'{profile_id}: existing run-profile configuration differs')
    for cell_id, raw in validation.resolved_configs.items():
        item = cells[cell_id]
        expanded = expected_cells[cell_id]
        distribution = spec.distributions[expanded.distribution_id]
        profile = spec.run_profiles[expanded.run_profile_id]
        factors = {
            **declared_composition(ExperimentCfg.model_validate(raw)),
            **distribution.factors,
        }
        expected = SuiteManifestCell(
            cell_id=cell_id,
            name=f'{spec.suite_id}/{expanded.distribution_id}/{expanded.run_profile_id}',
            distribution_id=expanded.distribution_id,
            run_profile_id=expanded.run_profile_id,
            family_id=distribution.family_id,
            family_label=distribution.family_label,
            origin='native',
            status=item.status,
            include_in_family_summary=distribution.include_in_family_summary,
            factors=factors,
            tags=distribution.tags,
            analysis_blocks=distribution.analysis_blocks,
            analysis_tier=distribution.analysis_tier,
            run_profile_factors=profile.factors,
            data_root=str(Path('distributions') / expanded.distribution_id / 'data' / 'schema-v5'),
            result_root=str(
                Path('distributions')
                / expanded.distribution_id
                / 'runs'
                / expanded.run_profile_id
                / 'attempts'
                / 'initial'
            ),
            resolved_config_path=str(
                Path('distributions')
                / expanded.distribution_id
                / 'runs'
                / expanded.run_profile_id
                / 'resolved_config.yaml'
            ),
            config_sha256=sha256_json(raw),
            dataset_sha256=dataset_hash(raw),
            run_profile_sha256=sha256_json(profile.config),
            nested_from=expanded.nested_from,
        )
        if (
            item.distribution_id != expected.distribution_id
            or item.run_profile_id != expected.run_profile_id
            or item.origin != expected.origin
            or item.data_root != expected.data_root
            or item.result_root != expected.result_root
            or item.resolved_config_path != expected.resolved_config_path
            or item.config_sha256 != expected.config_sha256
            or item.dataset_sha256 != expected.dataset_sha256
            or item.run_profile_sha256 != expected.run_profile_sha256
            or item.nested_from != expected.nested_from
            or _resolved_yaml_hash(root, item.resolved_config_path) != expected.config_sha256
        ):
            raise ValueError(f'{cell_id}: existing resolved configuration differs')


def _verify_derived(manifest: SuiteManifest, spec: SuiteSpec, results_dir: Path) -> None:
    if (
        manifest.suite_id != spec.suite_id
        or manifest.origin != 'derived'
        or manifest.source is None
        or spec.source is None
    ):
        raise ValueError(f'{spec.suite_id}: existing suite is not derived')
    if manifest.source != spec.source:
        raise ValueError(f'{spec.suite_id}: existing pinned source contract differs from the spec')
    source_root, source_manifest = load_pinned_source_manifest(
        results_dir=results_dir, source=spec.source
    )
    root = suite_root(results_dir, spec.suite_id)
    requested = set(spec.source.distribution_ids)
    source_distributions = {item.distribution_id: item for item in source_manifest.distributions}
    missing = requested - set(source_distributions)
    if missing:
        raise ValueError(f'{spec.suite_id}: source distributions are missing: {sorted(missing)}')
    source_cells = [cell for cell in source_manifest.cells if cell.distribution_id in requested]
    source_profile_ids = {cell.run_profile_id for cell in source_cells}
    profile_maps = {
        model_name: {
            profile_id: _derived_run_profile_id(profile_id, embedding_child_token(model_name))
            for profile_id in source_profile_ids
        }
        for model_name in spec.source.embedding_models
    }
    expected_cells = {
        f'{source_cell.distribution_id}__{profile_maps[model_name][source_cell.run_profile_id]}': (
            source_cell,
            model_name,
        )
        for model_name in spec.source.embedding_models
        for source_cell in source_cells
    }
    cells = {cell.cell_id: cell for cell in manifest.cells}
    if set(cells) != set(expected_cells):
        raise ValueError(f'{spec.suite_id}: existing derived cell surface differs from the spec')
    distributions = {item.distribution_id: item for item in manifest.distributions}
    if set(distributions) != requested:
        raise ValueError(f'{spec.suite_id}: existing derived distributions differ from the spec')
    for distribution_id in requested:
        source_distribution = source_distributions[distribution_id]
        expected = source_distribution.model_copy(
            update={
                'resolved_distribution_path': str(
                    Path('distributions') / distribution_id / 'resolved_distribution.yaml'
                )
            }
        )
        item = distributions[distribution_id]
        if item != expected or _resolved_yaml_hash(root, item.resolved_distribution_path) != (
            expected.distribution_sha256
        ):
            raise ValueError(f'{distribution_id}: existing derived distribution differs')
    expected_profiles = {
        profile_id for profile_map in profile_maps.values() for profile_id in profile_map.values()
    }
    profiles = {item.run_profile_id: item for item in manifest.run_profiles}
    if set(profiles) != expected_profiles:
        raise ValueError(f'{spec.suite_id}: existing derived profiles differ from the spec')
    source_profiles = {item.run_profile_id: item for item in source_manifest.run_profiles}
    profile_hashes: dict[str, str] = {}
    for model_name, profile_map in profile_maps.items():
        for source_profile_id, profile_id in profile_map.items():
            source_profile = source_profiles[source_profile_id]
            raw_profile = cast(
                dict[str, Any],
                read_yaml_mapping(
                    safe_relative(source_root, source_profile.resolved_run_profile_path)
                ),
            )
            embeddings = raw_profile.setdefault('embeddings', {})
            if not isinstance(embeddings, dict):
                raise ValueError(f'{source_profile_id}: embeddings profile is not a mapping')
            embeddings['model_name'] = model_name
            override = spec.source.embedding_overrides.get(model_name)
            if override is not None:
                embeddings['batch_size'] = override.batch_size
            profile_hash = sha256_json(raw_profile)
            profile_hashes[profile_id] = profile_hash
            expected = SuiteManifestRunProfile(
                run_profile_id=profile_id,
                factors={**source_profile.factors, 'embedding': model_name},
                resolved_run_profile_path=str(
                    Path('run_profiles') / profile_id / 'resolved_run_profile.yaml'
                ),
                run_profile_sha256=profile_hash,
            )
            item = profiles[profile_id]
            if item != expected or _resolved_yaml_hash(root, item.resolved_run_profile_path) != (
                expected.run_profile_sha256
            ):
                raise ValueError(f'{profile_id}: existing derived run profile differs')
    if manifest.comparison_groups != source_manifest.comparison_groups:
        raise ValueError(f'{spec.suite_id}: existing comparison groups differ from the source')
    for cell_id, (source_cell, model_name) in expected_cells.items():
        cell = cells[cell_id]
        profile_id = profile_maps[model_name][source_cell.run_profile_id]
        raw = cast(
            dict[str, Any],
            read_yaml_mapping(safe_relative(source_root, source_cell.resolved_config_path)),
        )
        embeddings = cast(dict[str, Any], raw['embeddings'])
        embeddings['model_name'] = model_name
        override = spec.source.embedding_overrides.get(model_name)
        if override is not None:
            embeddings['batch_size'] = override.batch_size
        expected_result_root = (
            Path('distributions')
            / source_cell.distribution_id
            / 'runs'
            / profile_id
            / 'attempts'
            / 'initial'
        )
        expected = SuiteManifestCell(
            cell_id=cell_id,
            name=f'{spec.suite_id}/{source_cell.distribution_id}/{profile_id}',
            distribution_id=source_cell.distribution_id,
            run_profile_id=profile_id,
            family_id=source_cell.family_id,
            family_label=source_cell.family_label,
            origin='derived',
            status=cell.status,
            include_in_family_summary=source_cell.include_in_family_summary,
            factors=source_cell.factors,
            tags=source_cell.tags,
            analysis_blocks=source_cell.analysis_blocks,
            analysis_tier=source_cell.analysis_tier,
            run_profile_factors={
                **source_cell.run_profile_factors,
                'embedding': model_name,
            },
            data_root=source_cell.data_root,
            result_root=str(expected_result_root),
            resolved_config_path=str(
                Path('distributions')
                / source_cell.distribution_id
                / 'runs'
                / profile_id
                / 'resolved_config.yaml'
            ),
            config_sha256=sha256_json(raw),
            dataset_sha256=source_cell.dataset_sha256,
            run_profile_sha256=profile_hashes[profile_id],
            source_suite_id=spec.source.suite_id,
            source_cell_id=source_cell.cell_id,
            source_manifest_sha256=spec.source.manifest_sha256,
            source_dataset_sha256=source_cell.dataset_sha256,
        )
        if cell != expected or _resolved_yaml_hash(root, cell.resolved_config_path) != (
            expected.config_sha256
        ):
            raise ValueError(f'{cell_id}: existing derived source mapping or paths differ')


def _resolved_yaml_hash(root: Path, relative_path: str) -> str:
    return sha256_json(read_yaml_mapping(safe_relative(root, relative_path)))


def _materialize_native(spec: SuiteSpec, root: Path, validation: ValidationResult) -> SuiteManifest:
    distributions: list[SuiteManifestDistribution] = []
    for distribution_id, distribution in spec.distributions.items():
        resolved = validation.resolved_distributions[distribution_id]
        cfg = ExperimentCfg.model_validate(resolved)
        dist_root = root / 'distributions' / distribution_id
        path = dist_root / 'resolved_distribution.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        factors = {**declared_composition(cfg), **distribution.factors}
        write_json(
            dist_root / 'distribution_metadata.json',
            {
                'layout_version': SUITE_LAYOUT_VERSION,
                'distribution_id': distribution_id,
                'family_id': distribution.family_id,
                'family_label': distribution.family_label,
                'factors': factors,
                'tags': distribution.tags,
                'analysis_blocks': distribution.analysis_blocks,
                'analysis_tier': distribution.analysis_tier,
                'nested_from': distribution.nested_from,
            },
        )
        distributions.append(
            SuiteManifestDistribution(
                distribution_id=distribution_id,
                family_id=distribution.family_id,
                family_label=distribution.family_label,
                factors=factors,
                tags=distribution.tags,
                analysis_blocks=distribution.analysis_blocks,
                analysis_tier=distribution.analysis_tier,
                include_in_family_summary=distribution.include_in_family_summary,
                nested_from=distribution.nested_from,
                resolved_distribution_path=str(path.relative_to(root)),
                distribution_sha256=sha256_json(resolved),
                dataset_sha256=dataset_hash(resolved),
            )
        )
    profiles: list[SuiteManifestRunProfile] = []
    for profile_id, profile in spec.run_profiles.items():
        path = root / 'run_profiles' / profile_id / 'resolved_run_profile.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(profile.config, sort_keys=False))
        write_json(
            path.parent / 'run_profile_metadata.json',
            {
                'layout_version': SUITE_LAYOUT_VERSION,
                'run_profile_id': profile_id,
                'factors': profile.factors,
            },
        )
        profiles.append(
            SuiteManifestRunProfile(
                run_profile_id=profile_id,
                factors=profile.factors,
                resolved_run_profile_path=str(path.relative_to(root)),
                run_profile_sha256=sha256_json(profile.config),
            )
        )
    cells: list[SuiteManifestCell] = []
    for expanded in spec.expanded_cells():
        distribution = spec.distributions[expanded.distribution_id]
        profile = spec.run_profiles[expanded.run_profile_id]
        resolved = validation.resolved_configs[expanded.cell_id]
        run_root = (
            root / 'distributions' / expanded.distribution_id / 'runs' / expanded.run_profile_id
        )
        config_path = run_root / 'resolved_config.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        factors = {
            **declared_composition(ExperimentCfg.model_validate(resolved)),
            **distribution.factors,
        }
        write_json(
            run_root / 'cell_metadata.json',
            {
                'layout_version': SUITE_LAYOUT_VERSION,
                'manifest_version': MANIFEST_VERSION,
                'cell_id': expanded.cell_id,
                'factors': factors,
                'tags': distribution.tags,
                'analysis_blocks': distribution.analysis_blocks,
                'analysis_tier': distribution.analysis_tier,
                'nested_from': expanded.nested_from,
                'run_profile_factors': profile.factors,
            },
        )
        data_root = root / 'distributions' / expanded.distribution_id / 'data' / 'schema-v5'
        result_root = run_root / 'attempts' / 'initial'
        cells.append(
            SuiteManifestCell(
                cell_id=expanded.cell_id,
                name=f'{spec.suite_id}/{expanded.distribution_id}/{expanded.run_profile_id}',
                distribution_id=expanded.distribution_id,
                run_profile_id=expanded.run_profile_id,
                family_id=distribution.family_id,
                family_label=distribution.family_label,
                origin='native',
                status='planned',
                include_in_family_summary=distribution.include_in_family_summary,
                factors=factors,
                tags=distribution.tags,
                analysis_blocks=distribution.analysis_blocks,
                analysis_tier=distribution.analysis_tier,
                run_profile_factors=profile.factors,
                data_root=str(data_root.relative_to(root)),
                result_root=str(result_root.relative_to(root)),
                resolved_config_path=str(config_path.relative_to(root)),
                config_sha256=sha256_json(resolved),
                dataset_sha256=dataset_hash(resolved),
                run_profile_sha256=sha256_json(profile.config),
                nested_from=expanded.nested_from,
            )
        )
    manifest = SuiteManifest(
        suite_id=spec.suite_id,
        origin='native',
        created_at=datetime.now(UTC).isoformat(),
        cells=cells,
        distributions=distributions,
        run_profiles=profiles,
        comparison_groups=spec.comparison_groups,
    )
    write_suite_manifest(root, manifest)
    return manifest


def _materialize_derived(spec: SuiteSpec, results_dir: Path, root: Path) -> SuiteManifest:
    assert spec.source is not None
    source_root, source_manifest = load_pinned_source_manifest(
        results_dir=results_dir, source=spec.source
    )
    source_distributions = {item.distribution_id: item for item in source_manifest.distributions}
    requested = set(spec.source.distribution_ids)
    missing = requested - set(source_distributions)
    if missing:
        raise ValueError(f'{spec.suite_id}: source distributions are missing: {sorted(missing)}')
    source_cells = [cell for cell in source_manifest.cells if cell.distribution_id in requested]
    incomplete = [cell.cell_id for cell in source_cells if cell.status != 'completed']
    if incomplete:
        raise ValueError(f'{spec.suite_id}: source cells are incomplete: {incomplete[:8]}')
    _validate_source_generation_artifacts(
        results_dir=results_dir,
        source_suite_id=spec.source.suite_id,
        source_cells=source_cells,
    )
    distributions: list[SuiteManifestDistribution] = []
    for distribution_id in spec.source.distribution_ids:
        source_distribution = source_distributions[distribution_id]
        source_path = safe_relative(source_root, source_distribution.resolved_distribution_path)
        target_path = root / 'distributions' / distribution_id / 'resolved_distribution.yaml'
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(source_path.read_text())
        distributions.append(
            source_distribution.model_copy(
                update={'resolved_distribution_path': str(target_path.relative_to(root))}
            )
        )
    source_profiles = {item.run_profile_id: item for item in source_manifest.run_profiles}
    selected_profile_ids = {cell.run_profile_id for cell in source_cells}
    profiles: list[SuiteManifestRunProfile] = []
    profile_maps: dict[str, dict[str, str]] = {}
    profile_hashes: dict[str, str] = {}
    for model_name in spec.source.embedding_models:
        token = embedding_child_token(model_name)
        profile_map = {
            source_profile_id: _derived_run_profile_id(source_profile_id, token)
            for source_profile_id in selected_profile_ids
        }
        profile_maps[model_name] = profile_map
        for source_profile_id in sorted(selected_profile_ids):
            source_profile = source_profiles[source_profile_id]
            raw_profile = cast(
                dict[str, Any],
                read_yaml_mapping(
                    safe_relative(source_root, source_profile.resolved_run_profile_path)
                ),
            )
            embeddings = raw_profile.setdefault('embeddings', {})
            if not isinstance(embeddings, dict):
                raise ValueError(f'{source_profile_id}: embeddings profile is not a mapping')
            embeddings['model_name'] = model_name
            override = spec.source.embedding_overrides.get(model_name)
            if override is not None:
                embeddings['batch_size'] = override.batch_size
            profile_id = profile_map[source_profile_id]
            target_path = root / 'run_profiles' / profile_id / 'resolved_run_profile.yaml'
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(yaml.safe_dump(raw_profile, sort_keys=False))
            factors = {**source_profile.factors, 'embedding': model_name}
            write_json(
                target_path.parent / 'run_profile_metadata.json',
                {
                    'layout_version': SUITE_LAYOUT_VERSION,
                    'run_profile_id': profile_id,
                    'factors': factors,
                    'source_run_profile_id': source_profile_id,
                },
            )
            profile_hashes[profile_id] = sha256_json(raw_profile)
            profiles.append(
                SuiteManifestRunProfile(
                    run_profile_id=profile_id,
                    factors=factors,
                    resolved_run_profile_path=str(target_path.relative_to(root)),
                    run_profile_sha256=profile_hashes[profile_id],
                )
            )
    cells: list[SuiteManifestCell] = []
    for model_name, profile_map in profile_maps.items():
        for source_cell in sorted(source_cells, key=lambda item: item.cell_id):
            raw_config = cast(
                dict[str, Any],
                read_yaml_mapping(safe_relative(source_root, source_cell.resolved_config_path)),
            )
            embeddings = raw_config.setdefault('embeddings', {})
            if not isinstance(embeddings, dict):
                raise ValueError(f'{source_cell.cell_id}: embeddings config is not a mapping')
            embeddings['model_name'] = model_name
            override = spec.source.embedding_overrides.get(model_name)
            if override is not None:
                embeddings['batch_size'] = override.batch_size
            ExperimentCfg.model_validate(raw_config)
            if dataset_hash(raw_config) != source_cell.dataset_sha256:
                raise ValueError(f'{source_cell.cell_id}: source dataset hash is stale')
            profile_id = profile_map[source_cell.run_profile_id]
            cell_id = f'{source_cell.distribution_id}__{profile_id}'
            run_root = root / 'distributions' / source_cell.distribution_id / 'runs' / profile_id
            config_path = run_root / 'resolved_config.yaml'
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))
            write_json(
                run_root / 'cell_metadata.json',
                {
                    'layout_version': SUITE_LAYOUT_VERSION,
                    'manifest_version': MANIFEST_VERSION,
                    'cell_id': cell_id,
                    'source_suite_id': spec.source.suite_id,
                    'source_cell_id': source_cell.cell_id,
                    'source_manifest_sha256': spec.source.manifest_sha256,
                    'source_dataset_sha256': source_cell.dataset_sha256,
                },
            )
            cells.append(
                SuiteManifestCell(
                    cell_id=cell_id,
                    name=f'{spec.suite_id}/{source_cell.distribution_id}/{profile_id}',
                    distribution_id=source_cell.distribution_id,
                    run_profile_id=profile_id,
                    family_id=source_cell.family_id,
                    family_label=source_cell.family_label,
                    origin='derived',
                    status='planned',
                    include_in_family_summary=source_cell.include_in_family_summary,
                    factors=source_cell.factors,
                    tags=source_cell.tags,
                    analysis_blocks=source_cell.analysis_blocks,
                    analysis_tier=source_cell.analysis_tier,
                    run_profile_factors={
                        **source_cell.run_profile_factors,
                        'embedding': model_name,
                    },
                    data_root=source_cell.data_root,
                    result_root=str((run_root / 'attempts' / 'initial').relative_to(root)),
                    resolved_config_path=str(config_path.relative_to(root)),
                    config_sha256=sha256_json(raw_config),
                    dataset_sha256=dataset_hash(raw_config),
                    run_profile_sha256=profile_hashes[profile_id],
                    source_suite_id=spec.source.suite_id,
                    source_cell_id=source_cell.cell_id,
                    source_manifest_sha256=spec.source.manifest_sha256,
                    source_dataset_sha256=source_cell.dataset_sha256,
                )
            )
    manifest = SuiteManifest(
        suite_id=spec.suite_id,
        origin='derived',
        created_at=datetime.now(UTC).isoformat(),
        source=spec.source,
        cells=cells,
        distributions=distributions,
        run_profiles=profiles,
        comparison_groups=source_manifest.comparison_groups,
    )
    write_suite_manifest(root, manifest)
    return manifest


def _derived_run_profile_id(source_profile_id: str, model_token: str) -> str:
    _, separator, suffix = source_profile_id.partition('_')
    return f'{model_token}_{suffix}' if separator else f'{model_token}_{source_profile_id}'


def _validate_source_generation_artifacts(
    *,
    results_dir: Path,
    source_suite_id: str,
    source_cells: list[SuiteManifestCell],
) -> None:
    from experiments.medical_dataset_gen.suites.runtime import SuiteRuntime

    runtime = SuiteRuntime.load(results_dir=results_dir, suite_id=source_suite_id)
    validated_surfaces: set[tuple[str, str, str]] = set()
    for cell in source_cells:
        cfg = runtime.load_config(cell)
        key = (
            cell.distribution_id,
            str(cfg.generation.chunk_text_style),
            f'{cfg.generation.query_structure}:{cfg.generation.focus_mode}',
        )
        if key in validated_surfaces:
            continue
        paths = runtime.paths(cell, cfg)
        missing = [
            table_name
            for table_name in SHARED_GENERATION_TABLES
            if not paths.table_path(table_name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f'{cell.cell_id}: source generation artifacts are missing: {missing}'
            )
        validated_surfaces.add(key)
