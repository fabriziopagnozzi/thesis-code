"""Runtime path resolution for materialized suite cells."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow.parquet as pq
import yaml

from experiments.medical_dataset_gen.suites.core import (
    SuiteManifestCell,
    _dataset_hash,
    _declared_composition,
    _sha256_json,
    load_suite_manifest,
    resolve_derived_source_cell,
    suite_root,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    SharedGenerationArtifactPaths,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


def resolve_manifest_cell(
    *, results_dir: Path, suite_id: str, cell_id: str
) -> tuple[Path, SuiteManifestCell]:
    root = suite_root(results_dir, suite_id)
    manifest = load_suite_manifest(results_dir, suite_id)
    for cell in manifest.cells:
        if cell.cell_id == cell_id:
            return root, cell
    available = ', '.join(cell.cell_id for cell in manifest.cells[:8])
    suffix = ', …' if len(manifest.cells) > 8 else ''
    raise KeyError(f'unknown suite cell {cell_id!r}; available: {available}{suffix}')


def load_cell_config(root: Path, cell: SuiteManifestCell) -> ExperimentCfg:
    path = _safe_relative(root, cell.resolved_config_path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'resolved config is not a mapping: {path}')
    cfg = ExperimentCfg.model_validate(raw)
    # Keep dataset-producing cells in one concrete distribution namespace so
    # the four wording profiles reuse base/chunk artifacts.  Evaluation
    # attempts remain profile-specific via ``artifact_root``.
    cfg.global_.output_experiment = cell.distribution_id
    return cfg


def suite_paths_for_cell(
    *,
    root: Path,
    cell: SuiteManifestCell,
    cfg: ExperimentCfg,
    attempt_id: str | None = None,
    create_attempt: bool = False,
) -> MedicalDatasetGenPaths:
    """Create stage/report paths without relying on legacy directory parsing."""
    if cell.origin == 'derived':
        source_root, source_cell = resolve_derived_source_cell(root=root, cell=cell)
        data_root = _safe_relative(source_root, source_cell.data_root)
    else:
        data_root = _safe_relative(root, cell.data_root)
    base_attempt_root = _safe_relative(root, cell.attempt_root)
    if attempt_id is None:
        artifact_root = base_attempt_root
    else:
        if not attempt_id or '/' in attempt_id or '\\' in attempt_id or attempt_id in {'.', '..'}:
            raise ValueError(f'invalid attempt identifier: {attempt_id!r}')
        artifact_root = base_attempt_root.parent / attempt_id
        if artifact_root.exists() and create_attempt:
            raise FileExistsError(f'evaluation attempt already exists: {artifact_root}')

    chunk_key = (
        'simple_c' if cfg.generation.chunk_text_style == 'ontology_explicit' else 'hardened_c'
    )
    if cfg.generation.query_structure == 'label_only':
        query_key = 'label_only_q_label_only_f'
    else:
        surface = 'biased' if cfg.generation.query_structure == 'unbalanced' else 'unbiased'
        query_key = f'{surface}_q_{cfg.generation.focus_mode}_f'
    shared = {
        'query_plans': data_root / 'base' / 'query_plans.parquet',
        'clinical_facts': data_root / 'base' / 'clinical_facts.parquet',
        'chunk_documents': data_root / 'chunks' / chunk_key / 'chunk_documents.parquet',
        'chunk_memberships': data_root / 'chunks' / chunk_key / 'chunk_memberships.parquet',
        'qrels': data_root / 'chunks' / chunk_key / 'qrels.parquet',
        'queries': data_root / 'queries' / query_key / 'queries.parquet',
        'gold_answers': data_root / 'queries' / query_key / 'gold_answers.parquet',
    }
    return MedicalDatasetGenPaths(
        cell.distribution_id,
        shared_generation_artifact_paths=cast(SharedGenerationArtifactPaths, shared),
        artifact_root=artifact_root,
        cache_namespace='v5',
    )


def write_attempt_metadata(
    *,
    paths: MedicalDatasetGenPaths,
    root: Path,
    cell: SuiteManifestCell,
    attempt_id: str,
) -> None:
    payload = {
        'layout_version': 5,
        'suite_cell_id': cell.cell_id,
        'origin': cell.origin,
        'dataset_schema_version': cell.dataset_schema_version,
        'evaluation_schema_version': 5,
        'attempt_id': attempt_id,
        'source_attempt': cell.attempt_root,
    }
    paths.experiment_dir.mkdir(parents=True, exist_ok=False)
    (paths.experiment_dir / 'attempt_metadata.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n'
    )


def register_completed_evaluation_attempt(
    *,
    root: Path,
    cell: SuiteManifestCell,
    attempt_id: str | None,
) -> None:
    """Publish a completed evaluation attempt without replacing older ones."""
    manifest_path = root / 'suite_manifest.json'
    raw = json.loads(manifest_path.read_text())
    cells = raw.get('cells')
    if not isinstance(cells, list):
        raise ValueError(f'invalid suite manifest: {manifest_path}')
    for entry in cells:
        if not isinstance(entry, dict) or entry.get('cell_id') != cell.cell_id:
            continue
        if attempt_id is None:
            entry['status'] = 'completed'
        else:
            attempts = entry.setdefault('extra_evaluation_attempts', [])
            if not isinstance(attempts, list):
                raise ValueError(f'invalid extra attempts in {manifest_path}')
            if attempt_id not in attempts:
                attempts.append(attempt_id)
        break
    else:
        raise KeyError(f'{manifest_path}: cell disappeared while publishing {cell.cell_id}')
    tmp_path = manifest_path.with_suffix('.json.tmp')
    tmp_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + '\n')
    tmp_path.replace(manifest_path)


def required_nested_scale_source(
    *, root: Path, cell: SuiteManifestCell
) -> SuiteManifestCell | None:
    """Return the terminal (largest) source for a nested scale cell.

    The suite specifies increasing supports as ``large.nested_from=medium``
    and ``medium.nested_from=small`` because that reads naturally in reports.
    Dataset construction runs in the reverse direction: create the terminal
    large support, then project exact smaller supports from it.
    """
    manifest = _manifest_for_root(root, cell)
    cells = {candidate.cell_id: candidate for candidate in manifest.cells}
    current = cell
    while True:
        children = [
            candidate
            for candidate in manifest.cells
            if candidate.nested_from == current.cell_id
            and candidate.run_profile_id == cell.run_profile_id
        ]
        if not children:
            return current if current.cell_id != cell.cell_id else None
        if len(children) != 1:
            raise ValueError(
                f'{cell.cell_id}: nested scale lineage must have one successor, found '
                f'{[candidate.cell_id for candidate in children]}'
            )
        current = children[0]
        if current.cell_id not in cells:
            raise ValueError(f'{cell.cell_id}: nested scale lineage escaped the suite manifest')


def project_nested_scale_parents(
    *, root: Path, source_cell: SuiteManifestCell, source_cfg: ExperimentCfg
) -> list[str]:
    """Project lower supports from a completed largest schema-v5 support.

    Only the terminal scale cell renders evidence.  Smaller cells hard-link
    the common plans/documents and filter the source's fact, membership, and
    qrel rows.  Stable v5 document IDs make the resulting candidates exact
    subsets, without copying or regenerating chunk text.
    """
    manifest = _manifest_for_root(root, source_cell)
    cells = {cell.cell_id: cell for cell in manifest.cells}
    if source_cell.nested_from is None:
        return []
    if any(
        candidate.nested_from == source_cell.cell_id
        and candidate.run_profile_id == source_cell.run_profile_id
        for candidate in manifest.cells
    ):
        # Only a terminal/large cell is allowed to materialize a lineage.
        return []
    if source_cell.origin != 'native' or source_cell.dataset_schema_version != 5:
        return []

    source_paths = suite_paths_for_cell(root=root, cell=source_cell, cfg=source_cfg)
    required = (
        'query_plans',
        'clinical_facts',
        'chunk_documents',
        'chunk_memberships',
        'qrels',
        'queries',
        'gold_answers',
    )
    missing = [name for name in required if not source_paths.table_path(name).is_file()]
    if missing:
        return []

    source_facts = read_parquet(source_paths, 'clinical_facts')
    source_memberships = read_parquet(source_paths, 'chunk_memberships')
    source_qrels = read_parquet(source_paths, 'qrels')
    projected: list[str] = []
    parent_id = source_cell.nested_from
    while parent_id is not None:
        parent = cells.get(parent_id)
        if parent is None:
            raise ValueError(f'{source_cell.cell_id}: missing nested parent {parent_id!r}')
        if parent.origin != 'native' or parent.dataset_schema_version != 5:
            raise ValueError(f'{source_cell.cell_id}: cannot project a non-native v5 parent')
        parent_cfg = load_cell_config(root, parent)
        parent_paths = suite_paths_for_cell(root=root, cell=parent, cfg=parent_cfg)
        _project_one_nested_scale_cell(
            source_paths=source_paths,
            source_facts=source_facts,
            source_memberships=source_memberships,
            source_qrels=source_qrels,
            target_paths=parent_paths,
            target_cfg=parent_cfg,
            source_cell=source_cell,
            target_cell=parent,
        )
        projected.append(parent.cell_id)
        parent_id = parent.nested_from
    return projected


def _project_one_nested_scale_cell(
    *,
    source_paths: MedicalDatasetGenPaths,
    source_facts: pl.DataFrame,
    source_memberships: pl.DataFrame,
    source_qrels: pl.DataFrame,
    target_paths: MedicalDatasetGenPaths,
    target_cfg: ExperimentCfg,
    source_cell: SuiteManifestCell,
    target_cell: SuiteManifestCell,
) -> None:
    base_files = (
        target_paths.table_path('query_plans'),
        target_paths.table_path('clinical_facts'),
    )
    chunk_surface_files = (
        target_paths.table_path('chunk_documents'),
        target_paths.table_path('chunk_memberships'),
        target_paths.table_path('qrels'),
    )
    query_surface_files = (
        target_paths.table_path('queries'),
        target_paths.table_path('gold_answers'),
    )
    base_existing = [path.exists() for path in base_files]
    chunk_surface_existing = [path.exists() for path in chunk_surface_files]
    query_surface_existing = [path.exists() for path in query_surface_files]
    if all(chunk_surface_existing) and all(query_surface_existing):
        if not all(base_existing):
            raise RuntimeError(
                f'{target_cell.cell_id}: nested projection has a completed surface but '
                'is missing its shared base data'
            )
        return
    # Chunk text and query wording are independently shared across run
    # profiles.  A prior profile can therefore leave a complete query surface
    # while this profile still needs its own chunk surface projected.
    if any(chunk_surface_existing) and not all(chunk_surface_existing):
        raise RuntimeError(
            f'{target_cell.cell_id}: nested projection found a partial chunk data tree; '
            'remove no files automatically and inspect the incomplete target manually'
        )
    if any(query_surface_existing) and not all(query_surface_existing):
        raise RuntimeError(
            f'{target_cell.cell_id}: nested projection found a partial query data tree; '
            'remove no files automatically and inspect the incomplete target manually'
        )
    if any(base_existing) and not all(base_existing):
        raise RuntimeError(
            f'{target_cell.cell_id}: nested projection has a partial shared base data tree; '
            'remove no files automatically and inspect the incomplete target manually'
        )
    if all(chunk_surface_existing) and not all(base_existing):
        raise RuntimeError(
            f'{target_cell.cell_id}: nested projection has a completed chunk surface but '
            'is missing its shared base data'
        )

    if not all(base_existing):
        _hard_link(source_paths.table_path('query_plans'), target_paths.table_path('query_plans'))
    if not all(chunk_surface_existing):
        projected_facts = _select_nested_scale_facts(source_facts, target_cfg)
        # ``fact_id`` contains a readable, randomly suffixed identifier and is not
        # a primary key: a collision can occur even within a cluster.  A v5 fact's
        # stable ``chunk_reuse_key`` is propagated as ``chunk_<key>`` into both
        # memberships and qrels, so it is the safe cross-table projection key.
        selected_chunk_keys = projected_facts.select(
            [
                'query_id',
                pl.concat_str([pl.lit('chunk_'), pl.col('chunk_reuse_key')]).alias('chunk_id'),
            ]
        )
        projected_memberships = source_memberships.join(
            selected_chunk_keys, on=['query_id', 'chunk_id'], how='inner'
        )
        projected_qrels = source_qrels.join(
            selected_chunk_keys, on=['query_id', 'chunk_id'], how='inner'
        )
        if projected_memberships.height != projected_qrels.height:
            raise RuntimeError(
                f'{target_cell.cell_id}: source memberships/qrels disagree during nested projection'
            )
        _hard_link(
            source_paths.table_path('chunk_documents'), target_paths.table_path('chunk_documents')
        )
        write_parquet(target_paths, 'chunk_memberships', projected_memberships)
        write_parquet(target_paths, 'qrels', projected_qrels)
        target_paths.table_path('qrels').with_suffix('.projection.json').write_text(
            json.dumps(
                {
                    'layout_version': 5,
                    'projection': 'nested_scale_subset',
                    'source_cell_id': source_cell.cell_id,
                    'target_cell_id': target_cell.cell_id,
                    'source_document_path': str(source_paths.table_path('chunk_documents')),
                    'candidate_rows': projected_qrels.height,
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )
        if not all(base_existing):
            write_parquet(target_paths, 'clinical_facts', projected_facts)
    if not all(query_surface_existing):
        _hard_link(source_paths.table_path('queries'), target_paths.table_path('queries'))
        _hard_link(source_paths.table_path('gold_answers'), target_paths.table_path('gold_answers'))


def _select_nested_scale_facts(
    source_facts: pl.DataFrame, target_cfg: ExperimentCfg
) -> pl.DataFrame:
    """Select the prefix of every stable v5 cluster required by a target scale."""
    pools = target_cfg.generation.chunk_pools
    gold_per_role = {
        'dominant_primary_gold': int(pools.dominant_primary.size or 0),
        'primary_gold': int(pools.other_primary.size or 0),
        'secondary_gold': int(pools.secondary.size or 0),
        'niche_gold': int(pools.niche.size or 0),
    }
    gold = source_facts.filter(pl.col('is_gold')).with_columns(
        pl.col('fact_id').rank('ordinal').over(['query_id', 'facet_id']).alias('_rank')
    )
    gold_limit = pl.lit(0)
    for role, count in gold_per_role.items():
        gold_limit = pl.when(pl.col('cluster_role') == role).then(count).otherwise(gold_limit)
    selected_gold = gold.filter(pl.col('_rank') <= gold_limit).drop('_rank')

    non_gold = source_facts.filter(~pl.col('is_gold')).with_columns(
        pl.col('fact_id').rank('ordinal').over(['query_id', 'cluster_id']).alias('_rank')
    )
    cluster_limit = pl.lit(0)
    near_miss_specs = target_cfg.generation.near_miss_specs
    if near_miss_specs is None:
        raise ValueError('nested v5 scale projections require explicit near_miss_specs')
    for index, spec in enumerate(near_miss_specs, start=1):
        cluster_limit = (
            pl.when(pl.col('cluster_id').str.contains(f'_v5_nm_s{index:02d}_'))
            .then(int(spec.chunks_per_cluster or 0))
            .otherwise(cluster_limit)
        )
    for index, spec in enumerate(pools.background_outliers, start=1):
        cluster_limit = (
            pl.when(pl.col('cluster_id').str.contains(f'_bg_s{index:02d}_'))
            .then(int(spec.chunks_per_cluster or 0))
            .otherwise(cluster_limit)
        )
    selected_non_gold = non_gold.filter(pl.col('_rank') <= cluster_limit).drop('_rank')
    selected = pl.concat([selected_gold, selected_non_gold], how='vertical_relaxed').sort(
        ['query_id', 'fact_id']
    )
    expected_per_query = (
        target_cfg.generation.total_gold_chunks() + target_cfg.generation.total_distractor_chunks()
    )
    counts = selected.group_by('query_id').len().filter(pl.col('len') != expected_per_query)
    if counts.height:
        examples = counts.head(5).to_dicts()
        raise RuntimeError(
            'nested projection did not produce the target candidate mass; '
            f'expected={expected_per_query}, examples={examples}'
        )
    return selected


def _hard_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise FileExistsError(f'nested projection refuses to overwrite {target}')
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError(f'nested projection requires same filesystem: {source} -> {target}')
    os.link(source, target)


def _manifest_for_root(root: Path, cell: SuiteManifestCell):
    # ``root`` already denotes _results/v5/suites/<suite>; derive the ID from
    # the trusted manifest rather than a directory-name convention elsewhere.
    raw = json.loads((root / 'suite_manifest.json').read_text())
    manifest = load_suite_manifest(root.parents[2], str(raw['suite_id']))
    if not any(candidate.cell_id == cell.cell_id for candidate in manifest.cells):
        raise ValueError(f'{cell.cell_id}: cell is not part of {root / "suite_manifest.json"}')
    return manifest


def verify_cell_artifacts(root: Path, cell: SuiteManifestCell) -> list[str]:
    """Return stale-artifact errors from the cell's immutable file manifest."""
    if cell.artifact_manifest_path is None:
        return []
    manifest_path = _safe_relative(root, cell.artifact_manifest_path)
    raw = json.loads(manifest_path.read_text())
    files = raw.get('files')
    if not isinstance(files, list):
        return [f'{manifest_path}: files must be a list']
    errors: list[str] = []
    import hashlib

    for entry in files:
        if not isinstance(entry, Mapping):
            errors.append(f'{manifest_path}: invalid file entry')
            continue
        raw_path = entry.get('path')
        expected = entry.get('sha256')
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            errors.append(f'{manifest_path}: file entry lacks path/hash')
            continue
        path = _safe_relative(root, raw_path)
        if not path.is_file():
            errors.append(f'missing artifact: {path}')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            errors.append(f'stale artifact hash: {path}')
    return errors


@dataclass(frozen=True)
class MaterializedSuiteValidation:
    errors: tuple[str, ...]
    checked_cells: int


def validate_materialized_suite(
    *, results_dir: Path, suite_id: str, verify_hashes: bool = False
) -> MaterializedSuiteValidation:
    """Validate manifest/config drift and completed nested-scale artifacts."""
    root = suite_root(results_dir, suite_id)
    manifest = load_suite_manifest(results_dir, suite_id)
    errors: list[str] = []
    cells = {cell.cell_id: cell for cell in manifest.cells}
    configs: dict[str, ExperimentCfg] = {}
    for cell in manifest.cells:
        if cell.origin == 'derived':
            try:
                resolve_derived_source_cell(root=root, cell=cell)
            except Exception as exc:
                errors.append(f'{cell.cell_id}: invalid derived source ({exc})')
                continue
        config_path = _safe_relative(root, cell.resolved_config_path)
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            errors.append(f'{cell.cell_id}: config is not a mapping')
            continue
        if _sha256_json(raw) != cell.config_sha256:
            errors.append(f'{cell.cell_id}: resolved configuration hash is stale')
        if _dataset_hash(raw) != cell.dataset_sha256:
            errors.append(f'{cell.cell_id}: dataset configuration hash is stale')
        try:
            cfg = load_cell_config(root, cell)
        except Exception as exc:
            errors.append(f'{cell.cell_id}: invalid resolved configuration ({exc})')
            continue
        configs[cell.cell_id] = cfg
        composition = _declared_composition(cfg)
        _compare_factor(errors, cell, 'gold_mass_vector', composition['gold_mass_vector'])
        _compare_factor(errors, cell, 'near_miss_mass', composition['near_miss_mass'])
        _compare_factor(errors, cell, 'background_mass', composition['background_mass'])
        if verify_hashes:
            errors.extend(verify_cell_artifacts(root, cell))

    for cell in manifest.cells:
        if cell.nested_from is None or cell.status != 'completed':
            continue
        parent = cells.get(cell.nested_from)
        if parent is None or parent.status != 'completed':
            errors.append(f'{cell.cell_id}: nested source {cell.nested_from!r} is not completed')
            continue
        cfg = configs.get(cell.cell_id)
        parent_cfg = configs.get(parent.cell_id)
        if cfg is None or parent_cfg is None:
            continue
        child_paths = suite_paths_for_cell(root=root, cell=cell, cfg=cfg)
        parent_paths = suite_paths_for_cell(root=root, cell=parent, cfg=parent_cfg)
        child_qrels = child_paths.table_path('qrels')
        parent_qrels = parent_paths.table_path('qrels')
        if not child_qrels.is_file() or not parent_qrels.is_file():
            errors.append(f'{cell.cell_id}: nested support needs both qrels artifacts')
            continue
        small_ids = set(pq.read_table(parent_qrels, columns=['chunk_id'])['chunk_id'].to_pylist())
        larger_ids = set(pq.read_table(child_qrels, columns=['chunk_id'])['chunk_id'].to_pylist())
        if not small_ids <= larger_ids:
            errors.append(
                f'{cell.cell_id}: nested qrels are not a chunk-id superset of {parent.cell_id}'
            )
        expected = cfg.generation.total_gold_chunks() + cfg.generation.total_distractor_chunks()
        observed = pl.read_parquet(child_qrels, columns=['query_id']).group_by('query_id').len()
        invalid = observed.filter(pl.col('len') != expected)
        if invalid.height:
            examples = invalid.head(5).to_dicts()
            errors.append(
                f'{cell.cell_id}: nested qrels do not preserve exact pool mass '
                f'expected={expected}, examples={examples}'
            )
    return MaterializedSuiteValidation(errors=tuple(errors), checked_cells=len(manifest.cells))


def _compare_factor(
    errors: list[str], cell: SuiteManifestCell, factor: str, actual: object
) -> None:
    if factor in cell.factors and cell.factors[factor] != actual:
        errors.append(
            f'{cell.cell_id}: declared {factor}={cell.factors[factor]!r} does not match {actual!r}'
        )


def _safe_relative(root: Path, raw_path: str) -> Path:
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'path escapes suite root: {raw_path!r}') from exc
    return path
