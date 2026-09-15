"""Typed runtime context for one materialized v5 suite."""

from __future__ import annotations

import fcntl
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

import polars as pl
import pyarrow.parquet as pq
import yaml

from experiments.medical_dataset_gen.suites.contracts import (
    SuiteManifest,
    SuiteManifestCell,
    SuiteManifestDistribution,
    SuiteManifestRunProfile,
)
from experiments.medical_dataset_gen.suites.io import safe_relative, sha256_json
from experiments.medical_dataset_gen.suites.manifests import (
    load_suite_manifest,
    resolve_derived_source_cell,
    suite_root,
    write_suite_manifest,
)
from experiments.medical_dataset_gen.suites.resolution import dataset_hash, declared_composition
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    EMBEDDING_ARTIFACT_FILENAMES,
    MedicalDatasetGenPaths,
    SharedEmbeddingArtifactPaths,
    SharedGenerationArtifactPaths,
)


@dataclass(frozen=True)
class SuiteRuntime:
    root: Path
    manifest: SuiteManifest
    cells: dict[str, SuiteManifestCell]
    profiles: dict[str, SuiteManifestRunProfile]
    distributions: dict[str, SuiteManifestDistribution]

    @classmethod
    def load(cls, *, results_dir: Path, suite_id: str) -> SuiteRuntime:
        manifest = load_suite_manifest(results_dir, suite_id)
        return cls(
            root=suite_root(results_dir, suite_id),
            manifest=manifest,
            cells={cell.cell_id: cell for cell in manifest.cells},
            profiles={profile.run_profile_id: profile for profile in manifest.run_profiles},
            distributions={
                distribution.distribution_id: distribution
                for distribution in manifest.distributions
            },
        )

    def cell(self, cell_id: str) -> SuiteManifestCell:
        try:
            return self.cells[cell_id]
        except KeyError as exc:
            available = ', '.join(list(self.cells)[:8])
            suffix = ', …' if len(self.cells) > 8 else ''
            raise KeyError(
                f'unknown suite cell {cell_id!r}; available: {available}{suffix}'
            ) from exc

    def load_config(self, cell: SuiteManifestCell) -> ExperimentCfg:
        path = safe_relative(self.root, cell.resolved_config_path)
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f'resolved config is not a mapping: {path}')
        cfg = ExperimentCfg.model_validate(raw)
        cfg.global_.output_experiment = cell.distribution_id
        return cfg

    def paths(
        self,
        cell: SuiteManifestCell,
        cfg: ExperimentCfg,
        *,
        prevalidated_data_root: Path | None = None,
    ) -> MedicalDatasetGenPaths:
        if prevalidated_data_root is not None:
            data_root = prevalidated_data_root
        elif cell.origin == 'derived':
            source_root, source_cell = resolve_derived_source_cell(root=self.root, cell=cell)
            data_root = safe_relative(source_root, source_cell.data_root)
        else:
            data_root = safe_relative(self.root, cell.data_root)
        artifact_root = safe_relative(self.root, cell.result_root)
        chunk_key = (
            'simple_c' if cfg.generation.chunk_text_style == 'ontology_explicit' else 'hardened_c'
        )
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
        embeddings = suite_shared_embedding_artifact_paths(
            root=self.root,
            distribution_id=cell.distribution_id,
            chunk_key=chunk_key,
            query_key=query_key,
            cfg=cfg,
        )
        return MedicalDatasetGenPaths(
            cell.distribution_id,
            shared_generation_artifact_paths=cast(SharedGenerationArtifactPaths, shared),
            shared_embedding_artifact_paths=embeddings,
            artifact_root=artifact_root,
            cache_namespace='v5',
        )

    def select(self, raw_where: str) -> list[SuiteManifestCell]:
        return [cell for cell in self.manifest.cells if suite_where_matches(cell, raw_where)]

    def mark_completed(self, cell: SuiteManifestCell) -> None:
        with _SuiteManifestUpdateLock(self.root):
            latest = load_suite_manifest(self.root.parents[2], self.manifest.suite_id)
            canonical = next(
                (candidate for candidate in latest.cells if candidate.cell_id == cell.cell_id),
                None,
            )
            if canonical is None:
                raise KeyError(f'{cell.cell_id}: cell disappeared from the suite manifest')
            if canonical.status == 'completed':
                return
            updated_cells = [
                candidate.model_copy(update={'status': 'completed'})
                if candidate.cell_id == cell.cell_id
                else candidate
                for candidate in latest.cells
            ]
            write_suite_manifest(
                self.root, latest.model_copy(update={'cells': updated_cells})
            )


def suite_shared_embedding_artifact_paths(
    *,
    root: Path,
    distribution_id: str,
    chunk_key: str,
    query_key: str,
    cfg: ExperimentCfg,
) -> SharedEmbeddingArtifactPaths:
    from experiments.medical_dataset_gen.dataset_generation.caches import (
        chunk_embedding_signature,
        query_embedding_signature,
    )

    chunk_root = (
        root
        / 'distributions'
        / distribution_id
        / 'shared_embeddings'
        / 'chunks'
        / chunk_key
        / chunk_embedding_signature(cfg)
    )
    query_root = (
        root
        / 'distributions'
        / distribution_id
        / 'shared_embeddings'
        / 'queries'
        / query_key
        / query_embedding_signature(cfg)
    )
    return {
        'chunk_vectors': chunk_root / EMBEDDING_ARTIFACT_FILENAMES['chunk_vectors'],
        'chunk_ids': chunk_root / EMBEDDING_ARTIFACT_FILENAMES['chunk_ids'],
        'query_vectors': query_root / EMBEDDING_ARTIFACT_FILENAMES['query_vectors'],
        'query_ids': query_root / EMBEDDING_ARTIFACT_FILENAMES['query_ids'],
    }


class SuiteDistributionLock:
    """A non-blocking lock around one distribution's shared v5 artifacts."""

    def __init__(self, *, root: Path, distribution_id: str) -> None:
        self.path = root / '.locks' / f'{distribution_id}.lock'
        self.handle: IO[str] | None = None

    def __enter__(self) -> SuiteDistributionLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+')
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f'{self.path.stem.removesuffix(".lock")}: suite distribution is already running'
            ) from exc
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def suite_distribution_lock(*, root: Path, distribution_id: str) -> SuiteDistributionLock:
    return SuiteDistributionLock(root=root, distribution_id=distribution_id)


class _SuiteManifestUpdateLock:
    def __init__(self, root: Path) -> None:
        self.path = root / '.locks' / 'suite_manifest.lock'
        self.handle: IO[str] | None = None

    def __enter__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+')
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def suite_where_matches(cell: SuiteManifestCell, raw_where: str) -> bool:
    for clause in raw_where.split(','):
        if '=' not in clause:
            raise ValueError(f'--where expects key=value clauses, got {clause!r}')
        key, expected = (part.strip() for part in clause.split('=', 1))
        if not key or not expected:
            raise ValueError(f'--where expects non-empty key=value clauses, got {clause!r}')
        expected_values = set(expected.split('|'))
        if key == 'tag':
            if not expected_values.intersection(cell.tags):
                return False
            continue
        if key == 'analysis_block':
            if not expected_values.intersection(cell.analysis_blocks):
                return False
            continue
        values: dict[str, object] = {
            'cell_id': cell.cell_id,
            'distribution_id': cell.distribution_id,
            'run_profile_id': cell.run_profile_id,
            'family_id': cell.family_id,
            'analysis_tier': cell.analysis_tier,
            **cell.run_profile_factors,
            **cell.factors,
        }
        if str(values.get(key)) not in expected_values:
            return False
    return True


def nested_depth(cell: SuiteManifestCell, by_id: Mapping[str, SuiteManifestCell]) -> int:
    depth = 0
    current = cell
    while current.nested_from is not None:
        parent = by_id.get(current.nested_from)
        if parent is None:
            break
        depth += 1
        current = parent
    return depth


@dataclass(frozen=True)
class MaterializedSuiteValidation:
    errors: tuple[str, ...]
    checked_cells: int


def validate_materialized_suite(*, results_dir: Path, suite_id: str) -> MaterializedSuiteValidation:
    runtime = SuiteRuntime.load(results_dir=results_dir, suite_id=suite_id)
    errors: list[str] = []
    configs: dict[str, ExperimentCfg] = {}
    for cell in runtime.manifest.cells:
        if cell.origin == 'derived':
            try:
                resolve_derived_source_cell(root=runtime.root, cell=cell)
            except Exception as exc:
                errors.append(f'{cell.cell_id}: invalid derived source ({exc})')
                continue
        raw = yaml.safe_load(safe_relative(runtime.root, cell.resolved_config_path).read_text())
        if not isinstance(raw, dict):
            errors.append(f'{cell.cell_id}: config is not a mapping')
            continue
        if sha256_json(raw) != cell.config_sha256:
            errors.append(f'{cell.cell_id}: resolved configuration hash is stale')
        if dataset_hash(raw) != cell.dataset_sha256:
            errors.append(f'{cell.cell_id}: dataset configuration hash is stale')
        try:
            cfg = runtime.load_config(cell)
        except Exception as exc:
            errors.append(f'{cell.cell_id}: invalid resolved configuration ({exc})')
            continue
        configs[cell.cell_id] = cfg
        composition = declared_composition(cfg)
        _compare_factor(errors, cell, 'gold_mass_vector', composition['gold_mass_vector'])
        _compare_factor(errors, cell, 'near_miss_mass', composition['near_miss_mass'])
        _compare_factor(errors, cell, 'background_mass', composition['background_mass'])
    for cell in runtime.manifest.cells:
        if cell.nested_from is None or cell.status != 'completed':
            continue
        parent = runtime.cells.get(cell.nested_from)
        if parent is None or parent.status != 'completed':
            errors.append(f'{cell.cell_id}: nested source {cell.nested_from!r} is not completed')
            continue
        cfg = configs.get(cell.cell_id)
        parent_cfg = configs.get(parent.cell_id)
        if cfg is None or parent_cfg is None:
            continue
        child_qrels = runtime.paths(cell, cfg).table_path('qrels')
        parent_qrels = runtime.paths(parent, parent_cfg).table_path('qrels')
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
            errors.append(
                f'{cell.cell_id}: nested qrels do not preserve exact pool mass '
                f'expected={expected}, examples={invalid.head(5).to_dicts()}'
            )
    return MaterializedSuiteValidation(
        errors=tuple(errors), checked_cells=len(runtime.manifest.cells)
    )


def _compare_factor(
    errors: list[str], cell: SuiteManifestCell, factor: str, actual: object
) -> None:
    if factor in cell.factors and cell.factors[factor] != actual:
        errors.append(
            f'{cell.cell_id}: declared {factor}={cell.factors[factor]!r} does not match {actual!r}'
        )
