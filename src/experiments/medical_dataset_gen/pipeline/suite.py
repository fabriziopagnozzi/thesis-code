"""Materialized-suite orchestration for the pipeline CLI."""

from __future__ import annotations

import argparse
import fcntl
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO

from experiments.medical_dataset_gen.dataset_generation.caches import chunk_embedding_signature
from experiments.medical_dataset_gen.embedding.artifacts import embedding_artifacts_ready
from experiments.medical_dataset_gen.pipeline.stages import PipelineStage
from experiments.medical_dataset_gen.suites.core import load_suite_manifest
from experiments.medical_dataset_gen.suites.runtime import (
    load_cell_config,
    project_nested_scale_parents,
    register_completed_evaluation_attempt,
    required_nested_scale_source,
    resolve_manifest_cell,
    suite_paths_for_cell,
    verify_cell_artifacts,
    write_attempt_metadata,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

from .cli import selected_stage_names, with_dataset_schema_version
from .normal import run_pipeline_stages
from .standalone import StandaloneRunSpec, run_standalone_script_sequence


def run_suite_mode(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    run_specs: list[StandaloneRunSpec] | None,
) -> None:
    if args.where is not None:
        run_suite_where(parser=parser, args=args, run_specs=run_specs)
    else:
        run_suite_cell(parser=parser, args=args, run_specs=run_specs)


def run_suite_cell(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    run_specs: list[StandaloneRunSpec] | None,
) -> None:
    assert args.suite is not None and args.cell is not None
    root, cell = resolve_manifest_cell(
        results_dir=MedicalDatasetGenPaths.results_dir,
        suite_id=args.suite,
        cell_id=args.cell,
    )
    if args.dry_run:
        stages = [] if run_specs is not None else selected_stage_names(parser, args)
        print(
            f'[pipeline] dry-run suite={args.suite} cell={cell.cell_id} '
            f'stages={stages or [spec.script for spec in run_specs or []]}'
        )
        return
    cfg = with_dataset_schema_version(load_cell_config(root, cell), args.version)
    stage_names = [] if run_specs is not None else selected_stage_names(parser, args)
    requested = set(stage_names)
    if run_specs is not None:
        requested.update(spec.script for spec in run_specs)

    generation_stages = {'plans', 'facts', 'chunks', 'queries_answers', 'qrels'}
    if cell.origin in {'migrated_v4', 'derived'} and requested & generation_stages:
        parser.error(
            'source-backed and migrated suite cells are immutable datasets; generation stages are forbidden. '
            'Use --stages embed,filter_queries,eval or --run eval with a new --attempt.'
        )
    if cell.origin == 'migrated_v4' and args.attempt is None:
        parser.error(
            'migrated v4 suite cells require --attempt for any pipeline action so '
            'migrated-v4 is never overwritten.'
        )
    nested_source = required_nested_scale_source(root=root, cell=cell)
    if nested_source is not None and requested & generation_stages:
        parser.error(
            f'{cell.cell_id} is a deterministic nested subset. Generate the terminal '
            f'large support first with --cell {nested_source.cell_id}; it will project this '
            'cell without regenerating evidence or chunks.'
        )
    with suite_distribution_lock(root=root, distribution_id=cell.distribution_id):
        if args.attempt is not None:
            paths = suite_paths_for_cell(
                root=root,
                cell=cell,
                cfg=cfg,
                attempt_id=args.attempt,
                create_attempt=True,
            )
            write_attempt_metadata(paths=paths, root=root, cell=cell, attempt_id=args.attempt)
        else:
            paths = suite_paths_for_cell(root=root, cell=cell, cfg=cfg)
        errors = verify_cell_artifacts(root, cell)
        if errors:
            parser.error('suite artifact validation failed:\n' + '\n'.join(errors[:20]))
        paths.ensure_dirs()
        reuse_nested_chunk_embeddings = reuse_nested_scale_chunk_embeddings(
            root=root,
            cell=cell,
            cfg=cfg,
            paths=paths,
            requested=requested,
        )
        if run_specs is not None:
            run_standalone_script_sequence(
                run_specs=run_specs,
                cfg=cfg,
                paths=paths,
                no_log_tee=args.no_log_tee,
            )
            if 'eval' in requested:
                register_completed_evaluation_attempt(root=root, cell=cell, attempt_id=args.attempt)
            return
        run_pipeline_stages(
            cfg=cfg,
            paths=paths,
            stage_names=stage_names,
            queries_only=args.queries_only or reuse_nested_chunk_embeddings,
        )
        projected = project_nested_scale_parents(root=root, source_cell=cell, source_cfg=cfg)
        if projected:
            print(f'[pipeline] projected nested scale cells from {cell.cell_id}: {projected}')
        if 'eval' in requested:
            register_completed_evaluation_attempt(root=root, cell=cell, attempt_id=args.attempt)


class SuiteDistributionLock:
    """A non-blocking lock around a distribution's shared v5 data area."""

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
                f'{self.path.stem.removesuffix(".lock")}: suite distribution is already running; '
                'wait for the active pipeline or use a different distribution'
            ) from exc
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def suite_distribution_lock(*, root: Path, distribution_id: str) -> SuiteDistributionLock:
    return SuiteDistributionLock(root=root, distribution_id=distribution_id)


def reuse_nested_scale_chunk_embeddings(
    *,
    root: Path,
    cell: object,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    requested: set[PipelineStage],
) -> bool:
    """Hard-link verified source vectors for nested-scale evaluation cells.

    Nested scales project memberships/qrels but deliberately share the same
    immutable chunk document surface.  Re-embedding it for each 64/128 subset
    would be both redundant and prone to cache contention.  Query vectors are
    still materialized per cell because wording and query IDs belong to the
    run profile.
    """
    from experiments.medical_dataset_gen.suites.core import SuiteManifestCell

    if 'embed' not in requested or not isinstance(cell, SuiteManifestCell):
        return False
    source = required_nested_scale_source(root=root, cell=cell)
    if source is None or source.cell_id == cell.cell_id:
        return False
    source_cfg = load_cell_config(root, source)
    if chunk_embedding_signature(source_cfg) != chunk_embedding_signature(cfg):
        return False
    source_paths = suite_paths_for_cell(root=root, cell=source, cfg=source_cfg)
    if not embedding_artifacts_ready(source_paths):
        return False
    if not os.path.samefile(
        source_paths.table_path('chunk_documents'), paths.table_path('chunk_documents')
    ):
        raise RuntimeError(
            f'{cell.cell_id}: nested scale does not share its source chunk documents; '
            'refusing to reuse source embedding vectors'
        )
    for artifact in ('chunk_vectors', 'chunk_ids'):
        source_path = source_paths.embeddings_paths(artifact)
        target_path = paths.embeddings_paths(artifact)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            if not os.path.samefile(source_path, target_path):
                raise RuntimeError(
                    f'{cell.cell_id}: existing {artifact} is not the verified nested source; '
                    'create a new evaluation attempt rather than overwriting it'
                )
            continue
        if source_path.stat().st_dev != target_path.parent.stat().st_dev:
            raise RuntimeError(
                f'{cell.cell_id}: nested embedding reuse requires one filesystem: '
                f'{source_path} -> {target_path}'
            )
        os.link(source_path, target_path)
    print(f'[pipeline] reusing verified chunk embeddings from {source.cell_id}')
    return True


def run_suite_where(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    run_specs: list[StandaloneRunSpec] | None,
) -> None:
    """Run a manifest-selected batch in dependency order.

    Selection is manifest metadata only.  A nested scale request that includes
    generation implicitly adds its terminal large support, then executes it
    first so smaller cells are projected rather than independently generated.
    """
    assert args.suite is not None and args.where is not None
    manifest = load_suite_manifest(MedicalDatasetGenPaths.results_dir, args.suite)
    selected = [cell for cell in manifest.cells if suite_where_matches(cell, args.where)]
    if not selected:
        parser.error(f'--where selected no cells: {args.where}')
    root = MedicalDatasetGenPaths.results_dir / 'v5' / 'suites' / args.suite
    by_id = {cell.cell_id: cell for cell in manifest.cells}
    selected_ids = {cell.cell_id for cell in selected}
    stages = [] if run_specs is not None else selected_stage_names(parser, args)
    generation_stages: set[PipelineStage] = {
        'plans',
        'facts',
        'chunks',
        'queries_answers',
        'qrels',
    }
    if generation_stages.intersection(stages):
        for cell in list(selected):
            source = required_nested_scale_source(root=root, cell=cell)
            if source is not None:
                selected_ids.add(source.cell_id)
    ordered = sorted(
        (by_id[cell_id] for cell_id in selected_ids),
        key=lambda cell: (-nested_depth(cell, by_id), cell.cell_id),
    )
    if args.dry_run:
        print(f'[pipeline] dry-run suite={args.suite} where={args.where} stages={stages}')
        for cell in ordered:
            implicit = (
                ' (nested source)'
                if cell.cell_id not in {item.cell_id for item in selected}
                else ''
            )
            print(f'  {cell.cell_id}{implicit}')
        return
    for cell in ordered:
        cell_args = argparse.Namespace(**vars(args))
        cell_args.cell = cell.cell_id
        cell_args.where = None
        if run_specs is None and required_nested_scale_source(root=root, cell=cell) is not None:
            # The terminal large cell above has already produced and projected
            # this cell's generation artifacts.  Running its normal stage
            # selection would ask the nested-cell guard to regenerate plans,
            # facts, chunks, queries, or qrels.  Retain only downstream work
            # so that chunk vectors can be hard-linked and query vectors plus
            # evaluation are materialized in the immutable target cell.
            downstream_stages = [
                stage
                for stage in stages
                if stage not in {'plans', 'facts', 'chunks', 'queries_answers', 'qrels'}
            ]
            if not downstream_stages:
                continue
            cell_args.from_stage = None
            cell_args.to_stage = None
            cell_args.stages = ','.join(downstream_stages)
        run_suite_cell(parser=parser, args=cell_args, run_specs=run_specs)


def suite_where_matches(cell: object, raw_where: str) -> bool:
    from experiments.medical_dataset_gen.suites.core import SuiteManifestCell

    if not isinstance(cell, SuiteManifestCell):
        return False
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


def nested_depth(cell: object, by_id: Mapping[str, object]) -> int:
    from experiments.medical_dataset_gen.suites.core import SuiteManifestCell

    if not isinstance(cell, SuiteManifestCell):
        return 0
    depth = 0
    current = cell
    while current.nested_from is not None:
        parent = by_id.get(current.nested_from)
        if not isinstance(parent, SuiteManifestCell):
            break
        depth += 1
        current = parent
    return depth
