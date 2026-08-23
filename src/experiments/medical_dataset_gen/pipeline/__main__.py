"""Command-line orchestration for the synthetic medical benchmark pipeline."""

from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

from experiments.medical_dataset_gen.dataset_generation.caches import chunk_embedding_signature
from experiments.medical_dataset_gen.embedding.artifacts import embedding_artifacts_ready
from experiments.medical_dataset_gen.embedding.stage import run_embed
from experiments.medical_dataset_gen.pipeline.stages import (
    PIPELINE_STAGE_SET,
    STAGE_BY_NAME,
    STAGE_SPECS,
    STANDALONE_SCRIPT_BY_NAME,
    STANDALONE_SCRIPT_SET,
    PipelineStage,
    StageSpec,
    StandalonePipelineScript,
    pipeline_stage,
    stage_index,
    standalone_script,
)
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
from experiments.medical_dataset_gen.utils.cli_parsing import parse_comma_separated_names
from experiments.medical_dataset_gen.utils.exp_naming import child_experiment_names
from experiments.medical_dataset_gen.utils.global_schemas import (
    DATASET_SCHEMA_VERSION_LIST,
    DatasetSchemaVersion,
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config,
    paths_for,
)
from experiments.medical_dataset_gen.utils.logging_utils import colorprint, setup_logging


@dataclass(frozen=True)
class StandaloneRunSpec:
    script: StandalonePipelineScript
    script_args: list[str]


def main() -> None:
    parser = _build_parser()
    args, unknown_args = parser.parse_known_args()
    if args.results_dir is not None:
        MedicalDatasetGenPaths.results_dir = args.results_dir.expanduser().resolve()

    if args.exp is None and args.suite is None:
        parser.error('missing experiment; pass --exp or --suite with --cell/--where')
    if args.exp is not None and args.suite is not None:
        parser.error('--exp and --suite are mutually exclusive')
    if args.suite is not None and (args.cell is None) == (args.where is None):
        parser.error('--suite requires exactly one of --cell or --where')

    # Standalone scripts own their secondary CLI arguments; the orchestrator
    # validates the outer command before forwarding the quoted arguments.
    run_specs = _parse_run_specs(parser, args.run) if args.run is not None else None
    if run_specs is not None:
        _validate_run_mode_args(parser, args, unknown_args)
    elif unknown_args:
        parser.error('unknown argument(s): ' + ' '.join(unknown_args))

    if args.suite is not None:
        if args.where is not None:
            _run_suite_where(parser=parser, args=args, run_specs=run_specs)
        else:
            _run_suite_cell(parser=parser, args=args, run_specs=run_specs)
        return

    assert args.exp is not None
    children = [] if args.parent else child_experiment_names(args.exp)
    if children:
        _run_child_experiments(children=children, parent_exp=args.exp)
        return

    # Resolve the effective configuration before constructing any artifact path.
    cfg = _with_dataset_schema_version(load_config(args.exp), args.version)
    paths = paths_for(cfg)

    if run_specs is not None:
        _run_standalone_script_sequence(
            run_specs=run_specs,
            cfg=cfg,
            paths=paths,
            no_log_tee=args.no_log_tee,
        )
        return

    stages_to_run = _selected_stage_names(parser, args)
    _run_pipeline_stages(
        cfg=cfg,
        paths=paths,
        stage_names=stages_to_run,
        queries_only=args.queries_only,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='Override the results root for this invocation; useful for an isolated suite.',
    )
    parser.add_argument(
        '--suite',
        type=str,
        default=None,
        help='Materialized v5 suite ID. Requires --cell and bypasses legacy directory discovery.',
    )
    parser.add_argument(
        '--cell', type=str, default=None, help='Explicit materialized suite cell ID.'
    )
    parser.add_argument(
        '--where',
        type=str,
        default=None,
        help='Batch-select materialized suite cells, for example tag=smoke|core or analysis_tier=scale.',
    )
    parser.add_argument(
        '--attempt',
        type=str,
        default=None,
        help='New immutable evaluation attempt ID for a suite cell.',
    )
    parser.add_argument(
        '--version',
        choices=[f'v{version}' for version in DATASET_SCHEMA_VERSION_LIST],
        default=None,
        help=(
            'Override dataset_schema_version from the resolved experiment config for this run, '
            'for example --version v4.'
        ),
    )

    # Normal pipeline selection.
    parser.add_argument('--from', dest='from_stage', choices=PIPELINE_STAGE_SET, default=None)
    parser.add_argument('--to', dest='to_stage', choices=PIPELINE_STAGE_SET, default=None)
    parser.add_argument('--stages', default=None)
    parser.add_argument(
        '--exclude',
        action='append',
        nargs='+',
        choices=PIPELINE_STAGE_SET,
        default=None,
        help='Exclude one or more stages from a normal pipeline run. May be repeated.',
    )

    # Secondary stage entrypoints and execution controls.
    parser.add_argument(
        '--run',
        action='append',
        default=None,
        help=(
            'Run a standalone pipeline script through the orchestrator entrypoint. '
            'May be repeated. Format: --run "eval --steps evaluation_stats".'
        ),
    )
    parser.add_argument('--no-log-tee', action='store_true')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Resolve suite selection and nested dependencies without writing or running stages.',
    )
    parser.add_argument(
        '--queries-only',
        action='store_true',
        help=(
            'At the embed stage, require and reuse existing chunk embeddings and write only '
            'query IDs and query vectors.'
        ),
    )
    parser.add_argument(
        '--parent',
        action='store_true',
        help='Run the parent experiment itself even if child subexperiments exist.',
    )
    return parser


def _with_dataset_schema_version(
    cfg: ExperimentCfg,
    raw_version: str | None,
) -> ExperimentCfg:
    if raw_version is None:
        return cfg

    version = cast(DatasetSchemaVersion, int(raw_version.removeprefix('v')))
    if cfg.dataset_schema_version != version:
        print(
            '[pipeline] overriding dataset_schema_version: '
            f'v{cfg.dataset_schema_version} -> v{version}'
        )
    raw_cfg = cfg.model_dump(mode='python', by_alias=True)
    raw_cfg['dataset_schema_version'] = version
    return ExperimentCfg.model_validate(raw_cfg)


def _run_suite_cell(
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
        stages = [] if run_specs is not None else _selected_stage_names(parser, args)
        print(
            f'[pipeline] dry-run suite={args.suite} cell={cell.cell_id} '
            f'stages={stages or [spec.script for spec in run_specs or []]}'
        )
        return
    cfg = _with_dataset_schema_version(load_cell_config(root, cell), args.version)
    stage_names = [] if run_specs is not None else _selected_stage_names(parser, args)
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
    with _suite_distribution_lock(root=root, distribution_id=cell.distribution_id):
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
        reuse_nested_chunk_embeddings = _reuse_nested_scale_chunk_embeddings(
            root=root,
            cell=cell,
            cfg=cfg,
            paths=paths,
            requested=requested,
        )
        if run_specs is not None:
            _run_standalone_script_sequence(
                run_specs=run_specs,
                cfg=cfg,
                paths=paths,
                no_log_tee=args.no_log_tee,
            )
            if 'eval' in requested:
                register_completed_evaluation_attempt(root=root, cell=cell, attempt_id=args.attempt)
            return
        _run_pipeline_stages(
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


class _SuiteDistributionLock:
    """A non-blocking lock around a distribution's shared v5 data area."""

    def __init__(self, *, root: Path, distribution_id: str) -> None:
        self.path = root / '.locks' / f'{distribution_id}.lock'
        self.handle: IO[str] | None = None

    def __enter__(self) -> _SuiteDistributionLock:
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


def _suite_distribution_lock(*, root: Path, distribution_id: str) -> _SuiteDistributionLock:
    return _SuiteDistributionLock(root=root, distribution_id=distribution_id)


def _reuse_nested_scale_chunk_embeddings(
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


def _run_suite_where(
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
    selected = [cell for cell in manifest.cells if _suite_where_matches(cell, args.where)]
    if not selected:
        parser.error(f'--where selected no cells: {args.where}')
    root = MedicalDatasetGenPaths.results_dir / 'v5' / 'suites' / args.suite
    by_id = {cell.cell_id: cell for cell in manifest.cells}
    selected_ids = {cell.cell_id for cell in selected}
    stages = [] if run_specs is not None else _selected_stage_names(parser, args)
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
        key=lambda cell: (-_nested_depth(cell, by_id), cell.cell_id),
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
        _run_suite_cell(parser=parser, args=cell_args, run_specs=run_specs)


def _suite_where_matches(cell: object, raw_where: str) -> bool:
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


def _nested_depth(cell: object, by_id: Mapping[str, object]) -> int:
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


def _validate_run_mode_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    unknown_args: list[str],
) -> None:
    if args.from_stage or args.to_stage or args.stages or args.exclude:
        parser.error('--run cannot be combined with --from, --to, --stages, or --exclude')
    if unknown_args:
        parser.error(
            'unknown argument(s) outside --run: '
            + ' '.join(unknown_args)
            + '. Put standalone script arguments inside the quoted --run value.'
        )


def _run_child_experiments(*, children: list[str], parent_exp: str) -> None:
    print(f'[pipeline] experiment={parent_exp} has children: {children}')
    for child_exp in children:
        print(f'\n[pipeline] launching child experiment: {child_exp}')
        subprocess.run(
            [
                sys.executable,
                '-m',
                'experiments.medical_dataset_gen.pipeline',
                *sys.argv[1:],
                '--exp',
                child_exp,
            ],
            check=True,
        )


def _parse_run_specs(
    parser: argparse.ArgumentParser,
    raw_runs: list[str],
) -> list[StandaloneRunSpec]:
    run_specs: list[StandaloneRunSpec] = []
    for raw_run in raw_runs:
        parts = shlex.split(raw_run)
        if not parts:
            parser.error('--run value cannot be empty')

        script_name = parts[0]
        if script_name not in STANDALONE_SCRIPT_SET:
            parser.error(
                f'unknown standalone script in --run: {script_name}. '
                + 'Valid scripts: '
                + ', '.join(sorted(STANDALONE_SCRIPT_SET))
            )
        run_specs.append(
            StandaloneRunSpec(
                script=standalone_script(script_name),
                script_args=parts[1:],
            )
        )
    return run_specs


def _run_standalone_script_sequence(
    *,
    run_specs: list[StandaloneRunSpec],
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    no_log_tee: bool,
) -> None:
    if not no_log_tee:
        setup_logging(paths)

    colorprint(
        'bright_blue',
        f'[pipeline] running standalone scripts: {[spec.script for spec in run_specs]}',
    )
    print(f'[pipeline] experiment={paths.exp_name} dir={paths.experiment_dir}')

    for run_spec in run_specs:
        script_spec = STANDALONE_SCRIPT_BY_NAME[run_spec.script]

        print(f'\n=== Script: {run_spec.script} ===')
        colorprint(
            'bright_blue',
            f'[pipeline] running standalone script: {run_spec.script} experiment={paths.exp_name}',
        )
        script_spec.run(cfg, paths, run_spec.script_args)


def _selected_stage_names(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[PipelineStage]:
    if args.stages is not None:
        if args.from_stage or args.to_stage:
            parser.error('--stages cannot be combined with --from or --to')
        try:
            parsed = parse_comma_separated_names(
                raw_value=args.stages,
                valid_names=PIPELINE_STAGE_SET,
                option_name='--stages',
            )
        except ValueError as exc:
            parser.error(str(exc))
        selected = [pipeline_stage(name) for name in parsed or []]
    else:
        start_idx = stage_index(pipeline_stage(args.from_stage)) if args.from_stage else 0
        stop_idx = (
            stage_index(pipeline_stage(args.to_stage)) if args.to_stage else len(STAGE_SPECS) - 1
        )
        if stop_idx < start_idx:
            parser.error('--to must be the same as or later than --from')
        selected = [spec.name for spec in STAGE_SPECS[start_idx : stop_idx + 1]]

    excluded = {pipeline_stage(stage) for group in args.exclude or [] for stage in group}
    selected_set = set(selected) - excluded

    return [spec.name for spec in STAGE_SPECS if spec.name in selected_set]


def _run_pipeline_stages(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    stage_names: list[PipelineStage],
    queries_only: bool = False,
) -> None:
    colorprint('bright_blue', f'\n[pipeline] running experiment: {paths.exp_name}')

    for stage_name in stage_names:
        spec = STAGE_BY_NAME[stage_name]

        if _should_skip_shared_stage(cfg, paths, spec):
            continue

        if spec.ready is not None and spec.ready(paths):
            print(f'[pipeline] skipping {stage_name}; artifacts already exist')
            continue

        colorprint('bright_green', f'\n{"=" * 3} Stage: {stage_name} {"=" * 3}')
        if stage_name == 'embed':
            run_embed(cfg, paths, queries_only=queries_only)
        else:
            spec.run(cfg, paths)


def _should_skip_shared_stage(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    spec: StageSpec,
) -> bool:
    if not spec.shared_outputs or not paths.uses_shared_generation():
        return False

    required_outputs = spec.shared_outputs
    if spec.name == 'queries_answers' and not cfg.retrieval.compute_answer_rouge:
        # Retrieval-only runs consume query text but never load answer references.
        required_outputs = ('queries',)

    missing = [table for table in required_outputs if not paths.table_path(table).exists()]
    if missing:
        return False

    print(
        f'[pipeline] skipping shared stage {spec.name}; existing outputs in '
        f'{paths.table_path(required_outputs[0]).parent}'
    )
    return True


if __name__ == '__main__':
    main()
