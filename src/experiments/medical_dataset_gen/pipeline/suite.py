"""Manifest-driven orchestration for schema-v5 suite cells."""

from __future__ import annotations

import argparse

from experiments.medical_dataset_gen.pipeline.stages import PipelineStage
from experiments.medical_dataset_gen.suites.nesting import (
    project_nested_scale_parents,
    required_nested_scale_source,
    reuse_nested_scale_chunk_embeddings,
)
from experiments.medical_dataset_gen.suites.runtime import (
    SuiteRuntime,
    nested_depth,
    suite_distribution_lock,
)
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

from .cli import selected_stage_names
from .normal import run_pipeline_stages
from .standalone import StandaloneRunSpec, run_standalone_script_sequence

GENERATION_STAGES: set[PipelineStage] = {
    'plans',
    'facts',
    'chunks',
    'queries_answers',
    'qrels',
}


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
    runtime: SuiteRuntime | None = None,
) -> None:
    assert args.suite is not None and args.cell is not None
    runtime = runtime or SuiteRuntime.load(
        results_dir=MedicalDatasetGenPaths.results_dir, suite_id=args.suite
    )
    cell = runtime.cell(args.cell)
    if args.dry_run:
        stages = [] if run_specs is not None else selected_stage_names(parser, args)
        print(
            f'[pipeline] dry-run suite={args.suite} cell={cell.cell_id} '
            f'stages={stages or [spec.script for spec in run_specs or []]}'
        )
        return
    cfg = runtime.load_config(cell)
    stage_names = [] if run_specs is not None else selected_stage_names(parser, args)
    requested = set(stage_names)
    if run_specs is not None:
        requested.update(spec.script for spec in run_specs)
    if cell.origin == 'derived' and requested & GENERATION_STAGES:
        parser.error(
            'derived suite cells reuse their pinned source dataset; generation stages are forbidden'
        )
    nested_source = required_nested_scale_source(runtime, cell)
    if nested_source is not None and requested & GENERATION_STAGES:
        parser.error(
            f'{cell.cell_id} is a deterministic nested subset. Generate the terminal '
            f'large support first with --cell {nested_source.cell_id}; it will project this cell.'
        )
    with suite_distribution_lock(root=runtime.root, distribution_id=cell.distribution_id):
        paths = runtime.paths(cell, cfg)
        paths.ensure_dirs()
        reuse_embeddings = reuse_nested_scale_chunk_embeddings(
            runtime=runtime,
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
                runtime.mark_completed(cell)
            return
        run_pipeline_stages(
            cfg=cfg,
            paths=paths,
            stage_names=stage_names,
            queries_only=args.queries_only or reuse_embeddings,
        )
        projected = project_nested_scale_parents(runtime, cell, cfg)
        if projected:
            print(f'[pipeline] projected nested scale cells from {cell.cell_id}: {projected}')
        if 'eval' in requested:
            runtime.mark_completed(cell)


def run_suite_where(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    run_specs: list[StandaloneRunSpec] | None,
) -> None:
    assert args.suite is not None and args.where is not None
    runtime = SuiteRuntime.load(
        results_dir=MedicalDatasetGenPaths.results_dir,
        suite_id=args.suite,
    )
    selected = runtime.select(args.where)
    if not selected:
        parser.error(f'--where selected no cells: {args.where}')
    selected_ids = {cell.cell_id for cell in selected}
    stages = [] if run_specs is not None else selected_stage_names(parser, args)
    if GENERATION_STAGES.intersection(stages):
        for cell in selected:
            source = required_nested_scale_source(runtime, cell)
            if source is not None:
                selected_ids.add(source.cell_id)
    ordered = sorted(
        (runtime.cells[cell_id] for cell_id in selected_ids),
        key=lambda cell: (-nested_depth(cell, runtime.cells), cell.cell_id),
    )
    if args.dry_run:
        print(f'[pipeline] dry-run suite={args.suite} where={args.where} stages={stages}')
        explicitly_selected = {cell.cell_id for cell in selected}
        for cell in ordered:
            implicit = ' (nested source)' if cell.cell_id not in explicitly_selected else ''
            print(f'  {cell.cell_id}{implicit}')
        return
    for cell in ordered:
        cell_args = argparse.Namespace(**vars(args))
        cell_args.cell = cell.cell_id
        cell_args.where = None
        if run_specs is None and required_nested_scale_source(runtime, cell) is not None:
            downstream = [stage for stage in stages if stage not in GENERATION_STAGES]
            if not downstream:
                continue
            cell_args.from_stage = None
            cell_args.to_stage = None
            cell_args.stages = ','.join(downstream)
        run_suite_cell(parser=parser, args=cell_args, run_specs=run_specs, runtime=runtime)
