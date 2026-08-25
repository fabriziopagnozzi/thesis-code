"""Normal experiment-mode orchestration for the pipeline CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys

from experiments.medical_dataset_gen.embedding.stage import run_embed
from experiments.medical_dataset_gen.pipeline.stages import (
    STAGE_BY_NAME,
    PipelineStage,
    StageSpec,
)
from experiments.medical_dataset_gen.utils.exp_naming import child_experiment_names
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config,
    paths_for,
)
from experiments.medical_dataset_gen.utils.logging_utils import colorprint

from .cli import selected_stage_names, with_dataset_schema_version
from .standalone import StandaloneRunSpec, run_standalone_script_sequence


def run_normal_mode(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    run_specs: list[StandaloneRunSpec] | None,
) -> None:
    assert args.exp is not None
    children = [] if args.parent else child_experiment_names(args.exp)
    if children:
        run_child_experiments(children=children, parent_exp=args.exp)
        return

    # Resolve the effective configuration before constructing any artifact path.
    cfg = with_dataset_schema_version(load_config(args.exp), args.version)
    paths = paths_for(cfg)

    if run_specs is not None:
        run_standalone_script_sequence(
            run_specs=run_specs,
            cfg=cfg,
            paths=paths,
            no_log_tee=args.no_log_tee,
        )
        return

    run_pipeline_stages(
        cfg=cfg,
        paths=paths,
        stage_names=selected_stage_names(parser, args),
        queries_only=args.queries_only,
    )


def run_child_experiments(*, children: list[str], parent_exp: str) -> None:
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


def run_pipeline_stages(
    *,
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    stage_names: list[PipelineStage],
    queries_only: bool = False,
) -> None:
    colorprint('bright_blue', f'\n[pipeline] running experiment: {paths.exp_name}')

    for stage_name in stage_names:
        spec = STAGE_BY_NAME[stage_name]

        if should_skip_shared_stage(cfg, paths, spec):
            continue

        if spec.ready is not None and spec.ready(paths):
            print(f'[pipeline] skipping {stage_name}; artifacts already exist')
            continue

        colorprint('bright_green', f'\n{"=" * 3} Stage: {stage_name} {"=" * 3}')
        if stage_name == 'embed':
            run_embed(cfg, paths, queries_only=queries_only)
        else:
            spec.run(cfg, paths)


def should_skip_shared_stage(
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
