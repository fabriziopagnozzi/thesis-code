from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

from experiments.medical_dataset_gen.dataset_generation.chunk_materialization import (
    run_make_chunks,
)
from experiments.medical_dataset_gen.dataset_generation.facts import run_make_facts
from experiments.medical_dataset_gen.dataset_generation.planning import run_make_query_plans
from experiments.medical_dataset_gen.dataset_generation.qrels import run_make_qrels
from experiments.medical_dataset_gen.dataset_generation.queries_answers import (
    run_make_queries_answers,
)
from experiments.medical_dataset_gen.embedding.artifacts import embedding_artifacts_ready
from experiments.medical_dataset_gen.embedding.stage import run_embed
from experiments.medical_dataset_gen.evaluation.plot_stage import (
    parse_plots_cli_args,
    run_eval_plots,
)
from experiments.medical_dataset_gen.evaluation.stage import (
    parse_evaluate_cli_args,
    run_evaluate,
)
from experiments.medical_dataset_gen.query_geometry.filtering import run_filter_queries
from experiments.medical_dataset_gen.query_geometry.plot_stage import (
    parse_geom_plots_cli_args,
    run_query_geom_plots,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    SharedGenerationTableName,
    get_literals,
)

type PipelineStage = Literal[
    'plans',
    'facts',
    'chunks',
    'queries_answers',
    'qrels',
    'embed',
    'filter_queries',
    'eval',
    'eval_plots',
    'geom_plots',
]
PIPELINE_STAGE_NAMES = list[PipelineStage](get_literals(PipelineStage))
PIPELINE_STAGE_SET = set[PipelineStage](PIPELINE_STAGE_NAMES)
type PipelineStageFn = Callable[[ExperimentCfg, MedicalDatasetGenPaths], object]
type StageReadyFn = Callable[[MedicalDatasetGenPaths], bool]


@dataclass(frozen=True)
class StageSpec:
    name: PipelineStage
    run: PipelineStageFn
    shared_outputs: tuple[SharedGenerationTableName, ...] = ()
    ready: StageReadyFn | None = None


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec('plans', run_make_query_plans, shared_outputs=('query_plans',)),
    StageSpec('facts', run_make_facts, shared_outputs=('clinical_facts',)),
    StageSpec('chunks', run_make_chunks, shared_outputs=('chunk_documents', 'chunk_memberships')),
    StageSpec(
        'queries_answers', run_make_queries_answers, shared_outputs=('queries', 'gold_answers')
    ),
    StageSpec('qrels', run_make_qrels, shared_outputs=('qrels',)),
    StageSpec('embed', run_embed, ready=embedding_artifacts_ready),
    StageSpec('filter_queries', run_filter_queries),
    StageSpec('eval', run_evaluate),
    StageSpec('eval_plots', run_eval_plots),
    StageSpec('geom_plots', run_query_geom_plots),
)
STAGE_BY_NAME = {spec.name: spec for spec in STAGE_SPECS}

type StandalonePipelineScript = Literal['eval', 'geom_plots', 'eval_plots']
STANDALONE_SCRIPT_NAMES = list[StandalonePipelineScript](get_literals(StandalonePipelineScript))
STANDALONE_SCRIPT_SET = set[StandalonePipelineScript](STANDALONE_SCRIPT_NAMES)
type StandaloneRunner = Callable[[ExperimentCfg, MedicalDatasetGenPaths, list[str]], object]


@dataclass(frozen=True)
class StandaloneScriptSpec:
    name: StandalonePipelineScript
    run: StandaloneRunner


def _run_evaluate_with_args(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    argv: list[str],
) -> object:
    return run_evaluate(cfg, paths, selected_steps=parse_evaluate_cli_args(argv))


def _run_eval_plots_with_args(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    argv: list[str],
) -> object:
    return run_eval_plots(cfg, paths, selected_plots=parse_plots_cli_args(argv))


def _run_geom_plots_with_args(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    argv: list[str],
) -> object:
    return run_query_geom_plots(cfg, paths, selected_plots=parse_geom_plots_cli_args(argv))


STANDALONE_SCRIPT_SPECS: tuple[StandaloneScriptSpec, ...] = (
    StandaloneScriptSpec('eval', _run_evaluate_with_args),
    StandaloneScriptSpec('geom_plots', _run_geom_plots_with_args),
    StandaloneScriptSpec('eval_plots', _run_eval_plots_with_args),
)
STANDALONE_SCRIPT_BY_NAME = {spec.name: spec for spec in STANDALONE_SCRIPT_SPECS}


def stage_index(name: PipelineStage) -> int:
    for index, spec in enumerate(STAGE_SPECS):
        if spec.name == name:
            return index
    raise KeyError(name)


def pipeline_stage(value: str) -> PipelineStage:
    if value not in PIPELINE_STAGE_SET:
        raise KeyError(value)
    return cast(PipelineStage, value)


def standalone_script(value: str) -> StandalonePipelineScript:
    if value not in STANDALONE_SCRIPT_SET:
        raise KeyError(value)
    return cast(StandalonePipelineScript, value)
