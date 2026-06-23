from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, NoReturn, get_args
from uuid import uuid4

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ChunkPoolScope,
    DistractorStr,
    PlanCalibrationMode,
)
from helpers.dir_paths import ROOT_DIR

type SyntheticMedicalDatasetTableName = Literal[
    'query_plans',
    'query_plan_calibration',
    'clinical_facts',
    'chunk_documents',
    'chunk_memberships',
    'queries',
    'gold_answers',
    'qrels',
    'generation_rejects',
    'embeddings',
    'geometry_stats',
    'geometry_slice_stats',
    'evaluation_results',
    'evaluation_stats',
    'evaluation_slice_stats',
    'lambda_pair_agreement',
    'query_geometry_points',
    'query_geometry_stats',
]
SYNTH_MEDICAL_DATASET_TABLE_NAMES = set[str](get_args(SyntheticMedicalDatasetTableName.__value__))


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class GlobalCfg(ConfigModel):
    seed: PositiveInt = 42
    conditions: PositiveInt = 4
    output_experiment: str = 'v2'
    result_dir_overrides: dict[SyntheticMedicalDatasetTableName, str] = Field(default_factory=dict)


class BackgroundOutlierCfg(ConfigModel):
    clusters_per_query: int = Field(default=1, ge=0)
    cluster_size: int = Field(default=8, ge=0)

    def total_chunks(self) -> int:
        return self.clusters_per_query * self.cluster_size


class DistractorConfigCfg(ConfigModel):
    same_condition_wrong_subgroup: int = Field(default=9, ge=0)
    same_subgroup_wrong_condition: int = Field(default=8, ge=0)
    same_axis_wrong_condition: int = Field(default=8, ge=0)
    same_condition_wrong_axis: int = Field(default=0, ge=0)
    background_outlier: BackgroundOutlierCfg = Field(default_factory=BackgroundOutlierCfg)

    def point_distractor_counts(self) -> dict[DistractorStr, int]:
        return {
            'same_condition_wrong_subgroup': self.same_condition_wrong_subgroup,
            'same_subgroup_wrong_condition': self.same_subgroup_wrong_condition,
            'same_axis_wrong_condition': self.same_axis_wrong_condition,
            'same_condition_wrong_axis': self.same_condition_wrong_axis,
        }

    def total_point_distractors(self) -> int:
        return sum(self.point_distractor_counts().values())

    def total_chunks(self) -> int:
        return self.total_point_distractors() + self.background_outlier.total_chunks()


class ChunkPoolConfig(ConfigModel):
    gold_chunks_calibrated_primary: PositiveInt = 24
    gold_chunks_other_primary: PositiveInt = 20
    gold_chunks_secondary: PositiveInt = 14

    niche_gold_clusters_per_query: int = Field(default=0, ge=0, le=2)
    gold_chunks_niche: PositiveInt = 4


class GenerationLlmConfig(ConfigModel):
    use_llm_chunk_generation: bool = False
    use_llm_chunk_rewriting: bool = False
    use_llm_query_paraphrase: bool = False
    model_name: str = 'gemma4:12b-ud_q8_xl'
    num_workers: PositiveInt = 4
    max_attempts: PositiveInt = 3
    temperature: PositiveFloat = 0.1
    num_ctx: PositiveInt = 4096


class GenerationCfg(ConfigModel):
    query_limit: PositiveInt | None = None
    ontology_path: str | None = None

    calibration_mode: PlanCalibrationMode = 'rotating'
    calibration_probe_chunks_per_facet: PositiveInt = 8

    chunk_pool_config: ChunkPoolConfig = Field(default_factory=ChunkPoolConfig)
    distractor_config: DistractorConfigCfg = Field(default_factory=DistractorConfigCfg)

    chunk_min_words: PositiveInt = 25
    chunk_max_words: PositiveInt = 90
    chunk_word_tolerance: PositiveInt = 2

    llm_config: GenerationLlmConfig = Field(default_factory=GenerationLlmConfig)

    @model_validator(mode='after')
    def _validate_niche_cluster_size(self) -> GenerationCfg:
        if (
            self.chunk_pool_config.niche_gold_clusters_per_query
            and self.chunk_pool_config.gold_chunks_niche
            >= self.chunk_pool_config.gold_chunks_secondary
        ):
            raise ValueError(
                'generation.chunk_pool_config.gold_chunks_niche must be smaller than gold_chunks_secondary '
                'when niche gold clusters are enabled'
            )
        return self

    def total_distractor_chunks(self) -> int:
        return self.distractor_config.total_chunks()


class EmbeddingCfg(ConfigModel):
    model_name: str = 'multi-qa-mpnet-base-cos-v1'
    batch_size: PositiveInt = 64
    device: str = 'cuda'
    devices: list[str] = Field(default_factory=list)
    query_prompt: str | None = None
    normalize: bool = True


class RetrievalCfg(ConfigModel):
    pool_scope: ChunkPoolScope = 'query_local'
    candidate_pool_n: PositiveInt = 300
    k_values: list[PositiveInt] = Field(default_factory=lambda: [5, 10, 20])
    lambda_values: list[NonNegativeFloat] = Field(default_factory=lambda: [0.3, 0.5, 0.7])
    strategies: set[Literal['top_k', 'mmr', 'fac_loc']] = Field(
        default_factory=lambda: set(['top_k', 'mmr', 'fac_loc'])
    )
    mmr_window: int | None = None
    only_pass_geometry: bool = True
    compute_answer_rouge: bool = True


class GeometryFilterCfg(ConfigModel):
    topk_k: PositiveInt = 20
    min_primary_axis_count: PositiveInt = 14
    max_topk_retrieved_facets: PositiveInt | None = 2
    min_in_minus_cross_similarity: PositiveFloat = 0.03
    min_same_axis_cohort_gap: PositiveFloat = 0.03
    min_same_cohort_axis_gap: PositiveFloat = 0.03
    min_distractors_in_pool: PositiveInt = 10


class MethodsComparisonKernelMetricCfg(ConfigModel):
    summary_metric: str = 'MeanFacetHitRate@k'
    enabled: bool = True
    weight: PositiveFloat = 1.0
    target_gain_vs_topk: float = 0.08
    gain_bandwidth: PositiveFloat = 0.01
    target_lower_bound_vs_topk: float = 0.03
    lower_bound_bandwidth: PositiveFloat = 0.0075


type KernelAggregationStrategy = Literal['geometric_mean', 'arithmetic_mean', 'minimum']


class MethodsComparisonKernelsCfg(ConfigModel):
    lambda_max: NonNegativeFloat = 0.80
    agreement_alpha: PositiveFloat = 3.0
    kernel_floor: float = Field(default=0.05, gt=0, le=1)
    pair_aggregation: KernelAggregationStrategy = 'geometric_mean'
    metrics: list[MethodsComparisonKernelMetricCfg] = Field(
        default_factory=lambda: [MethodsComparisonKernelMetricCfg()]
    )


class EvaluationCfg(ConfigModel):
    workers: PositiveInt | None = None
    fac_loc_mmr_comparison_kernels: MethodsComparisonKernelsCfg = Field(
        default_factory=MethodsComparisonKernelsCfg
    )


class QueryGeometryCfg(ConfigModel):
    n_queries: PositiveInt = 6
    query_ids: list[str] = Field(default_factory=list)
    query_selection: Literal['mixed', 'best'] = 'mixed'
    candidate_pool_n: PositiveInt | None = None
    plot_k: PositiveInt = 10
    reduction: Literal['umap', 'pca'] = 'umap'
    pca_dims: PositiveInt | None = None
    umap_metric: Literal['cosine', 'euclidean'] = 'cosine'
    umap_neighbors: PositiveInt = 15
    umap_min_dist: PositiveFloat = 0.08
    hdbscan_umap_dims: PositiveInt = 10
    hdbscan_min_cluster_size: PositiveInt = 5
    hdbscan_min_samples: PositiveInt | None = None
    random_state: PositiveInt = 42


class ExperimentCfg(ConfigModel):
    dataset_schema_version: Literal[2]
    global_: GlobalCfg = Field(alias='global')
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    embeddings: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    geometry_filter: GeometryFilterCfg = Field(default_factory=GeometryFilterCfg)
    evaluation: EvaluationCfg = Field(default_factory=EvaluationCfg)
    query_geometry: QueryGeometryCfg = Field(default_factory=QueryGeometryCfg)

    model_config = ConfigDict(populate_by_name=True, extra='forbid')

    @model_validator(mode='after')
    def _validate_v2_scope(self) -> ExperimentCfg:
        if self.retrieval.pool_scope != 'query_local':
            raise ValueError('dataset schema v2 supports only retrieval.pool_scope=query_local')
        local_pool_size = (
            self.generation.chunk_pool_config.gold_chunks_calibrated_primary
            + self.generation.chunk_pool_config.gold_chunks_other_primary
            + (2 - self.generation.chunk_pool_config.niche_gold_clusters_per_query)
            * self.generation.chunk_pool_config.gold_chunks_secondary
            + self.generation.chunk_pool_config.niche_gold_clusters_per_query
            * self.generation.chunk_pool_config.gold_chunks_niche
            + self.generation.total_distractor_chunks()
        )
        if self.retrieval.candidate_pool_n < local_pool_size:
            raise ValueError(
                'retrieval.candidate_pool_n must include the complete v2 query-local pool '
                f'({local_pool_size} chunks)'
            )
        return self


class MedicalDatasetGenPaths:
    root = ROOT_DIR / 'src' / 'experiments' / 'medical_dataset_gen'
    results_dir = root / '_results'
    default_ontology_path = root / 'data_templates' / 'medical_ontology.yaml'

    def __init__(
        self,
        exp_name: str,
        result_dir_overrides: dict[SyntheticMedicalDatasetTableName, str] | None = None,
    ):
        self.exp_name = exp_name
        self.experiment_dir = self.results_dir / exp_name
        self.logs_dir = self.experiment_dir / '_logs'
        self.figures_dir = self.experiment_dir / '_figures'
        self.config_path = self.experiment_dir / '_config.yaml'
        self.result_dir_overrides = dict(result_dir_overrides or {})
        self.embeddings_chunk_vectors_path = self.experiment_dir / 'embeddings_chunk_vectors.npy'
        self.embeddings_query_vectors_path = self.experiment_dir / 'embeddings_query_vectors.npy'
        self.embeddings_chunk_ids_path = self.experiment_dir / 'embeddings_chunk_ids.npy'
        self.embeddings_query_ids_path = self.experiment_dir / 'embeddings_query_ids.npy'
        self.embeddings_meta_path = self.experiment_dir / 'embeddings_metadata.json'

    def ensure_dirs(self) -> None:
        for path in [self.results_dir, self.experiment_dir, self.logs_dir, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def table_path(
        self,
        table: SyntheticMedicalDatasetTableName,
        ext: Literal['parquet', 'json', 'jsonl', 'csv'] = 'parquet',
    ) -> Path:
        override = self.result_dir_overrides.get(table)
        if override is None:
            return self.experiment_dir / f'{table}.{ext}'

        override_path = Path(override)
        if override_path.suffix:
            if override_path.is_absolute():
                return override_path
            return self.experiment_dir / override_path

        if override_path.is_absolute():
            return override_path / f'{table}.{ext}'
        return self.results_dir / override_path / f'{table}.{ext}'

    def get_result_dir(self, table: SyntheticMedicalDatasetTableName) -> Path:
        return self.table_path(table).parent


def load_config(exp: str | None = None) -> ExperimentCfg:
    exp_name = exp or os.getenv('EXP') or os.getenv('EXP_NAME')
    if exp_name is None:
        raise ValueError(
            'missing experiment name; pass --exp or set EXP/EXP_NAME so '
            'the config can be loaded from _results/<exp>/_config.yaml'
        )

    exp_name = _resolve_experiment_name(exp_name)
    cfg_path = MedicalDatasetGenPaths(exp_name).config_path
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'missing experiment config: {cfg_path}. '
            'Create it manually before running the pipeline.'
        )

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentCfg.model_validate(raw)
    cfg.global_.output_experiment = exp_name
    return cfg


def load_config_from_cli() -> ExperimentCfg:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exp', type=str, default=os.getenv('EXP') or os.getenv('EXP_NAME'))
    (args, _) = parser.parse_known_args()
    return load_config(exp=args.exp)  # a pydantic validated model


def paths_for(cfg: ExperimentCfg) -> MedicalDatasetGenPaths:
    paths = MedicalDatasetGenPaths(
        cfg.global_.output_experiment,
        result_dir_overrides=cfg.global_.result_dir_overrides,
    )
    paths.ensure_dirs()
    return paths


def unreachable_code(err: str) -> NoReturn:
    raise RuntimeError(err)


def setup_logging(paths: MedicalDatasetGenPaths, run_id: str | None = None) -> None:
    main = sys.modules['__main__']
    script_name = Path(main.__file__ if main.__file__ else f'unknown_script_{uuid4()}').stem
    suffix = run_id or datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    log_path = paths.logs_dir / f'{script_name}_{suffix}.log'

    class _Tee(io.TextIOBase):
        def __init__(self, filepath: Path):
            self._terminal = sys.stdout
            self._file = open(filepath, 'w')  # noqa: SIM115

        def write(self, msg: str) -> int:
            self._terminal.write(msg)
            self._file.write(msg)
            return len(msg)

        def flush(self) -> None:
            self._terminal.flush()
            self._file.flush()

    sys.stdout = _Tee(log_path)


def _resolve_experiment_name(
    exp_name: str, results_dir: Path = MedicalDatasetGenPaths.results_dir
) -> str:
    dir = results_dir / exp_name
    if dir.is_dir():
        return exp_name

    matches = sorted(results_dir.glob(f'{exp_name}*'))
    if len(matches) == 1:
        return matches[0].name
    elif len(matches) > 1:
        raise RuntimeError(
            f'{exp_name!r} is an ambiguous prefix, found many matches: {[m.name for m in matches]}'
        )
    else:
        raise FileNotFoundError(f'no experiment directory prefixed {exp_name!r} in {results_dir}. ')
