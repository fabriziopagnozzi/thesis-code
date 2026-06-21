from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Literal, NoReturn
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ChunkPoolScope,
    PlanCalibrationMode,
    QueryType,
)
from helpers.dir_paths import ROOT_DIR

type SyntheticMedicalTableName = Literal[
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
    'evaluation_results',
    'evaluation_stats',
    'lambda_pair_agreement',
    'embedding_geometry_points',
    'embedding_geometry_query_stats',
]


class GlobalCfg(BaseModel):
    seed: PositiveInt = 42
    n_queries: PositiveInt = 120
    conditions: PositiveInt = 4
    output_experiment: str = 'mvp'
    result_dir_overrides: dict[SyntheticMedicalTableName, str] = Field(default_factory=dict)


class GenerationCfg(BaseModel):
    ontology_path: str | None = None
    query_types: list[QueryType] = Field(default_factory=lambda: ['subgroup_comparison'])
    dominance_mode: PlanCalibrationMode = 'rotating'
    dominance_probe_chunks_per_facet: PositiveInt = 8
    calibration_min_probe_margin: float | None = Field(default=None, ge=0.0)
    gold_chunks_dominant: PositiveInt = 25
    gold_chunks_complementary: PositiveInt = 14
    distractors_per_query: PositiveInt = 30
    background_outlier_clusters_per_query: int = Field(default=1, ge=0)
    background_outlier_cluster_size: int = Field(default=12, ge=0)
    chunk_min_words: PositiveInt = 25
    chunk_max_words: PositiveInt = 90
    chunk_word_tolerance: PositiveInt = 2
    llm_name: str = 'gemma4:12b-ud_q8_xl'
    llm_workers: PositiveInt = 1
    use_llm_chunk_generation: bool = True
    use_llm_chunk_rewriting: bool = False
    use_llm_query_paraphrase: bool = False
    llm_chunk_max_attempts: PositiveInt = 3
    llm_temperature: PositiveFloat = 0.1
    llm_num_ctx: PositiveInt = 4096


class EmbeddingCfg(BaseModel):
    model_name: str = 'multi-qa-mpnet-base-cos-v1'
    batch_size: PositiveInt = 64
    device: str = 'cuda'
    devices: list[str] = Field(default_factory=list)
    query_prompt: str | None = None
    normalize: bool = True


class RetrievalCfg(BaseModel):
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


class GeometryCfg(BaseModel):
    topk_dominance_k: PositiveInt = 10
    primary_topk_dominance_k: PositiveInt = 20
    min_topk_dominant_count: PositiveInt = 5
    max_topk_retrieved_facets: PositiveInt | None = 2
    min_in_minus_cross_similarity: PositiveFloat = 0.03
    min_distractors_in_pool: PositiveInt = 10


class MethodsComparisonKernelMetricCfg(BaseModel):
    summary_metric: str = 'MeanFacetHitRate@k'
    enabled: bool = True
    weight: PositiveFloat = 1.0
    target_gain_vs_topk: float = 0.08
    gain_bandwidth: PositiveFloat = 0.01
    target_lower_bound_vs_topk: float = 0.03
    lower_bound_bandwidth: PositiveFloat = 0.0075


class MethodsComparisonKernelsCfg(BaseModel):
    lambda_max: NonNegativeFloat = 0.80
    agreement_alpha: PositiveFloat = 3.0
    kernel_floor: float = Field(default=0.05, gt=0, le=1)
    pair_aggregation: Literal['geometric_mean', 'arithmetic_mean', 'minimum'] = 'geometric_mean'
    metrics: list[MethodsComparisonKernelMetricCfg] = Field(
        default_factory=lambda: [MethodsComparisonKernelMetricCfg()]
    )


class EvaluationCfg(BaseModel):
    workers: PositiveInt | None = None
    fac_loc_mmr_comparison_kernels: MethodsComparisonKernelsCfg = Field(
        default_factory=MethodsComparisonKernelsCfg
    )


class EmbeddingGeometryCfg(BaseModel):
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


class ExperimentCfg(BaseModel):
    global_: GlobalCfg = Field(alias='global')
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    embeddings: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    geometry: GeometryCfg = Field(default_factory=GeometryCfg)
    evaluation: EvaluationCfg = Field(default_factory=EvaluationCfg)
    embedding_geometry: EmbeddingGeometryCfg = Field(default_factory=EmbeddingGeometryCfg)

    model_config = {'populate_by_name': True, 'extra': 'ignore'}


class MedicalDatasetGenPaths:
    root = ROOT_DIR / 'src' / 'experiments' / 'medical_dataset_gen'
    results_dir = root / '_results'
    default_ontology_path = root / 'data_templates' / 'medical_ontology.yaml'

    def __init__(
        self,
        exp_name: str,
        result_dir_overrides: dict[SyntheticMedicalTableName, str] | None = None,
    ):
        self.exp_name = exp_name
        self.experiment_dir = self.results_dir / exp_name
        self.logs_dir = self.experiment_dir / '_logs'
        self.figures_dir = self.experiment_dir / '_figures'
        self.config_path = self.experiment_dir / '_config.yaml'
        self.result_dir_overrides = dict(result_dir_overrides or {})
        self.embeddings_npz_path = self.experiment_dir / 'embeddings.npz'
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
        table: SyntheticMedicalTableName,
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

    def get_result_dir(self, table: SyntheticMedicalTableName) -> Path:
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


def setup_logging(paths: MedicalDatasetGenPaths) -> None:
    main = sys.modules['__main__']
    script_name = Path(main.__file__ if main.__file__ else f'unknown_script_{uuid4()}').stem
    log_path = paths.logs_dir / f'{script_name}.log'

    class _Tee(io.TextIOBase):
        def __init__(self, filepath: Path):
            self._terminal = sys.stdout
            self._file = open(filepath, 'a')  # noqa: SIM115

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
    exact_dir = results_dir / exp_name
    if exact_dir.is_dir():
        return exp_name

    matches = sorted(
        path.name
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(exp_name)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f'exp={exp_name!r} is ambiguous in {results_dir}: {matches}')

    raise FileNotFoundError(
        f'no experiment directory matching {exp_name!r} in {results_dir}. '
        'Use the full directory name or a unique prefix.'
    )
