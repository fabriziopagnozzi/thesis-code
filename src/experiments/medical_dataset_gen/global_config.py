from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, NoReturn, get_args
from uuid import uuid4

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    AxisPairConditionOverride,
    ChunkPoolScope,
    ClinicalAxis,
    ConditionKey,
    PlanCalibrationMode,
)
from experiments.medical_dataset_gen.schemas.metrics_schemas import METRIC_NAME_TO_FIELD
from helpers.dir_paths import ROOT_DIR
from helpers.embedder import EmbeddingModelName


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class GlobalCfg(ConfigModel):
    seed: PositiveInt = 42
    conditions: PositiveInt = 4
    output_experiment: str = 'v2'
    result_dir_overrides: dict[SyntheticMedicalDatasetTableName, str] = Field(default_factory=dict)


type DistractorChange = Literal['condition', 'subgroup', 'axis', 'axis_value_bin']


class DistractorSpec(ConfigModel):
    size: int = Field(ge=1)
    changes: list[DistractorChange] = Field(min_length=1)

    @model_validator(mode='before')
    @classmethod
    def _normalize_legacy_same_different_spec(cls, data: object) -> object:
        if not isinstance(data, dict) or 'changes' in data:
            return data
        if not {'condition', 'subgroup', 'axis_config'} <= set(data):
            return data

        axis_config = data['axis_config']
        if not isinstance(axis_config, dict):
            return data

        condition_value = data['condition']
        subgroup_value = data['subgroup']
        axis_value = axis_config.get('axis')
        value_bin = axis_config.get('value_bin')
        allowed_values = {'same', 'different'}
        if condition_value not in allowed_values:
            raise ValueError('condition must be "same" or "different"')
        if subgroup_value not in allowed_values:
            raise ValueError('subgroup must be "same" or "different"')
        if axis_value not in allowed_values:
            raise ValueError('axis_config.axis must be "same" or "different"')
        if value_bin is not None and value_bin not in allowed_values:
            raise ValueError('axis_config.value_bin must be "same" or "different"')

        normalized = dict(data)
        normalized.pop('condition')
        normalized.pop('subgroup')
        normalized.pop('axis_config')
        changes: list[DistractorChange] = []
        if condition_value == 'different':
            changes.append('condition')
        if subgroup_value == 'different':
            changes.append('subgroup')
        if axis_value == 'different':
            if value_bin is not None:
                raise ValueError(
                    'axis_config.value_bin is only allowed when axis_config.axis="same"'
                )
            changes.append('axis')
        elif value_bin == 'different':
            changes.append('axis_value_bin')
        normalized['changes'] = changes
        return normalized

    @model_validator(mode='after')
    def _validate_changes(self) -> DistractorSpec:
        if len(self.changes) != len(set(self.changes)):
            raise ValueError('DistractorSpec.changes must not contain duplicates')
        if 'axis' in self.changes and 'axis_value_bin' in self.changes:
            raise ValueError('DistractorSpec.changes cannot include both axis and axis_value_bin')
        if 'axis_value_bin' in self.changes and not {'condition', 'subgroup'} & set(self.changes):
            raise ValueError(
                'DistractorSpec.changes=axis_value_bin requires condition or subgroup to change too'
            )
        return self

    def changes_condition(self) -> bool:
        return 'condition' in self.changes

    def changes_subgroup(self) -> bool:
        return 'subgroup' in self.changes

    def changes_axis(self) -> bool:
        return 'axis' in self.changes

    def changes_axis_value_bin(self) -> bool:
        return 'axis_value_bin' in self.changes


class BackgroundDistractorSpec(DistractorSpec):
    size: int = Field(default=8, ge=1)
    num_clusters: int = Field(default=1, ge=1)
    changes: list[DistractorChange] = Field(
        default_factory=lambda: ['condition', 'subgroup', 'axis'],
        min_length=1,
    )


class LocalChunkPoolCfg(ConfigModel):
    size: PositiveInt
    distractors: list[DistractorSpec] = Field(default_factory=list)

    def total_distractor_chunks(self) -> int:
        return sum(spec.size for spec in self.distractors)


class NicheChunkPoolCfg(LocalChunkPoolCfg):
    num_clusters_per_query: int = Field(default=0, ge=0, le=2)


class ChunkPoolsCfg(ConfigModel):
    chunk_min_words: PositiveInt = 25
    chunk_max_words: PositiveInt = 90
    chunk_word_tolerance: PositiveInt = 2

    primary_calibrated: LocalChunkPoolCfg = Field(
        default_factory=lambda: LocalChunkPoolCfg(
            size=24,
            distractors=[
                DistractorSpec(size=3, changes=['subgroup']),
                DistractorSpec(size=2, changes=['condition']),
                DistractorSpec(size=2, changes=['condition', 'subgroup']),
                DistractorSpec(size=1, changes=['subgroup', 'axis']),
            ],
        )
    )
    other_primary: LocalChunkPoolCfg = Field(
        default_factory=lambda: LocalChunkPoolCfg(
            size=20,
            distractors=[
                DistractorSpec(size=2, changes=['subgroup']),
                DistractorSpec(size=2, changes=['condition']),
                DistractorSpec(size=2, changes=['condition', 'subgroup']),
            ],
        )
    )
    secondary: LocalChunkPoolCfg = Field(
        default_factory=lambda: LocalChunkPoolCfg(
            size=14,
            distractors=[
                DistractorSpec(size=2, changes=['subgroup']),
                DistractorSpec(size=2, changes=['condition']),
                DistractorSpec(size=2, changes=['condition', 'subgroup']),
            ],
        )
    )
    niche: NicheChunkPoolCfg = Field(
        default_factory=lambda: NicheChunkPoolCfg(
            size=4,
            num_clusters_per_query=0,
            distractors=[
                DistractorSpec(size=2, changes=['subgroup']),
                DistractorSpec(size=2, changes=['condition']),
                DistractorSpec(size=2, changes=['condition', 'subgroup']),
            ],
        )
    )
    background_outliers: list[BackgroundDistractorSpec] = Field(
        default_factory=lambda: [BackgroundDistractorSpec()]
    )

    def gold_chunks_per_query(self) -> int:
        return (
            self.primary_calibrated.size
            + self.other_primary.size
            + (2 - self.niche.num_clusters_per_query) * self.secondary.size
            + self.niche.num_clusters_per_query * self.niche.size
        )

    def point_distractor_chunks_per_query(self) -> int:
        return (
            self.primary_calibrated.total_distractor_chunks()
            + self.other_primary.total_distractor_chunks()
            + (2 - self.niche.num_clusters_per_query) * self.secondary.total_distractor_chunks()
            + self.niche.num_clusters_per_query * self.niche.total_distractor_chunks()
        )

    def background_outlier_chunks_per_query(self) -> int:
        return sum(spec.size * spec.num_clusters for spec in self.background_outliers)

    def total_distractor_chunks(self) -> int:
        return self.point_distractor_chunks_per_query() + self.background_outlier_chunks_per_query()


class GenerationLlmConfig(ConfigModel):
    use_llm_chunk_generation: bool = False
    use_llm_chunk_rewriting: bool = False
    use_llm_query_paraphrase: bool = False
    model_name: str = 'gemma4:12b-ud_q8_xl'
    num_workers: PositiveInt = 4
    max_attempts: PositiveInt = 3
    temperature: PositiveFloat = 0.1
    num_ctx: PositiveInt = 4096


class AxisPairPolicyOverrideCfg(ConfigModel):
    axes: tuple[ClinicalAxis, ClinicalAxis]
    allowed_primary_axes: list[ClinicalAxis] | None = None
    blocked_profile_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None
    condition_overrides: list[AxisPairConditionOverride] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_override(self) -> AxisPairPolicyOverrideCfg:
        axis_set = set(self.axes)
        if len(axis_set) != 2:
            raise ValueError('generation.axis_pair_policy_overrides.axes must contain two axes')
        if self.allowed_primary_axes is not None and not set(self.allowed_primary_axes) <= axis_set:
            raise ValueError(
                'generation.axis_pair_policy_overrides.allowed_primary_axes must stay within the pair'
            )
        seen_conditions: set[ConditionKey] = set()
        for override in self.condition_overrides:
            if override.condition_id in seen_conditions:
                raise ValueError(
                    'generation.axis_pair_policy_overrides.condition_overrides must be unique per condition'
                )
            seen_conditions.add(override.condition_id)
            if (
                override.allowed_primary_axes is not None
                and not set(override.allowed_primary_axes) <= axis_set
            ):
                raise ValueError(
                    'generation.axis_pair_policy_overrides.condition_overrides.allowed_primary_axes '
                    'must stay within the pair'
                )
        return self

    def matches(self, left: ClinicalAxis, right: ClinicalAxis) -> bool:
        return set(self.axes) == {left, right}


class GenerationCfg(ConfigModel):
    query_limit: PositiveInt | None = None
    ontology_path: str | None = None

    calibration_mode: PlanCalibrationMode = 'rotating'
    calibration_probe_chunks_per_facet: PositiveInt = 8
    axis_pair_policy_overrides: list[AxisPairPolicyOverrideCfg] = Field(default_factory=list)

    chunk_pools: ChunkPoolsCfg = Field(default_factory=ChunkPoolsCfg)
    llm_config: GenerationLlmConfig = Field(default_factory=GenerationLlmConfig)

    @model_validator(mode='after')
    def _validate_niche_cluster_size(self) -> GenerationCfg:
        if (
            self.chunk_pools.niche.num_clusters_per_query
            and self.chunk_pools.niche.size >= self.chunk_pools.secondary.size
        ):
            raise ValueError(
                'generation.chunk_pools.niche.size must be smaller than '
                'generation.chunk_pools.secondary.size when niche clusters are enabled'
            )
        seen_pairs: set[frozenset[ClinicalAxis]] = set()
        for override in self.axis_pair_policy_overrides:
            key = frozenset(override.axes)
            if key in seen_pairs:
                raise ValueError(
                    'generation.axis_pair_policy_overrides must be unique per unordered axis pair'
                )
            seen_pairs.add(key)
        return self

    def total_gold_chunks(self) -> int:
        return self.chunk_pools.gold_chunks_per_query()

    def total_distractor_chunks(self) -> int:
        return self.chunk_pools.total_distractor_chunks()

    def axis_pair_policy_override(
        self, left: ClinicalAxis, right: ClinicalAxis
    ) -> AxisPairPolicyOverrideCfg | None:
        for override in self.axis_pair_policy_overrides:
            if override.matches(left, right):
                return override
        return None


class EmbeddingCfg(ConfigModel):
    model_name: EmbeddingModelName = 'multi-qa-mpnet-base-cos-v1'
    batch_size: PositiveInt = 64
    device: str = 'cuda'
    devices: list[str] = Field(default_factory=list)
    query_prompt: str | None = None
    document_prompt: str | None = None
    normalize: bool = True


class LambdaGridCfg(ConfigModel):
    start: float = Field(ge=0.0, le=1.0)
    stop: float = Field(ge=0.0, le=1.0)
    num_values: PositiveInt

    @model_validator(mode='after')
    def _validate_bounds(self) -> LambdaGridCfg:
        if self.start > self.stop:
            raise ValueError('lambda grid start must be <= stop')
        return self

    def values(self) -> list[float]:
        return sorted({
            float(round(value, 6)) for value in np.linspace(self.start, self.stop, self.num_values)
        })


class RetrievalCfg(ConfigModel):
    pool_scope: ChunkPoolScope = 'query_local'
    candidate_pool_n: PositiveInt = 300
    k_values: list[PositiveInt] = Field(default_factory=lambda: [5, 10, 20])
    lambdas_mmr: LambdaGridCfg = Field(
        default_factory=lambda: LambdaGridCfg(start=0.02, stop=0.98, num_values=30)
    )
    lambdas_fac_loc: LambdaGridCfg = Field(
        default_factory=lambda: LambdaGridCfg(start=0.02, stop=0.40, num_values=30)
    )
    strategies: set[Literal['top_k', 'mmr', 'fac_loc']] = Field(
        default_factory=lambda: set(['top_k', 'mmr', 'fac_loc'])
    )
    mmr_window: int | None = None
    only_pass_geometry: bool = True
    compute_answer_rouge: bool = True

    def lambda_values_for_strategy(
        self,
        strategy: Literal['top_k', 'mmr', 'fac_loc'],
    ) -> list[float] | list[None]:
        if strategy == 'top_k':
            return [None]
        if strategy == 'mmr':
            return self.lambdas_mmr.values()
        if strategy == 'fac_loc':
            return self.lambdas_fac_loc.values()
        raise ValueError(f'unknown retrieval strategy: {strategy!r}')


class GeometryFilterCfg(ConfigModel):
    topk_k: PositiveInt = 20
    min_primary_axis_count: PositiveInt = 14
    max_topk_retrieved_facets: PositiveInt | None = 2
    min_in_minus_cross_similarity: PositiveFloat = 0.03
    min_same_axis_cohort_gap: PositiveFloat = 0.03
    min_same_cohort_axis_gap: PositiveFloat = 0.03
    min_distractors_in_pool: PositiveInt = 10


class MethodsComparisonKernelMetricCfg(ConfigModel):
    summary_metric: str = 'FacetCoverage@k'
    enabled: bool = True
    weight: PositiveFloat = 1.0
    target_gain_vs_topk: float = 0.08
    gain_bandwidth: PositiveFloat = 0.01
    target_lower_bound_vs_topk: float = 0.03
    lower_bound_bandwidth: PositiveFloat = 0.0075

    @field_validator('summary_metric')
    @classmethod
    def _validate_summary_metric(cls, value: str) -> str:
        if value not in METRIC_NAME_TO_FIELD:
            raise ValueError(f'unknown evaluation summary_metric: {value}')
        return value


type KernelAggregationStrategy = Literal['geometric_mean', 'arithmetic_mean', 'minimum']


class MethodsComparisonKernelsCfg(ConfigModel):
    lambda_max: NonNegativeFloat = 0.80
    agreement_alpha: PositiveFloat = 3.0
    kernel_floor: float = Field(default=0.05, gt=0, le=1)
    pair_aggregation: KernelAggregationStrategy = 'geometric_mean'
    metrics: list[MethodsComparisonKernelMetricCfg] = Field(
        default_factory=lambda: [MethodsComparisonKernelMetricCfg()]
    )


type LambdaSelectionTieBreak = Literal['lower_lambda', 'higher_lambda']


class LambdaSelectionCfg(ConfigModel):
    tie_break: LambdaSelectionTieBreak = 'lower_lambda'


class EvaluationCfg(ConfigModel):
    workers: PositiveInt | None = None
    all_clean_rate_precision_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    lambda_selection: LambdaSelectionCfg = Field(default_factory=LambdaSelectionCfg)
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
    dataset_schema_version: Literal[2] = 2
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
            self.generation.total_gold_chunks() + self.generation.total_distractor_chunks()
        )
        if self.retrieval.candidate_pool_n < local_pool_size:
            raise ValueError(
                'retrieval.candidate_pool_n must include the complete v2 query-local pool '
                f'({local_pool_size} chunks)'
            )
        return self


# Utils ----------------------------------------------------------------------

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
