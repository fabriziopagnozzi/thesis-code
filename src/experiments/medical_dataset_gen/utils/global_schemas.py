from __future__ import annotations

from math import ceil
from typing import Literal, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from experiments.medical_dataset_gen.dataset_generation.schemas import (
    CLINICAL_AXIS_LIST,
    ChunkPoolScope,
    ChunkSurfacePolicy,
    ChunkTextStyle,
    ClinicalAxis,
    QueryFocusMode,
    QueryStructure,
)
from experiments.medical_dataset_gen.utils.global_utils import ResultDirOverrides, get_literals
from helpers.embedder import EmbeddingModelName


# Configuration models reject unknown fields so a typo cannot silently change a run.
class BasePydanticCfgModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class GlobalCfg(BasePydanticCfgModel):
    seed: PositiveInt = 42
    conditions: PositiveInt = 4
    output_experiment: str = 'v4'
    use_shared: bool = True
    result_dir_overrides: ResultDirOverrides = Field(default_factory=dict)


# Dataset construction and distractor-pool settings.
type EvalPlotTheme = Literal['dark', 'light']
type DatasetSchemaVersion = Literal[2, 3, 4, 5]
DATASET_SCHEMA_VERSION_LIST = list[DatasetSchemaVersion](get_literals(DatasetSchemaVersion))
type DistractorChange = Literal['condition', 'subgroup', 'axis', 'axis_value_bin']


class DistractorSpec(BasePydanticCfgModel):
    # ``size`` is retained as the compiled, total number of chunks for the
    # legacy generator.  Schema-v5 authoring uses the explicit cluster
    # vocabulary below.  Keeping both at the boundary means old experiments
    # remain readable while new specifications never have to encode topology
    # in an overloaded scalar.
    size: int | None = Field(default=None, ge=1)
    num_clusters: PositiveInt = 1
    chunks_per_cluster: PositiveInt | None = None
    changes: list[DistractorChange] = Field(min_length=1)

    @model_validator(mode='before')
    @classmethod
    def _normalize_legacy_same_different_spec(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if 'changes' in data:
            return cls._normalize_cluster_size(data)
        if not {'condition', 'subgroup', 'axis_config'} <= set(data):
            return cls._normalize_cluster_size(data)

        axis_config = data['axis_config']
        if not isinstance(axis_config, dict):
            return cls._normalize_cluster_size(data)

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
        return cls._normalize_cluster_size(normalized)

    @staticmethod
    def _normalize_cluster_size(data: dict[object, object]) -> dict[object, object]:
        """Compile v5 cluster fields into the legacy total-size field."""
        normalized = dict(data)
        raw_size = normalized.get('size')
        raw_clusters = normalized.get('num_clusters', 1)
        raw_per_cluster = normalized.get('chunks_per_cluster')
        if raw_per_cluster is None:
            if raw_size is None:
                # Let Pydantic provide the useful missing-field error.
                return normalized
            normalized['chunks_per_cluster'] = raw_size
            normalized.setdefault('num_clusters', 1)
            return normalized
        try:
            expected_size = int(cast(int | str, raw_clusters)) * int(
                cast(int | str, raw_per_cluster)
            )
        except (TypeError, ValueError):
            return normalized
        if raw_size is not None and int(cast(int | str, raw_size)) != expected_size:
            raise ValueError(
                'size must equal num_clusters * chunks_per_cluster when both are provided'
            )
        normalized['size'] = expected_size
        return normalized

    @model_validator(mode='after')
    def _validate_changes(self) -> DistractorSpec:
        if self.size is None or self.chunks_per_cluster is None:
            raise ValueError('distractor requires size or chunks_per_cluster')
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
    # Background ``size`` has historically meant chunks *per* cluster.  The
    # v5 vocabulary keeps that semantic meaning and makes it explicit.
    size: int | None = Field(default=8, ge=1)
    num_clusters: int = Field(default=1, ge=1)
    chunks_per_cluster: PositiveInt | None = None
    changes: list[DistractorChange] = Field(
        default_factory=lambda: ['condition', 'subgroup', 'axis'],
        min_length=1,
    )

    @model_validator(mode='before')
    @classmethod
    def _normalize_legacy_same_different_spec(cls, data: object) -> object:
        """Background keeps ``size`` as a per-cluster quantity.

        This intentionally overrides ``DistractorSpec``'s normalizer: local
        distractor sizes are totals whereas background sizes have always been
        per-cluster.  Using the same validator name prevents Pydantic from
        applying the local-pool interpretation first.
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if 'changes' not in normalized and {'condition', 'subgroup', 'axis_config'} <= set(
            normalized
        ):
            axis_config = normalized.pop('axis_config')
            condition = normalized.pop('condition')
            subgroup = normalized.pop('subgroup')
            if not isinstance(axis_config, dict):
                return data
            changes: list[DistractorChange] = []
            if condition == 'different':
                changes.append('condition')
            if subgroup == 'different':
                changes.append('subgroup')
            if axis_config.get('axis') == 'different':
                changes.append('axis')
            elif axis_config.get('value_bin') == 'different':
                changes.append('axis_value_bin')
            normalized['changes'] = changes
        raw_per_cluster = normalized.get('chunks_per_cluster')
        raw_size = normalized.get('size')
        if raw_per_cluster is None:
            if raw_size is not None:
                normalized['chunks_per_cluster'] = raw_size
            return normalized
        if raw_size is not None and int(raw_size) != int(raw_per_cluster):
            raise ValueError('background size must equal chunks_per_cluster when both are provided')
        normalized['size'] = raw_per_cluster
        return normalized


class LocalChunkPoolCfg(BasePydanticCfgModel):
    # v5 accepts explicit cluster/support values instead of an overloaded
    # scalar. Suite-level contracts decide whether those values describe
    # topology (near misses/background) or only compile a gold facet's total
    # support. ``size`` remains the materialized total for generator code.
    size: PositiveInt | None = None
    num_clusters: PositiveInt = 1
    chunks_per_cluster: PositiveInt | None = None
    distractors: list[DistractorSpec] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _normalize_cluster_size(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        raw_size = normalized.get('size')
        raw_clusters = normalized.get('num_clusters', 1)
        raw_per_cluster = normalized.get('chunks_per_cluster')
        if raw_per_cluster is None:
            if raw_size is not None:
                normalized['chunks_per_cluster'] = raw_size
                normalized.setdefault('num_clusters', 1)
            return normalized
        try:
            total = int(raw_clusters) * int(raw_per_cluster)
        except (TypeError, ValueError):
            return normalized
        if raw_size is not None and int(raw_size) != total:
            raise ValueError(
                'pool size must equal num_clusters * chunks_per_cluster when both are provided'
            )
        normalized['size'] = total
        return normalized

    @model_validator(mode='after')
    def _validate_cluster_size(self) -> LocalChunkPoolCfg:
        if self.size is None or self.chunks_per_cluster is None:
            raise ValueError('pool requires size or chunks_per_cluster')
        return self

    def total_distractor_chunks(self) -> int:
        return sum(int(spec.size or 0) for spec in self.distractors)


class NicheChunkPoolCfg(LocalChunkPoolCfg):
    num_clusters_per_query: int = Field(default=0, ge=0, le=2)


class ChunkPoolsCfg(BasePydanticCfgModel):
    chunk_min_words: PositiveInt = 1
    chunk_max_words: PositiveInt = 999999
    chunk_word_tolerance: PositiveInt = 2

    dominant_primary: LocalChunkPoolCfg
    other_primary: LocalChunkPoolCfg
    secondary: LocalChunkPoolCfg
    niche: NicheChunkPoolCfg
    background_outliers: list[BackgroundDistractorSpec] = Field(
        default_factory=lambda: [BackgroundDistractorSpec()]
    )

    def gold_chunks_per_query(self) -> int:
        return (
            int(self.dominant_primary.size or 0)
            + int(self.other_primary.size or 0)
            + (2 - self.niche.num_clusters_per_query) * int(self.secondary.size or 0)
            + self.niche.num_clusters_per_query * int(self.niche.size or 0)
        )

    def near_miss_distractors_per_query(self) -> int:
        return (
            self.dominant_primary.total_distractor_chunks()
            + self.other_primary.total_distractor_chunks()
            + (2 - self.niche.num_clusters_per_query) * self.secondary.total_distractor_chunks()
            + self.niche.num_clusters_per_query * self.niche.total_distractor_chunks()
        )

    def background_outliers_per_query(self) -> int:
        return sum(int(spec.size or 0) * spec.num_clusters for spec in self.background_outliers)

    def total_distractor_chunks(self) -> int:
        return self.near_miss_distractors_per_query() + self.background_outliers_per_query()


class GenerationCfg(BasePydanticCfgModel):
    query_limit: PositiveInt | None = None
    ontology_path: str | None = None
    chunk_text_style: ChunkTextStyle = 'ontology_explicit'
    focus_mode: QueryFocusMode = 'list'
    query_structure: QueryStructure = 'unbalanced'
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout'
    excluded_clinical_axes: list[ClinicalAxis] = Field(default_factory=list)

    chunk_pools: ChunkPoolsCfg
    # Optional schema-v5 global near-miss specifications.  They make total
    # near-miss mass and topology independently controllable instead of
    # coupling them to whichever gold facet a legacy local pool happens to
    # target.  When absent, the archived local-pool behavior is unchanged.
    near_miss_specs: list[DistractorSpec] | None = None

    @model_validator(mode='after')
    def _validate_niche_cluster_size(self) -> GenerationCfg:
        if self.query_structure == 'label_only':
            # Label-only has one surface by design; focus mode must not create hidden variants.
            self.focus_mode = 'natural'

        if self.chunk_pools.niche.num_clusters_per_query and cast(
            int, self.chunk_pools.niche.size
        ) >= cast(int, self.chunk_pools.secondary.size):
            raise ValueError(
                'generation.chunk_pools.niche.size must be smaller than '
                'generation.chunk_pools.secondary.size when niche clusters are enabled'
            )

        if len(self.excluded_clinical_axes) != len(set(self.excluded_clinical_axes)):
            raise ValueError('generation.excluded_clinical_axes must not contain duplicates')

        active_axes = set(CLINICAL_AXIS_LIST) - set(self.excluded_clinical_axes)
        if len(active_axes) < 2:
            raise ValueError('generation.excluded_clinical_axes must leave at least two axes')
        return self

    def total_gold_chunks(self) -> int:
        return self.chunk_pools.gold_chunks_per_query()

    def total_distractor_chunks(self) -> int:
        near_miss = (
            sum(int(spec.size or 0) for spec in self.near_miss_specs)
            if self.near_miss_specs is not None
            else self.chunk_pools.near_miss_distractors_per_query()
        )
        return near_miss + self.chunk_pools.background_outliers_per_query()


# Embedding and retrieval settings.
class EmbeddingCfg(BasePydanticCfgModel):
    model_name: EmbeddingModelName = 'multi-qa-mpnet-base-cos-v1'
    batch_size: PositiveInt = 64
    device: str = 'cuda'
    devices: list[str] = Field(default_factory=list)
    query_prompt: str | None = None
    document_prompt: str | None = None
    normalize: bool = True


class LambdaGridCfg(BasePydanticCfgModel):
    start: float = Field(ge=0.0, le=1.0)
    stop: float = Field(ge=0.0, le=1.0)
    num_values: PositiveInt

    @model_validator(mode='after')
    def _validate_bounds(self) -> LambdaGridCfg:
        if self.start > self.stop:
            raise ValueError('lambda grid start must be <= stop')
        return self

    def values(self) -> list[float]:
        return sorted(
            {
                float(round(value, 6))
                for value in np.linspace(self.start, self.stop, self.num_values)
            }
        )


class RetrievalCfg(BasePydanticCfgModel):
    pool_scope: ChunkPoolScope = 'query_local'
    candidate_pool_n: PositiveInt = 999_999_999
    k_values: list[PositiveInt] = Field(default_factory=lambda: [6, 10, 14])
    lambdas_mmr: LambdaGridCfg = Field(
        default_factory=lambda: LambdaGridCfg(start=0.01, stop=0.99, num_values=120)
    )
    lambdas_fac_loc: LambdaGridCfg = Field(
        default_factory=lambda: LambdaGridCfg(start=0.01, stop=0.99, num_values=120)
    )
    strategies: set[Literal['top_k', 'mmr', 'fac_loc']] = Field(
        default_factory=lambda: set(['top_k', 'mmr', 'fac_loc'])
    )
    mmr_window: int | None = None
    compute_answer_rouge: bool = False

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


# Geometry and evaluation settings.
type GeometryStressHorizonBasis = Literal['competitive_pool_mass']


class GeometryFilterCfg(BasePydanticCfgModel):
    """Predeclared validity and relative-depth stress criteria for query geometry."""

    stress_horizon_basis: GeometryStressHorizonBasis = 'competitive_pool_mass'
    stress_horizon_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    stress_horizon_min_k: PositiveInt = 4
    stress_horizon_max_k: PositiveInt = 24
    min_primary_axis_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    max_retrieved_facet_fraction: float = Field(default=0.50, gt=0.0, le=1.0)

    @model_validator(mode='after')
    def _validate_stress_horizon_bounds(self) -> GeometryFilterCfg:
        if self.stress_horizon_min_k > self.stress_horizon_max_k:
            raise ValueError('stress_horizon_min_k must not exceed stress_horizon_max_k')
        return self

    def stress_horizon(self, *, competitive_pool_mass: int) -> int:
        """Resolve a fixed-within-distribution rank horizon before retrieval is observed."""
        if competitive_pool_mass < 1:
            raise ValueError('competitive_pool_mass must be positive')
        if self.stress_horizon_basis == 'competitive_pool_mass':
            raw_horizon = ceil(self.stress_horizon_fraction * competitive_pool_mass)
        else:
            raise ValueError(f'unsupported stress horizon basis: {self.stress_horizon_basis}')
        return min(max(raw_horizon, self.stress_horizon_min_k), self.stress_horizon_max_k)


type LambdaSelectionTieBreak = Literal['lower_lambda', 'higher_lambda']
type EvaluationMode = Literal['exploring', 'testing']


class LambdaSelectionCfg(BasePydanticCfgModel):
    tie_break: LambdaSelectionTieBreak = 'lower_lambda'


class EvaluationRerankerCfg(BasePydanticCfgModel):
    model_name: str = 'Qwen/Qwen3-Reranker-0.6B'
    batch_size: PositiveInt = 16
    device: str = 'cuda'
    max_length: PositiveInt | None = None
    candidate_pool_n: PositiveInt | None = None
    prompt: str | None = (
        'Given a clinical evidence query, retrieve passages that answer the requested '
        'clinical aspects.'
    )
    show_progress_bar: bool = True
    trust_remote_code: bool = False


class EvaluationCfg(BasePydanticCfgModel):
    mode: EvaluationMode = 'testing'
    workers: PositiveInt | None = None
    use_reranker: bool = False
    reranker: EvaluationRerankerCfg = Field(default_factory=EvaluationRerankerCfg)
    all_clean_rate_precision_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    plot_theme: EvalPlotTheme = 'light'
    lambda_selection: LambdaSelectionCfg = Field(default_factory=LambdaSelectionCfg)


# Query-geometry plotting settings are kept separate from evaluation settings.
class QueryGeometryCfg(BasePydanticCfgModel):
    n_queries: PositiveInt = 9
    query_ids: list[str] = Field(default_factory=list)
    query_selection: Literal['mixed', 'best'] = 'mixed'
    candidate_pool_n: PositiveInt | None = None
    plot_k: PositiveInt = 10
    compute_clusters: bool = False
    reduction: Literal['umap', 'pca'] = 'umap'
    pca_dims: PositiveInt | None = None
    umap_metric: Literal['cosine', 'euclidean'] = 'cosine'
    umap_neighbors: PositiveInt = 15
    umap_min_dist: PositiveFloat = 0.08
    hdbscan_umap_dims: PositiveInt = 10
    hdbscan_min_cluster_size: PositiveInt = 5
    hdbscan_min_samples: PositiveInt | None = None
    random_state: PositiveInt = 42


class ExperimentCfg(BasePydanticCfgModel):
    # v2--v4 remain readable for archived experiments; new suite construction
    # uses v5.
    dataset_schema_version: DatasetSchemaVersion = 4
    global_: GlobalCfg = Field(alias='global')
    generation: GenerationCfg
    embeddings: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    geometry_filter: GeometryFilterCfg = Field(default_factory=GeometryFilterCfg)
    evaluation: EvaluationCfg = Field(default_factory=EvaluationCfg)
    query_geometry: QueryGeometryCfg = Field(default_factory=QueryGeometryCfg)

    model_config = ConfigDict(populate_by_name=True, extra='forbid')

    @model_validator(mode='before')
    @classmethod
    def _discard_archived_v2_geometry_option(cls, data: object) -> object:
        if not isinstance(data, dict) or data.get('dataset_schema_version') != 2:
            return data
        retrieval = data.get('retrieval')
        if not isinstance(retrieval, dict) or 'only_pass_geometry' not in retrieval:
            return data

        normalized = dict(data)
        normalized_retrieval = dict(retrieval)
        normalized_retrieval.pop('only_pass_geometry')
        normalized['retrieval'] = normalized_retrieval
        return normalized

    @model_validator(mode='after')
    def _validate_query_local_generation_scope(self) -> ExperimentCfg:
        if self.dataset_schema_version not in {3, 4, 5}:
            return self
        if self.retrieval.pool_scope != 'query_local':
            raise ValueError(
                f'dataset schema v{self.dataset_schema_version} supports only '
                'retrieval.pool_scope=query_local'
            )
        local_pool_size = (
            self.generation.total_gold_chunks() + self.generation.total_distractor_chunks()
        )
        if self.retrieval.candidate_pool_n < local_pool_size:
            raise ValueError(
                'retrieval.candidate_pool_n must include the complete query-local pool '
                f'({local_pool_size} chunks)'
            )
        return self
