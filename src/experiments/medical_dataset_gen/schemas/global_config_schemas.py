from __future__ import annotations

from math import ceil
from typing import Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    CLINICAL_AXIS_LIST,
    AxisPairConditionOverride,
    ChunkPoolScope,
    ChunkSurfacePolicy,
    ChunkTextStyle,
    ClinicalAxis,
    ConditionKey,
    QueryFocusMode,
    QueryStructure,
)
from experiments.medical_dataset_gen.utils.global_utils import ResultDirOverrides
from helpers.embedder import EmbeddingModelName


class BasePydanticCfgModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class GlobalCfg(BasePydanticCfgModel):
    seed: PositiveInt = 42
    conditions: PositiveInt = 4
    output_experiment: str = 'v2'
    use_shared: bool = True
    result_dir_overrides: ResultDirOverrides = Field(default_factory=dict)


type DistractorChange = Literal['condition', 'subgroup', 'axis', 'axis_value_bin']


class DistractorSpec(BasePydanticCfgModel):
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


class LocalChunkPoolCfg(BasePydanticCfgModel):
    size: PositiveInt
    distractors: list[DistractorSpec] = Field(default_factory=list)

    def total_distractor_chunks(self) -> int:
        return sum(spec.size for spec in self.distractors)


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
            self.dominant_primary.size
            + self.other_primary.size
            + (2 - self.niche.num_clusters_per_query) * self.secondary.size
            + self.niche.num_clusters_per_query * self.niche.size
        )

    def near_miss_distractors_per_query(self) -> int:
        return (
            self.dominant_primary.total_distractor_chunks()
            + self.other_primary.total_distractor_chunks()
            + (2 - self.niche.num_clusters_per_query) * self.secondary.total_distractor_chunks()
            + self.niche.num_clusters_per_query * self.niche.total_distractor_chunks()
        )

    def background_outliers_per_query(self) -> int:
        return sum(spec.size * spec.num_clusters for spec in self.background_outliers)

    def total_distractor_chunks(self) -> int:
        return self.near_miss_distractors_per_query() + self.background_outliers_per_query()


class GenerationLlmConfig(BasePydanticCfgModel):
    use_llm_chunk_generation: bool = False
    use_llm_chunk_rewriting: bool = False
    use_llm_query_paraphrase: bool = False
    model_name: str = 'gemma4:12b-ud_q8_xl'
    num_workers: PositiveInt = 4
    max_attempts: PositiveInt = 3
    temperature: PositiveFloat = 0.1
    num_ctx: PositiveInt = 4096


class AxisPairPolicyOverrideCfg(BasePydanticCfgModel):
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


class GenerationCfg(BasePydanticCfgModel):
    query_limit: PositiveInt | None = None
    ontology_path: str | None = None
    chunk_text_style: ChunkTextStyle = 'ontology_explicit'
    focus_mode: QueryFocusMode = 'list'
    query_structure: QueryStructure = 'unbalanced'
    chunk_surface_policy: ChunkSurfacePolicy = 'split_heldout'
    excluded_clinical_axes: list[ClinicalAxis] = Field(
        # default_factory=lambda: ['diagnostic_evidence_type']
        default_factory=list
    )

    axis_pair_policy_overrides: list[AxisPairPolicyOverrideCfg] = Field(default_factory=list)

    chunk_pools: ChunkPoolsCfg
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

        if len(self.excluded_clinical_axes) != len(set(self.excluded_clinical_axes)):
            raise ValueError('generation.excluded_clinical_axes must not contain duplicates')

        active_axes = set(CLINICAL_AXIS_LIST) - set(self.excluded_clinical_axes)
        if len(active_axes) < 2:
            raise ValueError('generation.excluded_clinical_axes must leave at least two axes')
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
        return sorted({
            float(round(value, 6)) for value in np.linspace(self.start, self.stop, self.num_values)
        })


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
    # Deprecated no-op retained so older saved configs still validate. Evaluation
    # now scores all queries and records the geometry pass flag per query.
    only_pass_geometry: bool | None = None
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


type EvalPlotTheme = Literal['dark', 'light']


class EvaluationCfg(BasePydanticCfgModel):
    mode: EvaluationMode = 'testing'
    workers: PositiveInt | None = None
    all_clean_rate_precision_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    plot_theme: EvalPlotTheme = 'light'
    lambda_selection: LambdaSelectionCfg = Field(default_factory=LambdaSelectionCfg)


class QueryGeometryCfg(BasePydanticCfgModel):
    n_queries: PositiveInt = 6
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
    dataset_schema_version: Literal[2] = 2
    global_: GlobalCfg = Field(alias='global')
    generation: GenerationCfg
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
