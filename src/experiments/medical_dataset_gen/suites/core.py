"""Declarative, immutable v5 experiment-suite contracts.

Schema v1 materialized one record per distribution x wording cell.  Schema v2
keeps the legacy view readable, but makes the reusable distribution and its
evaluation crossings first-class.  This matters for the matched thesis suite:
one medium-scale pool can be a reference, a scale anchor, and a member of more
than one contrast without silently creating three datasets.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.medical_dataset_gen.utils.exp_naming import embedding_child_token
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import (
    SHARED_GENERATION_TABLES,
    MedicalDatasetGenPaths,
)

SUITE_LAYOUT_VERSION = 5
MANIFEST_VERSION = 2
type SuiteOrigin = Literal['native', 'derived', 'migrated_v4']
type TransformOp = Literal[
    'set_gold_mass_vector',
    'set_near_miss_mass',
    'set_near_miss_mix',
    'set_background_mass',
    'set_background_components',
    'set_structural_shell',
    'set_cluster_topology',
    'scale_support',
]


class SuiteModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SuiteTransform(SuiteModel):
    """A typed, auditable composition operation.

    ``components`` intentionally stays structural YAML rather than a second
    duplicate background schema.  It is validated by ``ExperimentCfg`` after
    resolution, which is the authoritative generator contract.
    """

    op: TransformOp
    masses: list[int] | None = None
    niche_count: int | None = Field(default=None, ge=0, le=2)
    mass: int | None = Field(default=None, ge=0)
    one_change_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    components: list[dict[str, Any]] | None = None
    shell: Literal['near', 'intermediate', 'far'] | None = None
    component: Literal['near_miss', 'background'] | None = None
    num_clusters: int | None = Field(default=None, ge=1)
    chunks_per_cluster: int | None = Field(default=None, ge=1)
    multiplier: float | None = Field(default=None, gt=0.0)
    preserve_cluster_count: bool = True

    @model_validator(mode='after')
    def _validate_operation(self) -> SuiteTransform:
        if self.op == 'set_gold_mass_vector':
            if (
                self.masses is None
                or len(self.masses) != 4
                or any(value < 1 for value in self.masses)
            ):
                raise ValueError('set_gold_mass_vector requires four positive masses')
            if self.niche_count is None:
                # Read schema-v1 specs conservatively.  Native-v5 specs must
                # declare this explicitly; only the compatibility adapter may
                # rely on the historical inference.
                self.niche_count = 0 if self.masses[2] == self.masses[3] else 1
        elif self.op in {'set_near_miss_mass', 'set_background_mass'} and self.mass is None:
            raise ValueError(f'{self.op} requires mass')
        elif self.op == 'set_near_miss_mix' and self.one_change_fraction is None:
            raise ValueError('set_near_miss_mix requires one_change_fraction')
        elif self.op == 'set_background_components' and not self.components:
            raise ValueError('set_background_components requires non-empty components')
        elif self.op == 'set_structural_shell' and self.shell is None:
            raise ValueError('set_structural_shell requires shell')
        elif self.op == 'set_cluster_topology':
            if (
                self.component is None
                or self.num_clusters is None
                or self.chunks_per_cluster is None
            ):
                raise ValueError('set_cluster_topology requires component and topology fields')
        elif self.op == 'scale_support' and self.multiplier is None:
            raise ValueError('scale_support requires multiplier')
        return self


class Distribution(SuiteModel):
    family_id: str
    family_label: str
    config: dict[str, Any]
    transforms: list[SuiteTransform] = Field(default_factory=list)
    factors: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    analysis_blocks: list[str] = Field(default_factory=list)
    analysis_tier: Literal['core', 'scale', 'interaction', 'geometry', 'smoke'] = 'core'
    include_in_family_summary: bool = True
    nested_from: str | None = None


# Kept as an import-compatible name for callers that constructed a v1 spec.
DistributionBase = Distribution


class RunProfile(SuiteModel):
    config: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    factors: dict[str, str] = Field(default_factory=dict)


class Evaluation(SuiteModel):
    evaluation_id: str
    distribution_ids: list[str] = Field(min_length=1)
    run_profile_ids: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    factors: dict[str, Any] = Field(default_factory=dict)


class SuiteCell(SuiteModel):
    """Compatibility representation for a fully expanded v1 cell."""

    cell_id: str
    distribution_id: str
    run_profile_id: str | None = None
    run_profile_ids: list[str] | None = None
    transforms: list[SuiteTransform] = Field(default_factory=list)
    factors: dict[str, Any] = Field(default_factory=dict)
    include_in_causal_summaries: bool = True
    nested_from: str | None = None

    @model_validator(mode='after')
    def _validate_run_profile_selection(self) -> SuiteCell:
        if (self.run_profile_id is None) == (self.run_profile_ids is None):
            raise ValueError('cell requires exactly one of run_profile_id or run_profile_ids')
        if self.run_profile_ids is not None and not self.run_profile_ids:
            raise ValueError('run_profile_ids must not be empty')
        return self


class ComparisonGroup(SuiteModel):
    comparison_id: str
    # v2 contracts reference reusable distributions.  ``cells`` and
    # ``varying_factor`` preserve migrated-v4/schema-v1 report support.
    distribution_ids: list[str] = Field(default_factory=list)
    cells: list[str] = Field(default_factory=list)
    varying_factors: list[str] = Field(default_factory=list, max_length=2)
    varying_factor: str | None = None
    matching_factors: list[str] = Field(default_factory=list)
    factor_levels: dict[str, list[Any]] = Field(default_factory=dict)
    reference_levels: dict[str, Any] = Field(default_factory=dict)
    owned_paths: dict[str, list[str]] = Field(default_factory=dict)
    analysis_block: str | None = None
    run_profile_ids: list[str] = Field(default_factory=list)
    strict: bool = True

    @model_validator(mode='after')
    def _validate_shape(self) -> ComparisonGroup:
        if not self.distribution_ids and not self.cells:
            raise ValueError('comparison requires distribution_ids or legacy cells')
        factors = self.all_varying_factors
        if not factors:
            raise ValueError('comparison requires one or two varying factors')
        if len(factors) > 2:
            raise ValueError('comparison supports at most two varying factors')
        return self

    @property
    def all_varying_factors(self) -> list[str]:
        return self.varying_factors or ([self.varying_factor] if self.varying_factor else [])


class AnalysisSeriesPoint(SuiteModel):
    """One intentionally non-rectangular analysis point.

    A proportional-budget scale diagonal crosses different distributions, run
    profiles, and retrieval budgets. It therefore cannot be represented by a
    normal comparison group, whose cells share a profile and budget schedule.
    """

    point_id: str
    distribution_id: str
    run_profile_id: str
    k: int = Field(ge=1)
    factors: dict[str, Any] = Field(default_factory=dict)


class AnalysisSeries(SuiteModel):
    """Manifest-declared ordered series for a non-rectangular analysis."""

    series_id: str
    analysis_block: str
    points: list[AnalysisSeriesPoint] = Field(min_length=2)
    reference_point_id: str
    lambda_source_k: int = Field(ge=1)
    strict: bool = True

    @model_validator(mode='after')
    def _validate_points(self) -> AnalysisSeries:
        point_ids = [point.point_id for point in self.points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError(f'{self.series_id}: duplicate analysis-series point IDs')
        if self.reference_point_id not in point_ids:
            raise ValueError(
                f'{self.series_id}: unknown reference_point_id {self.reference_point_id!r}'
            )
        return self


class DerivedSuiteSource(SuiteModel):
    """Immutable source contract for a suite that only reruns downstream stages."""

    suite_id: str
    manifest_sha256: str = Field(min_length=64, max_length=64)
    distribution_ids: list[str] = Field(min_length=1)
    embedding_models: list[str] = Field(min_length=1)

    @model_validator(mode='before')
    @classmethod
    def _adapt_single_embedding_model(cls, data: object) -> object:
        if not isinstance(data, Mapping) or 'embedding_models' in data:
            return data
        raw = dict(data)
        model_name = raw.pop('embedding_model', None)
        if isinstance(model_name, str):
            raw['embedding_models'] = [model_name]
        return raw

    @model_validator(mode='after')
    def _validate_source(self) -> DerivedSuiteSource:
        _validate_identifier(self.suite_id, 'source suite_id')
        if len(self.distribution_ids) != len(set(self.distribution_ids)):
            raise ValueError('derived source distribution_ids must be unique')
        for distribution_id in self.distribution_ids:
            _validate_identifier(distribution_id, 'source distribution_id')
        if len(self.embedding_models) != len(set(self.embedding_models)):
            raise ValueError('derived source embedding_models must be unique')
        if len(self.manifest_sha256) != 64 or any(
            character not in '0123456789abcdef' for character in self.manifest_sha256
        ):
            raise ValueError('derived source manifest_sha256 must be a lowercase SHA-256 digest')
        return self


class SuiteSpec(SuiteModel):
    layout_version: Literal[5] = SUITE_LAYOUT_VERSION
    suite_id: str
    origin: Literal['native', 'derived'] = 'native'
    dataset_schema_version: Literal[5] = 5
    description: str = ''
    source: DerivedSuiteSource | None = None
    distributions: dict[str, Distribution] = Field(default_factory=dict)
    run_profiles: dict[str, RunProfile] = Field(default_factory=dict)
    evaluations: list[Evaluation] = Field(default_factory=list)
    comparison_groups: list[ComparisonGroup] = Field(default_factory=list)
    analysis_series: list[AnalysisSeries] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _adapt_v1_spec(cls, data: object) -> object:
        """Read old native specs while all newly written specs use v2 names."""
        if not isinstance(data, Mapping) or 'distributions' in data:
            return data
        if 'distribution_bases' not in data or 'cells' not in data:
            return data
        raw = deepcopy(dict(data))
        bases = cast(Mapping[str, Mapping[str, Any]], raw.pop('distribution_bases'))
        cells = cast(list[Mapping[str, Any]], raw.pop('cells'))
        distributions: dict[str, dict[str, Any]] = {}
        evaluations: list[dict[str, Any]] = []
        for old_cell in cells:
            cell = dict(old_cell)
            identifier = str(cell['cell_id'])
            base_id = str(cell['distribution_id'])
            base = deepcopy(dict(bases[base_id]))
            base['transforms'] = cell.pop('transforms', [])
            base['factors'] = cell.pop('factors', {})
            base['nested_from'] = cell.pop('nested_from', None)
            base['include_in_family_summary'] = bool(cell.pop('include_in_causal_summaries', True))
            distributions[identifier] = base
            profile_ids = cell.pop('run_profile_ids', None)
            if profile_ids is None:
                profile_ids = [cell.pop('run_profile_id')]
            evaluations.append(
                {
                    'evaluation_id': identifier,
                    'distribution_ids': [identifier],
                    'run_profile_ids': profile_ids,
                }
            )
        raw['distributions'] = distributions
        raw['evaluations'] = evaluations
        return raw

    @model_validator(mode='after')
    def _validate_references(self) -> SuiteSpec:
        if self.origin == 'derived':
            if self.source is None:
                raise ValueError('derived suite requires a source contract')
            if self.distributions or self.run_profiles or self.evaluations:
                raise ValueError(
                    'derived suite derives its distributions and run profiles from its source'
                )
            if self.comparison_groups or self.analysis_series:
                raise ValueError(
                    'derived suite inherits comparisons and analysis series from its source'
                )
            return self
        if self.source is not None:
            raise ValueError('native suite must not declare a source contract')
        if not self.evaluations:
            raise ValueError('suite requires at least one evaluation cross')
        for distribution_id, distribution in self.distributions.items():
            _validate_identifier(distribution_id, 'distribution_id')
            if (
                distribution.nested_from is not None
                and distribution.nested_from not in self.distributions
            ):
                raise ValueError(
                    f'{distribution_id}: unknown nested_from {distribution.nested_from!r}'
                )
        for evaluation in self.evaluations:
            unknown_distributions = sorted(
                set(evaluation.distribution_ids) - set(self.distributions)
            )
            unknown_profiles = sorted(set(evaluation.run_profile_ids) - set(self.run_profiles))
            if unknown_distributions or unknown_profiles:
                raise ValueError(
                    f'{evaluation.evaluation_id}: unknown distributions={unknown_distributions}, '
                    f'run_profiles={unknown_profiles}'
                )
        expanded_ids = [cell.cell_id for cell in self.expanded_cells()]
        if len(expanded_ids) != len(set(expanded_ids)):
            raise ValueError('evaluation crosses produce duplicate distribution/run-profile cells')
        for group in self.comparison_groups:
            if group.distribution_ids:
                unknown = sorted(set(group.distribution_ids) - set(self.distributions))
                if unknown:
                    raise ValueError(f'{group.comparison_id}: unknown distributions {unknown}')
            elif set(group.cells) - set(expanded_ids):
                raise ValueError(f'{group.comparison_id}: unknown legacy cells')
        for series in self.analysis_series:
            for point in series.points:
                cell_id = f'{point.distribution_id}__{point.run_profile_id}'
                if cell_id not in expanded_ids:
                    raise ValueError(
                        f'{series.series_id}: unknown analysis-series cell {cell_id!r}'
                    )
        return self

    def expanded_cells(self) -> list[SuiteCell]:
        cells: list[SuiteCell] = []
        for evaluation in self.evaluations:
            for distribution_id in evaluation.distribution_ids:
                distribution = self.distributions[distribution_id]
                for profile_id in evaluation.run_profile_ids:
                    cells.append(
                        SuiteCell(
                            cell_id=f'{distribution_id}__{profile_id}',
                            distribution_id=distribution_id,
                            run_profile_id=profile_id,
                            transforms=distribution.transforms,
                            factors={**distribution.factors, **evaluation.factors},
                            include_in_causal_summaries=distribution.include_in_family_summary,
                            nested_from=(
                                f'{distribution.nested_from}__{profile_id}'
                                if distribution.nested_from is not None
                                and profile_id
                                in self.profiles_for_distribution(distribution.nested_from)
                                else None
                            ),
                        )
                    )
        return cells

    def profiles_for_distribution(self, distribution_id: str) -> set[str]:
        return {
            profile_id
            for evaluation in self.evaluations
            if distribution_id in evaluation.distribution_ids
            for profile_id in evaluation.run_profile_ids
        }


class SuiteManifestDistribution(SuiteModel):
    distribution_id: str
    family_id: str
    family_label: str
    factors: dict[str, Any]
    tags: list[str]
    analysis_blocks: list[str]
    analysis_tier: str
    include_in_family_summary: bool
    nested_from: str | None = None
    resolved_distribution_path: str
    distribution_sha256: str
    dataset_sha256: str


class SuiteManifestRunProfile(SuiteModel):
    """Immutable rendering/retrieval profile reused by v2 evaluations."""

    run_profile_id: str
    factors: dict[str, str]
    resolved_run_profile_path: str
    run_profile_sha256: str


class SuiteManifestCell(SuiteModel):
    cell_id: str
    name: str
    distribution_id: str
    distribution_base_id: str | None = None
    evaluation_id: str | None = None
    run_profile_id: str
    family_id: str
    family_label: str
    origin: SuiteOrigin
    dataset_schema_version: Literal[4, 5]
    evaluation_schema_version: int
    status: Literal['planned', 'completed']
    include_in_causal_summaries: bool = True
    include_in_family_summary: bool = True
    factors: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    analysis_blocks: list[str] = Field(default_factory=list)
    analysis_tier: str = 'core'
    run_profile_factors: dict[str, str] = Field(default_factory=dict)
    data_root: str
    attempt_root: str
    resolved_config_path: str
    config_sha256: str
    dataset_sha256: str
    run_profile_sha256: str
    artifact_manifest_path: str | None = None
    nested_from: str | None = None
    extra_evaluation_attempts: list[str] = Field(default_factory=list)
    source_suite_id: str | None = None
    source_cell_id: str | None = None
    source_manifest_sha256: str | None = None
    source_dataset_sha256: str | None = None

    @model_validator(mode='after')
    def _validate_source_reference(self) -> SuiteManifestCell:
        source_fields = (
            self.source_suite_id,
            self.source_cell_id,
            self.source_manifest_sha256,
            self.source_dataset_sha256,
        )
        if self.origin == 'derived':
            if any(value is None for value in source_fields):
                raise ValueError('derived suite cell requires a complete source reference')
        elif any(value is not None for value in source_fields):
            raise ValueError('only derived suite cells may declare a source reference')
        return self


class SuiteManifest(SuiteModel):
    manifest_version: Literal[1, 2] = MANIFEST_VERSION
    layout_version: Literal[5] = SUITE_LAYOUT_VERSION
    suite_id: str
    origin: SuiteOrigin
    created_at: str
    source: DerivedSuiteSource | None = None
    # ``cells`` is deliberately retained as the v1 compatibility view used by
    # the migrated archive and generic report discovery.
    cells: list[SuiteManifestCell] = Field(default_factory=list)
    distributions: list[SuiteManifestDistribution] = Field(default_factory=list)
    run_profiles: list[SuiteManifestRunProfile] = Field(default_factory=list)
    evaluations: list[SuiteManifestCell] = Field(default_factory=list)
    comparison_groups: list[ComparisonGroup] = Field(default_factory=list)
    analysis_series: list[AnalysisSeries] = Field(default_factory=list)

    @model_validator(mode='after')
    def _normalize_v2_view(self) -> SuiteManifest:
        if self.manifest_version == 2:
            if not self.cells and self.evaluations:
                self.cells = list(self.evaluations)
            elif not self.evaluations and self.cells:
                self.evaluations = list(self.cells)
            if len(self.cells) != len(self.evaluations):
                raise ValueError('v2 manifest cells/evaluations disagree')
        if self.origin == 'derived' and self.source is None:
            raise ValueError('derived manifest requires a source contract')
        if self.origin != 'derived' and self.source is not None:
            raise ValueError('only derived manifests may declare a source contract')
        return self


@dataclass(frozen=True)
class ValidationResult:
    warnings: tuple[str, ...]
    resolved_configs: dict[str, dict[str, Any]]
    resolved_distributions: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class LogicalSuite:
    """A report-facing suite plus the physical root for every cell."""

    manifest: SuiteManifest
    roots_by_cell_id: dict[str, Path]


def suite_spec_root() -> Path:
    return MedicalDatasetGenPaths.root / 'experiment_specs'


def suite_root(results_dir: Path, suite_id: str) -> Path:
    _validate_identifier(suite_id, 'suite_id')
    return results_dir / 'v5' / 'suites' / suite_id


def load_suite_spec(suite_id_or_path: str | Path) -> SuiteSpec:
    path = Path(suite_id_or_path)
    if not path.suffix:
        path = suite_spec_root() / f'{path}.yaml'
    return SuiteSpec.model_validate(_read_yaml_mapping(path))


def load_suite_manifest(results_dir: Path, suite_id: str) -> SuiteManifest:
    path = suite_root(results_dir, suite_id) / 'suite_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'missing suite manifest: {path}')
    return SuiteManifest.model_validate(json.loads(path.read_text()))


def load_pinned_source_manifest(
    *, results_dir: Path, source: DerivedSuiteSource
) -> tuple[Path, SuiteManifest]:
    """Load a source suite only when its on-disk manifest is exactly pinned."""
    root = suite_root(results_dir, source.suite_id)
    manifest_path = root / 'suite_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'missing derived-suite source manifest: {manifest_path}')
    observed = _sha256_file(manifest_path)
    if observed != source.manifest_sha256:
        raise ValueError(
            f'{source.suite_id}: source manifest hash changed; expected '
            f'{source.manifest_sha256}, observed {observed}'
        )
    manifest = SuiteManifest.model_validate(json.loads(manifest_path.read_text()))
    if manifest.origin != 'native':
        raise ValueError(f'{source.suite_id}: derived suites require a native source suite')
    return root, manifest


def resolve_derived_source_cell(
    *, root: Path, cell: SuiteManifestCell
) -> tuple[Path, SuiteManifestCell]:
    """Resolve and verify the immutable source data backing one derived cell."""
    if cell.origin != 'derived':
        raise ValueError(f'{cell.cell_id}: cell is not source-backed')
    results_dir = root.parents[2]
    manifest = load_suite_manifest(results_dir, root.name)
    source = manifest.source
    if source is None:
        raise ValueError(f'{cell.cell_id}: derived suite manifest has no source contract')
    source_root, source_manifest = load_pinned_source_manifest(
        results_dir=results_dir, source=source
    )
    if (
        cell.source_suite_id != source.suite_id
        or cell.source_manifest_sha256 != source.manifest_sha256
    ):
        raise ValueError(f'{cell.cell_id}: source reference disagrees with derived suite contract')
    source_cells = {candidate.cell_id: candidate for candidate in source_manifest.cells}
    source_cell_id = cast(str, cell.source_cell_id)
    source_cell = source_cells.get(source_cell_id)
    if source_cell is None:
        raise ValueError(f'{cell.cell_id}: source cell {source_cell_id!r} is missing')
    if source_cell.status != 'completed':
        raise ValueError(f'{cell.cell_id}: source cell {source_cell.cell_id} is not completed')
    if source_cell.distribution_id != cell.distribution_id:
        raise ValueError(f'{cell.cell_id}: source distribution does not match derived distribution')
    if source_cell.dataset_sha256 != cell.source_dataset_sha256:
        raise ValueError(f'{cell.cell_id}: source dataset hash changed')
    if cell.dataset_sha256 != cell.source_dataset_sha256:
        raise ValueError(f'{cell.cell_id}: derived dataset hash differs from its source')
    _validate_source_generation_artifacts(source_root=source_root, source_cell=source_cell)
    return source_root, source_cell


def load_logical_suite(results_dir: Path, suite_id: str) -> LogicalSuite:
    """Expose a derived suite as its pinned source plus its local evaluations."""
    root = suite_root(results_dir, suite_id)
    manifest = load_suite_manifest(results_dir, suite_id)
    if manifest.origin != 'derived':
        return LogicalSuite(
            manifest=manifest,
            roots_by_cell_id={cell.cell_id: root for cell in manifest.cells},
        )
    assert manifest.source is not None
    source_root, source_manifest = load_pinned_source_manifest(
        results_dir=results_dir, source=manifest.source
    )
    selected_distributions = set(manifest.source.distribution_ids)
    source_cells = [
        cell for cell in source_manifest.cells if cell.distribution_id in selected_distributions
    ]
    if any(cell.status != 'completed' for cell in source_cells):
        incomplete = [cell.cell_id for cell in source_cells if cell.status != 'completed']
        raise ValueError(f'{suite_id}: incomplete source cells: {incomplete[:8]}')
    for cell in manifest.cells:
        resolve_derived_source_cell(root=root, cell=cell)
    all_cells = source_cells + list(manifest.cells)
    cell_ids = [cell.cell_id for cell in all_cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError(f'{suite_id}: source and derived cells must have distinct IDs')
    logical_manifest = manifest.model_copy(
        update={
            # Retain all source distributions so the report's validity filter
            # can still remove legacy shells before strict contrasts are built.
            'distributions': source_manifest.distributions,
            'run_profiles': source_manifest.run_profiles + manifest.run_profiles,
            'cells': all_cells,
            'evaluations': all_cells,
            'comparison_groups': source_manifest.comparison_groups,
            'analysis_series': source_manifest.analysis_series + manifest.analysis_series,
        }
    )
    return LogicalSuite(
        manifest=logical_manifest,
        roots_by_cell_id={
            **{cell.cell_id: source_root for cell in source_cells},
            **{cell.cell_id: root for cell in manifest.cells},
        },
    )


def validate_suite(spec: SuiteSpec) -> ValidationResult:
    if spec.origin == 'derived':
        return ValidationResult(warnings=(), resolved_configs={}, resolved_distributions={})
    warnings: list[str] = []
    resolved_configs: dict[str, dict[str, Any]] = {}
    resolved_distributions: dict[str, dict[str, Any]] = {}
    cells = spec.expanded_cells()
    cells_by_id = {cell.cell_id: cell for cell in cells}
    for distribution_id, distribution in spec.distributions.items():
        raw = _resolve_distribution_config(distribution)
        # Resolve once to make all subsequent profile-specific config hashes
        # derive from the exact same composition snapshot.
        _validate_single_region_gold_facets(raw)
        ExperimentCfg.model_validate(raw)
        resolved_distributions[distribution_id] = raw
    for cell in cells:
        resolved = resolve_cell_config(spec, cell)
        ExperimentCfg.model_validate(resolved)
        resolved_configs[cell.cell_id] = resolved
        if cell.nested_from is not None:
            parent = cells_by_id[cell.nested_from]
            if parent.run_profile_id != cell.run_profile_id:
                raise ValueError(f'{cell.cell_id}: nested support must keep its run profile')
    _validate_comparison_groups(spec, cells, resolved_configs)
    _validate_analysis_series(spec, cells, resolved_configs)
    _validate_declared_composition_factors(spec, resolved_distributions)
    _validate_family_semantics(spec, resolved_configs, warnings)
    return ValidationResult(
        warnings=tuple(warnings),
        resolved_configs=resolved_configs,
        resolved_distributions=resolved_distributions,
    )


def _validate_analysis_series(
    spec: SuiteSpec,
    cells: Sequence[SuiteCell],
    resolved: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate declared non-rectangular analyses against executable cells."""
    cells_by_id = {cell.cell_id: cell for cell in cells}
    for series in spec.analysis_series:
        seen_cells: set[str] = set()
        for point in series.points:
            cell_id = f'{point.distribution_id}__{point.run_profile_id}'
            if cell_id in seen_cells:
                raise ValueError(f'{series.series_id}: repeated analysis-series cell {cell_id!r}')
            seen_cells.add(cell_id)
            cell = cells_by_id[cell_id]
            cfg = ExperimentCfg.model_validate(resolved[cell.cell_id])
            if point.k not in cfg.retrieval.k_values:
                raise ValueError(
                    f'{series.series_id}: k={point.k} is absent from {cell_id} retrieval.k_values'
                )
        reference = next(
            point for point in series.points if point.point_id == series.reference_point_id
        )
        if series.lambda_source_k != reference.k:
            raise ValueError(
                f'{series.series_id}: lambda_source_k must equal the reference point budget '
                f'({reference.k})'
            )


def _validate_declared_composition_factors(
    spec: SuiteSpec,
    resolved_distributions: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject metadata that contradicts the generator-resolved composition."""
    for distribution_id, distribution in spec.distributions.items():
        composition = _declared_composition(
            ExperimentCfg.model_validate(resolved_distributions[distribution_id])
        )
        actual: dict[str, Any] = {
            key: composition[key]
            for key in (
                'gold_mass_vector',
                'gold_mass',
                'niche_count',
                'near_miss_mass',
                'background_mass',
                'pool_mass',
            )
        }
        masses = composition['gold_mass_vector']
        actual['dominance_share'] = max(masses) / sum(masses)
        actual['near_miss_load_ratio'] = composition['near_miss_mass'] / composition['gold_mass']
        actual['near_miss_topology'] = _topology_label(composition['near_miss_topology'])
        actual['background_topology'] = _topology_label(
            composition['background_topology_components']
        )
        actual['one_change_fraction'] = _one_change_fraction(composition['near_miss_topology'])
        actual['background_shell'] = _background_shell(
            composition['background_topology_components']
        )
        for key, expected in distribution.factors.items():
            if key not in actual or actual[key] is None:
                continue
            if _canonical_json(expected) != _canonical_json(actual[key]):
                raise ValueError(
                    f'{distribution_id}: declared factor {key}={expected!r} disagrees with '
                    f'resolved composition {actual[key]!r}'
                )


def _topology_label(components: object) -> str | None:
    if not isinstance(components, list):
        return None
    if not components:
        return 'none'
    support_counts = {int(component['chunks_per_cluster']) for component in components}
    if len(support_counts) != 1:
        return None
    clusters = sum(int(component['num_clusters']) for component in components)
    return f'{clusters}x{support_counts.pop()}'


def _one_change_fraction(components: object) -> float | None:
    if not isinstance(components, list) or not components:
        return None
    total = sum(int(component['mass']) for component in components)
    if not total:
        return None
    one_change = sum(
        int(component['mass']) for component in components if len(component['changes']) == 1
    )
    return one_change / total


def _background_shell(components: object) -> str | None:
    if not isinstance(components, list) or not components:
        return None
    shells = {tuple(component['changes']) for component in components}
    if len(shells) != 1:
        return None
    return {
        ('subgroup',): 'near',
        ('condition', 'subgroup'): 'intermediate',
        ('condition', 'subgroup', 'axis'): 'far',
    }.get(shells.pop())


def _resolve_distribution_config(distribution: Distribution) -> dict[str, Any]:
    raw = deepcopy(distribution.config)
    raw['dataset_schema_version'] = 5
    for transform in distribution.transforms:
        _apply_transform(raw, transform)
    return raw


def _validate_single_region_gold_facets(raw: Mapping[str, Any]) -> None:
    """Keep native-v5 gold support aligned with the four-facet task.

    The shared configuration schema retains ``num_clusters`` for legacy
    compatibility and for non-gold components. In this suite contract it is
    not a gold-topology control: every target facet maps to exactly one gold
    region and its support is controlled only by the gold mass vector.
    """
    generation = cast(Mapping[str, Any], raw['generation'])
    pools = cast(Mapping[str, Mapping[str, Any]], generation['chunk_pools'])
    for pool_name in ('dominant_primary', 'other_primary', 'secondary', 'niche'):
        if int(pools[pool_name].get('num_clusters', 1)) != 1:
            raise ValueError(
                f'{pool_name}: native-v5 gold facets must have one materialized gold region; '
                'use set_gold_mass_vector to change support'
            )


def resolve_cell_config(spec: SuiteSpec, cell: SuiteCell) -> dict[str, Any]:
    if cell.run_profile_id is None:
        raise ValueError('resolve an expanded suite cell')
    distribution = spec.distributions[cell.distribution_id]
    profile = spec.run_profiles[cell.run_profile_id]
    raw = _deep_merge(_resolve_distribution_config(distribution), profile.config)
    raw.setdefault('global', {})
    raw['global']['output_experiment'] = cell.distribution_id
    return raw


def materialize_suite(
    spec: SuiteSpec,
    *,
    results_dir: Path,
    overwrite: bool = False,
    replace_planned: bool = False,
    refresh_planned_execution: bool = False,
    prune_planned: bool = False,
    prune_removed_embeddings: bool = False,
) -> SuiteManifest:
    """Materialize a v5 suite without generating data or embeddings."""
    if spec.origin == 'derived':
        return _materialize_derived_suite(
            spec=spec,
            results_dir=results_dir,
            replace_planned=replace_planned,
            prune_removed_embeddings=prune_removed_embeddings,
        )
    validation = validate_suite(spec)
    root = suite_root(results_dir, spec.suite_id)
    manifest_path = root / 'suite_manifest.json'
    if manifest_path.exists():
        existing = load_suite_manifest(results_dir, spec.suite_id)
        if existing.origin != 'native':
            raise ValueError(f'{root} is not a native suite')
        if replace_planned:
            _remove_planned_native_suite(root, existing)
        elif prune_planned:
            _prune_metadata_only_planned_cells(
                root=root,
                existing=existing,
                spec=spec,
                validation=validation,
            )
        elif refresh_planned_execution:
            _validate_planned_execution_refresh(root=root, existing=existing, validation=validation)
        elif not overwrite:
            return existing
        else:
            raise ValueError('--overwrite is unsafe for materialized suites; use --replace-planned')

    from datetime import UTC, datetime

    distributions: list[SuiteManifestDistribution] = []
    for distribution_id, distribution in spec.distributions.items():
        resolved = validation.resolved_distributions[distribution_id]
        cfg = ExperimentCfg.model_validate(resolved)
        dist_root = root / 'distributions' / distribution_id
        distribution_path = dist_root / 'resolved_distribution.yaml'
        distribution_path.parent.mkdir(parents=True, exist_ok=True)
        distribution_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        factors = {**_declared_composition(cfg), **distribution.factors}
        _write_json(
            dist_root / 'distribution_metadata.json',
            {
                'layout_version': SUITE_LAYOUT_VERSION,
                'distribution_id': distribution_id,
                'family_id': distribution.family_id,
                'family_label': distribution.family_label,
                'factors': factors,
                'tags': distribution.tags,
                'analysis_blocks': distribution.analysis_blocks,
                'analysis_tier': distribution.analysis_tier,
                'nested_from': distribution.nested_from,
            },
        )
        distributions.append(
            SuiteManifestDistribution(
                distribution_id=distribution_id,
                family_id=distribution.family_id,
                family_label=distribution.family_label,
                factors=factors,
                tags=distribution.tags,
                analysis_blocks=distribution.analysis_blocks,
                analysis_tier=distribution.analysis_tier,
                include_in_family_summary=distribution.include_in_family_summary,
                nested_from=distribution.nested_from,
                resolved_distribution_path=str(distribution_path.relative_to(root)),
                distribution_sha256=_sha256_json(resolved),
                dataset_sha256=_dataset_hash(resolved),
            )
        )

    manifest_profiles: list[SuiteManifestRunProfile] = []
    for profile_id, profile in spec.run_profiles.items():
        profile_root = root / 'run_profiles' / profile_id
        profile_path = profile_root / 'resolved_run_profile.yaml'
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(yaml.safe_dump(profile.config, sort_keys=False))
        profile_factors = {
            str(key): str(value) for key, value in {**profile.metadata, **profile.factors}.items()
        }
        _write_json(
            profile_root / 'run_profile_metadata.json',
            {
                'layout_version': SUITE_LAYOUT_VERSION,
                'run_profile_id': profile_id,
                'factors': profile_factors,
            },
        )
        manifest_profiles.append(
            SuiteManifestRunProfile(
                run_profile_id=profile_id,
                factors=profile_factors,
                resolved_run_profile_path=str(profile_path.relative_to(root)),
                run_profile_sha256=_sha256_json(profile.config),
            )
        )

    cells: list[SuiteManifestCell] = []
    for cell in spec.expanded_cells():
        assert cell.run_profile_id is not None
        distribution = spec.distributions[cell.distribution_id]
        profile = spec.run_profiles[cell.run_profile_id]
        resolved = validation.resolved_configs[cell.cell_id]
        cfg = ExperimentCfg.model_validate(resolved)
        dist_root = root / 'distributions' / cell.distribution_id
        run_root = dist_root / 'runs' / cell.run_profile_id
        attempt_root = run_root / 'attempts' / 'initial'
        config_path = run_root / 'resolved_config.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        resolved_factors = {**_declared_composition(cfg), **distribution.factors, **cell.factors}
        evaluation = next(
            candidate
            for candidate in spec.evaluations
            if cell.distribution_id in candidate.distribution_ids
            and cell.run_profile_id in candidate.run_profile_ids
        )
        _write_json(
            run_root / 'cell_metadata.json',
            {
                'layout_version': SUITE_LAYOUT_VERSION,
                'manifest_version': MANIFEST_VERSION,
                'cell_id': cell.cell_id,
                'evaluation_id': evaluation.evaluation_id,
                'factors': resolved_factors,
                'tags': sorted(set(distribution.tags + evaluation.tags)),
                'analysis_blocks': distribution.analysis_blocks,
                'analysis_tier': distribution.analysis_tier,
                'nested_from': cell.nested_from,
                'run_profile_factors': {**profile.metadata, **profile.factors},
            },
        )
        data_root = dist_root / 'data' / 'schema-v5'
        cells.append(
            SuiteManifestCell(
                cell_id=cell.cell_id,
                name=f'{spec.suite_id}/{cell.distribution_id}/{cell.run_profile_id}',
                distribution_id=cell.distribution_id,
                distribution_base_id=cell.distribution_id,
                evaluation_id=evaluation.evaluation_id,
                run_profile_id=cell.run_profile_id,
                family_id=distribution.family_id,
                family_label=distribution.family_label,
                origin='native',
                dataset_schema_version=5,
                evaluation_schema_version=5,
                status='planned',
                # Scale, topology, and interaction cells remain causal
                # contrasts even when they are intentionally excluded from a
                # five-family descriptive aggregate.
                include_in_causal_summaries=True,
                include_in_family_summary=distribution.include_in_family_summary,
                factors=resolved_factors,
                tags=sorted(set(distribution.tags + evaluation.tags)),
                analysis_blocks=distribution.analysis_blocks,
                analysis_tier=distribution.analysis_tier,
                run_profile_factors={
                    str(key): str(value)
                    for key, value in {**profile.metadata, **profile.factors}.items()
                },
                data_root=str(data_root.relative_to(root)),
                attempt_root=str(attempt_root.relative_to(root)),
                resolved_config_path=str(config_path.relative_to(root)),
                config_sha256=_sha256_json(resolved),
                dataset_sha256=_dataset_hash(resolved),
                run_profile_sha256=_sha256_json(profile.config),
                nested_from=cell.nested_from,
            )
        )
    manifest = SuiteManifest(
        manifest_version=MANIFEST_VERSION,
        suite_id=spec.suite_id,
        origin='native',
        created_at=datetime.now(UTC).isoformat(),
        distributions=distributions,
        run_profiles=manifest_profiles,
        evaluations=cells,
        cells=cells,
        comparison_groups=spec.comparison_groups,
        analysis_series=spec.analysis_series,
    )
    _write_json(manifest_path, manifest.model_dump(mode='json'))
    return manifest


def _materialize_derived_suite(
    *,
    spec: SuiteSpec,
    results_dir: Path,
    replace_planned: bool,
    prune_removed_embeddings: bool,
) -> SuiteManifest:
    """Materialize model-specific evaluations against immutable source data."""
    assert spec.origin == 'derived' and spec.source is not None
    root = suite_root(results_dir, spec.suite_id)
    manifest_path = root / 'suite_manifest.json'
    if manifest_path.exists():
        existing = load_suite_manifest(results_dir, spec.suite_id)
        if prune_removed_embeddings:
            return _prune_removed_derived_embeddings(root=root, existing=existing, spec=spec)
        if not replace_planned:
            return existing
        _remove_planned_derived_suite(root, existing)

    source_root, source_manifest = load_pinned_source_manifest(
        results_dir=results_dir, source=spec.source
    )
    source_distributions = {
        distribution.distribution_id: distribution for distribution in source_manifest.distributions
    }
    requested_distributions = set(spec.source.distribution_ids)
    missing_distributions = sorted(requested_distributions - set(source_distributions))
    if missing_distributions:
        raise ValueError(
            f'{spec.suite_id}: source distributions are missing: {missing_distributions}'
        )

    source_cells = [
        cell for cell in source_manifest.cells if cell.distribution_id in requested_distributions
    ]
    if not source_cells:
        raise ValueError(f'{spec.suite_id}: source selection produced no cells')
    incomplete_cells = [cell.cell_id for cell in source_cells if cell.status != 'completed']
    if incomplete_cells:
        raise ValueError(f'{spec.suite_id}: source cells are incomplete: {incomplete_cells[:8]}')
    selected_counts = {
        distribution_id: sum(cell.distribution_id == distribution_id for cell in source_cells)
        for distribution_id in requested_distributions
    }
    if any(count == 0 for count in selected_counts.values()):
        raise ValueError(f'{spec.suite_id}: selected source distribution has no completed profiles')
    for source_cell in source_cells:
        _validate_source_generation_artifacts(source_root=source_root, source_cell=source_cell)

    from datetime import UTC, datetime

    distributions: list[SuiteManifestDistribution] = []
    for distribution_id in spec.source.distribution_ids:
        source_distribution = source_distributions[distribution_id]
        source_path = _safe_relative(source_root, source_distribution.resolved_distribution_path)
        target_path = root / 'distributions' / distribution_id / 'resolved_distribution.yaml'
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(source_path.read_text())
        distributions.append(
            source_distribution.model_copy(
                update={'resolved_distribution_path': str(target_path.relative_to(root))}
            )
        )

    source_profiles = {profile.run_profile_id: profile for profile in source_manifest.run_profiles}
    profile_ids = {cell.run_profile_id for cell in source_cells}
    missing_profiles = sorted(profile_ids - set(source_profiles))
    if missing_profiles:
        raise ValueError(f'{spec.suite_id}: source profiles are missing: {missing_profiles}')
    manifest_profiles: list[SuiteManifestRunProfile] = []
    derived_profile_hashes: dict[str, str] = {}
    profile_maps: dict[str, dict[str, str]] = {}
    for model_name in spec.source.embedding_models:
        model_token = embedding_child_token(model_name)
        profile_map = {
            profile_id: _derived_run_profile_id(profile_id, model_token)
            for profile_id in profile_ids
        }
        if len(set(profile_map.values())) != len(profile_map):
            raise ValueError(f'{spec.suite_id}: derived run-profile IDs collide for {model_name}')
        profile_maps[model_name] = profile_map
        for source_profile_id in sorted(profile_ids):
            source_profile = source_profiles[source_profile_id]
            source_path = _safe_relative(source_root, source_profile.resolved_run_profile_path)
            raw_profile = _read_yaml_mapping(source_path)
            raw_embeddings = raw_profile.setdefault('embeddings', {})
            if not isinstance(raw_embeddings, dict):
                raise ValueError(f'{source_profile_id}: source embeddings profile is not a mapping')
            raw_embeddings['model_name'] = model_name
            derived_profile_id = profile_map[source_profile_id]
            target_path = root / 'run_profiles' / derived_profile_id / 'resolved_run_profile.yaml'
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(yaml.safe_dump(raw_profile, sort_keys=False))
            factors = {**source_profile.factors, 'embedding': model_name}
            _write_json(
                target_path.parent / 'run_profile_metadata.json',
                {
                    'layout_version': SUITE_LAYOUT_VERSION,
                    'run_profile_id': derived_profile_id,
                    'factors': factors,
                    'source_run_profile_id': source_profile_id,
                },
            )
            profile_hash = _sha256_json(raw_profile)
            manifest_profiles.append(
                SuiteManifestRunProfile(
                    run_profile_id=derived_profile_id,
                    factors=factors,
                    resolved_run_profile_path=str(target_path.relative_to(root)),
                    run_profile_sha256=profile_hash,
                )
            )
            derived_profile_hashes[derived_profile_id] = profile_hash

    cells: list[SuiteManifestCell] = []
    for model_name, profile_map in profile_maps.items():
        model_token = embedding_child_token(model_name)
        for source_cell in sorted(source_cells, key=lambda item: item.cell_id):
            source_config_path = _safe_relative(source_root, source_cell.resolved_config_path)
            raw_config = _read_yaml_mapping(source_config_path)
            raw_embeddings = raw_config.setdefault('embeddings', {})
            if not isinstance(raw_embeddings, dict):
                raise ValueError(
                    f'{source_cell.cell_id}: source embeddings config is not a mapping'
                )
            raw_embeddings['model_name'] = model_name
            ExperimentCfg.model_validate(raw_config)
            if _dataset_hash(raw_config) != source_cell.dataset_sha256:
                raise ValueError(f'{source_cell.cell_id}: source dataset hash is stale')
            derived_profile_id = profile_map[source_cell.run_profile_id]
            cell_id = f'{source_cell.distribution_id}__{derived_profile_id}'
            run_root = (
                root / 'distributions' / source_cell.distribution_id / 'runs' / derived_profile_id
            )
            config_path = run_root / 'resolved_config.yaml'
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False))
            _write_json(
                run_root / 'cell_metadata.json',
                {
                    'layout_version': SUITE_LAYOUT_VERSION,
                    'manifest_version': MANIFEST_VERSION,
                    'cell_id': cell_id,
                    'source_suite_id': spec.source.suite_id,
                    'source_cell_id': source_cell.cell_id,
                    'source_manifest_sha256': spec.source.manifest_sha256,
                    'source_dataset_sha256': source_cell.dataset_sha256,
                },
            )
            cells.append(
                SuiteManifestCell(
                    cell_id=cell_id,
                    name=f'{spec.suite_id}/{source_cell.distribution_id}/{derived_profile_id}',
                    distribution_id=source_cell.distribution_id,
                    distribution_base_id=source_cell.distribution_base_id,
                    evaluation_id=f'{source_cell.evaluation_id}__{model_token}',
                    run_profile_id=derived_profile_id,
                    family_id=source_cell.family_id,
                    family_label=source_cell.family_label,
                    origin='derived',
                    dataset_schema_version=source_cell.dataset_schema_version,
                    evaluation_schema_version=source_cell.evaluation_schema_version,
                    status='planned',
                    include_in_causal_summaries=source_cell.include_in_causal_summaries,
                    include_in_family_summary=source_cell.include_in_family_summary,
                    factors=dict(source_cell.factors),
                    tags=list(source_cell.tags),
                    analysis_blocks=list(source_cell.analysis_blocks),
                    analysis_tier=source_cell.analysis_tier,
                    run_profile_factors={
                        **source_cell.run_profile_factors,
                        'embedding': model_name,
                    },
                    data_root=source_cell.data_root,
                    attempt_root=str((run_root / 'attempts' / 'initial').relative_to(root)),
                    resolved_config_path=str(config_path.relative_to(root)),
                    config_sha256=_sha256_json(raw_config),
                    dataset_sha256=_dataset_hash(raw_config),
                    run_profile_sha256=derived_profile_hashes[derived_profile_id],
                    source_suite_id=spec.source.suite_id,
                    source_cell_id=source_cell.cell_id,
                    source_manifest_sha256=spec.source.manifest_sha256,
                    source_dataset_sha256=source_cell.dataset_sha256,
                )
            )
    analysis_series = [
        series.model_copy(
            update={
                'series_id': f'{series.series_id}__{embedding_child_token(model_name)}',
                'points': [
                    point.model_copy(
                        update={'run_profile_id': profile_maps[model_name][point.run_profile_id]}
                    )
                    for point in series.points
                ],
            }
        )
        for model_name in spec.source.embedding_models
        for series in source_manifest.analysis_series
    ]
    comparison_groups = list(source_manifest.comparison_groups)
    manifest = SuiteManifest(
        manifest_version=MANIFEST_VERSION,
        suite_id=spec.suite_id,
        origin='derived',
        created_at=datetime.now(UTC).isoformat(),
        source=spec.source,
        distributions=distributions,
        run_profiles=manifest_profiles,
        evaluations=cells,
        cells=cells,
        comparison_groups=comparison_groups,
        analysis_series=analysis_series,
    )
    _write_json(manifest_path, manifest.model_dump(mode='json'))
    return manifest


def _prune_removed_derived_embeddings(
    *, root: Path, existing: SuiteManifest, spec: SuiteSpec
) -> SuiteManifest:
    if existing.origin != 'derived' or existing.source is None or spec.source is None:
        raise ValueError('embedding pruning requires an existing derived suite and source contract')
    existing_source = existing.source
    desired_source = spec.source
    if (
        existing_source.suite_id != desired_source.suite_id
        or existing_source.manifest_sha256 != desired_source.manifest_sha256
        or existing_source.distribution_ids != desired_source.distribution_ids
    ):
        raise ValueError('embedding pruning requires an unchanged pinned source and distributions')

    existing_models = set(existing_source.embedding_models)
    desired_models = set(desired_source.embedding_models)
    unknown_models = desired_models - existing_models
    removed_models = existing_models - desired_models
    if unknown_models:
        raise ValueError(f'embedding pruning cannot add models: {sorted(unknown_models)}')
    if not removed_models:
        raise ValueError('embedding pruning requires at least one removed embedding model')

    retained_cells = [
        cell for cell in existing.cells if cell.run_profile_factors.get('embedding') in desired_models
    ]
    removed_cells = [
        cell for cell in existing.cells if cell.run_profile_factors.get('embedding') in removed_models
    ]
    if len(retained_cells) + len(removed_cells) != len(existing.cells):
        raise ValueError('derived manifest contains a cell with an unknown embedding model')
    if not retained_cells or not removed_cells:
        raise ValueError('embedding pruning produced an invalid empty arm')
    retained_profiles = [
        profile
        for profile in existing.run_profiles
        if profile.factors.get('embedding') in desired_models
    ]
    removed_profiles = [
        profile
        for profile in existing.run_profiles
        if profile.factors.get('embedding') in removed_models
    ]
    if len(retained_profiles) + len(removed_profiles) != len(existing.run_profiles):
        raise ValueError('derived manifest contains a run profile with an unknown embedding model')

    root_resolved = root.resolve()
    paths_to_remove = [
        root / profile.resolved_run_profile_path for profile in removed_profiles
    ] + [
        root / cell.resolved_config_path for cell in removed_cells
    ]
    removable_roots = {path.parent for path in paths_to_remove}
    for path in removable_roots:
        if not path.resolve().is_relative_to(root_resolved):
            raise ValueError(f'refusing to remove a path outside the suite root: {path}')
    for path in sorted(removable_roots, key=lambda item: len(item.parts), reverse=True):
        if path.exists():
            shutil.rmtree(path)

    removed_tokens = {embedding_child_token(model_name) for model_name in removed_models}
    retained_series = [
        series
        for series in existing.analysis_series
        if not any(series.series_id.endswith(f'__{token}') for token in removed_tokens)
    ]
    pruned = existing.model_copy(
        update={
            'source': desired_source,
            'cells': retained_cells,
            'evaluations': retained_cells,
            'run_profiles': retained_profiles,
            'analysis_series': retained_series,
        }
    )
    _write_json(root / 'suite_manifest.json', pruned.model_dump(mode='json'))
    return pruned


def _derived_run_profile_id(source_profile_id: str, model_token: str) -> str:
    _, separator, suffix = source_profile_id.partition('_')
    return f'{model_token}_{suffix}' if separator else f'{model_token}_{source_profile_id}'


def _validate_source_generation_artifacts(
    *, source_root: Path, source_cell: SuiteManifestCell
) -> None:
    """Fail before downstream work when a pinned source is incomplete on disk."""
    from experiments.medical_dataset_gen.suites.runtime import (
        load_cell_config,
        suite_paths_for_cell,
    )

    cfg = load_cell_config(source_root, source_cell)
    paths = suite_paths_for_cell(root=source_root, cell=source_cell, cfg=cfg)
    missing = [
        table_name
        for table_name in SHARED_GENERATION_TABLES
        if not paths.table_path(table_name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f'{source_cell.cell_id}: source generation artifacts are missing: {missing}'
        )


def _remove_planned_derived_suite(root: Path, manifest: SuiteManifest) -> None:
    """Replace only a metadata-only derived suite before any model run begins."""
    if manifest.origin != 'derived' or any(cell.status != 'planned' for cell in manifest.cells):
        raise ValueError('can replace only a wholly planned derived suite')
    artifact_suffixes = {'.parquet', '.npy', '.npz'}
    forbidden_names = {'artifact_manifest.json', 'attempt_metadata.json'}
    artifacts = [
        path
        for path in root.rglob('*')
        if path.is_file() and (path.suffix in artifact_suffixes or path.name in forbidden_names)
    ]
    if artifacts:
        example = artifacts[0].relative_to(root)
        raise ValueError(f'cannot replace derived suite with artifacts: {example}')
    if root.name != manifest.suite_id or root.parent.name != 'suites':
        raise ValueError(f'refusing to remove non-suite root: {root}')
    shutil.rmtree(root)


def _remove_planned_native_suite(root: Path, manifest: SuiteManifest) -> None:
    """Delete only a known metadata-only native suite after strict checks."""
    if manifest.origin != 'native' or any(cell.status != 'planned' for cell in manifest.cells):
        raise ValueError('can replace only a wholly planned native suite')
    artifact_suffixes = {'.parquet', '.npy', '.npz'}
    forbidden_names = {'artifact_manifest.json', 'attempt_metadata.json'}
    artifacts = [
        path
        for path in root.rglob('*')
        if path.is_file() and (path.suffix in artifact_suffixes or path.name in forbidden_names)
    ]
    if artifacts:
        examples = ', '.join(str(path.relative_to(root)) for path in artifacts[:5])
        raise ValueError(f'cannot replace native suite with artifacts: {examples}')
    if root.name != manifest.suite_id or root.parent.name != 'suites':
        raise ValueError(f'refusing to remove non-suite root: {root}')
    shutil.rmtree(root)


def _prune_metadata_only_planned_cells(
    *,
    root: Path,
    existing: SuiteManifest,
    spec: SuiteSpec,
    validation: ValidationResult,
) -> None:
    """Remove obsolete planned metadata without touching generated data.

    This deliberately supports only subtraction from an all-planned suite.  It
    is for correcting a planned design after a smoke dataset has been created:
    retained distributions and cells must resolve byte-for-byte identically,
    and every removed distribution must contain metadata/configuration files
    only.  It cannot add or alter an evaluation.
    """
    if existing.origin != 'native' or any(cell.status != 'planned' for cell in existing.cells):
        raise ValueError('can prune only a wholly planned native suite')

    existing_distributions = {item.distribution_id: item for item in existing.distributions}
    desired_distribution_ids = set(spec.distributions)
    removed_distribution_ids = set(existing_distributions) - desired_distribution_ids
    if not removed_distribution_ids:
        raise ValueError('planned prune requires at least one removed distribution')
    if desired_distribution_ids - set(existing_distributions):
        raise ValueError('planned prune cannot add distributions')

    existing_cells = {cell.cell_id: cell for cell in existing.cells}
    desired_cells = validation.resolved_configs
    if set(desired_cells) - set(existing_cells):
        raise ValueError('planned prune cannot add evaluation cells')
    retained_cell_ids = set(existing_cells) & set(desired_cells)
    if any(
        existing_cells[cell_id].distribution_id in removed_distribution_ids
        for cell_id in retained_cell_ids
    ):
        raise ValueError('planned prune cannot retain a cell for a removed distribution')
    if any(
        cell.distribution_id not in removed_distribution_ids
        for cell_id, cell in existing_cells.items()
        if cell_id not in desired_cells
    ):
        raise ValueError('planned prune may remove cells only through removed distributions')

    for distribution_id in desired_distribution_ids:
        existing_distribution = existing_distributions[distribution_id]
        resolved = validation.resolved_distributions[distribution_id]
        if existing_distribution.distribution_sha256 != _sha256_json(resolved):
            raise ValueError(
                f'{distribution_id}: planned prune would alter its distribution configuration'
            )
        if existing_distribution.dataset_sha256 != _dataset_hash(resolved):
            raise ValueError(
                f'{distribution_id}: planned prune would alter its dataset configuration'
            )
    for cell_id in retained_cell_ids:
        existing_cell = existing_cells[cell_id]
        resolved = desired_cells[cell_id]
        if existing_cell.config_sha256 != _sha256_json(resolved):
            raise ValueError(f'{cell_id}: planned prune would alter its resolved configuration')
        if existing_cell.dataset_sha256 != _dataset_hash(resolved):
            raise ValueError(f'{cell_id}: planned prune would alter its dataset configuration')

    artifact_suffixes = {'.parquet', '.npy', '.npz'}
    forbidden_names = {'artifact_manifest.json', 'attempt_metadata.json'}
    for distribution_id in sorted(removed_distribution_ids):
        distribution_root = root / 'distributions' / distribution_id
        if not distribution_root.is_dir():
            raise ValueError(f'missing planned distribution directory: {distribution_root}')
        artifacts = [
            path
            for path in distribution_root.rglob('*')
            if path.is_file() and (path.suffix in artifact_suffixes or path.name in forbidden_names)
        ]
        if artifacts:
            example = artifacts[0].relative_to(root)
            raise ValueError(f'cannot prune planned distribution with artifacts: {example}')
        shutil.rmtree(distribution_root)


def _validate_planned_execution_refresh(
    *, root: Path, existing: SuiteManifest, validation: ValidationResult
) -> None:
    """Permit a run-profile-only refresh while preserving generated v5 data.

    This is intentionally narrower than replacement.  It makes an execution
    setting such as a device choice reproducible in the snapshots before any
    completed evaluation exists, but refuses a change that could invalidate
    generated facts, chunks, qrels, or an immutable attempt.
    """
    if existing.origin != 'native' or any(cell.status != 'planned' for cell in existing.cells):
        raise ValueError('can refresh execution profiles only for a wholly planned native suite')
    attempt_artifacts = [
        path
        for path in (root / 'distributions').glob('*/runs/*/attempts/**/*')
        if path.is_file()
        and (path.suffix in {'.npy', '.npz', '.parquet'} or path.name.endswith('.json'))
    ]
    if attempt_artifacts:
        example = attempt_artifacts[0].relative_to(root)
        raise ValueError(
            'cannot refresh execution profiles while attempt artifacts exist; '
            f'archive or remove the incomplete attempt first ({example})'
        )
    existing_cells = {cell.cell_id: cell for cell in existing.cells}
    if set(existing_cells) != set(validation.resolved_configs):
        raise ValueError('cannot refresh execution profiles after the suite cell set changes')
    for cell_id, raw in validation.resolved_configs.items():
        if _dataset_hash(raw) != existing_cells[cell_id].dataset_sha256:
            raise ValueError(
                f'{cell_id}: refresh would change dataset configuration; '
                'use a new suite or replace an artifact-free suite instead'
            )


def _apply_transform(raw: dict[str, Any], transform: SuiteTransform) -> None:
    pools = cast(dict[str, Any], raw['generation']['chunk_pools'])
    if transform.op == 'set_gold_mass_vector':
        assert transform.masses is not None and transform.niche_count is not None
        dominant, other, secondary_a, secondary_b = transform.masses
        pools['dominant_primary'] = _with_mass(pools['dominant_primary'], dominant)
        pools['other_primary'] = _with_mass(pools['other_primary'], other)
        if transform.niche_count == 0:
            if secondary_a != secondary_b:
                raise ValueError('zero niche facets require equal final two gold masses')
            pools['secondary'] = _with_mass(pools['secondary'], secondary_a)
            pools['niche'] = _with_mass(pools['niche'], 1)
            pools['niche']['num_clusters_per_query'] = 0
        elif transform.niche_count == 1:
            pools['secondary'] = _with_mass(pools['secondary'], secondary_a)
            pools['niche'] = _with_mass(pools['niche'], secondary_b)
            pools['niche']['num_clusters_per_query'] = 1
        else:
            if secondary_a != secondary_b:
                raise ValueError('two niche facets require equal final two gold masses')
            # The generator keeps a dormant ``secondary`` template even when
            # both complementary facets are niche facets, and validates that
            # a niche template is smaller.  Its support is not materialized
            # when ``num_clusters_per_query=2``; set a one-chunk sentinel
            # above the niche support solely to satisfy that invariant.
            pools['secondary'] = _with_mass(pools['secondary'], secondary_a + 1)
            pools['niche'] = _with_mass(pools['niche'], secondary_a)
            pools['niche']['num_clusters_per_query'] = 2
        return
    if transform.op == 'set_near_miss_mass':
        assert transform.mass is not None
        raw['generation']['near_miss_specs'] = _resize_near_miss_specs(
            raw['generation'].get('near_miss_specs') or _default_near_miss_specs(), transform.mass
        )
        return
    if transform.op == 'set_near_miss_mix':
        assert transform.one_change_fraction is not None
        mass = _near_miss_mass(
            raw['generation'].get('near_miss_specs') or _default_near_miss_specs()
        )
        raw['generation']['near_miss_specs'] = _near_miss_mix_specs(
            mass, transform.one_change_fraction
        )
        return
    if transform.op == 'set_background_mass':
        assert transform.mass is not None
        if transform.mass == 0:
            pools['background_outliers'] = []
            return
        current = pools['background_outliers'][0]
        clusters = transform.num_clusters or int(current.get('num_clusters', 1))
        if transform.mass % clusters:
            raise ValueError('background mass must be divisible by num_clusters')
        pools['background_outliers'] = [
            {**current, 'num_clusters': clusters, 'chunks_per_cluster': transform.mass // clusters}
        ]
        return
    if transform.op == 'set_background_components':
        assert transform.components is not None
        pools['background_outliers'] = deepcopy(transform.components)
        return
    if transform.op == 'set_structural_shell':
        assert transform.shell is not None
        changes = {
            'near': ['subgroup'],
            'intermediate': ['condition', 'subgroup'],
            'far': ['condition', 'subgroup', 'axis'],
        }[transform.shell]
        pools['background_outliers'] = [
            {**component, 'changes': changes} for component in pools['background_outliers']
        ]
        return
    if transform.op == 'set_cluster_topology':
        assert transform.component is not None
        assert transform.num_clusters is not None and transform.chunks_per_cluster is not None
        if transform.component == 'background':
            if len(pools['background_outliers']) != 1:
                raise ValueError('background topology requires exactly one background component')
            pools['background_outliers'] = [
                {
                    **pools['background_outliers'][0],
                    'num_clusters': transform.num_clusters,
                    'chunks_per_cluster': transform.chunks_per_cluster,
                }
            ]
        elif transform.component == 'near_miss':
            specs = raw['generation'].get('near_miss_specs') or _default_near_miss_specs()
            requested_total = transform.num_clusters * transform.chunks_per_cluster
            if _near_miss_mass(specs) != requested_total:
                raise ValueError('near-miss mass must equal requested topology mass')
            if transform.num_clusters % len(specs):
                raise ValueError(
                    'near-miss topology must allocate clusters equally across active types'
                )
            clusters_per_spec = transform.num_clusters // len(specs)
            raw['generation']['near_miss_specs'] = [
                {
                    **spec,
                    'num_clusters': clusters_per_spec,
                    'chunks_per_cluster': transform.chunks_per_cluster,
                }
                for spec in specs
            ]
        return
    if transform.op == 'scale_support':
        assert transform.multiplier is not None
        pool_names = ['dominant_primary', 'other_primary', 'secondary']
        if int(pools['niche'].get('num_clusters_per_query', 0)) > 0:
            pool_names.append('niche')
        for pool_name in pool_names:
            _scale_pool(
                pools[pool_name],
                transform.multiplier,
                transform.preserve_cluster_count,
                total_size=True,
            )
        for component in pools['background_outliers']:
            _scale_pool(
                component, transform.multiplier, transform.preserve_cluster_count, total_size=False
            )
        for spec in raw['generation'].get('near_miss_specs') or []:
            _scale_pool(
                spec, transform.multiplier, transform.preserve_cluster_count, total_size=True
            )
        return
    raise ValueError(f'unsupported transform {transform.op!r}')


def _with_mass(pool: Mapping[str, Any], mass: int) -> dict[str, Any]:
    clusters = int(pool.get('num_clusters', 1))
    if mass % clusters:
        clusters = 1
    return {**pool, 'num_clusters': clusters, 'chunks_per_cluster': mass // clusters, 'size': mass}


def _scale_pool(
    pool: dict[str, Any], multiplier: float, preserve_cluster_count: bool, *, total_size: bool
) -> None:
    fraction = Fraction(str(multiplier))
    clusters = int(pool.get('num_clusters', 1))
    per_cluster = int(pool.get('chunks_per_cluster', pool.get('size', 1)))
    if preserve_cluster_count:
        scaled = Fraction(per_cluster) * fraction
        if scaled.denominator != 1 or scaled < 1:
            raise ValueError(
                f'exact scale_support requires integral chunks_per_cluster: {per_cluster} x {multiplier}'
            )
        pool['num_clusters'] = clusters
        pool['chunks_per_cluster'] = int(scaled)
    else:
        scaled = Fraction(clusters) * fraction
        if scaled.denominator != 1 or scaled < 1:
            raise ValueError(
                f'exact scale_support requires integral cluster count: {clusters} x {multiplier}'
            )
        pool['num_clusters'] = int(scaled)
        pool['chunks_per_cluster'] = per_cluster
    if total_size:
        pool['size'] = int(pool['num_clusters']) * int(pool['chunks_per_cluster'])
    else:
        # Background's legacy ``size`` has per-cluster semantics.
        pool['size'] = int(pool['chunks_per_cluster'])


def _default_near_miss_specs() -> list[dict[str, Any]]:
    return [
        {'changes': ['condition'], 'num_clusters': 1, 'chunks_per_cluster': 4},
        {'changes': ['subgroup'], 'num_clusters': 1, 'chunks_per_cluster': 4},
        {'changes': ['condition', 'axis_value_bin'], 'num_clusters': 1, 'chunks_per_cluster': 4},
        {'changes': ['subgroup', 'axis_value_bin'], 'num_clusters': 1, 'chunks_per_cluster': 4},
    ]


def _near_miss_mass(specs: Iterable[Mapping[str, Any]]) -> int:
    return sum(
        int(spec.get('num_clusters', 1)) * int(spec.get('chunks_per_cluster', spec.get('size', 1)))
        for spec in specs
    )


def _resize_near_miss_specs(specs: list[dict[str, Any]], mass: int) -> list[dict[str, Any]]:
    if mass == 0:
        return []
    total_clusters = sum(int(spec.get('num_clusters', 1)) for spec in specs)
    if not total_clusters or mass % total_clusters:
        raise ValueError(
            f'near-miss mass {mass} must divide evenly across {total_clusters} stable clusters'
        )
    per_cluster = mass // total_clusters
    return [
        {
            **spec,
            'chunks_per_cluster': per_cluster,
            'size': per_cluster * int(spec.get('num_clusters', 1)),
        }
        for spec in specs
    ]


def _near_miss_mix_specs(mass: int, one_change_fraction: float) -> list[dict[str, Any]]:
    if mass == 0:
        return []
    total_clusters = 4
    one_clusters = int(total_clusters * one_change_fraction)
    if one_clusters not in {0, 2, 4} or mass % total_clusters:
        raise ValueError(
            'near-miss mix needs 0%, 50%, or 100% one-change mass at four fixed clusters'
        )
    per_cluster = mass // total_clusters
    groups = [
        (['condition'], one_clusters // 2),
        (['subgroup'], one_clusters // 2),
        (['condition', 'axis_value_bin'], (total_clusters - one_clusters) // 2),
        (['subgroup', 'axis_value_bin'], (total_clusters - one_clusters) // 2),
    ]
    return [
        {'changes': changes, 'num_clusters': clusters, 'chunks_per_cluster': per_cluster}
        for changes, clusters in groups
        if clusters
    ]


def _validate_comparison_groups(
    spec: SuiteSpec,
    cells: Sequence[SuiteCell],
    resolved: Mapping[str, Mapping[str, Any]],
) -> None:
    by_distribution_profile = {(cell.distribution_id, cell.run_profile_id): cell for cell in cells}
    by_id = {cell.cell_id: cell for cell in cells}
    for group in spec.comparison_groups:
        if group.distribution_ids:
            profiles = (
                set(group.run_profile_ids)
                if group.run_profile_ids
                else set.intersection(
                    *(
                        spec.profiles_for_distribution(identifier)
                        for identifier in group.distribution_ids
                    )
                )
            )
            grouped_cells = [
                by_distribution_profile[(distribution_id, profile)]
                for profile in sorted(profiles)
                for distribution_id in group.distribution_ids
            ]
        else:
            grouped_cells = [by_id[cell_id] for cell_id in group.cells]
        factors = group.all_varying_factors
        for factor in factors:
            values = {
                _canonical_json(_comparison_factor_value(group, cell, factor))
                for cell in grouped_cells
            }
            if any(_comparison_factor_value(group, cell, factor) is None for cell in grouped_cells):
                raise ValueError(f'{group.comparison_id}: a cell lacks varying factor {factor!r}')
            if group.strict and len(values) < 2:
                raise ValueError(
                    f'{group.comparison_id}: varying factor {factor!r} has fewer than two values'
                )
            declared_order = group.factor_levels.get(factor)
            if declared_order is not None and set(map(_canonical_json, declared_order)) != values:
                raise ValueError(
                    f'{group.comparison_id}: declared factor_levels differ from the selected cells'
                )
        for factor in group.matching_factors:
            values = {_canonical_json(cell.factors.get(factor)) for cell in grouped_cells}
            if len(values) != 1:
                raise ValueError(f'{group.comparison_id}: matching factor {factor!r} is confounded')
        for profile in {cell.run_profile_id for cell in grouped_cells}:
            profile_cells = [cell for cell in grouped_cells if cell.run_profile_id == profile]
            _validate_group_factor_cross(group, profile_cells)
            _validate_config_diffs(group, profile_cells, resolved)


def _validate_group_factor_cross(group: ComparisonGroup, cells: Sequence[SuiteCell]) -> None:
    """Require every strict one- or two-factor design to be complete exactly once."""
    if not group.strict:
        return
    factors = group.all_varying_factors
    observed = [
        tuple(_canonical_json(_comparison_factor_value(group, cell, factor)) for factor in factors)
        for cell in cells
    ]
    if len(observed) != len(set(observed)):
        raise ValueError(f'{group.comparison_id}: duplicate factor combination in one run profile')
    expected = [
        tuple(_canonical_json(value) for value in values)
        for values in _factor_product(group, factors)
    ]
    if set(observed) != set(expected):
        raise ValueError(f'{group.comparison_id}: incomplete declared factor cross')


def _factor_product(group: ComparisonGroup, factors: Sequence[str]) -> list[tuple[Any, ...]]:
    if len(factors) == 1:
        return [(value,) for value in group.factor_levels.get(factors[0], [])]
    first, second = factors
    return [
        (first_value, second_value)
        for first_value in group.factor_levels.get(first, [])
        for second_value in group.factor_levels.get(second, [])
    ]


def _comparison_factor_value(group: ComparisonGroup, cell: SuiteCell, factor: str) -> object:
    """Use an explicit group reference level for a reusable control cell.

    A balanced reference participates in several ladders. Requiring it to
    carry unrelated labels such as ``dominance_level=control`` and
    ``background_mass=16`` would put analysis semantics back into the dataset
    definition, so only a comparison supplies those labels.
    """
    return cell.factors.get(factor, group.reference_levels.get(factor))


def _validate_config_diffs(
    group: ComparisonGroup, cells: Sequence[SuiteCell], resolved: Mapping[str, Mapping[str, Any]]
) -> None:
    if len(cells) < 2:
        return
    paths = [
        path for factor in group.all_varying_factors for path in group.owned_paths.get(factor, [])
    ]
    if not paths:
        raise ValueError(f'{group.comparison_id}: every varying factor needs owned_paths')
    reference = _flatten_for_diff(resolved[cells[0].cell_id])
    for cell in cells[1:]:
        candidate = _flatten_for_diff(resolved[cell.cell_id])
        all_paths = set(reference) | set(candidate)
        illegal = sorted(
            path
            for path in all_paths
            if reference.get(path) != candidate.get(path)
            and not any(path == allowed or path.startswith(f'{allowed}.') for allowed in paths)
        )
        if illegal:
            raise ValueError(
                f'{group.comparison_id}: undeclared configuration drift for {cell.cell_id}: {illegal[:5]}'
            )


def _flatten_for_diff(value: object, prefix: str = '') -> dict[str, object]:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if prefix == 'global' and key == 'output_experiment':
                continue
            result.update(_flatten_for_diff(child, f'{prefix}.{key}' if prefix else str(key)))
        return result
    if isinstance(value, list):
        result: dict[str, object] = {}
        for index, child in enumerate(value):
            result.update(_flatten_for_diff(child, f'{prefix}.{index}'))
        return result
    return {prefix: value}


def _validate_family_semantics(
    spec: SuiteSpec, configs: Mapping[str, Mapping[str, Any]], warnings: list[str]
) -> None:
    allowed = {
        'balanced_clean',
        'dominance',
        'sparse_niche',
        'near_miss_heavy',
        'background_variant',
        'interaction',
    }
    for cell in spec.expanded_cells():
        distribution = spec.distributions[cell.distribution_id]
        composition = _declared_composition(ExperimentCfg.model_validate(configs[cell.cell_id]))
        if distribution.family_id == 'balanced_clean':
            masses = composition['gold_mass_vector']
            if max(masses) / min(masses) > 1.35:
                raise ValueError(
                    f'{cell.cell_id}: balanced_clean gold masses are too uneven: {masses}'
                )
            if composition['near_miss_mass'] > composition['gold_mass']:
                raise ValueError(f'{cell.cell_id}: balanced_clean cannot be near-miss dominated')
        if distribution.family_id not in allowed:
            warnings.append(f'{cell.cell_id}: non-thesis family_id {distribution.family_id!r}')


def _declared_composition(cfg: ExperimentCfg) -> dict[str, Any]:
    pools = cfg.generation.chunk_pools
    niche_count = pools.niche.num_clusters_per_query
    if niche_count == 0:
        vector = [
            int(pools.dominant_primary.size or 0),
            int(pools.other_primary.size or 0),
            int(pools.secondary.size or 0),
            int(pools.secondary.size or 0),
        ]
    elif niche_count == 1:
        vector = [
            int(pools.dominant_primary.size or 0),
            int(pools.other_primary.size or 0),
            int(pools.secondary.size or 0),
            int(pools.niche.size or 0),
        ]
    else:
        vector = [
            int(pools.dominant_primary.size or 0),
            int(pools.other_primary.size or 0),
            int(pools.niche.size or 0),
            int(pools.niche.size or 0),
        ]
    specs = cfg.generation.near_miss_specs or []
    near_miss_topology = [
        {
            'changes': list(spec.changes),
            'num_clusters': int(spec.num_clusters),
            'chunks_per_cluster': int(spec.chunks_per_cluster or 0),
            'mass': int(spec.size or 0),
        }
        for spec in specs
    ]
    background_topology = [
        {
            'changes': list(spec.changes),
            'num_clusters': int(spec.num_clusters),
            'chunks_per_cluster': int(spec.chunks_per_cluster or 0),
            'mass': int(spec.size or 0) * int(spec.num_clusters),
        }
        for spec in pools.background_outliers
    ]
    near_miss = sum(row['mass'] for row in near_miss_topology)
    background = sum(row['mass'] for row in background_topology)
    return {
        'gold_mass_vector': vector,
        'gold_mass': sum(vector),
        'niche_count': niche_count,
        'near_miss_mass': near_miss,
        'near_miss_topology': near_miss_topology,
        'background_mass': background,
        'background_topology_components': background_topology,
        'pool_mass': sum(vector) + near_miss + background,
    }


def _dataset_hash(raw: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            'dataset_schema_version': raw.get('dataset_schema_version'),
            'global': {
                key: value
                for key, value in cast(Mapping[str, Any], raw.get('global', {})).items()
                if key not in {'output_experiment', 'result_dir_overrides'}
            },
            'generation': raw.get('generation'),
        }
    )


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'missing suite spec: {path}')
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'suite spec must be a mapping: {path}')
    return cast(dict[str, Any], raw)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: Path, raw_path: str) -> Path:
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'path escapes suite root: {raw_path!r}') from exc
    return path


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        merged[key] = (
            _deep_merge(cast(Mapping[str, Any], existing), cast(Mapping[str, Any], value))
            if isinstance(existing, Mapping) and isinstance(value, Mapping)
            else deepcopy(value)
        )
    return merged


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _validate_identifier(value: str, label: str) -> None:
    if not value or '/' in value or '\\' in value or value in {'.', '..'}:
        raise ValueError(f'{label} must be a simple non-empty identifier: {value!r}')
