"""Strict schema-v5 contracts for the frozen thesis experiment suites."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

SUITE_LAYOUT_VERSION = 5
MANIFEST_VERSION = 2
type SuiteOrigin = Literal['native', 'derived']


class SuiteModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SetGoldMassVector(SuiteModel):
    op: Literal['set_gold_mass_vector']
    masses: list[int] = Field(min_length=4, max_length=4)
    niche_count: int = Field(ge=0, le=2)

    @model_validator(mode='after')
    def _positive_masses(self) -> SetGoldMassVector:
        if any(mass < 1 for mass in self.masses):
            raise ValueError('set_gold_mass_vector requires four positive masses')
        return self


class SetNearMissMass(SuiteModel):
    op: Literal['set_near_miss_mass']
    mass: int = Field(ge=0)


class SetNearMissMix(SuiteModel):
    op: Literal['set_near_miss_mix']
    one_change_fraction: float = Field(ge=0.0, le=1.0)


class SetBackgroundTopology(SuiteModel):
    op: Literal['set_background_topology']
    mass: int = Field(ge=0)
    num_clusters: int | None = Field(default=None, ge=1)


class SetBackgroundComponents(SuiteModel):
    op: Literal['set_background_components']
    components: list[dict[str, Any]] = Field(min_length=1)


class SetNearMissTopology(SuiteModel):
    op: Literal['set_near_miss_topology']
    num_clusters: int = Field(ge=1)
    chunks_per_cluster: int = Field(ge=1)


class ScaleSupport(SuiteModel):
    op: Literal['scale_support']
    multiplier: float = Field(gt=0.0)


type SuiteTransform = Annotated[
    SetGoldMassVector
    | SetNearMissMass
    | SetNearMissMix
    | SetBackgroundTopology
    | SetBackgroundComponents
    | SetNearMissTopology
    | ScaleSupport,
    Field(discriminator='op'),
]


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


class RunProfile(SuiteModel):
    config: dict[str, Any]
    factors: dict[str, str] = Field(default_factory=dict)


class ComparisonGroup(SuiteModel):
    comparison_id: str
    distribution_ids: list[str] = Field(min_length=2)
    varying_factors: list[str] = Field(min_length=1, max_length=2)
    matching_factors: list[str] = Field(default_factory=list)
    factor_levels: dict[str, list[Any]] = Field(default_factory=dict)
    reference_levels: dict[str, Any] = Field(default_factory=dict)
    owned_paths: dict[str, list[str]] = Field(default_factory=dict)
    analysis_block: str


class DerivedEmbeddingOverrides(SuiteModel):
    batch_size: PositiveInt


class DerivedSuiteSource(SuiteModel):
    suite_id: str
    manifest_sha256: str = Field(min_length=64, max_length=64)
    distribution_ids: list[str] = Field(min_length=1)
    embedding_models: list[str] = Field(min_length=1)
    embedding_overrides: dict[str, DerivedEmbeddingOverrides] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _validate_source(self) -> DerivedSuiteSource:
        validate_identifier(self.suite_id, 'source suite_id')
        if len(self.distribution_ids) != len(set(self.distribution_ids)):
            raise ValueError('derived source distribution_ids must be unique')
        for distribution_id in self.distribution_ids:
            validate_identifier(distribution_id, 'source distribution_id')
        if len(self.embedding_models) != len(set(self.embedding_models)):
            raise ValueError('derived source embedding_models must be unique')
        unknown_overrides = set(self.embedding_overrides) - set(self.embedding_models)
        if unknown_overrides:
            raise ValueError(
                'derived source embedding_overrides reference undeclared models: '
                f'{sorted(unknown_overrides)}'
            )
        if any(character not in '0123456789abcdef' for character in self.manifest_sha256):
            raise ValueError('derived source manifest_sha256 must be a lowercase SHA-256 digest')
        return self


@dataclass(frozen=True)
class ExpandedCell:
    cell_id: str
    distribution_id: str
    run_profile_id: str
    factors: dict[str, Any]
    nested_from: str | None


class SuiteSpec(SuiteModel):
    layout_version: Literal[5] = SUITE_LAYOUT_VERSION
    suite_id: str
    origin: SuiteOrigin = 'native'
    dataset_schema_version: Literal[5]
    description: str = ''
    source: DerivedSuiteSource | None = None
    distributions: dict[str, Distribution] = Field(default_factory=dict)
    run_profiles: dict[str, RunProfile] = Field(default_factory=dict)
    comparison_groups: list[ComparisonGroup] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_references(self) -> SuiteSpec:
        validate_identifier(self.suite_id, 'suite_id')
        if self.origin == 'derived':
            if self.source is None:
                raise ValueError('derived suite requires a source contract')
            if self.distributions or self.run_profiles or self.comparison_groups:
                raise ValueError('derived suites inherit their scientific design from the source')
            return self
        if self.source is not None:
            raise ValueError('native suite must not declare a source contract')
        if not self.distributions or not self.run_profiles:
            raise ValueError('native suite requires distributions and run profiles')
        for distribution_id, distribution in self.distributions.items():
            validate_identifier(distribution_id, 'distribution_id')
            if (
                distribution.nested_from is not None
                and distribution.nested_from not in self.distributions
            ):
                raise ValueError(
                    f'{distribution_id}: unknown nested_from {distribution.nested_from!r}'
                )
            seen = {distribution_id}
            parent_id = distribution.nested_from
            while parent_id is not None:
                if parent_id in seen:
                    raise ValueError(f'{distribution_id}: nested_from contains a cycle')
                seen.add(parent_id)
                parent_id = self.distributions[parent_id].nested_from
        for profile_id in self.run_profiles:
            validate_identifier(profile_id, 'run_profile_id')
        comparison_ids = [group.comparison_id for group in self.comparison_groups]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError('native suite contains duplicate comparison IDs')
        for group in self.comparison_groups:
            validate_identifier(group.comparison_id, 'comparison_id')
            unknown = sorted(set(group.distribution_ids) - set(self.distributions))
            if unknown:
                raise ValueError(f'{group.comparison_id}: unknown distributions {unknown}')
        return self

    def expanded_cells(self) -> list[ExpandedCell]:
        profiles = tuple(self.run_profiles)
        return [
            ExpandedCell(
                cell_id=f'{distribution_id}__{profile_id}',
                distribution_id=distribution_id,
                run_profile_id=profile_id,
                factors=dict(distribution.factors),
                nested_from=(
                    f'{distribution.nested_from}__{profile_id}'
                    if distribution.nested_from is not None
                    else None
                ),
            )
            for distribution_id, distribution in self.distributions.items()
            for profile_id in profiles
        ]


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
    run_profile_id: str
    factors: dict[str, str]
    resolved_run_profile_path: str
    run_profile_sha256: str


class SuiteManifestCell(SuiteModel):
    cell_id: str
    name: str
    distribution_id: str
    run_profile_id: str
    family_id: str
    family_label: str
    origin: SuiteOrigin
    status: Literal['planned', 'completed']
    include_in_family_summary: bool = True
    factors: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    analysis_blocks: list[str] = Field(default_factory=list)
    analysis_tier: str = 'core'
    run_profile_factors: dict[str, str] = Field(default_factory=dict)
    data_root: str
    result_root: str
    resolved_config_path: str
    config_sha256: str
    dataset_sha256: str
    run_profile_sha256: str
    nested_from: str | None = None
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
        if self.origin == 'derived' and any(value is None for value in source_fields):
            raise ValueError('derived suite cell requires a complete source reference')
        if self.origin == 'native' and any(value is not None for value in source_fields):
            raise ValueError('only derived suite cells may declare a source reference')
        return self


class SuiteManifest(SuiteModel):
    manifest_version: Literal[2] = MANIFEST_VERSION
    layout_version: Literal[5] = SUITE_LAYOUT_VERSION
    suite_id: str
    origin: SuiteOrigin
    created_at: str
    source: DerivedSuiteSource | None = None
    cells: list[SuiteManifestCell] = Field(min_length=1)
    distributions: list[SuiteManifestDistribution] = Field(min_length=1)
    run_profiles: list[SuiteManifestRunProfile] = Field(min_length=1)
    comparison_groups: list[ComparisonGroup] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_manifest(self) -> SuiteManifest:
        validate_identifier(self.suite_id, 'suite_id')
        if self.origin == 'derived' and self.source is None:
            raise ValueError('derived manifest requires a source contract')
        if self.origin == 'native' and self.source is not None:
            raise ValueError('only derived manifests may declare a source contract')
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError('manifest contains duplicate cell IDs')
        raw_distribution_ids = [item.distribution_id for item in self.distributions]
        raw_profile_ids = [item.run_profile_id for item in self.run_profiles]
        if len(raw_distribution_ids) != len(set(raw_distribution_ids)):
            raise ValueError('manifest contains duplicate distribution IDs')
        if len(raw_profile_ids) != len(set(raw_profile_ids)):
            raise ValueError('manifest contains duplicate run-profile IDs')
        distribution_ids = set(raw_distribution_ids)
        profile_ids = set(raw_profile_ids)
        cells_by_id = {cell.cell_id: cell for cell in self.cells}
        if self.source is not None and set(self.source.distribution_ids) != distribution_ids:
            raise ValueError('derived manifest distributions disagree with its source contract')
        for cell in self.cells:
            if cell.origin != self.origin:
                raise ValueError(f'{cell.cell_id}: cell origin disagrees with its manifest')
            if cell.distribution_id not in distribution_ids:
                raise ValueError(f'{cell.cell_id}: unknown manifest distribution')
            if cell.run_profile_id not in profile_ids:
                raise ValueError(f'{cell.cell_id}: unknown manifest run profile')
            if cell.cell_id != f'{cell.distribution_id}__{cell.run_profile_id}':
                raise ValueError(f'{cell.cell_id}: cell ID disagrees with its dimensions')
            if self.source is not None and (
                cell.source_suite_id != self.source.suite_id
                or cell.source_manifest_sha256 != self.source.manifest_sha256
            ):
                raise ValueError(f'{cell.cell_id}: source pin disagrees with its manifest')
            seen = {cell.cell_id}
            parent_id = cell.nested_from
            while parent_id is not None:
                if parent_id in seen:
                    raise ValueError(f'{cell.cell_id}: nested_from contains a cycle')
                seen.add(parent_id)
                parent = cells_by_id.get(parent_id)
                if parent is None:
                    raise ValueError(f'{cell.cell_id}: unknown nested_from cell {parent_id!r}')
                if parent.run_profile_id != cell.run_profile_id:
                    raise ValueError(f'{cell.cell_id}: nested cells must share a run profile')
                parent_id = parent.nested_from
        comparison_ids = [group.comparison_id for group in self.comparison_groups]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError('manifest contains duplicate comparison IDs')
        for group in self.comparison_groups:
            unknown = set(group.distribution_ids) - distribution_ids
            if unknown:
                raise ValueError(
                    f'{group.comparison_id}: unknown manifest distributions {sorted(unknown)}'
                )
        return self


def canonical_manifest(raw: object) -> SuiteManifest:
    if not isinstance(raw, Mapping):
        raise ValueError('suite manifest must be a mapping')
    data = dict(raw)
    if data.get('manifest_version') != 2 or data.get('layout_version') != 5:
        raise ValueError('only v5 manifest version 2 is supported')
    if data.get('origin') not in {'native', 'derived'}:
        raise ValueError('only native and derived v5 manifests are supported')
    series = data.pop('analysis_series', [])
    if series:
        raise ValueError('analysis series are not part of the frozen thesis design')
    raw_cells = data.get('cells')
    if not isinstance(raw_cells, list):
        raise ValueError('v5 manifest requires a canonical cells array')
    raw_evaluations = data.pop('evaluations', None)
    if raw_evaluations is not None:
        if not isinstance(raw_evaluations, list):
            raise ValueError('manifest evaluations compatibility view must be an array')
        cell_ids = {_raw_cell_id(cell) for cell in raw_cells}
        evaluation_ids = {_raw_cell_id(cell) for cell in raw_evaluations}
        if len(cell_ids) != len(raw_cells):
            raise ValueError('manifest cells array contains duplicate cell IDs')
        if cell_ids != evaluation_ids or len(raw_cells) != len(raw_evaluations):
            raise ValueError('manifest cells/evaluations contain different cell-ID sets')
        canonical_cells = {
            cell['cell_id']: cell for cell in (_canonical_cell(raw) for raw in raw_cells)
        }
        for raw_evaluation in raw_evaluations:
            evaluation = _canonical_cell(raw_evaluation)
            canonical = canonical_cells[evaluation['cell_id']]
            if _without_status(evaluation) != _without_status(canonical):
                raise ValueError(
                    f'{evaluation["cell_id"]}: cells/evaluations duplicate fields disagree'
                )
        data['cells'] = list(canonical_cells.values())
    else:
        data['cells'] = [_canonical_cell(cell) for cell in raw_cells]
    data['comparison_groups'] = [
        _canonical_comparison(group)
        for group in cast(list[object], data.get('comparison_groups', []))
    ]
    return SuiteManifest.model_validate(data)


def _raw_cell_id(raw: object) -> str:
    if not isinstance(raw, Mapping) or not isinstance(raw.get('cell_id'), str):
        raise ValueError('manifest cell requires a string cell_id')
    return cast(str, raw['cell_id'])


def _without_status(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cell.items() if key != 'status'}


def _canonical_cell(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError('manifest cell must be a mapping')
    cell = dict(raw)
    cell_id = _raw_cell_id(cell)
    if cell.pop('dataset_schema_version', 5) != 5:
        raise ValueError(f'{cell_id}: only dataset schema v5 is supported')
    if cell.pop('evaluation_schema_version', 5) != 5:
        raise ValueError(f'{cell_id}: only evaluation schema v5 is supported')
    base_id = cell.pop('distribution_base_id', cell.get('distribution_id'))
    if base_id != cell.get('distribution_id'):
        raise ValueError(f'{cell_id}: distribution_base_id must equal distribution_id')
    evaluation_id = cell.pop('evaluation_id', None)
    if evaluation_id is not None and not isinstance(evaluation_id, str):
        raise ValueError(f'{cell_id}: evaluation_id must be a string when present')
    if cell.pop('include_in_causal_summaries', True) is not True:
        raise ValueError(f'{cell_id}: causal exclusion is not part of the frozen design')
    if cell.pop('artifact_manifest_path', None) is not None:
        raise ValueError(f'{cell_id}: artifact manifests are unsupported')
    if cell.pop('extra_evaluation_attempts', []):
        raise ValueError(f'{cell_id}: named evaluation attempts are unsupported')
    attempt_root = cell.pop('attempt_root', None)
    result_root = cell.get('result_root')
    if result_root is None:
        if not isinstance(attempt_root, str):
            raise ValueError(f'{cell_id}: missing result root')
        cell['result_root'] = attempt_root
    elif attempt_root is not None and attempt_root != result_root:
        raise ValueError(f'{cell_id}: result_root and attempt_root disagree')
    return cell


def _canonical_comparison(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError('comparison group must be a mapping')
    group = dict(raw)
    if group.pop('cells', []):
        raise ValueError('legacy cell-based comparison groups are unsupported')
    if group.pop('varying_factor', None) is not None:
        raise ValueError('legacy singular varying_factor is unsupported')
    if group.pop('run_profile_ids', []):
        raise ValueError('profile-restricted comparison groups are unsupported')
    if group.pop('strict', True) is not True:
        raise ValueError('non-strict comparison groups are unsupported')
    return group


def validate_identifier(value: str, label: str) -> None:
    if not value or '/' in value or '\\' in value or value in {'.', '..'}:
        raise ValueError(f'{label} must be a simple non-empty identifier: {value!r}')
