"""Public facade for the frozen schema-v5 thesis suite implementation."""

from experiments.medical_dataset_gen.suites.contracts import (
    MANIFEST_VERSION,
    SUITE_LAYOUT_VERSION,
    ComparisonGroup,
    DerivedEmbeddingOverrides,
    DerivedSuiteSource,
    Distribution,
    ExpandedCell,
    RunProfile,
    SuiteManifest,
    SuiteManifestCell,
    SuiteManifestDistribution,
    SuiteManifestRunProfile,
    SuiteSpec,
    SuiteTransform,
)
from experiments.medical_dataset_gen.suites.io import sha256_json as _sha256_json
from experiments.medical_dataset_gen.suites.manifests import (
    LogicalSuite,
    load_logical_suite,
    load_logical_suite_family,
    load_pinned_source_manifest,
    load_suite_manifest,
    load_suite_spec,
    resolve_derived_source_cell,
    suite_root,
    suite_spec_root,
)
from experiments.medical_dataset_gen.suites.materialization import materialize_suite
from experiments.medical_dataset_gen.suites.resolution import (
    ValidationResult,
    resolve_cell_config,
    validate_suite,
)
from experiments.medical_dataset_gen.suites.resolution import (
    dataset_hash as _dataset_hash,
)
from experiments.medical_dataset_gen.suites.resolution import (
    declared_composition as _declared_composition,
)

__all__ = [
    'MANIFEST_VERSION',
    'SUITE_LAYOUT_VERSION',
    'ComparisonGroup',
    'DerivedEmbeddingOverrides',
    'DerivedSuiteSource',
    'Distribution',
    'ExpandedCell',
    'LogicalSuite',
    'RunProfile',
    'SuiteManifest',
    'SuiteManifestCell',
    'SuiteManifestDistribution',
    'SuiteManifestRunProfile',
    'SuiteSpec',
    'SuiteTransform',
    'ValidationResult',
    '_dataset_hash',
    '_declared_composition',
    '_sha256_json',
    'load_logical_suite',
    'load_logical_suite_family',
    'load_pinned_source_manifest',
    'load_suite_manifest',
    'load_suite_spec',
    'materialize_suite',
    'resolve_cell_config',
    'resolve_derived_source_cell',
    'suite_root',
    'suite_spec_root',
    'validate_suite',
]
