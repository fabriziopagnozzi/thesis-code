from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, cast, get_args

import yaml

from experiments.medical_dataset_gen.utils.exp_naming import resolve_experiment_name
from helpers.dir_paths import ROOT_DIR

if TYPE_CHECKING:
    from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg


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
    'evaluation_selection_stats',
    'evaluation_report_grid_stats',
    'evaluation_slice_stats',
    'query_geometry_points',
    'query_geometry_stats',
]
SYNTH_MEDICAL_DATASET_TABLE_NAMES = set[str](get_args(SyntheticMedicalDatasetTableName.__value__))

type EmbeddingArtifactName = Literal[
    'chunk_vectors', 'query_vectors', 'chunk_ids', 'query_ids', 'metadata'
]

type ResultDirOverrides = dict[SyntheticMedicalDatasetTableName | EmbeddingArtifactName, str]

type YamlMapping = dict[str, object]


class MedicalDatasetGenPaths:
    root = ROOT_DIR / 'src' / 'experiments' / 'medical_dataset_gen'
    results_dir = root / '_results'
    cache_dir = root / '_cache'
    default_ontology_path = root / 'data_templates' / 'medical_ontology.yaml'
    experiment_aliases_path = results_dir / '_experiment_aliases.json'

    def __init__(
        self,
        exp_name: str,
        result_dir_overrides: ResultDirOverrides | None = None,
    ):
        self.exp_name = exp_name
        self.experiment_dir = self.results_dir / exp_name
        if not exp_name:
            raise ValueError('experiment name cannot be empty')

        exp_path = Path(exp_name)
        exp_parts = exp_path.parts
        if len(exp_parts) > 2:
            raise ValueError(f'subexperiments support only one nesting level: {exp_name!r}')

        parent_exp_name = exp_parts[0] if len(exp_parts) == 2 else exp_path.name
        self.parent_experiment_dir = self.results_dir / parent_exp_name
        self.logs_dir = self.experiment_dir / '_logs'
        self.figures_dir = self.experiment_dir / '_figures'
        self.config_path = self.experiment_dir / '_config.yaml'
        self.parent_config_path = self.parent_experiment_dir / '_config.yaml'
        self.subconfig_path = self.experiment_dir / '_subconfig.yaml'
        self.result_dir_overrides = dict(result_dir_overrides or {})

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
                return override_path / f'{table}.{ext}'
            return self.experiment_dir / override_path / f'{table}.{ext}'

        if override_path.is_absolute():
            return override_path / f'{table}.{ext}'
        return self.results_dir / override_path / f'{table}.{ext}'

    def embeddings_paths(self, name: EmbeddingArtifactName) -> Path:
        override = self.result_dir_overrides.get(name)
        artifact_disk_old_name = f'embeddings_{name}.{"json" if name == "metadata" else "npy"}'  # for compatibility with old exps

        if override is None:
            return self.experiment_dir / artifact_disk_old_name
        else:
            return Path(override) / artifact_disk_old_name

    def get_result_dir(self, table: SyntheticMedicalDatasetTableName) -> Path:
        return self.table_path(table).parent

    def deterministic_chunk_documents_cache_dir(self, render_signature: str) -> Path:
        return self.cache_dir / 'deterministic_chunk_documents' / render_signature

    def deterministic_chunk_documents_bucket_path(
        self,
        render_signature: str,
        bucket: str,
    ) -> Path:
        return self.deterministic_chunk_documents_cache_dir(render_signature) / f'{bucket}.parquet'

    def deterministic_chunk_documents_lock_path(
        self,
        render_signature: str,
        bucket: str,
    ) -> Path:
        return self.deterministic_chunk_documents_bucket_path(render_signature, bucket).with_suffix(
            '.parquet.lock'
        )

    def chunk_embeddings_cache_dir(self, embedding_signature: str) -> Path:
        return self.cache_dir / 'chunk_embeddings' / embedding_signature

    def chunk_embeddings_bucket_path(self, embedding_signature: str, bucket: str) -> Path:
        return self.chunk_embeddings_cache_dir(embedding_signature) / f'{bucket}.parquet'

    def chunk_embeddings_lock_path(self, embedding_signature: str, bucket: str) -> Path:
        return self.chunk_embeddings_bucket_path(embedding_signature, bucket).with_suffix(
            '.parquet.lock'
        )

    def config_source_paths(self) -> tuple[Path, ...]:
        if self.is_subexperiment():
            return (self.parent_config_path, self.subconfig_path)
        return (self.config_path,)

    def is_subexperiment(self) -> bool:
        return len(Path(self.exp_name).parts) == 2


def load_config(exp: str | None = None) -> ExperimentCfg:
    from experiments.medical_dataset_gen.schemas.global_config_schemas import ExperimentCfg

    exp_name = exp or os.getenv('EXP') or os.getenv('EXP_NAME')
    if exp_name is None:
        raise ValueError(
            'missing experiment name; pass --exp or set EXP/EXP_NAME so '
            'the config can be loaded from _results/<exp>/_config.yaml'
        )

    exp_name = resolve_experiment_name(exp_name)
    paths = MedicalDatasetGenPaths(exp_name)
    raw = load_raw_experiment_config(paths)
    cfg = ExperimentCfg.model_validate(raw)
    cfg.global_.output_experiment = exp_name
    return cfg


def load_raw_experiment_config(paths: MedicalDatasetGenPaths) -> YamlMapping:
    if not paths.is_subexperiment():
        return _read_yaml_mapping(
            paths.config_path,
            missing_message='Create it manually before running the pipeline.',
        )

    parent_raw = _read_yaml_mapping(
        paths.parent_config_path,
        missing_message='Subexperiments require a parent _config.yaml.',
    )
    sub_raw = _read_yaml_mapping(
        paths.subconfig_path,
        missing_message='Create _subconfig.yaml for the subexperiment overrides.',
    )
    if 'generation' in sub_raw:
        raise ValueError(
            f'{paths.subconfig_path} cannot override generation. '
            'Subexperiments must keep the parent dataset distribution unchanged.'
        )
    return _deep_merge_config(parent_raw, sub_raw)


def _read_yaml_mapping(path: Path, missing_message: str) -> YamlMapping:
    if not path.exists():
        raise FileNotFoundError(f'missing experiment config: {path}. {missing_message}')
    with open(path) as file:
        raw = yaml.safe_load(file)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f'experiment config must be a YAML mapping: {path}')
    return {str(key): value for key, value in raw.items()}


def _deep_merge_config(base: YamlMapping, overrides: YamlMapping) -> YamlMapping:
    merged = dict(base)
    for key, override_value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _deep_merge_config(
                {str(child_key): value for child_key, value in base_value.items()},
                {str(child_key): value for child_key, value in override_value.items()},
            )
        else:
            merged[key] = cast(object, override_value)
    return merged


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
