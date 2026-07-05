from __future__ import annotations

import argparse
import io
import os
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, cast, get_args
from uuid import uuid4

import yaml

from experiments.medical_dataset_gen.utils.constants import COLOR_CODES, RESET, ColorName
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
                return override_path
            return self.experiment_dir / override_path

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


def resolve_experiment_name(exp_name: str, results_dir: Path | None = None) -> str:
    results_dir = results_dir or MedicalDatasetGenPaths.results_dir
    exp_path = Path(exp_name)
    if not exp_path.parts:
        raise ValueError('experiment name cannot be empty')
    if exp_path.is_absolute() or len(exp_path.parts) > 2:
        raise ValueError(
            f'experiment names are relative and support at most one child level: {exp_name!r}'
        )

    dir = results_dir / exp_path
    if dir.is_dir():
        return exp_name

    if len(exp_path.parts) == 2:
        parent_prefix, child_prefix = exp_path.parts
        parent_name = _resolve_experiment_dir_prefix(parent_prefix, results_dir).name
        child_dir = _resolve_experiment_dir_prefix(child_prefix, results_dir / parent_name)
        return f'{parent_name}/{child_dir.name}'

    matches = sorted(path for path in results_dir.glob(f'{exp_name}*') if path.is_dir())
    if len(matches) == 1:
        return matches[0].name
    elif len(matches) > 1:
        raise RuntimeError(
            f'{exp_name!r} is an ambiguous prefix, found many matches: {[m.name for m in matches]}'
        )
    else:
        raise FileNotFoundError(f'no experiment directory prefixed {exp_name!r} in {results_dir}. ')


def child_experiment_names(
    parent_exp_name: str,
    results_dir: Path | None = None,
) -> list[str]:
    results_dir = results_dir or MedicalDatasetGenPaths.results_dir
    parent_name = resolve_experiment_name(parent_exp_name, results_dir=results_dir)
    parent_path = results_dir / parent_name
    if len(Path(parent_name).parts) != 1:
        return []
    return [
        f'{parent_name}/{child_path.name}'
        for child_path in sorted(parent_path.iterdir())
        if child_path.is_dir() and (child_path / '_subconfig.yaml').is_file()
    ]


def _resolve_experiment_dir_prefix(prefix: str, parent_dir: Path) -> Path:
    exact_dir = parent_dir / prefix
    if exact_dir.is_dir():
        return exact_dir

    matches = sorted(path for path in parent_dir.glob(f'{prefix}*') if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f'{prefix!r} is an ambiguous prefix in {parent_dir}, '
            f'found many matches: {[path.name for path in matches]}'
        )
    raise FileNotFoundError(f'no experiment directory prefixed {prefix!r} in {parent_dir}. ')


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
    raw = _load_raw_experiment_config(paths)
    cfg = ExperimentCfg.model_validate(raw)
    cfg.global_.output_experiment = exp_name
    return cfg


def _load_raw_experiment_config(paths: MedicalDatasetGenPaths) -> YamlMapping:
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


def colored(color: ColorName, text: str) -> str:
    return f'\033[38;5;{COLOR_CODES[color]}m{text}{RESET}'


def colorprint(color: ColorName, text: str) -> None:
    print(colored(color, text))
