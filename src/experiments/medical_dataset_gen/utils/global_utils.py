from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, get_args
from uuid import uuid4

import yaml

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


def resolve_experiment_name(
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


def load_config(exp: str | None = None) -> ExperimentCfg:
    exp_name = exp or os.getenv('EXP') or os.getenv('EXP_NAME')
    if exp_name is None:
        raise ValueError(
            'missing experiment name; pass --exp or set EXP/EXP_NAME so '
            'the config can be loaded from _results/<exp>/_config.yaml'
        )

    exp_name = resolve_experiment_name(exp_name)
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
