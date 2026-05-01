import argparse
import io
import os
import sys
from os import getenv
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import lancedb
import polars as pl
import torch
import yaml
from duckdb import DuckDBPyConnection
from pydantic import BaseModel, Field, PositiveInt, computed_field

from experiments.mimic.utils.constants import (
    MimicTable,
)
from helpers.dir_paths import DATASETS_DIR, ROOT_DIR, THIRDPARTY_CODE_DIR


class GlobalCfg(BaseModel):
    model_config = {'populate_by_name': True, 'extra': 'ignore'}

    num_conditions: PositiveInt = Field(alias='n_conditions')
    result_dir_overrides: dict[str, str] = {}

    prefilter_n: int = 1_000_000_000  # basically infinity
    chunks_vec_table: str
    query_retrieval_instruction: str | None = (
        'Instruct: Given a multi-aspect clinical query comparing patient cohorts, '
        'retrieve relevant hospital discharge summaries containing matching patient '
        'demographics, comorbidities, and treatment outcomes.\nQuery: '
    )

    embedding_model: str

    sections_filter: frozenset[str] = frozenset({'BRIEF HOSPITAL COURSE', 'DISCHARGE MEDICATIONS'})

    @computed_field
    @property
    def sections_filter_sql(self) -> str:
        quoted = ', '.join(f"'{s}'" for s in sorted(self.sections_filter))
        return f'section_name IN ({quoted})'


def _resolve_exp_name() -> str:
    exp = getenv('EXP') or getenv('EXP_NAME')
    if exp is None:
        raise RuntimeError('Specify EXP (or EXP_NAME) environment variable.')

    results_dir = ROOT_DIR / 'src' / 'experiments' / 'mimic' / '_results'
    if (results_dir / exp).is_dir():
        return exp

    matches = sorted(results_dir.glob(f'{exp}-*'))
    if len(matches) == 1:
        return matches[0].name
    if len(matches) > 1:
        raise RuntimeError(f'EXP={exp!r} is ambiguous: {[m.name for m in matches]}')

    raise RuntimeError(f'No experiment directory matching {exp!r} in {results_dir}')


class MimicPaths:
    exp_name = _resolve_exp_name()
    mimic_root = ROOT_DIR / 'src' / 'experiments' / 'mimic'
    results_dir = mimic_root / '_results'

    vector_db_dir = results_dir / '_vector_db'
    figures_dir = results_dir / 'figures'
    experiment_dir = results_dir / exp_name

    config_path = experiment_dir / '_config.yaml'
    logs_dir = experiment_dir / '_logs'

    init_sql = mimic_root / '_mimic_init.sql'
    duckdb_concepts_dir = THIRDPARTY_CODE_DIR / 'mimic_code' / 'mimic-iv' / 'concepts_duckdb'

    hosp_dir = DATASETS_DIR / 'mimic-iv' / 'hosp'
    icu_dir = DATASETS_DIR / 'mimic-iv' / 'icu'
    note_dir = DATASETS_DIR / 'mimic-iv' / 'note'
    bhc_dir = DATASETS_DIR / 'mimic-iv' / 'ext-bhc'


type TopLevelConfigFileKeys = Literal[
    'global', 'chunking', 'embeddings', 'queries', 'evaluation', 'pool_analysis'
]


def load_global_cfg(path: str | Path = MimicPaths.config_path, cfg: GlobalCfg | None = None):
    if cfg is not None:
        global_cfg = cfg
        return global_cfg
    with open(path) as f:
        _loaded_cfg = yaml.safe_load(f)

    global_cfg = GlobalCfg.model_validate(_loaded_cfg['global'])
    return global_cfg


def load_default_config(
    key: TopLevelConfigFileKeys, path: str | Path | None = None
) -> dict[str, Any]:
    with open(path or MimicPaths.config_path) as f:
        return yaml.safe_load(f)[key]


def load_config_from_main(key: TopLevelConfigFileKeys) -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--config', type=str, default=None, help='Path to a custom YAML config file'
    )
    args, _ = parser.parse_known_args()
    return load_default_config(key=key, path=args.config)


def get_result_dir(table: MimicTable) -> Path:
    subdir = global_cfg.result_dir_overrides.get(table)
    if subdir is not None:
        return MimicPaths.results_dir / subdir
    else:
        return MimicPaths.experiment_dir


def get_table_path(
    table: MimicTable,
    ext: Literal['csv', 'jsonl', 'json', 'parquet'] = 'parquet',
) -> Path:
    shared_tables = [
        'admissions_metadata',
        'age',
        'charlson',
        'icd9_to_icd10_cm_gem',
        'unified_diagnoses',
    ]
    if table in shared_tables:
        return MimicPaths.results_dir / '_shared' / f'{table}.{ext}'
    else:
        return get_result_dir(table) / f'{table}.{ext}'


def read_parquet(table: MimicTable):
    return pl.read_parquet(get_table_path(table, ext='parquet'))


# -- Phase 2 --
def setup_logging() -> None:
    import sys

    main = sys.modules['__main__']
    script_name = Path(main.__file__ if main.__file__ else f'unknown_script_{uuid4()}').stem
    log_path = MimicPaths.logs_dir / f'{script_name}.log'

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


def silent_excepthook(type, value, traceback):
    if type in [KeyboardInterrupt, SystemExit]:
        return
    sys.__excepthook__(type, value, traceback)


sys.excepthook = silent_excepthook


# <side_effects>

torch.cuda.empty_cache()
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

for p in [MimicPaths.results_dir, MimicPaths.logs_dir, MimicPaths.vector_db_dir]:
    p.mkdir(parents=True, exist_ok=True)

global_cfg: GlobalCfg = load_global_cfg()
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb  # noqa: E402

duckdb_con: DuckDBPyConnection = connect_mimic_duckdb()
lancedb_con: lancedb.DBConnection = lancedb.connect(MimicPaths.vector_db_dir)

# </side_effects>
