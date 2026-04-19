import argparse
import io
import re
from os import getenv
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, PositiveInt, computed_field, model_validator

from helpers.dir_paths import MIMIC_IV_DIR
from helpers.query_algorithms import ScoringFunction


class SharedQueriesCfg(BaseModel):
    charlson_labels: dict[str, str]
    prefilter_n: int

    @computed_field
    @property
    def label_to_charlson_col(self) -> dict[str, str]:
        return {v: k for k, v in self.charlson_labels.items()}


class GlobalCfg(BaseModel):
    model_config = {'populate_by_name': True, 'extra': 'ignore'}

    num_conditions: PositiveInt = Field(alias='n_conditions')
    embedding_model: str
    vector_db_dir: str
    shared_queries_cfg: SharedQueriesCfg = Field(alias='phase_3_shared')
    result_dir_overrides: dict[str, str] = {}


exp_name = getenv('EXP_NAME', MIMIC_IV_DIR)
MIMIC_RESULTS_DIR = MIMIC_IV_DIR / '_results' / exp_name
CONFIG_FILES_DIR = MIMIC_RESULTS_DIR / '_configs'
LOGS_DIR = MIMIC_RESULTS_DIR / '_logs'


def load_global_cfg(
    path: Path = CONFIG_FILES_DIR / 'global_config.yaml',
    cfg: GlobalCfg | None = None,
):
    global global_cfg
    if cfg is not None:
        global_cfg = cfg
        return

    with open(path) as f:
        _loaded_cfg = yaml.safe_load(f)
    global_cfg = GlobalCfg.model_validate(_loaded_cfg)
    return


load_global_cfg()
VECTOR_DB_DIR = MIMIC_IV_DIR / global_cfg.vector_db_dir


def load_default_config(phase: int, path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = CONFIG_FILES_DIR / f'phase_{phase}_config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def load_config_from_main(phase: int) -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--config', type=str, default=None, help='Path to a custom YAML config file'
    )
    args, _ = parser.parse_known_args()
    return load_default_config(phase, path=args.config)


def get_result_dir(table: str) -> Path:
    subdir = global_cfg.result_dir_overrides.get(table)
    if subdir is not None:
        return MIMIC_IV_DIR / '_results' / subdir
    return MIMIC_RESULTS_DIR


def get_parquet_path(table: str) -> Path:
    return get_result_dir(table) / f'{table}.parquet'


# PYDANTIC MODELS FOR CONFIGS
# -- Phase 1 --
class ConditionsStatsCfg(BaseModel):
    min_admissions: int

    @classmethod
    def load(cls) -> ConditionsStatsCfg:
        return cls(**load_default_config(phase=1)['conditions_stats'])


class NoteChunkingCfg(BaseModel):
    model_config = {'arbitrary_types_allowed': True}

    keep_sections: set[str]
    skip_sections: set[str]
    metadata_only_sections: set[str]
    max_tokens: int = 512
    stride_tokens: int = 128
    min_chunk_tokens: int = 15

    @computed_field
    @property
    def all_sections(self) -> set[str]:
        return self.keep_sections | self.skip_sections | self.metadata_only_sections

    @computed_field
    @property
    def tag_re(self) -> re.Pattern:
        return re.compile(
            rf'<({"|".join(re.escape(s) for s in self.all_sections)})>',
            re.IGNORECASE,
        )

    @classmethod
    def load(cls) -> NoteChunkingCfg:
        data = load_default_config(phase=1)['note_chunking']
        return cls(**data)


class DedupCfg(BaseModel):
    boilerplate_sections: list[str]

    @computed_field
    @property
    def boilerplate_sections_set(self) -> set[str]:
        return set(self.boilerplate_sections)

    @classmethod
    def load(cls) -> DedupCfg:
        return cls(**load_default_config(phase=1)['dedup'])


# -- Phase 2 --
class EmbedCfg(BaseModel):
    model_name: str = ''
    batch_size: int
    commit_every: int
    device: str

    @classmethod
    def load(cls) -> EmbedCfg:
        data = load_default_config(phase=2)['embed']
        data.setdefault('model_name', global_cfg.embedding_model)
        return cls(**data)


# -- Phase 3 --
class DemographicModifier(BaseModel):
    text: str
    column: str
    op: str
    value: float | int


class BuildQueryPromptsCfg(BaseModel):
    n_grounding_patients: int
    max_modifiers: int
    high_value_sections: list[str]
    demographic_modifiers: list[DemographicModifier]
    prompt_template: str

    @computed_field
    @property
    def demographic_modifiers_text(self) -> list[str]:
        return [m.text for m in self.demographic_modifiers]

    @computed_field
    @property
    def demographic_filters(self) -> dict[str, tuple]:
        return {m.text: (m.column, m.op, m.value) for m in self.demographic_modifiers}

    @classmethod
    def load(cls) -> BuildQueryPromptsCfg:
        return cls(**load_default_config(phase=3)['build_query_prompts'])


class GenQueriesCfg(BaseModel):
    save_every: int
    model: str | None = None
    temperature: float = 0.3
    top_p: float | None = None
    top_k: int | None = None
    think: bool = False

    @classmethod
    def load(cls) -> GenQueriesCfg:
        return cls(**load_default_config(phase=3)['gen_queries_llm'])


class FilterQueriesCfg(BaseModel):
    k_values: list[int]
    lam_values: list[float]
    jaccard_threshold: float

    @model_validator(mode='before')
    @classmethod
    def _compat_single_k_lam(cls, data: dict) -> dict:
        if 'k' in data and 'k_values' not in data:
            data['k_values'] = [data.pop('k')]
        if 'lam' in data and 'lam_values' not in data:
            data['lam_values'] = [data.pop('lam')]
        return data

    @classmethod
    def load(cls) -> FilterQueriesCfg:
        return cls(**load_default_config(phase=3)['filter_queries'])


class GoldAnnotationCfg(BaseModel):
    batch_size: int
    resume_batch_size: int | None = None
    wide_pool_n: int = 10000
    final_pool_n: int = 3000
    min_per_modifier: int = 50
    num_ctx: int | None = None
    num_predict: int | None = None
    model: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    think: bool = False
    stream: bool = False
    tagging_system_prompt: str
    tagging_user_template: str

    @classmethod
    def load(cls) -> GoldAnnotationCfg:
        return cls(**load_default_config(phase=3)['gold_annotation'])


# -- Phase 4 --
class EvaluateCfg(BaseModel):
    vector_col: str
    strategies: list[ScoringFunction]
    k_values: list[int]
    lam_values: list[float]
    prefilter_n: int
    device: str

    @classmethod
    def load(cls) -> EvaluateCfg:
        data = load_default_config(phase=4)['evaluate']
        data.setdefault('embedding_model', global_cfg.embedding_model)
        data.setdefault('vector_col', col_for_model(global_cfg.embedding_model))
        return cls(**data)


def col_for_model(model_name: str) -> str:
    safe = re.sub(r'[/\-.]', '_', model_name)
    return f'vector_{safe}'


def setup_logging() -> None:
    import sys

    main = sys.modules['__main__']
    script_name = Path(main.__file__ if main.__file__ else f'unknown_script_{uuid4()}').stem
    log_path = LOGS_DIR / f'{script_name}.log'
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

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
