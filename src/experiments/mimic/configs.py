import argparse
import io
import os
import re
import sys
from os import getenv
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import torch
import yaml
from pydantic import BaseModel, Field, PositiveInt, computed_field, model_validator

from experiments.mimic.constants import CHARLSON_LABELS_TO_STR, MimicTable
from experiments.mimic.prompts_default import (
    GOLD_TAGS_SYSTEM_PROMPT_DEF,
    GOLD_TAGS_TEMPLATE_DEF,
    QUERY_GENERATING_TEMPLATE_DEF,
)
from experiments.mimic.utils import get_vec_col_name
from helpers.dir_paths import MIMIC_IV_DIR
from helpers.query_algorithms import ScoringFunction

torch.cuda.empty_cache()
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

type TopLevelConfigKeys = Literal[
    'global',
    'chunking',
    'embeddings',
    'queries',
    'evaluation',
]

MIMIC_RESULTS_DIR = MIMIC_IV_DIR / '_results'
VECTOR_DB_DIR = MIMIC_RESULTS_DIR / '_vector_db'

MIMIC_EXPERIMENT_DIR = MIMIC_RESULTS_DIR / getenv('EXP_NAME', MIMIC_IV_DIR)
LOGS_DIR = MIMIC_EXPERIMENT_DIR / '_logs'
CONFIG_DIR = MIMIC_EXPERIMENT_DIR / '_config.yaml'


class GlobalCfg(BaseModel):
    model_config = {'populate_by_name': True, 'extra': 'ignore'}

    num_conditions: PositiveInt = Field(alias='n_conditions')
    result_dir_overrides: dict[str, str] = {}

    prefilter_n: int
    embedding_model: str
    chunks_vec_table: str
    query_retrieval_instruction: str | None = (
        'Instruct: Given a multi-aspect clinical query comparing patient cohorts, '
        'retrieve relevant hospital discharge summaries containing matching patient '
        'demographics, comorbidities, and treatment outcomes.\nQuery: '
    )

    @computed_field
    @property
    def vector_column(self) -> str:
        return get_vec_col_name(self.embedding_model)

    @computed_field
    @property
    def label_to_charlson_col(self) -> dict[str, str]:
        return {v: k for k, v in CHARLSON_LABELS_TO_STR.items()}


def load_global_cfg(path: str | Path = CONFIG_DIR, cfg: GlobalCfg | None = None):
    global global_cfg
    if cfg is not None:
        global_cfg = cfg
        return
    with open(path) as f:
        _loaded_cfg = yaml.safe_load(f)

    global_cfg = GlobalCfg.model_validate(_loaded_cfg['global'])
    return


load_global_cfg()


def load_default_config(key: TopLevelConfigKeys, path: str | Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_DIR) as f:
        return yaml.safe_load(f)[key]


def load_config_from_main(key: TopLevelConfigKeys) -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--config', type=str, default=None, help='Path to a custom YAML config file'
    )
    args, _ = parser.parse_known_args()
    return load_default_config(key=key, path=args.config)


def get_result_dir(table: MimicTable) -> Path:
    subdir = global_cfg.result_dir_overrides.get(table)
    if subdir is not None:
        return MIMIC_RESULTS_DIR / subdir
    else:
        return MIMIC_EXPERIMENT_DIR


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
        return MIMIC_RESULTS_DIR / '_shared' / f'{table}.{ext}'
    else:
        return get_result_dir(table) / f'{table}.{ext}'


# PYDANTIC MODELS FOR CONFIGS
# -- Phase 1 --
class ConditionsStatsCfg(BaseModel):
    min_admissions: int = 0
    cond_processing_llm: str = 'gemma4-31b-text'

    @classmethod
    def load(cls) -> ConditionsStatsCfg:
        return cls(**load_default_config(key='chunking')['conditions_stats'])


class NoteChunkingCfg(BaseModel):
    model_config = {'arbitrary_types_allowed': True}

    keep_sections: set[str] = {
        'HISTORY OF PRESENT ILLNESS',
        'DISCHARGE MEDICATIONS',
        'DISCHARGE DIAGNOSIS',
    }
    skip_sections: set[str] = {
        'PERTINENT RESULTS',
        'SEX',
        'SERVICE',
        'ALLERGIES',
        'ATTENDING',
        'SOCIAL HISTORY',
        'FOLLOWUP INSTRUCTIONS',
        'FACILITY',
        'MAJOR SURGICAL OR INVASIVE PROCEDURE',
        'FAMILY HISTORY',
        'MEDICATIONS ON ADMISSION',
        'DISCHARGE DISPOSITION',
        'DISCHARGE CONDITION',
        'DISCHARGE INSTRUCTIONS',
        'PHYSICAL EXAM',
        'PAST MEDICAL HISTORY',
    }
    metadata_only_sections: set[str] = {'CHIEF COMPLAINT'}
    max_tokens: int = 1024
    stride_tokens: int = 512
    min_chunk_tokens: int = 256

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
        return cls(**load_default_config(key='chunking')['note_chunking'])


class DedupCfg(BaseModel):
    boilerplate_sections: list[str]

    @computed_field
    @property
    def boilerplate_sections_set(self) -> set[str]:
        return set(self.boilerplate_sections)

    @classmethod
    def load(cls) -> DedupCfg:
        return cls(**load_default_config(key='chunking')['dedup'])


# -- Phase 2 --
class EmbedCfg(BaseModel):
    batch_size: int
    commit_every: int
    device: str

    @classmethod
    def load(cls) -> EmbedCfg:
        return cls(**load_default_config(key='embeddings'))


# -- Phase 3 --
class DemographicModifier(BaseModel):
    text: str
    column: str
    op: str
    value: float | int


class BuildQueryPromptsCfg(BaseModel):
    n_grounding_patients: int
    max_modifiers: int
    min_modifier_admissions: int = 5
    min_condition_admissions: int | None = None
    max_condition_admissions: int | None = None
    n_strata: int = 3
    stratify_scale: Literal['linear', 'log'] = 'log'
    stratify_seed: int = 42
    high_value_sections: list[str] = [
        'BRIEF HOSPITAL COURSE',
        'HISTORY OF PRESENT ILLNESS',
        'DISCHARGE DIAGNOSIS',
        'DISCHARGE MEDICATIONS',
    ]
    demographic_modifiers: list[DemographicModifier]
    prompt_template: str = QUERY_GENERATING_TEMPLATE_DEF

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
        return cls(**load_default_config(key='queries')['build_prompts'])


class GenQueriesCfg(BaseModel):
    save_every: int
    model: str | None = None
    temperature: float = 0.3
    top_p: float | None = None
    top_k: int | None = None
    think: bool = False

    @classmethod
    def load(cls) -> GenQueriesCfg:
        return cls(**load_default_config(key='queries')['llm_generation'])


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
        return cls(**load_default_config(key='queries')['filtering'])


class GoldAnnotationCfg(BaseModel):
    model_config = {'populate_by_name': True, 'extra': 'ignore'}

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
    tagging_system_prompt: str = GOLD_TAGS_SYSTEM_PROMPT_DEF
    tagging_user_template: str = GOLD_TAGS_TEMPLATE_DEF

    @classmethod
    def load(cls) -> GoldAnnotationCfg:
        return cls(**load_default_config(key='queries')['gold_annotation'])


# -- Phase 4 --
class EvaluateCfg(BaseModel):
    strategies: list[ScoringFunction]
    k_values: list[int]
    lam_values: list[float]
    device: str
    gold_mode: Literal['llm', 'structural'] = 'llm'

    @classmethod
    def load(cls) -> EvaluateCfg:
        return cls(**load_default_config(key='evaluation'))


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


def silent_excepthook(type, value, traceback):
    if type in [KeyboardInterrupt, SystemExit]:
        return
    sys.__excepthook__(type, value, traceback)


sys.excepthook = silent_excepthook
