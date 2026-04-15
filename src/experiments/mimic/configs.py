import argparse
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveInt, computed_field

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
    results_subdir: str
    result_dir_overrides: dict[str, str] = {}


def load_global_cfg(
    path: Path = MIMIC_IV_DIR / 'global_config.yaml',
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


def col_for_model(model_name: str) -> str:
    safe = re.sub(r'[/\-.]', '_', model_name)
    return f'vector_{safe}'


MIMIC_RESULTS_DIR = MIMIC_IV_DIR / '_results' / global_cfg.results_subdir
CONFIG_FILES_DIR = MIMIC_RESULTS_DIR / '_configs'


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
    embedding_model: str = ''
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
        data.setdefault('embedding_model', global_cfg.embedding_model)
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
    k: int
    lam: float
    jaccard_threshold: float

    @classmethod
    def load(cls) -> FilterQueriesCfg:
        return cls(**load_default_config(phase=3)['filter_queries'])


class GoldAnnotationCfg(BaseModel):
    batch_size: int
    num_ctx: int | None = None
    num_predict: int | None = None
    model: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    think: bool = False
    stream: bool = False
    map_system_prompt: str
    map_user_template: str

    @classmethod
    def load(cls) -> GoldAnnotationCfg:
        return cls(**load_default_config(phase=3)['gold_annotation'])


# -- Phase 4 --
class EvaluateCfg(BaseModel):
    vector_col: str = 'vector'
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
