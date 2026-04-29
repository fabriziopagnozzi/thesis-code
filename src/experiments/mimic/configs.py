import argparse
import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import polars as pl
import torch
import yaml
from pydantic import BaseModel, Field, PositiveInt, computed_field, model_validator

from experiments.mimic.utils.charlson import CHARLSON_LABELS_TO_STR
from experiments.mimic.utils.constants import (
    MimicPaths,
    MimicTable,
)
from experiments.mimic.utils.prompts_default import MimicDefaultPrompts
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

    @computed_field
    @property
    def label_to_charlson_col(self) -> dict[str, str]:
        return {v: k for k, v in CHARLSON_LABELS_TO_STR.items()}


def load_global_cfg(path: str | Path = MimicPaths.config, cfg: GlobalCfg | None = None):
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
    with open(path or MimicPaths.config) as f:
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
        return MimicPaths.results / subdir
    else:
        return MimicPaths.experiment


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
        return MimicPaths.results / '_shared' / f'{table}.{ext}'
    else:
        return get_result_dir(table) / f'{table}.{ext}'


def read_parquet(table: MimicTable):
    return pl.read_parquet(get_table_path(table, ext='parquet'))


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
    models: list[str]
    batch_sizes: list[int]
    commit_every: int

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
    prompt_template: str = MimicDefaultPrompts.query_gen_template

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
    tagging_system_prompt: str = MimicDefaultPrompts.gold_tags_system
    tagging_user_template: str = MimicDefaultPrompts.gold_tags_template

    @classmethod
    def load(cls) -> GoldAnnotationCfg:
        return cls(**load_default_config(key='queries')['gold_annotation'])


# -- Phase 4 --
class EvaluateCfg(BaseModel):
    embedding_model: str
    strategies: list[ScoringFunction]
    k_values: list[int]
    lam_values: list[float]
    gold_mode: Literal['llm', 'structural'] = 'llm'

    @classmethod
    def load(cls) -> EvaluateCfg:
        return cls(**load_default_config(key='evaluation'))


class PoolAnalysisCfg(BaseModel):
    model_config = {'arbitrary_types_allowed': True}

    embedding_model: str = 'Qwen/Qwen3-Embedding-0.6B'
    pool_n: int = 2000

    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.05
    umap_metric: str = 'cosine'
    umap_dim_for_cluster: int = 10
    umap_random_state: int = 42

    hdbscan_min_cluster_size: int = 30
    hdbscan_min_samples: int | None = None

    kmeans_k_min: int = 2
    kmeans_k_max: int = 12
    kmeans_random_state: int = 42

    lof_n_neighbors: int = 20
    lof_contamination: float | str = 'auto'

    knn_k: int = 10

    cv_n_splits: int = 5
    n_figures: int = 30

    output_subdir: str = 'pool_analysis'

    @property
    def output_dir(self) -> Path:
        return MimicPaths.experiment / self.output_subdir

    @classmethod
    def load(cls, path: str | Path | None = None) -> PoolAnalysisCfg:
        cfg_path = Path(path) if path else MimicPaths.config
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
        block = data.get('pool_analysis') or {}
        return cls(**block)


def setup_logging() -> None:
    import sys

    main = sys.modules['__main__']
    script_name = Path(main.__file__ if main.__file__ else f'unknown_script_{uuid4()}').stem
    log_path = MimicPaths.logs / f'{script_name}.log'

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
