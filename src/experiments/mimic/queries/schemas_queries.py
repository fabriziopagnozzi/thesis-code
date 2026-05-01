import json
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from experiments.mimic.global_configs import load_default_config
from experiments.mimic.utils.charlson import CHARLSON_STR_TO_LABEL
from experiments.mimic.utils.prompts_default import MimicDefaultPrompts


# CONFIGS
class DemographicModifier(BaseModel):
    text: str
    column: str
    op: str
    value: float | int


class BuildQueryPromptsCfg(BaseModel):
    n_grounding_patients: int
    max_comorbidity_modifiers: int
    min_modifier_admissions: int = 5
    min_condition_admissions: int | None = None
    max_condition_admissions: int | None = None
    n_strata: int = 3
    stratify_scale: Literal['linear', 'log'] = 'log'
    stratify_seed: int = 42
    high_value_sections: list[str] = [
        'BRIEF HOSPITAL COURSE',
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


type QueryModifierLabelId = str


class QueryModifier(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal['comorbidity', 'demographic']
    text: str

    _demographic_filters = BuildQueryPromptsCfg.load().demographic_filters

    @computed_field
    @property
    def label(self) -> QueryModifierLabelId:
        if self.type == 'comorbidity':
            return CHARLSON_STR_TO_LABEL[self.text]
        elif self.type == 'demographic':
            return self.text
        else:
            raise RuntimeError('[ERROR]: unsupported modifier type')

    @classmethod
    def parse_list(cls, json_str: str) -> list[QueryModifier]:
        return [cls(**d) for d in json.loads(json_str or '[]')]

    def format_subgroup_info(self):
        if self.type == 'comorbidity':
            return f'patients with the following comorbidity: "{self.text}"'
        elif self.type == 'demographic':
            return self.text
        else:
            raise RuntimeError('[ERROR]: unsupported modifier type')

    def sql_predicate(self) -> str:
        """Build a LanceDB-compatible SQL predicate for a single modifier.
        Comorbidity: uses the Charlson column already in the lance table (m.label).
        Demographic: looks up the (column, op, value) tuple from the demographic_filters config.
        """
        if self.type == 'comorbidity':
            return f'{self.label} > 0'

        demo = self._demographic_filters.get(self.label)
        if demo is None:
            raise ValueError(f'Unknown demographic modifier: {self.label!r}')
        column, op, value = demo

        return f'{column} {op} {value}'


type ModifierStats = dict[QueryModifier, dict[Literal['n_admissions', 'n_chunks'], int]]


class GroundingChunkSample(TypedDict):
    header: str
    text: str
    hadm_id: int


class DivergenceMetrics(TypedDict):
    """Return type of compute_divergence in c_filter_queries.py."""

    jaccard: float
    jaccard_div: float
    fac_gap: float
    fac_topk: float
    fac_fl: float
    pool_size: int


# PARQUET OUTPUTS
class QueryPromptRow(TypedDict):
    """queries_prompts.parquet - one row per (condition, modifier-set) prompt."""

    query_id: int
    icd10_3char: str
    condition_name: str
    stratum: int
    modifiers_json: str  # JSON list of objects of type QueryModifier
    modifier_stats_json: str  # JSON object of type ModifierStats
    n_modifiers: int
    n_condition_admissions: int
    n_condition_chunks: int
    n_grounding_chunks: int
    grounding_hadm_ids: list[int]
    full_prompt: str


class QueryRow(TypedDict):
    """queries.parquet - QueryPromptRow minus full_prompt, plus query_text."""

    query_id: int
    icd10_3char: str
    condition_name: str
    stratum: int
    modifiers_json: str
    n_modifiers: int
    n_condition_admissions: int
    n_condition_chunks: int
    modifier_stats_json: str  # JSON: {modifier_text: {n_admissions: int, n_chunks: int}}
    n_grounding_chunks: int
    grounding_hadm_ids: list[int]
    query_text: str


class QueryRowPostFiltering(QueryRow, total=False):
    """queries.parquet - after running the query filtering step, which adds stats
    and a filter_<vec_col> boolean column (name depends on embedding model)."""

    jaccard_div: float
    fac_gap: float
    fac_topk: float
    fac_fl: float
    pool_size: int


class GoldAnnotationRow(TypedDict):
    """gold_annotations.parquet - one row per annotated query."""

    query_id: int
    icd10_3char: str
    condition_name: str
    modifiers_json: str
    query_text: str
    facets_json: str  # JSON dict: facet_label → list[chunk_id]
    answer_text: str  # unified comparative answer prose
    n_facets: int
    n_gold_chunks: int


class TagDecision(BaseModel):
    chunk_id: str
    reason: str = ''
