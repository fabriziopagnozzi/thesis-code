import re

from pydantic import BaseModel, PositiveInt, computed_field

from helpers.query_algorithms import ScoringFunction

from .config_loaders import load_default_config


# PYDANTIC MODELS FOR CONFIGS
# Global
class GlobalCfg(BaseModel):
    num_conditions: PositiveInt
    results_subdir: str | None


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
        return cls(**load_default_config(phase=1)['note_chunking'])


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
    model_name: str
    batch_size: int
    commit_every: int
    device: str

    @classmethod
    def load(cls) -> EmbedCfg:
        return cls(**load_default_config(phase=2)['embed'])


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
    charlson_labels: dict[str, str]
    demographic_modifiers: list[DemographicModifier]
    personas: dict[str, str]
    prompt_template: str

    @computed_field
    @property
    def demographic_modifiers_text(self) -> list[str]:
        return [m.text for m in self.demographic_modifiers]

    @computed_field
    @property
    def demographic_filters(self) -> dict[str, tuple]:
        return {m.text: (m.column, m.op, m.value) for m in self.demographic_modifiers}

    @computed_field
    @property
    def label_to_charlson_col(self) -> dict[str, str]:
        return {v: k for k, v in self.charlson_labels.items()}

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
    prefilter_n: int

    @classmethod
    def load(cls) -> FilterQueriesCfg:
        return cls(**load_default_config(phase=3)['filter_queries'])


class GoldAnnotationCfg(BaseModel):
    prefilter_n: int
    batch_size: int
    num_ctx: int | None = None
    num_predict: int | None = None
    model: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    think: bool = False
    map_system_prompt: str
    map_user_template: str

    @classmethod
    def load(cls) -> GoldAnnotationCfg:
        return cls(**load_default_config(phase=3)['gold_annotation'])


# -- Phase 4 --
class EvaluateCfg(BaseModel):
    embedding_model: str
    strategies: list[ScoringFunction]
    k_values: list[int]
    lam_values: list[float]
    prefilter_n: int
    device: str

    @classmethod
    def load(cls) -> EvaluateCfg:
        return cls(**load_default_config(phase=4)['evaluate'])
