from __future__ import annotations

from pydantic import BaseModel

from experiments.mimic.chunking.schemas_chunking import ChunkRow
from experiments.mimic.global_configs import load_default_config


class EmbedCfg(BaseModel):
    models: list[str]
    batch_sizes: list[int]
    commit_every: int

    @classmethod
    def load(cls) -> EmbedCfg:
        return cls(**load_default_config(key='embeddings'))


class EmbedJoinedRow(ChunkRow, total=False):
    """chunks LEFT JOIN admissions_metadata (selected cols) iterated in embed_whole_corpus.py."""

    age: float | None
    gender: str
    race: str
    primary_icd_description: str
    top_icd_descriptions: str
    admission_type: str | None
    charlson_comorbidity_index: int
    myocardial_infarct: int | None
    congestive_heart_failure: int | None
    peripheral_vascular_disease: int | None
    cerebrovascular_disease: int | None
    dementia: int | None
    chronic_pulmonary_disease: int | None
    rheumatic_disease: int | None
    peptic_ulcer_disease: int | None
    mild_liver_disease: int | None
    severe_liver_disease: int | None
    diabetes_without_cc: int | None
    diabetes_with_cc: int | None
    paraplegia: int | None
    renal_disease: int | None
    malignant_cancer: int | None
    metastatic_solid_tumor: int | None
    aids: int | None
