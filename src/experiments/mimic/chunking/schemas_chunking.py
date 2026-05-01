import re
from typing import TypedDict

from pydantic import BaseModel, computed_field

from experiments.mimic.global_configs import load_default_config


class ConditionsStatsCfg(BaseModel):
    min_admissions: int = 0
    cond_processing_llm: str = 'gemma4-31b-text'

    @classmethod
    def load(cls) -> ConditionsStatsCfg:
        return cls(**load_default_config(key='chunking')['conditions_stats'])


class NoteChunkingCfg(BaseModel):
    model_config = {'arbitrary_types_allowed': True}

    keep_sections: set[str] = {
        'DISCHARGE MEDICATIONS',
    }
    skip_sections: set[str] = {
        'HISTORY OF PRESENT ILLNESS',
        'DISCHARGE DIAGNOSIS',
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


# ---------------------------------------------------------------------------
# Phase 1 - corpus construction
class ConditionStatsRow(TypedDict):
    """conditions_stats.parquet - one row per ICD-10 3-char prefix condition."""

    icd10_3char: str
    condition_name: str
    n_admissions: int
    mean_comorbidity_count: float
    top_comorbidity_mods_json: str  # JSON array of modifier dicts


class ChunkRow(TypedDict):
    """chunks.parquet - one row per overlapping text window."""

    text: str
    chunk_id: str
    note_id: str
    subject_id: int
    hadm_id: int
    section_name: str
    chief_complaint: str | None
    char_count: int
    approx_tokens: int
    contextual_prefix: str | None


class _AdmissionMetadataBase(TypedDict):
    """Always-present fields sourced from the admissions table."""

    hadm_id: int
    subject_id: int
    age: float | None
    gender: str
    race: str
    insurance: str
    marital_status: str
    admission_type: str
    discharge_location: str
    hospital_expire_flag: int | None
    primary_icd_code: str
    primary_icd_description: str
    top_icd_descriptions: str


class AdmissionMetadataRow(_AdmissionMetadataBase, total=False):
    """admissions_metadata.parquet - base fields + Charlson fields (null when absent from Charlson view)."""

    age_score: int
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


class AdmissionMetaSlimRow(TypedDict):
    """Slim projection of admissions_metadata used in query-prompt grounding."""

    hadm_id: int
    age: float | None
    gender: str
    race: str
