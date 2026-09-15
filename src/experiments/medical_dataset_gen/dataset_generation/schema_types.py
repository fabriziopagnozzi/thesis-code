"""Shared vocabularies and validation base for benchmark construction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from experiments.medical_dataset_gen.utils.global_utils import get_literals


class BenchmarkPydanticModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


type ClinicalAxis = Literal[
    'treatment_duration',
    'rehab_outcome',
    'complication_burden',
    'acute_clinical_course',
    'care_intensity',
    'diagnostic_evidence_type',
]
CLINICAL_AXIS_LIST = list[ClinicalAxis](get_literals(ClinicalAxis))

type ClusterRole = Literal[
    'dominant_primary_gold',
    'primary_gold',
    'secondary_gold',
    'niche_gold',
    'hard_distractor',
    'background_outlier',
]

type QueryType = Literal['prioritized_subgroup_comparison']
type CohortContrastFamily = Literal[
    'demographic',
    'comorbidity_present_absent',
    'distinct_comorbidity',
]

type ChunkPoolScope = Literal['query_local']
type ChunkTextStyle = Literal['ontology_explicit', 'semantic_hardened']
CHUNK_TEXT_STYLE_LIST = list[ChunkTextStyle](get_literals(ChunkTextStyle))

type QueryFocusMode = Literal['list', 'natural']
QUERY_FOCUS_MODE_LIST = list[QueryFocusMode](get_literals(QueryFocusMode))

type QueryStructure = Literal['unbalanced', 'balanced']
QUERY_STRUCTURE_LIST = list[QueryStructure](get_literals(QueryStructure))

type QueryWordingMode = tuple[QueryStructure, QueryFocusMode]
QUERY_WORDING_MODE_LIST: list[QueryWordingMode] = [
    ('unbalanced', 'list'),
    ('unbalanced', 'natural'),
    ('balanced', 'list'),
    ('balanced', 'natural'),
]

type ChunkSurfaceGroup = Literal['seen', 'heldout']
type ChunkSurfacePolicy = Literal['split_heldout', 'seen_only', 'heldout_only']
type ConditionAnchor = Literal['outer_template', 'axis_evidence']
type AxisTemplateFamily = Literal[
    'direct_fact',
    'temporal_course',
    'clinical_assessment',
    'contrast_or_alternative',
]
AXIS_TEMPLATE_FAMILY_LIST = list[AxisTemplateFamily](get_literals(AxisTemplateFamily))

type SubgroupAxis = Literal['demographic', 'comorbidity']
type SubgroupKey = str
type ConditionKey = str
type PatientSex = Literal['female', 'male']
type DataSplit = Literal['validation', 'test']
