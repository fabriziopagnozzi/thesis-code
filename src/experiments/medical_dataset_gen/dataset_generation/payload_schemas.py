"""Validated condition-specific payloads used to realize clinical facts."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field

from experiments.medical_dataset_gen.dataset_generation.schema_types import (
    BenchmarkPydanticModel,
    ClinicalAxis,
)


class TreatmentDurationPayload(BenchmarkPydanticModel):
    axis: Literal['treatment_duration']
    duration_days: int
    treatment: str
    treatment_course_id: str


class RehabOutcomePayload(BenchmarkPydanticModel):
    axis: Literal['rehab_outcome']
    outcome: str


class ComplicationBurdenPayload(BenchmarkPydanticModel):
    axis: Literal['complication_burden']
    detail: str


class AcuteClinicalCoursePayload(BenchmarkPydanticModel):
    axis: Literal['acute_clinical_course']
    detail: str


class CareIntensityPayload(BenchmarkPydanticModel):
    axis: Literal['care_intensity']
    detail: str


class DiagnosticEvidencePayload(BenchmarkPydanticModel):
    axis: Literal['diagnostic_evidence_type']
    detail: str


type AxisFactPayload = Annotated[
    TreatmentDurationPayload
    | RehabOutcomePayload
    | ComplicationBurdenPayload
    | AcuteClinicalCoursePayload
    | CareIntensityPayload
    | DiagnosticEvidencePayload,
    Field(discriminator='axis'),
]


def parse_axis_payload(value: str) -> AxisFactPayload:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError('axis payload must be a JSON object')
    axis = payload.get('axis')
    if not isinstance(axis, str):
        raise ValueError('axis payload must contain a string axis')
    model_by_axis = {
        'treatment_duration': TreatmentDurationPayload,
        'rehab_outcome': RehabOutcomePayload,
        'complication_burden': ComplicationBurdenPayload,
        'acute_clinical_course': AcuteClinicalCoursePayload,
        'care_intensity': CareIntensityPayload,
        'diagnostic_evidence_type': DiagnosticEvidencePayload,
    }
    try:
        model = model_by_axis[axis]
    except KeyError as exc:
        raise ValueError(f'unsupported axis payload: {axis!r}') from exc
    return model.model_validate(payload)


def axis_payload_required_phrase(payload: AxisFactPayload) -> str:
    if isinstance(payload, TreatmentDurationPayload):
        return f'{payload.duration_days} days of {payload.treatment}'
    if isinstance(payload, RehabOutcomePayload):
        return payload.outcome
    return payload.detail


def validate_axis_payload(axis: ClinicalAxis, axis_payload_json: str) -> AxisFactPayload:
    payload = parse_axis_payload(axis_payload_json)
    if payload.axis != axis:
        raise ValueError(f'axis payload {payload.axis!r} does not match {axis!r}')
    return payload
