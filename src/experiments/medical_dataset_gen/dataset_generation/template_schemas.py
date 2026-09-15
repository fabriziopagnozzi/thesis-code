"""Schemas for authored query, answer, and clinical-document templates."""

from __future__ import annotations

from pydantic import Field, model_validator

from experiments.medical_dataset_gen.dataset_generation.schema_types import (
    AXIS_TEMPLATE_FAMILY_LIST,
    CLINICAL_AXIS_LIST,
    QUERY_FOCUS_MODE_LIST,
    QUERY_STRUCTURE_LIST,
    AxisTemplateFamily,
    BenchmarkPydanticModel,
    ChunkSurfaceGroup,
    ClinicalAxis,
    ConditionAnchor,
    QueryFocusMode,
    QueryStructure,
)


class TemplateSpec(BenchmarkPydanticModel):
    id: str
    template: str


class SurfaceTemplateBucket(BenchmarkPydanticModel):
    seen: list[TemplateSpec] = Field(min_length=1)
    heldout: list[TemplateSpec] = Field(min_length=1)

    def templates_for_group(self, surface_group: ChunkSurfaceGroup) -> list[TemplateSpec]:
        if surface_group == 'seen':
            return self.seen
        return self.heldout


class CohortEvidenceTemplates(BenchmarkPydanticModel):
    comorbidity_present: SurfaceTemplateBucket
    comorbidity_reference: SurfaceTemplateBucket


class NoteStyleTemplates(BenchmarkPydanticModel):
    outer_template: dict[str, SurfaceTemplateBucket]
    axis_evidence: dict[str, SurfaceTemplateBucket]

    @model_validator(mode='after')
    def _validate_matching_families(self) -> NoteStyleTemplates:
        if set(self.outer_template) != set(self.axis_evidence):
            raise ValueError(
                'note-style families must match for outer_template and axis_evidence anchors'
            )
        return self

    def templates_for_anchor(
        self,
        condition_anchor: ConditionAnchor,
    ) -> dict[str, SurfaceTemplateBucket]:
        if condition_anchor == 'outer_template':
            return self.outer_template
        return self.axis_evidence


class PairedAxisSentenceTemplates(BenchmarkPydanticModel):
    """Clinical evidence wording shared by the two document-text controls."""

    seen: dict[AxisTemplateFamily, list[TemplateSpec]]
    heldout: dict[AxisTemplateFamily, list[TemplateSpec]]

    @model_validator(mode='after')
    def _validate_surface_counts(self) -> PairedAxisSentenceTemplates:
        self._validate_group('seen', self.seen)
        self._validate_group('heldout', self.heldout)
        return self

    @staticmethod
    def _validate_group(
        surface_group: ChunkSurfaceGroup,
        templates: dict[AxisTemplateFamily, list[TemplateSpec]],
    ) -> None:
        expected = set(AXIS_TEMPLATE_FAMILY_LIST)
        actual = set(templates)
        if actual != expected:
            raise ValueError(
                f'paired axis templates {surface_group} families must be {sorted(expected)}; '
                f'got {sorted(actual)}'
            )
        empty = [family for family, specs in templates.items() if not specs]
        if empty:
            raise ValueError(f'paired axis templates {surface_group} families are empty: {empty}')

    def template_specs(
        self,
        surface_group: ChunkSurfaceGroup,
    ) -> list[tuple[AxisTemplateFamily, TemplateSpec]]:
        templates = self.seen if surface_group == 'seen' else self.heldout
        return [
            (family, spec)
            for family in AXIS_TEMPLATE_FAMILY_LIST
            for spec in templates.get(family, [])
        ]


class ChunkAxisSentenceTemplates(BenchmarkPydanticModel):
    paired: dict[ClinicalAxis, PairedAxisSentenceTemplates]


class ChunkTemplateUtils(BenchmarkPydanticModel):
    hidden_benchmark_terms: list[str]
    note_style_templates: NoteStyleTemplates
    cohort_evidence_templates: CohortEvidenceTemplates
    treatment_course_templates: dict[str, SurfaceTemplateBucket]
    axis_sentence_templates: ChunkAxisSentenceTemplates
    simple_interpretations: dict[ClinicalAxis, dict[str, SurfaceTemplateBucket]]

    @model_validator(mode='after')
    def _validate_template_inventory(self) -> ChunkTemplateUtils:
        template_ids: set[str] = set()
        duplicate_ids: set[str] = set()
        for templates_by_family in (
            self.note_style_templates.outer_template,
            self.note_style_templates.axis_evidence,
        ):
            for bucket in templates_by_family.values():
                _record_template_ids(bucket, template_ids, duplicate_ids)

        for bucket in (
            self.cohort_evidence_templates.comorbidity_present,
            self.cohort_evidence_templates.comorbidity_reference,
        ):
            _record_template_ids(bucket, template_ids, duplicate_ids)

        if not self.treatment_course_templates:
            raise ValueError('treatment_course_templates must not be empty')
        for bucket in self.treatment_course_templates.values():
            _record_template_ids(bucket, template_ids, duplicate_ids)

        expected_axes = set(CLINICAL_AXIS_LIST)
        actual_axes = set(self.axis_sentence_templates.paired)
        if actual_axes != expected_axes:
            raise ValueError(
                f'paired axis templates must define every clinical axis; '
                f'missing={sorted(expected_axes - actual_axes)}, '
                f'unexpected={sorted(actual_axes - expected_axes)}'
            )
        for axis, paired in self.axis_sentence_templates.paired.items():
            seen_specs = paired.template_specs('seen')
            heldout_specs = paired.template_specs('heldout')
            expected_count = len(AXIS_TEMPLATE_FAMILY_LIST)
            if len(seen_specs) != expected_count or len(heldout_specs) != expected_count:
                raise ValueError(
                    f'paired {axis} templates must define exactly {expected_count} '
                    'seen and heldout families'
                )
            for _, spec in [*seen_specs, *heldout_specs]:
                _record_template_id(spec, template_ids, duplicate_ids)

        if set(self.simple_interpretations) != expected_axes:
            raise ValueError('simple_interpretations must define every clinical axis')
        for axis, interpretations_by_bin in self.simple_interpretations.items():
            if not interpretations_by_bin:
                raise ValueError(f'simple_interpretations for {axis!r} must not be empty')
            for value_bin, bucket in interpretations_by_bin.items():
                if not value_bin.strip():
                    raise ValueError(f'simple_interpretations for {axis!r} contains an empty bin')
                _record_template_ids(bucket, template_ids, duplicate_ids)

        if duplicate_ids:
            raise ValueError(f'duplicate chunk template ids: {sorted(duplicate_ids)}')
        return self


class AnswerTemplateSpec(BenchmarkPydanticModel):
    template: str


class QueryTemplateData(BenchmarkPydanticModel):
    query_templates: dict[QueryStructure, dict[QueryFocusMode, list[TemplateSpec]]]
    answer_template: AnswerTemplateSpec

    @model_validator(mode='after')
    def _validate_query_template_structures(self) -> QueryTemplateData:
        if set(self.query_templates) != set(QUERY_STRUCTURE_LIST):
            raise ValueError(
                f'query_templates must define exactly these structures: {QUERY_STRUCTURE_LIST}'
            )
        expected_ids: list[str] | None = None
        for structure in QUERY_STRUCTURE_LIST:
            specs_by_mode = self.query_templates[structure]
            expected_focus_modes = QUERY_FOCUS_MODE_LIST
            if set(specs_by_mode) != set(expected_focus_modes):
                raise ValueError(
                    f'query_templates.{structure} must define exactly these focus modes: '
                    f'{expected_focus_modes}'
                )
            for focus_mode in expected_focus_modes:
                specs = specs_by_mode[focus_mode]
                ids = [spec.id for spec in specs]
                if not ids:
                    raise ValueError(f'query_templates.{structure}.{focus_mode} must not be empty')
                if len(ids) != len(set(ids)):
                    raise ValueError(
                        f'duplicate query template ids for {structure}/{focus_mode}: {ids}'
                    )
                if expected_ids is None:
                    expected_ids = ids
                elif ids != expected_ids:
                    raise ValueError(
                        'query template structures and focus modes must define the same ids '
                        'in the same order'
                    )
        return self


def _record_template_ids(
    bucket: SurfaceTemplateBucket,
    template_ids: set[str],
    duplicate_ids: set[str],
) -> None:
    for spec in [*bucket.seen, *bucket.heldout]:
        _record_template_id(spec, template_ids, duplicate_ids)


def _record_template_id(
    spec: TemplateSpec,
    template_ids: set[str],
    duplicate_ids: set[str],
) -> None:
    if spec.id in template_ids:
        duplicate_ids.add(spec.id)
    template_ids.add(spec.id)
