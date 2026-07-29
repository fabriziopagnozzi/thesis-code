from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from experiments.medical_dataset_gen.dataset_generation.chunk_rendering import (
    render_canonical_chunk,
)
from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    CHUNK_TEMPLATE_DATA,
    validate_chunk_template_sources,
    validate_chunk_text,
)
from experiments.medical_dataset_gen.dataset_generation.facts import (
    _payload_required_phrase,
)
from experiments.medical_dataset_gen.dataset_generation.ontology_utils import (
    get_axis_pair_profiles,
    load_ontology,
)
from experiments.medical_dataset_gen.dataset_generation.planning import _materialize_plan
from experiments.medical_dataset_gen.dataset_generation.query_templates import (
    query_template_ids,
    render_query_template,
)
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    CHUNK_TEXT_STYLE_LIST,
    CLINICAL_AXIS_LIST,
    QUERY_WORDING_MODE_LIST,
    AcuteClinicalCoursePayload,
    AxisFactPayload,
    CareIntensityPayload,
    ChunkSurfaceGroup,
    ClinicalAxis,
    ClinicalFact,
    ComplicationBurdenPayload,
    DiagnosticEvidencePayload,
    MedicalOntology,
    QueryPlanSpec,
    RehabOutcomePayload,
    TreatmentDurationAxisValues,
    TreatmentDurationPayload,
)
from experiments.medical_dataset_gen.utils.global_schemas import ExperimentCfg
from experiments.medical_dataset_gen.utils.global_utils import MedicalDatasetGenPaths

DEFAULT_OUTPUT_DIR = MedicalDatasetGenPaths.results_dir / '_v4_language_review'


def main() -> None:
    args = _parse_args()
    cfg = _review_cfg()
    ontology = load_ontology(cfg)
    validate_chunk_template_sources(ontology)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    surface_inventory = pl.from_dicts(_surface_inventory_rows(), infer_schema_length=None)
    chunk_samples = pl.from_dicts(_chunk_sample_rows(cfg), infer_schema_length=None)
    query_samples = pl.from_dicts(_query_sample_rows(cfg), infer_schema_length=None)

    surface_inventory.write_parquet(args.output_dir / 'surface_inventory.parquet')
    chunk_samples.write_parquet(args.output_dir / 'chunk_samples.parquet')
    query_samples.write_parquet(args.output_dir / 'query_samples.parquet')
    (args.output_dir / 'README.md').write_text(
        _readme(
            surface_inventory=surface_inventory,
            chunk_samples=chunk_samples,
            query_samples=query_samples,
        )
    )
    print(f'[write] v4 language review -> {args.output_dir}')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _review_cfg() -> ExperimentCfg:
    return ExperimentCfg.model_validate({
        'dataset_schema_version': 4,
        'global': {'seed': 42, 'conditions': 22, 'output_experiment': '_v4_language_review'},
        'generation': {
            'chunk_pools': {
                'dominant_primary': {'size': 24},
                'other_primary': {'size': 20},
                'secondary': {'size': 14},
                'niche': {'size': 4, 'num_clusters_per_query': 0},
            },
        },
        'retrieval': {'pool_scope': 'query_local'},
    })


def _surface_inventory_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_anchor in ('outer_template', 'axis_evidence'):
        templates = CHUNK_TEMPLATE_DATA.note_style_templates.templates_for_anchor(condition_anchor)
        for family, bucket in templates.items():
            rows.extend(_bucket_rows('note_style', f'{condition_anchor}:{family}', bucket))
    for family, bucket in CHUNK_TEMPLATE_DATA.cohort_evidence_templates.model_dump().items():
        for group, specs in bucket.items():
            for spec in specs:
                rows.append({
                    'source': 'cohort_evidence',
                    'family': family,
                    'surface_group': group,
                    'template_id': spec['id'],
                    'template': spec['template'],
                })
    for course_id, bucket in CHUNK_TEMPLATE_DATA.treatment_course_templates.items():
        rows.extend(_bucket_rows('treatment_course', course_id, bucket))
    for axis, templates in CHUNK_TEMPLATE_DATA.axis_sentence_templates.paired.items():
        for surface_group in ('seen', 'heldout'):
            for family, spec in templates.template_specs(surface_group):
                rows.append({
                    'source': 'axis_sentence',
                    'family': f'{axis}:{family}',
                    'surface_group': surface_group,
                    'template_id': spec.id,
                    'template': spec.template,
                })
    for axis, interpretations_by_bin in CHUNK_TEMPLATE_DATA.simple_interpretations.items():
        for value_bin, bucket in interpretations_by_bin.items():
            rows.extend(_bucket_rows('simple_interpretation', f'{axis}:{value_bin}', bucket))
    return rows


def _bucket_rows(source: str, family: str, bucket) -> list[dict[str, object]]:
    return [
        {
            'source': source,
            'family': family,
            'surface_group': surface_group,
            'template_id': spec.id,
            'template': spec.template,
        }
        for surface_group in ('seen', 'heldout')
        for spec in bucket.templates_for_group(surface_group)
    ]


def _chunk_sample_rows(cfg: ExperimentCfg) -> list[dict[str, object]]:
    ontology = load_ontology(cfg)
    rows: list[dict[str, object]] = []
    note_styles = list(CHUNK_TEMPLATE_DATA.note_style_templates.outer_template)
    for condition_id, condition in ontology.conditions.items():
        for axis in CLINICAL_AXIS_LIST:
            for value_bin in ontology.clinical_axes[axis].bins:
                payloads = _all_axis_payloads(ontology, condition_id, axis, value_bin)
                for payload_index, payload in enumerate(payloads):
                    for surface_group in ('seen', 'heldout'):
                        for note_style in note_styles:
                            fact = _sample_fact(
                                cfg=cfg,
                                ontology=ontology,
                                condition_id=condition_id,
                                axis=axis,
                                value_bin=value_bin,
                                surface_group=surface_group,
                                payload=payload,
                                payload_index=payload_index,
                                note_style=note_style,
                            )
                            for style in CHUNK_TEXT_STYLE_LIST:
                                rendered = render_canonical_chunk(
                                    fact,
                                    ontology,
                                    text_style=style,
                                )
                                validation = validate_chunk_text(
                                    rendered.text,
                                    fact,
                                    ontology,
                                    text_style=style,
                                )
                                if validation.hard_errors:
                                    raise RuntimeError(
                                        f'invalid sample '
                                        f'{condition_id}/{axis}/{value_bin}/{style}: '
                                        + '; '.join(validation.hard_errors)
                                    )
                                rows.append({
                                    'condition_id': condition_id,
                                    'condition_display': condition.display,
                                    'axis': axis,
                                    'value_bin': value_bin,
                                    'axis_payload_json': fact.axis_payload_json,
                                    'condition_anchor': fact.condition_anchor,
                                    'note_style': note_style,
                                    'chunk_text_style': style,
                                    'surface_group': surface_group,
                                    'outer_template_id': rendered.provenance.outer_template_id,
                                    'axis_template_family': rendered.provenance.axis_template_family,
                                    'axis_template_id': rendered.provenance.axis_template_id,
                                    'text': rendered.text,
                                })
    return rows


def _sample_fact(
    *,
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    condition_id: str,
    axis: ClinicalAxis,
    value_bin: str,
    surface_group: ChunkSurfaceGroup,
    payload: AxisFactPayload,
    payload_index: int,
    note_style: str,
) -> ClinicalFact:
    condition = ontology.conditions[condition_id]
    subgroup_id = 'age_under_50'
    subgroup = ontology.subgroups[subgroup_id]
    payload_json = json.dumps(payload.model_dump(mode='json'), sort_keys=True)
    axis_bin_term = ontology.clinical_axes[axis].bin_terms[value_bin][0]
    reuse_key = _stable_hash(
        condition_id,
        axis,
        value_bin,
        surface_group,
        payload_index,
        note_style,
        payload_json,
    )
    required_payload = _payload_required_phrase(payload)
    condition_anchor = (
        'axis_evidence'
        if condition.display.casefold() in required_payload.casefold()
        else 'outer_template'
    )
    return ClinicalFact(
        query_id=f'review_{condition_id}_{axis}_{value_bin}',
        evidence_profile_id='review_profile',
        pool_id='review_pool',
        primary_axis=axis,
        secondary_axis=next(other for other in CLINICAL_AXIS_LIST if other != axis),
        dominant_primary_facet_id='review_f1',
        fact_id=f'review_fact_{reuse_key[:12]}',
        chunk_reuse_key=reuse_key,
        facet_id='review_f1',
        target_facet_id='review_f1',
        cluster_id='review_c1',
        cluster_role='dominant_primary_gold',
        condition_id=condition_id,
        condition_display=condition.display,
        subgroup_id=subgroup_id,
        subgroup_label=subgroup.label,
        subgroup_axis=subgroup.axis,
        subgroup_field=subgroup.field,
        subgroup_value=subgroup.value,
        subgroup_dimension_id=subgroup.dimension_id,
        subgroup_level_id=subgroup.level_id,
        subgroup_is_reference=subgroup.is_reference,
        axis=axis,
        value_bin=value_bin,
        axis_bin_term=axis_bin_term,
        axis_payload_json=payload_json,
        condition_anchor=condition_anchor,
        facet_priority='primary',
        is_gold=True,
        distractor_type=None,
        admission_id=f'adm_review_{reuse_key[:12]}',
        patient_id=f'pat_review_{subgroup_id}',
        patient_age=subgroup.patient_age_range[0] if subgroup.patient_age_range else 42,
        patient_sex='female',
        clinical_subgroup_phrase=subgroup.surface_phrases[0],
        note_style=note_style,
        chunk_surface_group=surface_group,
        split='validation' if surface_group == 'seen' else 'test',
        must_mention=[
            condition.display,
            ontology.clinical_axes[axis].label,
            axis_bin_term,
            required_payload,
        ],
        must_not_mention=[],
    )


def _all_axis_payloads(
    ontology: MedicalOntology,
    condition_id: str,
    axis: ClinicalAxis,
    value_bin: str,
) -> list[AxisFactPayload]:
    axis_values = ontology.conditions[condition_id].axis_values[axis]
    if axis == 'treatment_duration':
        if not isinstance(axis_values, TreatmentDurationAxisValues):
            raise TypeError(f'{condition_id}/{axis} has the wrong axis-values payload')
        return [
            TreatmentDurationPayload(
                axis='treatment_duration',
                duration_days=duration_days,
                treatment=treatment,
                treatment_course_id=course_id,
            )
            for course_id, course in axis_values.treatments.items()
            for duration_days in course.bins[value_bin]
            for treatment in course.surface_forms
        ]

    raw_values = axis_values.bins[value_bin]
    if any(not isinstance(value, str) for value in raw_values):
        raise TypeError(f'{condition_id}/{axis}/{value_bin} contains a non-text value')
    values = [str(value) for value in raw_values]
    if axis == 'rehab_outcome':
        return [RehabOutcomePayload(axis=axis, outcome=value) for value in values]
    if axis == 'complication_burden':
        return [ComplicationBurdenPayload(axis=axis, detail=value) for value in values]
    if axis == 'acute_clinical_course':
        return [AcuteClinicalCoursePayload(axis=axis, detail=value) for value in values]
    if axis == 'care_intensity':
        return [CareIntensityPayload(axis=axis, detail=value) for value in values]
    return [
        DiagnosticEvidencePayload(axis='diagnostic_evidence_type', detail=value) for value in values
    ]


def _query_sample_rows(cfg: ExperimentCfg) -> list[dict[str, object]]:
    ontology = load_ontology(cfg)
    contrast = ontology.cohort_contrasts[0]
    profile = get_axis_pair_profiles(ontology, 'care_intensity', 'treatment_duration')[0]
    spec = QueryPlanSpec(
        evidence_profile_id='review_query_profile',
        cohort_contrast_id=contrast.id,
        cohort_contrast_family='demographic',
        cohort_dimension_id=contrast.dimension_id,
        axis_a='care_intensity',
        axis_b='treatment_duration',
        profile_id=profile.id,
        cohort_a_bins=profile.cohort_a_bins,
        cohort_b_bins=profile.cohort_b_bins,
        condition_key='pneumonia',
        condition_display=ontology.conditions['pneumonia'].display,
        subgroup_a_id=contrast.cohort_a_id,
        subgroup_a=ontology.subgroups[contrast.cohort_a_id],
        subgroup_b_id=contrast.cohort_b_id,
        subgroup_b=ontology.subgroups[contrast.cohort_b_id],
    )
    plan = _materialize_plan(
        cfg,
        ontology,
        spec,
        primary_axis='care_intensity',
        secondary_axis='treatment_duration',
        query_id='review_q1',
    )
    rows: list[dict[str, object]] = []
    for structure, focus_mode in QUERY_WORDING_MODE_LIST:
        for template_id in query_template_ids(structure, focus_mode):
            rows.append({
                'query_structure': structure,
                'focus_mode': 'label_only' if structure == 'label_only' else focus_mode,
                'template_id': template_id,
                'query_text': render_query_template(
                    plan,
                    ontology,
                    template_id=template_id,
                    focus_mode=focus_mode,
                    query_structure=structure,
                ),
            })
    return rows


def _readme(
    *,
    surface_inventory: pl.DataFrame,
    chunk_samples: pl.DataFrame,
    query_samples: pl.DataFrame,
) -> str:
    return (
        '# V4 Language Review\n\n'
        'This directory is generated by `render_v4_language_review.py` without embeddings or '
        'retrieval evaluation.\n\n'
        f'- `surface_inventory.parquet`: {surface_inventory.height:,} authored template rows.\n'
        f'- `chunk_samples.parquet`: {chunk_samples.height:,} rendered chunk rows covering '
        'every ontology payload, note style, chunk style, and surface group combination.\n'
        f'- `query_samples.parquet`: {query_samples.height:,} rendered query rows covering '
        'all query structures, focus modes, and template IDs.\n'
    )


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


if __name__ == '__main__':
    main()
