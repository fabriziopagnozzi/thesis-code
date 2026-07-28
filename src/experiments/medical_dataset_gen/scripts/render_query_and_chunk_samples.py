from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import cast

import polars as pl

from experiments.medical_dataset_gen.dataset_generation.chunk_rendering import (
    render_canonical_chunk,
)
from experiments.medical_dataset_gen.dataset_generation.facts import make_gold_fact
from experiments.medical_dataset_gen.dataset_generation.ontology_utils import load_ontology
from experiments.medical_dataset_gen.dataset_generation.query_templates import (
    render_query_template,
)
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    CHUNK_TEXT_STYLE_LIST,
    QUERY_FOCUS_MODE_LIST,
    QUERY_STRUCTURE_LIST,
    ChunkSurfacePolicy,
    ChunkTextStyle,
    ClinicalAxis,
    ClinicalFact,
    MedicalOntology,
    QueryFocusMode,
    QueryLogicalForm,
    QueryPlan,
    QueryPlanFacet,
    QueryStructure,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config,
    paths_for,
)

type JsonObject = dict[str, object]
type QueryChunkMode = tuple[ChunkTextStyle, QueryFocusMode, QueryStructure]

DEFAULT_PRIMARY_AXIS: ClinicalAxis = 'care_intensity'
DEFAULT_SECONDARY_AXIS: ClinicalAxis = 'treatment_duration'
DEFAULT_OUTPUT = MedicalDatasetGenPaths.root / 'docs' / 'code_docs' / 'query_and_chunk_modes.md'


@dataclass(frozen=True)
class CliArgs:
    exp: str
    results_dir: Path | None
    output: Path
    query_id: str | None
    condition_id: str | None
    subgroup_a_id: str | None
    subgroup_b_id: str | None
    primary_axis: ClinicalAxis
    secondary_axis: ClinicalAxis


def main() -> None:
    args = _parse_args()
    if args.results_dir is not None:
        MedicalDatasetGenPaths.results_dir = args.results_dir

    cfg = load_config(args.exp)
    paths = paths_for(cfg)
    ontology = load_ontology(cfg)
    plan = _select_plan(
        _load_plans(paths.table_path('query_plans')),
        query_id=args.query_id,
        condition_id=args.condition_id,
        subgroup_a_id=args.subgroup_a_id,
        subgroup_b_id=args.subgroup_b_id,
        primary_axis=args.primary_axis,
        secondary_axis=args.secondary_axis,
    )
    facts = _representative_gold_facts(
        paths=paths,
        plan=plan,
        ontology=ontology,
        chunk_surface_policy=cfg.generation.chunk_surface_policy,
    )
    rendered = _render_markdown(
        plan=plan,
        facts=facts,
        ontology=ontology,
        exp_name=paths.exp_name,
        configured_chunk_text_style=cfg.generation.chunk_text_style,
        configured_focus_mode=cfg.generation.focus_mode,
        configured_query_structure=cfg.generation.query_structure,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f'[write] {args.output}')


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description=(
            'Render one fixed query plan and one gold chunk per facet under all '
            'chunk/query surface modalities.'
        )
    )
    parser.add_argument(
        '--exp',
        default=os.getenv('EXP') or os.getenv('EXP_NAME'),
        help='Experiment name containing query_plans.parquet. Defaults to EXP/EXP_NAME.',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='Optional override for the medical_dataset_gen _results directory.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Markdown output path. Defaults to {DEFAULT_OUTPUT}.',
    )
    parser.add_argument(
        '--query-id',
        default=None,
        help='Specific query_id to render. Overrides the other plan-selection filters.',
    )
    parser.add_argument(
        '--condition-id',
        default=None,
        help='Optional fixed condition id filter.',
    )
    parser.add_argument(
        '--subgroup-a-id',
        default=None,
        help='Optional fixed subgroup A id filter.',
    )
    parser.add_argument(
        '--subgroup-b-id',
        default=None,
        help='Optional fixed subgroup B id filter.',
    )
    parser.add_argument(
        '--primary-axis',
        choices=['care_intensity', 'treatment_duration'],
        default=DEFAULT_PRIMARY_AXIS,
        help='Primary axis for the fixed plan.',
    )
    parser.add_argument(
        '--secondary-axis',
        choices=['care_intensity', 'treatment_duration'],
        default=DEFAULT_SECONDARY_AXIS,
        help='Secondary axis for the fixed plan.',
    )
    parsed = parser.parse_args()
    if parsed.exp is None:
        parser.error('missing --exp or EXP/EXP_NAME')
    if parsed.primary_axis == parsed.secondary_axis:
        parser.error('--primary-axis and --secondary-axis must differ')

    return CliArgs(
        exp=str(parsed.exp),
        results_dir=cast(Path | None, parsed.results_dir),
        output=cast(Path, parsed.output),
        query_id=cast(str | None, parsed.query_id),
        condition_id=cast(str | None, parsed.condition_id),
        subgroup_a_id=cast(str | None, parsed.subgroup_a_id),
        subgroup_b_id=cast(str | None, parsed.subgroup_b_id),
        primary_axis=cast(ClinicalAxis, parsed.primary_axis),
        secondary_axis=cast(ClinicalAxis, parsed.secondary_axis),
    )


def _load_plans(path: Path) -> list[QueryPlan]:
    if not path.exists():
        raise FileNotFoundError(f'missing query plans: {path}')
    frame = pl.read_parquet(path)
    return [_query_plan_from_row(row) for row in frame.iter_rows(named=True)]


def _query_plan_from_row(row: Mapping[str, object]) -> QueryPlan:
    facets = [
        QueryPlanFacet.model_construct(_fields_set=None, **facet)
        for facet in cast(list[JsonObject], json.loads(str(row['facets_json'])))
    ]
    logical_form_raw = cast(JsonObject, json.loads(str(row['logical_form_json'])))
    logical_form = QueryLogicalForm.model_construct(
        query_type=logical_form_raw['type'],
        condition=logical_form_raw['condition'],
        subgroups=logical_form_raw['subgroups'],
        axes=logical_form_raw['axes'],
        facets=logical_form_raw['facets'],
        cohort_contrast_family=logical_form_raw['cohort_contrast_family'],
        primary_axis=logical_form_raw['primary_axis'],
        secondary_axis=logical_form_raw['secondary_axis'],
        dominant_primary_facet_id=logical_form_raw['dominant_primary_facet_id'],
    )
    payload = {
        key: value for key, value in row.items() if key not in {'facets_json', 'logical_form_json'}
    }
    return QueryPlan.model_construct(
        _fields_set=None,
        **payload,
        facets=facets,
        logical_form=logical_form,
    )


def _select_plan(
    plans: Sequence[QueryPlan],
    *,
    query_id: str | None,
    condition_id: str | None,
    subgroup_a_id: str | None,
    subgroup_b_id: str | None,
    primary_axis: ClinicalAxis,
    secondary_axis: ClinicalAxis,
) -> QueryPlan:
    if query_id is not None:
        for plan in plans:
            if plan.query_id == query_id:
                return plan
        raise KeyError(f'unknown query_id: {query_id!r}')

    matches = [
        plan
        for plan in plans
        if plan.primary_axis == primary_axis
        and plan.secondary_axis == secondary_axis
        and (condition_id is None or plan.condition_id == condition_id)
        and (subgroup_a_id is None or plan.subgroup_a_id == subgroup_a_id)
        and (subgroup_b_id is None or plan.subgroup_b_id == subgroup_b_id)
    ]
    if matches:
        return matches[0]

    raise LookupError(
        'no query plan matched '
        f'primary_axis={primary_axis!r}, secondary_axis={secondary_axis!r}, '
        f'condition_id={condition_id!r}, subgroup_a_id={subgroup_a_id!r}, '
        f'subgroup_b_id={subgroup_b_id!r}'
    )


def _representative_gold_facts(
    *,
    paths: MedicalDatasetGenPaths,
    plan: QueryPlan,
    ontology: MedicalOntology,
    chunk_surface_policy: ChunkSurfacePolicy,
) -> list[ClinicalFact]:
    facts_path = paths.table_path('clinical_facts')
    if facts_path.exists():
        facts = _load_representative_gold_facts_from_artifact(facts_path, plan)
    else:
        facts = _make_representative_gold_facts(
            plan,
            ontology,
            chunk_surface_policy=chunk_surface_policy,
        )
    _assert_one_fact_per_facet(plan, facts)
    return facts


def _load_representative_gold_facts_from_artifact(
    facts_path: Path,
    plan: QueryPlan,
) -> list[ClinicalFact]:
    frame = (
        pl.scan_parquet(facts_path)
        .filter((pl.col('query_id') == plan.query_id) & pl.col('is_gold'))
        .collect()
    )
    by_facet: dict[str, ClinicalFact] = {}
    for row in frame.iter_rows(named=True):
        fact = ClinicalFact.model_construct(**row)
        if fact.facet_id is not None and fact.facet_id not in by_facet:
            by_facet[fact.facet_id] = fact
    return [by_facet[facet.facet_id] for facet in plan.facets if facet.facet_id in by_facet]


def _make_representative_gold_facts(
    plan: QueryPlan,
    ontology: MedicalOntology,
    *,
    chunk_surface_policy: ChunkSurfacePolicy,
) -> list[ClinicalFact]:
    rng = Random(plan.plan_seed)
    return [
        make_gold_fact(
            plan,
            facet,
            ontology,
            local_idx=0,
            rng=rng,
            chunk_surface_policy=chunk_surface_policy,
        )
        for facet in plan.facets
    ]


def _assert_one_fact_per_facet(plan: QueryPlan, facts: Sequence[ClinicalFact]) -> None:
    expected = {facet.facet_id for facet in plan.facets}
    observed = {fact.facet_id for fact in facts}
    missing = expected - observed
    if missing:
        raise RuntimeError(f'missing representative gold fact(s) for facets: {sorted(missing)}')
    if len(facts) != len(expected):
        raise RuntimeError(
            f'expected {len(expected)} representative gold facts, observed {len(facts)}'
        )


def _render_markdown(
    *,
    plan: QueryPlan,
    facts: Sequence[ClinicalFact],
    ontology: MedicalOntology,
    exp_name: str,
    configured_chunk_text_style: ChunkTextStyle,
    configured_focus_mode: QueryFocusMode,
    configured_query_structure: QueryStructure,
) -> str:
    lines = [
        '# Query and Chunk Mode Samples',
        '',
        'This file renders one fixed query setup under all eight surface modalities.',
        '',
        '## Fixed setup',
        '',
        f'- experiment: `{exp_name}`',
        f'- query_id: `{plan.query_id}`',
        f'- configured modality: `{configured_chunk_text_style}` / '
        f'`{configured_focus_mode}` / `{configured_query_structure}`',
        f'- condition: `{plan.condition_display}` (`{plan.condition_id}`)',
        f'- subgroup A: `{plan.subgroup_a_label}` (`{plan.subgroup_a_id}`)',
        f'- subgroup B: `{plan.subgroup_b_label}` (`{plan.subgroup_b_id}`)',
        f'- axes: `{plan.primary_axis}` primary, `{plan.secondary_axis}` secondary',
        f'- query template: `{plan.template_id}`',
        '',
        '## Facets',
        '',
    ]
    for fact in facts:
        lines.extend(
            [
                f'- `{fact.facet_id}`: `{fact.cluster_role}`, `{fact.subgroup_label}`, '
                f'`{fact.axis}`, `{fact.value_bin}`',
            ]
        )
    lines.extend(['', '## Modalities', ''])

    for index, (chunk_text_style, focus_mode, query_structure) in enumerate(_all_modes(), start=1):
        query_text = render_query_template(
            plan,
            ontology,
            focus_mode=focus_mode,
            query_structure=query_structure,
        )
        lines.extend(
            [
                f'### {index}. `{chunk_text_style}` / `{focus_mode}` / `{query_structure}`',
                '',
                '**Query**',
                '',
                query_text,
                '',
                '**Gold facet chunks**',
                '',
            ]
        )
        for chunk_index, fact in enumerate(facts, start=1):
            rendered = render_canonical_chunk(fact, ontology, text_style=chunk_text_style)
            lines.extend(
                [
                    f'#### Chunk {chunk_index}: `{fact.facet_id}`',
                    '',
                    f'- subgroup: `{fact.subgroup_label}`',
                    f'- axis/bin: `{fact.axis}` / `{fact.value_bin}`',
                    f'- role: `{fact.cluster_role}`',
                    f'- fact_id: `{fact.fact_id}`',
                    '',
                    rendered.text,
                    '',
                ]
            )
    return '\n'.join(lines).rstrip() + '\n'


def _all_modes() -> list[QueryChunkMode]:
    return [
        (chunk_text_style, focus_mode, query_structure)
        for chunk_text_style in CHUNK_TEXT_STYLE_LIST
        for focus_mode in QUERY_FOCUS_MODE_LIST
        for query_structure in QUERY_STRUCTURE_LIST
    ]


if __name__ == '__main__':
    main()
