from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.dataset_generation.chunk_rendering import (
    chunk_id,
    finalize_chunk_row,
    new_chunk_state,
    render_canonical_chunk,
)
from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    validate_chunk_text,
    validate_chunk_template_sources,
)
from experiments.medical_dataset_gen.dataset_generation.ontology_utils import load_ontology
from experiments.medical_dataset_gen.dataset_generation.schemas import (
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.utils.global_schemas import (
    ExperimentCfg,
)
from experiments.medical_dataset_gen.utils.global_utils import (
    MedicalDatasetGenPaths,
    load_config_from_cli,
    paths_for,
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet


def run_make_chunks(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    validate_chunk_template_sources(ontology)
    facts = read_parquet(paths, 'clinical_facts')
    print(f'[chunks] deterministic v4 rendering for {len(facts):,} facts')
    return _render_chunks_deterministic_parallel(
        cfg=cfg,
        paths=paths,
        facts=facts,
        ontology=ontology,
    )


def _render_chunks_deterministic_parallel(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    facts: pl.DataFrame,
    ontology: MedicalOntology,
) -> pl.DataFrame:
    cfg_dump = cfg.model_dump(mode='python')
    ontology_dump = ontology.model_dump(mode='python')
    n_batches = facts.select(pl.col('query_id').n_unique()).item()
    workers = max(1, os.cpu_count() or 1)
    rows_all: list[dict[str, object]] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_deterministic_worker,
        initargs=(cfg_dump, ontology_dump),
    ) as executor:
        batch_iter = _iter_fact_batches(facts)
        for _query_id, rows in tqdm(
            executor.map(_render_deterministic_chunk_batch, batch_iter, chunksize=1),
            total=n_batches,
            desc='Rendering chunks',
            dynamic_ncols=True,
        ):
            rows_all.extend(rows)

    chunk_rows = _chunk_rows_frame(rows_all) if rows_all else pl.DataFrame()
    chunk_documents, chunk_memberships = _write_normalized_chunks(paths, chunk_rows)

    print(
        f'[chunks] normalized deterministic rows: '
        f'{len(chunk_documents):,} documents, {len(chunk_memberships):,} memberships'
    )
    return chunk_documents


_deterministic_worker_cfg: ExperimentCfg | None = None
_deterministic_worker_ontology: MedicalOntology | None = None


def _init_deterministic_worker(
    cfg_dump: dict[str, object], ontology_dump: dict[str, object]
) -> None:
    global _deterministic_worker_cfg, _deterministic_worker_ontology
    _deterministic_worker_cfg = ExperimentCfg.model_validate(cfg_dump)
    _deterministic_worker_ontology = MedicalOntology.model_validate(ontology_dump)


def _render_deterministic_chunk_batch(
    batch: tuple[int, list[dict[str, object]]],
) -> tuple[str, list[dict[str, object]]]:
    if _deterministic_worker_cfg is None or _deterministic_worker_ontology is None:
        raise RuntimeError('deterministic chunk worker was not initialized')

    start_index, fact_rows = batch
    first_fact = ClinicalFact.model_validate(fact_rows[0])
    rows: list[dict[str, object]] = []

    for offset, fact_row in enumerate(fact_rows):
        fact = ClinicalFact.model_validate(fact_row)
        rendered_draft = render_canonical_chunk(
            fact,
            _deterministic_worker_ontology,
            _deterministic_worker_cfg.generation.chunk_text_style,
        )
        draft_text = rendered_draft.text
        state = new_chunk_state(
            draft_text,
            validation=validate_chunk_text(
                draft_text,
                fact,
                _deterministic_worker_ontology,
                text_style=_deterministic_worker_cfg.generation.chunk_text_style,
            ),
            rendered_template=rendered_draft,
        )
        row = finalize_chunk_row(
            cfg=_deterministic_worker_cfg,
            fact=fact,
            ontology=_deterministic_worker_ontology,
            index=start_index + offset,
            state=state,
        )

        rows.append(row.model_dump(mode='python'))

    return first_fact.query_id, rows


def _iter_fact_batches(facts: pl.DataFrame):
    current_query_id: str | None = None
    current_rows: list[dict[str, object]] = []
    start_index = 0

    for fact_index, fact_row in enumerate(facts.iter_rows(named=True)):
        query_id = str(fact_row['query_id'])
        if current_query_id is None:
            current_query_id = query_id
            start_index = fact_index
        elif query_id != current_query_id:
            yield start_index, current_rows
            current_rows = []
            current_query_id = query_id
            start_index = fact_index
        current_rows.append(fact_row)

    if current_rows:
        yield start_index, current_rows


def _chunk_rows_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.from_dicts(rows, infer_schema_length=None)


def _write_normalized_chunks(
    paths: MedicalDatasetGenPaths,
    chunk_rows: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if len(chunk_rows) == 0:
        chunk_documents = pl.DataFrame()
        chunk_memberships = pl.DataFrame()
        write_parquet(paths, 'chunk_documents', chunk_documents)
        write_parquet(paths, 'chunk_memberships', chunk_memberships)
        return chunk_documents, chunk_memberships

    duplicate_text_keys = (
        chunk_rows.group_by('chunk_reuse_key')
        .agg(pl.col('text').n_unique().alias('n_texts'))
        .filter(pl.col('n_texts') > 1)
    )
    if len(duplicate_text_keys):
        examples = duplicate_text_keys['chunk_reuse_key'].head(5).to_list()
        raise RuntimeError(
            'chunk_reuse_key must map to exactly one text after canonical rendering; '
            f'found {len(duplicate_text_keys):,} violating key(s), examples={examples}'
        )

    chunk_rows = chunk_rows.with_columns(
        pl.col('text')
        .str.to_lowercase()
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
        .alias('_normalized_text')
    )
    before_dedup = len(chunk_rows)
    retained_rows = chunk_rows.unique(
        subset=['query_id', '_normalized_text'],
        keep='first',
        maintain_order=True,
    )
    dropped_duplicate_memberships = before_dedup - len(retained_rows)
    if dropped_duplicate_memberships:
        print(
            f'[chunks] dropped {dropped_duplicate_memberships:,} query-local duplicate '
            'membership row(s) by normalized text'
        )

    doc_keys = retained_rows.select('chunk_reuse_key').unique(maintain_order=True)
    doc_key_to_id = _doc_key_to_chunk_id(doc_keys['chunk_reuse_key'].to_list())

    with_doc_id = retained_rows.with_columns(
        pl.col('chunk_reuse_key')
        .replace_strict(doc_key_to_id, return_dtype=pl.String)
        .alias('chunk_id'),
        pl.col('chunk_id').alias('membership_id'),
    )

    doc_cols = [
        'chunk_id',
        'chunk_reuse_key',
        'text',
        'approx_words',
        'condition_id',
        'condition_display',
        'subgroup_id',
        'subgroup_label',
        'subgroup_axis',
        'subgroup_field',
        'subgroup_value',
        'axis',
        'value_bin',
        'axis_bin_term',
        'axis_payload_json',
        'subgroup_dimension_id',
        'subgroup_level_id',
        'subgroup_is_reference',
        'patient_age',
        'patient_sex',
        'clinical_subgroup_phrase',
        'note_style',
        'chunk_surface_group',
        'outer_template_family',
        'outer_template_id',
        'axis_template_family',
        'axis_template_id',
    ]
    membership_cols = [
        'membership_id',
        'chunk_id',
        'query_id',
        'evidence_profile_id',
        'pool_id',
        'primary_axis',
        'secondary_axis',
        'dominant_primary_facet_id',
        'fact_id',
        'facet_id',
        'target_facet_id',
        'cluster_id',
        'cluster_role',
        'axis',
        'value_bin',
        'axis_payload_json',
        'facet_priority',
        'is_gold',
        'distractor_type',
        'split',
    ]

    chunk_documents = (
        with_doc_id.select([col for col in doc_cols if col in with_doc_id.columns])
        .unique(subset=['chunk_id'], keep='first', maintain_order=True)
        .sort('chunk_id')
    )
    chunk_memberships = with_doc_id.select(
        [col for col in membership_cols if col in with_doc_id.columns]
    )

    duplicate_memberships = (
        chunk_memberships.group_by('query_id', 'chunk_id')
        .agg(pl.len().alias('n'))
        .filter(pl.col('n') > 1)
    )
    if len(duplicate_memberships):
        examples = duplicate_memberships.select('query_id', 'chunk_id').head(5).to_dicts()
        raise RuntimeError(
            'a query may only contain one membership for each chunk document; '
            f'found {len(duplicate_memberships):,} duplicate pair(s), examples={examples}'
        )

    invalid_gold_coverage = (
        chunk_memberships.filter(pl.col('is_gold'))
        .group_by('query_id')
        .agg(pl.col('facet_id').n_unique().alias('n_gold_facets'))
        .filter(pl.col('n_gold_facets') != 4)
    )
    if len(invalid_gold_coverage):
        examples = invalid_gold_coverage.head(5).to_dicts()
        raise RuntimeError(
            f'query-local duplicate dropping removed a required gold facet; examples={examples}'
        )

    write_parquet(paths, 'chunk_documents', chunk_documents)
    write_parquet(paths, 'chunk_memberships', chunk_memberships)
    print(
        f'[chunks] normalized {len(chunk_rows):,} generated row(s) -> '
        f'{len(chunk_documents):,} chunk document(s), '
        f'{len(chunk_memberships):,} query membership(s)'
    )
    return chunk_documents, chunk_memberships


def _doc_key_to_chunk_id(chunk_reuse_keys: list[str]) -> dict[str, str]:
    return {key: chunk_id(idx) for idx, key in enumerate(chunk_reuse_keys)}


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.logging_utils import (
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_make_chunks(cfg, paths)
