from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from experiments.medical_dataset_gen.dataset_generation.chunk_rendering import (
    chunk_id,
    finalize_chunk_row,
    new_chunk_state,
    render_canonical_chunk,
)
from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
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
)
from experiments.medical_dataset_gen.utils.io_utils import read_parquet, write_parquet

_DOCUMENT_COLUMNS = (
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
)
_MEMBERSHIP_COLUMNS = (
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
)
_STREAMING_ROW_GROUP_BATCHES = 32


def run_make_chunks(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> None:
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
) -> None:
    cfg_dump = cfg.model_dump(mode='python')
    ontology_dump = ontology.model_dump(mode='python')
    n_batches = facts.select(pl.col('query_id').n_unique()).item()
    workers = max(1, os.cpu_count() or 1)
    # ``ProcessPoolExecutor.map`` eagerly queues the whole iterator on Python
    # 3.12.  With large nested-scale sources that means retaining millions of
    # fact dictionaries and rendered rows at once.  Bound submitted batches
    # and append parquet row groups instead, keeping the v5 terminal build
    # proportional to one query-local pool rather than the complete suite.
    rendered_batches = _rendered_batches_bounded(
        cfg_dump=cfg_dump,
        ontology_dump=ontology_dump,
        facts=facts,
        workers=workers,
    )
    if cfg.dataset_schema_version >= 5:
        _write_v5_chunks_streaming(
            paths=paths,
            rendered_batches=rendered_batches,
            total_batches=n_batches,
            stable_document_ids=True,
        )
        return
    rows_all = [row for rows in rendered_batches for row in rows]
    chunk_rows = pl.from_dicts(rows_all, infer_schema_length=None) if rows_all else pl.DataFrame()
    chunk_documents, chunk_memberships = _write_normalized_chunks(
        paths,
        chunk_rows,
        stable_document_ids=False,
    )
    print(
        f'[chunks] normalized deterministic rows: '
        f'{len(chunk_documents):,} documents, {len(chunk_memberships):,} memberships'
    )


def _rendered_batches_bounded(
    *,
    cfg_dump: dict[str, object],
    ontology_dump: dict[str, object],
    facts: pl.DataFrame,
    workers: int,
) -> Iterator[list[dict[str, object]]]:
    pending: deque[object] = deque()
    batches = _iter_fact_batches(facts)
    max_pending = max(1, workers * 2)

    # Each worker reconstructs the validated config and ontology once, then
    # renders one query-local fact batch at a time.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_deterministic_worker,
        initargs=(cfg_dump, ontology_dump),
    ) as executor:
        for _ in range(max_pending):
            try:
                pending.append(executor.submit(_render_deterministic_chunk_batch, next(batches)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            yield future.result()  # type: ignore[union-attr]
            with suppress(StopIteration):
                pending.append(executor.submit(_render_deterministic_chunk_batch, next(batches)))


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
) -> list[dict[str, object]]:
    if _deterministic_worker_cfg is None or _deterministic_worker_ontology is None:
        raise RuntimeError('deterministic chunk worker was not initialized')

    start_index, fact_rows = batch
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

    return rows


def _iter_fact_batches(facts: pl.DataFrame) -> Iterator[tuple[int, list[dict[str, object]]]]:
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


def _write_v5_chunks_streaming(
    *,
    paths: MedicalDatasetGenPaths,
    rendered_batches: Iterator[list[dict[str, object]]],
    total_batches: int,
    stable_document_ids: bool,
) -> None:
    """Append query-local chunk rows without retaining a full pool in RAM."""
    if not stable_document_ids:
        raise ValueError('streaming chunk materialization is reserved for schema-v5 stable IDs')
    documents_path = paths.table_path('chunk_documents')
    memberships_path = paths.table_path('chunk_memberships')
    documents_tmp = _parquet_temp_path(documents_path)
    memberships_tmp = _parquet_temp_path(memberships_path)
    documents_writer: pq.ParquetWriter | None = None
    memberships_writer: pq.ParquetWriter | None = None
    document_schema: pa.Schema | None = None
    membership_schema: pa.Schema | None = None
    seen_document_keys: set[str] = set()
    generated_rows = 0
    document_count = 0
    membership_count = 0
    document_buffer: list[pl.DataFrame] = []
    membership_buffer: list[pl.DataFrame] = []
    try:
        for rows in tqdm(
            rendered_batches,
            total=total_batches,
            desc='Rendering chunks',
            dynamic_ncols=True,
        ):
            if not rows:
                continue
            generated_rows += len(rows)
            documents, memberships = _normalized_v5_batch(rows)
            _validate_batch_memberships(memberships)

            new_document_keys = [
                str(key)
                for key in documents['chunk_reuse_key'].to_list()
                if str(key) not in seen_document_keys
            ]
            if new_document_keys:
                new_key_set = set(new_document_keys)
                new_documents = documents.filter(pl.col('chunk_reuse_key').is_in(new_key_set))
                seen_document_keys.update(new_document_keys)
                document_buffer.append(new_documents)
                document_count += len(new_documents)
            membership_buffer.append(memberships)
            membership_count += len(memberships)
            if len(membership_buffer) >= _STREAMING_ROW_GROUP_BATCHES:
                documents_writer, document_schema, memberships_writer, membership_schema = (
                    _flush_chunk_buffers(
                        document_buffer=document_buffer,
                        membership_buffer=membership_buffer,
                        documents_writer=documents_writer,
                        document_schema=document_schema,
                        memberships_writer=memberships_writer,
                        membership_schema=membership_schema,
                        documents_path=documents_tmp,
                        memberships_path=memberships_tmp,
                    )
                )
        documents_writer, document_schema, memberships_writer, membership_schema = (
            _flush_chunk_buffers(
                document_buffer=document_buffer,
                membership_buffer=membership_buffer,
                documents_writer=documents_writer,
                document_schema=document_schema,
                memberships_writer=memberships_writer,
                membership_schema=membership_schema,
                documents_path=documents_tmp,
                memberships_path=memberships_tmp,
            )
        )
        if documents_writer is None or memberships_writer is None:
            write_parquet(paths, 'chunk_documents', pl.DataFrame())
            write_parquet(paths, 'chunk_memberships', pl.DataFrame())
            return
    except Exception:
        _close_parquet_writer(documents_writer)
        _close_parquet_writer(memberships_writer)
        documents_tmp.unlink(missing_ok=True)
        memberships_tmp.unlink(missing_ok=True)
        raise
    else:
        _close_parquet_writer(documents_writer)
        _close_parquet_writer(memberships_writer)
        documents_tmp.replace(documents_path)
        memberships_tmp.replace(memberships_path)
    print(
        f'[chunks] normalized {generated_rows:,} generated row(s) -> '
        f'{document_count:,} chunk document(s), {membership_count:,} query membership(s)'
    )


def _normalized_v5_batch(rows: list[dict[str, object]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    chunk_rows = pl.from_dicts(rows, infer_schema_length=None)
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
    # In v5, each stable reuse key is an intentional support unit.  Two such
    # units may render identical text for a query, yet collapsing them would
    # alter the declared cluster mass and break exact nested scale counts.
    # Earlier schemas preserve their historical text-level de-duplication.
    retained_rows = chunk_rows
    with_doc_id = retained_rows.with_columns(
        pl.concat_str([pl.lit('chunk_'), pl.col('chunk_reuse_key')]).alias('chunk_id')
    ).with_columns(pl.col('chunk_id').alias('membership_id'))
    documents = with_doc_id.select(
        [column for column in _DOCUMENT_COLUMNS if column in with_doc_id.columns]
    ).unique(subset=['chunk_id'], keep='first', maintain_order=True)
    memberships = with_doc_id.select(
        [column for column in _MEMBERSHIP_COLUMNS if column in with_doc_id.columns]
    )
    return documents, memberships


def _validate_batch_memberships(memberships: pl.DataFrame) -> None:
    duplicates = memberships.group_by('query_id', 'chunk_id').len().filter(pl.col('len') > 1)
    if len(duplicates):
        raise RuntimeError('a query may only contain one membership for each chunk document')
    invalid_gold_coverage = (
        memberships.filter(pl.col('is_gold'))
        .group_by('query_id')
        .agg(pl.col('facet_id').n_unique().alias('n_gold_facets'))
        .filter(pl.col('n_gold_facets') != 4)
    )
    if len(invalid_gold_coverage):
        examples = invalid_gold_coverage.head(5).to_dicts()
        raise RuntimeError(
            f'query-local duplicate dropping removed a required gold facet; examples={examples}'
        )


def _flush_chunk_buffers(
    *,
    document_buffer: list[pl.DataFrame],
    membership_buffer: list[pl.DataFrame],
    documents_writer: pq.ParquetWriter | None,
    document_schema: pa.Schema | None,
    memberships_writer: pq.ParquetWriter | None,
    membership_schema: pa.Schema | None,
    documents_path: Path,
    memberships_path: Path,
) -> tuple[pq.ParquetWriter | None, pa.Schema | None, pq.ParquetWriter | None, pa.Schema | None]:
    if document_buffer:
        documents_writer, document_schema = _append_parquet_batch(
            writer=documents_writer,
            schema=document_schema,
            path=documents_path,
            frame=pl.concat(document_buffer, how='vertical_relaxed'),
        )
        document_buffer.clear()
    if membership_buffer:
        memberships_writer, membership_schema = _append_parquet_batch(
            writer=memberships_writer,
            schema=membership_schema,
            path=memberships_path,
            frame=pl.concat(membership_buffer, how='vertical_relaxed'),
        )
        membership_buffer.clear()
    return documents_writer, document_schema, memberships_writer, membership_schema


def _append_parquet_batch(
    *,
    writer: pq.ParquetWriter | None,
    schema: pa.Schema | None,
    path: Path,
    frame: pl.DataFrame,
) -> tuple[pq.ParquetWriter, pa.Schema]:
    table = frame.to_arrow()
    if writer is None:
        schema = table.schema
        writer = pq.ParquetWriter(path, schema, compression='zstd')
        writer.write_table(table)
        return writer, schema
    assert schema is not None
    writer.write_table(table.cast(schema, safe=False))
    return writer, schema


def _parquet_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f'.{path.name}.{uuid4().hex}.tmp')


def _close_parquet_writer(writer: pq.ParquetWriter | None) -> None:
    if writer is not None:
        writer.close()


def _write_normalized_chunks(
    paths: MedicalDatasetGenPaths,
    chunk_rows: pl.DataFrame,
    *,
    stable_document_ids: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if len(chunk_rows) == 0:
        chunk_documents = pl.DataFrame()
        chunk_memberships = pl.DataFrame()
        write_parquet(paths, 'chunk_documents', chunk_documents)
        write_parquet(paths, 'chunk_memberships', chunk_memberships)
        return chunk_documents, chunk_memberships

    # A reuse key is allowed to identify only one rendered document surface.
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

    # Deduplicate identical text only within a query. The same document can
    # remain shared across queries through its reuse key.
    if stable_document_ids:
        retained_rows = chunk_rows
    else:
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

    # Schema v5 gives documents semantic IDs derived from their stable reuse
    # keys.  A scale suite can therefore generate its largest support once and
    # project exact smaller candidate sets without renumbering the shared
    # evidence.  v2--v4 retain their positional IDs verbatim so archived
    # artifact semantics (and their embedding mappings) remain untouched.
    doc_keys = retained_rows.select('chunk_reuse_key').unique(maintain_order=True)
    if stable_document_ids:
        doc_key_to_id = {key: f'chunk_{key}' for key in doc_keys['chunk_reuse_key'].to_list()}
    else:
        doc_key_to_id = {
            key: chunk_id(index) for index, key in enumerate(doc_keys['chunk_reuse_key'].to_list())
        }

    with_doc_id = retained_rows.with_columns(
        pl.col('chunk_reuse_key')
        .replace_strict(doc_key_to_id, return_dtype=pl.String)
        .alias('chunk_id')
    ).with_columns(pl.col('chunk_id').alias('membership_id'))

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

    # Keep document payload and query-local membership metadata in separate
    # tables; evaluation consumes the latter while embeddings consume the former.
    chunk_documents = (
        with_doc_id.select([col for col in doc_cols if col in with_doc_id.columns])
        .unique(subset=['chunk_id'], keep='first', maintain_order=True)
        .sort('chunk_id')
    )
    chunk_memberships = with_doc_id.select(
        [col for col in membership_cols if col in with_doc_id.columns]
    )

    # These checks protect the four-facet gold structure from accidental
    # collapse during text-level deduplication.
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
