from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from experiments.medical_dataset_gen.dataset_generation.chunk_cache import (
    GENERATION_CACHE_VERSION,
    REWRITE_CACHE_VERSION,
    GenerationCache,
    append_generation_cache,
    cached_chunk_state,
    cached_rewrite_chunk_state,
    chunk_generation_cache_key,
    chunk_rewrite_cache_key,
    load_generation_cache,
    remember_cache_entry,
)
from experiments.medical_dataset_gen.dataset_generation.chunk_grouped_llm import (
    render_chunks_grouped_llm,
    render_chunks_grouped_rewrite,
)
from experiments.medical_dataset_gen.dataset_generation.chunk_rendering import (
    chunk_id,
    finalize_chunk_row,
    generate_llm_chunk,
    new_chunk_state,
    reject_row,
    rejects_frame,
    render_canonical_chunk_text,
    rewrite_llm_chunk,
    row_from_state,
)
from experiments.medical_dataset_gen.dataset_generation.chunk_templates import (
    validate_chunk_text,
)
from experiments.medical_dataset_gen.dataset_generation.deterministic_caches import (
    deterministic_chunk_id,
    deterministic_render_signature,
    materialize_global_deterministic_documents,
)
from experiments.medical_dataset_gen.dataset_generation.ontology_utils import load_ontology
from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ChunkRow,
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.schemas.global_config_schemas import (
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
    facts = read_parquet(paths, 'clinical_facts')
    if cfg.generation.llm_config.use_llm_chunk_generation:
        print(f'[chunks] LLM generation enabled for all {len(facts):,} facts')
        if cfg.generation.llm_config.num_workers > 1:
            print(
                f'[chunks] using {cfg.generation.llm_config.num_workers} parallel workers; '
                f'configure the Ollama server with OLLAMA_NUM_PARALLEL>={cfg.generation.llm_config.num_workers} '
                'for actual concurrent decoding'
            )
    else:
        print('[chunks] LLM generation disabled; using deterministic clinical fallback')
        if cfg.generation.llm_config.use_llm_chunk_rewriting:
            print(
                '[chunks] deterministic chunks will be LLM-rewritten with the isolated global '
                'rewrite cache'
            )
            if cfg.generation.llm_config.num_workers > 1:
                print(
                    f'[chunks] using {cfg.generation.llm_config.num_workers} parallel workers for rewrite '
                    'calls; configure the Ollama server with '
                    f'OLLAMA_NUM_PARALLEL>={cfg.generation.llm_config.num_workers} for actual concurrent decoding'
                )
    if (
        cfg.generation.llm_config.use_llm_chunk_generation
        and cfg.generation.llm_config.use_llm_chunk_rewriting
    ):
        print(
            '[chunks] use_llm_chunk_rewriting is ignored when use_llm_chunk_generation is enabled'
        )

    local_cache_path = paths.experiment_dir / 'chunk_generation_cache.jsonl'
    shared_cache_path = paths.root / '_cache' / 'chunk_generation_cache.jsonl'
    rewrite_cache_path = paths.root / '_cache' / 'chunk_rewrite_cache.jsonl'
    generation_cache = (
        load_generation_cache(
            [shared_cache_path, local_cache_path],
            cache_version=GENERATION_CACHE_VERSION,
        )
        if cfg.generation.llm_config.use_llm_chunk_generation
        else GenerationCache(by_fact_id={}, by_reuse_key={})
    )
    if generation_cache.loaded_rows:
        print(
            f'[chunks] loaded {generation_cache.loaded_rows:,} cached chunk generation row(s) '
            f'({len(generation_cache.by_fact_id):,} exact fact keys, '
            f'{len(generation_cache.by_reuse_key):,} reusable keys)'
        )
    rewrite_cache = (
        load_generation_cache([rewrite_cache_path], cache_version=REWRITE_CACHE_VERSION)
        if (
            not cfg.generation.llm_config.use_llm_chunk_generation
            and cfg.generation.llm_config.use_llm_chunk_rewriting
        )
        else GenerationCache(by_fact_id={}, by_reuse_key={})
    )
    if rewrite_cache.loaded_rows:
        print(
            f'[chunks] loaded {rewrite_cache.loaded_rows:,} cached template rewrite row(s) '
            f'({len(rewrite_cache.by_fact_id):,} exact fact keys, '
            f'{len(rewrite_cache.by_reuse_key):,} reusable keys)'
        )

    if (
        not cfg.generation.llm_config.use_llm_chunk_generation
        and not cfg.generation.llm_config.use_llm_chunk_rewriting
    ):
        return _render_chunks_deterministic_parallel(
            cfg=cfg,
            paths=paths,
            facts=facts,
            ontology=ontology,
        )

    fact_rows = [ClinicalFact.model_validate(row) for row in facts.iter_rows(named=True)]

    if cfg.generation.llm_config.use_llm_chunk_generation:
        rows, rejects, failed_queries = render_chunks_grouped_llm(
            cfg=cfg,
            facts=fact_rows,
            ontology=ontology,
            cache=generation_cache,
            local_cache_path=local_cache_path,
            shared_cache_path=shared_cache_path,
            cache_version=GENERATION_CACHE_VERSION,
        )
    elif (
        cfg.generation.llm_config.use_llm_chunk_rewriting
        and cfg.generation.llm_config.num_workers > 1
    ):
        rows, rejects, failed_queries = render_chunks_grouped_rewrite(
            cfg=cfg,
            facts=fact_rows,
            ontology=ontology,
            cache=rewrite_cache,
            rewrite_cache_path=rewrite_cache_path,
            cache_version=REWRITE_CACHE_VERSION,
        )
    else:
        rows, rejects, failed_queries = render_chunks_sequential(
            cfg=cfg,
            facts=fact_rows,
            ontology=ontology,
            generation_cache=generation_cache,
            generation_cache_path=local_cache_path,
            rewrite_cache=rewrite_cache,
            rewrite_cache_path=rewrite_cache_path,
        )

    chunk_rows = pl.from_dicts(
        [row.model_dump(mode='python') for row in rows], infer_schema_length=None
    )
    chunk_documents, chunk_memberships = _write_normalized_chunks(paths, chunk_rows)

    write_parquet(paths, 'generation_rejects', rejects_frame(rejects))
    soft_warning_count = sum(row.validation_soft_warning_count for row in rows)
    chunks_with_soft_warnings = sum(1 for row in rows if row.validation_soft_warning_count > 0)
    if chunks_with_soft_warnings:
        print(
            f'[chunks] kept {chunks_with_soft_warnings:,} chunk(s) with soft warnings '
            f'({soft_warning_count:,} warnings total)'
        )
    if failed_queries:
        print(
            f'[chunks] dropped {len(failed_queries):,} query/queries after LLM validation failure; '
            f'kept {len(chunk_memberships):,}/{len(facts):,} chunk membership rows'
        )
    return chunk_documents


def render_chunks_sequential(
    cfg: ExperimentCfg,
    facts: list[ClinicalFact],
    ontology: MedicalOntology,
    generation_cache: GenerationCache,
    generation_cache_path: Path,
    rewrite_cache: GenerationCache,
    rewrite_cache_path: Path,
) -> tuple[list[ChunkRow], list[dict[str, object]], set[str]]:
    rows: list[ChunkRow] = []
    rejects: list[dict[str, object]] = []
    failed_queries: set[str] = set()

    for i, fact in tqdm(
        enumerate(facts),
        total=len(facts),
        desc='Rendering chunks',
        dynamic_ncols=True,
    ):
        if fact.query_id in failed_queries:
            continue
        cached = cached_chunk_state(cfg, fact, ontology, generation_cache)
        if cached is not None:
            rows.append(row_from_state(i, fact, cached[0]))
            continue

        cache_key: str | None = None

        if cfg.generation.llm_config.use_llm_chunk_generation:
            final_text, attempt_errors = generate_llm_chunk(cfg=cfg, fact=fact, ontology=ontology)
            if attempt_errors:
                rejects.append(reject_row(fact, '; '.join(attempt_errors), final_text))
                failed_queries.add(fact.query_id)
                print(
                    f'[chunks] dropping query {fact.query_id} after failed chunk {fact.fact_id}: '
                    + '; '.join(attempt_errors)
                )
                continue
            validation = validate_chunk_text(
                final_text,
                fact,
                ontology,
                text_style=cfg.generation.chunk_text_style,
            )
            state = new_chunk_state(
                final_text,
                text_generation_source='llm',
                llm_attempted=True,
                llm_rejected=False,
                validation=validation,
            )
        else:
            draft_text = render_canonical_chunk_text(
                fact,
                ontology,
                cfg.generation.chunk_text_style,
            )
            if cfg.generation.llm_config.use_llm_chunk_rewriting:
                cached = cached_rewrite_chunk_state(
                    cfg=cfg,
                    fact=fact,
                    ontology=ontology,
                    cache=rewrite_cache,
                    draft_text=draft_text,
                )
                if cached is not None:
                    rows.append(row_from_state(i, fact, cached[0]))
                    continue

                rewrite_key = chunk_rewrite_cache_key(cfg, fact, draft_text)
                rewritten_text, attempt_errors = rewrite_llm_chunk(
                    cfg=cfg,
                    fact=fact,
                    ontology=ontology,
                    draft_text=draft_text,
                )
                if attempt_errors:
                    print(
                        f'[chunks] keeping deterministic template for {fact.query_id} after rewrite '
                        f'failure in {fact.fact_id}: ' + '; '.join(attempt_errors)
                    )
                    final_text = draft_text
                    state = new_chunk_state(
                        final_text,
                        text_generation_source='fallback',
                        llm_attempted=True,
                        llm_rejected=True,
                        validation=validate_chunk_text(
                            final_text,
                            fact,
                            ontology,
                            text_style=cfg.generation.chunk_text_style,
                        ),
                    )
                    cache_key = None
                else:
                    final_text = rewritten_text
                    state = new_chunk_state(
                        final_text,
                        text_generation_source='llm',
                        llm_attempted=True,
                        llm_rejected=False,
                        validation=validate_chunk_text(
                            final_text,
                            fact,
                            ontology,
                            text_style=cfg.generation.chunk_text_style,
                        ),
                    )
                    cache_key = rewrite_key
            else:
                final_text = draft_text
                state = new_chunk_state(
                    final_text,
                    text_generation_source='fallback',
                    llm_attempted=False,
                    llm_rejected=False,
                    validation=validate_chunk_text(
                        final_text,
                        fact,
                        ontology,
                        text_style=cfg.generation.chunk_text_style,
                    ),
                )

        try:
            row, cache_entry = finalize_chunk_row(
                cfg=cfg,
                fact=fact,
                ontology=ontology,
                index=i,
                state=state,
                should_cache=cfg.generation.llm_config.use_llm_chunk_generation
                or (
                    cfg.generation.llm_config.use_llm_chunk_rewriting
                    and state.text_generation_source == 'llm'
                ),
                cache_key=cache_key
                if not cfg.generation.llm_config.use_llm_chunk_generation
                else None,
                cache_version=(
                    GENERATION_CACHE_VERSION
                    if cfg.generation.llm_config.use_llm_chunk_generation
                    else REWRITE_CACHE_VERSION
                ),
                cache_key_fn=chunk_generation_cache_key,
            )
        except RuntimeError as exc:
            rejects.append(reject_row(fact, str(exc), state.final_text))
            failed_queries.add(fact.query_id)
            print(
                f'[chunks] dropping query {fact.query_id} after final validation failure in {fact.fact_id}: {exc}'
            )
            continue

        if cache_entry is not None:
            if cfg.generation.llm_config.use_llm_chunk_generation:
                append_generation_cache(generation_cache_path, cache_entry)
                remember_cache_entry(generation_cache, cache_entry)
            else:
                append_generation_cache(rewrite_cache_path, cache_entry)
                remember_cache_entry(rewrite_cache, cache_entry)
        rows.append(row)

    kept_rows = [row for row in rows if row.query_id not in failed_queries]
    return kept_rows, rejects, failed_queries


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
    kept_rows = 0
    soft_warning_count = 0
    chunks_with_soft_warnings = 0
    rejects: list[dict[str, object]] = []
    failed_queries: set[str] = set()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_deterministic_worker,
        initargs=(cfg_dump, ontology_dump),
    ) as executor:
        batch_iter = _iter_fact_batches(facts)
        for (
            _query_id,
            rows,
            batch_soft_warning_count,
            batch_chunks_with_soft_warnings,
            reject_rows,
            failed,
        ) in tqdm(
            executor.map(_render_deterministic_chunk_batch, batch_iter, chunksize=1),
            total=n_batches,
            desc='Rendering chunks',
            dynamic_ncols=True,
        ):
            soft_warning_count += batch_soft_warning_count
            chunks_with_soft_warnings += batch_chunks_with_soft_warnings
            rejects.extend(reject_rows)
            if failed is not None:
                failed_queries.add(failed)
                print(
                    f'[chunks] dropping query {failed} after deterministic validation failure: '
                    f'{reject_rows[0]["reason"] if reject_rows else "validation failure"}'
                )
                continue

            rows_all.extend(rows)
            kept_rows += len(rows)

    chunk_rows = _chunk_rows_frame(rows_all) if rows_all else pl.DataFrame()
    render_signature = deterministic_render_signature(cfg)
    chunk_documents, chunk_memberships = _write_normalized_chunks(
        paths,
        chunk_rows,
        render_signature=render_signature,
    )

    write_parquet(paths, 'generation_rejects', rejects_frame(rejects))
    if chunks_with_soft_warnings:
        print(
            f'[chunks] kept {chunks_with_soft_warnings:,} chunk(s) with soft warnings '
            f'({soft_warning_count:,} warnings total)'
        )
    if failed_queries:
        print(
            f'[chunks] dropped {len(failed_queries):,} query/queries after deterministic '
            f'validation failure; kept {kept_rows:,}/{len(facts):,} chunk membership rows'
        )
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
) -> tuple[str, list[dict[str, object]], int, int, list[dict[str, object]], str | None]:
    if _deterministic_worker_cfg is None or _deterministic_worker_ontology is None:
        raise RuntimeError('deterministic chunk worker was not initialized')

    start_index, fact_rows = batch
    first_fact = ClinicalFact.model_validate(fact_rows[0])
    rows: list[dict[str, object]] = []
    rejects: list[dict[str, object]] = []
    soft_warning_count = 0
    chunks_with_soft_warnings = 0

    for offset, fact_row in enumerate(fact_rows):
        fact = ClinicalFact.model_validate(fact_row)
        draft_text = render_canonical_chunk_text(
            fact,
            _deterministic_worker_ontology,
            _deterministic_worker_cfg.generation.chunk_text_style,
        )
        state = new_chunk_state(
            draft_text,
            text_generation_source='fallback',
            llm_attempted=False,
            llm_rejected=False,
            validation=validate_chunk_text(
                draft_text,
                fact,
                _deterministic_worker_ontology,
                text_style=_deterministic_worker_cfg.generation.chunk_text_style,
            ),
        )
        try:
            row, _ = finalize_chunk_row(
                cfg=_deterministic_worker_cfg,
                fact=fact,
                ontology=_deterministic_worker_ontology,
                index=start_index + offset,
                state=state,
                should_cache=False,
            )
        except RuntimeError as exc:
            reject = reject_row(fact, str(exc), state.final_text)
            rejects.append(reject)
            return fact.query_id, [], 0, 0, rejects, fact.query_id

        rows.append(row.model_dump(mode='python'))
        soft_warning_count += row.validation_soft_warning_count
        if row.validation_soft_warning_count > 0:
            chunks_with_soft_warnings += 1

    return first_fact.query_id, rows, soft_warning_count, chunks_with_soft_warnings, rejects, None


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
    render_signature: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if len(chunk_rows) == 0:
        chunk_documents = pl.DataFrame()
        chunk_memberships = pl.DataFrame()
        write_parquet(paths, 'chunk_documents', chunk_documents)
        write_parquet(paths, 'chunk_memberships', chunk_memberships)
        return chunk_documents, chunk_memberships

    duplicate_text_keys = (
        chunk_rows
        .group_by('chunk_reuse_key')
        .agg(pl.col('text').n_unique().alias('n_texts'))
        .filter(pl.col('n_texts') > 1)
    )
    if len(duplicate_text_keys):
        examples = duplicate_text_keys['chunk_reuse_key'].head(5).to_list()
        raise RuntimeError(
            'chunk_reuse_key must map to exactly one text after canonical rendering; '
            f'found {len(duplicate_text_keys):,} violating key(s), examples={examples}'
        )

    doc_keys = chunk_rows.select('chunk_reuse_key').unique(maintain_order=True)
    doc_key_to_id = _doc_key_to_chunk_id(doc_keys['chunk_reuse_key'].to_list(), render_signature)

    with_doc_id = chunk_rows.with_columns(
        pl
        .col('chunk_reuse_key')
        .replace_strict(doc_key_to_id, return_dtype=pl.String)
        .alias('chunk_id'),
        pl.col('chunk_id').alias('membership_id'),
    )

    doc_cols = [
        'chunk_id',
        'chunk_reuse_key',
        'text',
        'approx_words',
        'text_generation_source',
        'llm_attempted',
        'llm_rejected',
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
        'validation_soft_warning_count',
        'validation_soft_warnings_json',
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
        'facet_priority',
        'is_gold',
        'distractor_type',
        'split',
    ]

    chunk_documents = (
        with_doc_id
        .select([col for col in doc_cols if col in with_doc_id.columns])
        .unique(subset=['chunk_id'], keep='first', maintain_order=True)
        .sort('chunk_id')
    )
    if render_signature is not None:
        chunk_documents = materialize_global_deterministic_documents(
            paths,
            render_signature,
            chunk_documents,
        ).sort('chunk_id')
    chunk_memberships = with_doc_id.select([
        col for col in membership_cols if col in with_doc_id.columns
    ])

    duplicate_memberships = (
        chunk_memberships
        .group_by('query_id', 'chunk_id')
        .agg(pl.len().alias('n'))
        .filter(pl.col('n') > 1)
    )
    if len(duplicate_memberships):
        examples = duplicate_memberships.select('query_id', 'chunk_id').head(5).to_dicts()
        raise RuntimeError(
            'a query may only contain one membership for each chunk document; '
            f'found {len(duplicate_memberships):,} duplicate pair(s), examples={examples}'
        )

    write_parquet(paths, 'chunk_documents', chunk_documents)
    write_parquet(paths, 'chunk_memberships', chunk_memberships)
    print(
        f'[chunks] normalized {len(chunk_rows):,} generated row(s) -> '
        f'{len(chunk_documents):,} chunk document(s), '
        f'{len(chunk_memberships):,} query membership(s)'
    )
    return chunk_documents, chunk_memberships


def _doc_key_to_chunk_id(
    chunk_reuse_keys: list[str],
    render_signature: str | None,
) -> dict[str, str]:
    if render_signature is None:
        return {key: chunk_id(idx) for idx, key in enumerate(chunk_reuse_keys)}
    return {key: deterministic_chunk_id(render_signature, key) for key in chunk_reuse_keys}


if __name__ == '__main__':
    from experiments.medical_dataset_gen.utils.logging import (
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    run_make_chunks(cfg, paths)
