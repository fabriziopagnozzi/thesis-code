from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from experiments.medical_dataset_gen.generation.chunk_cache import (
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
from experiments.medical_dataset_gen.generation.chunk_grouped_llm import (
    render_chunks_grouped_llm,
    render_chunks_grouped_rewrite,
)
from experiments.medical_dataset_gen.generation.chunk_rendering import (
    finalize_chunk_row,
    generate_llm_chunk,
    new_chunk_state,
    reject_row,
    rejects_frame,
    rewrite_llm_chunk,
    row_from_state,
)
from experiments.medical_dataset_gen.generation.ontology import load_ontology
from experiments.medical_dataset_gen.generation.schemas import (
    ChunkRow,
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.generation.text_templates import (
    render_chunk_text_template,
    validate_chunk_text,
)
from experiments.medical_dataset_gen.global_configs import (
    ExperimentCfg,
    MedicalDatasetGenPaths,
    read_parquet,
    write_parquet,
)


def run_make_chunks(cfg: ExperimentCfg, paths: MedicalDatasetGenPaths) -> pl.DataFrame:
    ontology = load_ontology(cfg)
    facts = read_parquet(paths, 'clinical_facts')
    rng = Random(cfg.global_.seed + 1000)
    if cfg.generation.use_llm_chunk_generation:
        print(f'[chunks] LLM generation enabled for all {len(facts):,} facts')
        if cfg.generation.llm_workers > 1:
            print(
                f'[chunks] using {cfg.generation.llm_workers} parallel workers; '
                f'configure the Ollama server with OLLAMA_NUM_PARALLEL>={cfg.generation.llm_workers} '
                'for actual concurrent decoding'
            )
    else:
        print('[chunks] LLM generation disabled; using deterministic clinical fallback')
        if cfg.generation.use_llm_chunk_rewriting:
            print(
                '[chunks] deterministic chunks will be LLM-rewritten with the isolated global '
                'rewrite cache'
            )
            if cfg.generation.llm_workers > 1:
                print(
                    f'[chunks] using {cfg.generation.llm_workers} parallel workers for rewrite '
                    'calls; configure the Ollama server with '
                    f'OLLAMA_NUM_PARALLEL>={cfg.generation.llm_workers} for actual concurrent decoding'
                )
    if cfg.generation.use_llm_chunk_generation and cfg.generation.use_llm_chunk_rewriting:
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
        if cfg.generation.use_llm_chunk_generation
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
        if (not cfg.generation.use_llm_chunk_generation and cfg.generation.use_llm_chunk_rewriting)
        else GenerationCache(by_fact_id={}, by_reuse_key={})
    )
    if rewrite_cache.loaded_rows:
        print(
            f'[chunks] loaded {rewrite_cache.loaded_rows:,} cached template rewrite row(s) '
            f'({len(rewrite_cache.by_fact_id):,} exact fact keys, '
            f'{len(rewrite_cache.by_reuse_key):,} reusable keys)'
        )

    if not cfg.generation.use_llm_chunk_generation and not cfg.generation.use_llm_chunk_rewriting:
        return _render_chunks_deterministic_parallel(
            cfg=cfg,
            paths=paths,
            facts=facts,
            ontology=ontology,
            output_path=paths.table_path('chunks'),
        )

    fact_rows = [ClinicalFact.model_validate(row) for row in facts.iter_rows(named=True)]

    if cfg.generation.use_llm_chunk_generation:
        rows, rejects, failed_queries = render_chunks_grouped_llm(
            cfg=cfg,
            facts=fact_rows,
            ontology=ontology,
            cache=generation_cache,
            local_cache_path=local_cache_path,
            shared_cache_path=shared_cache_path,
            cache_version=GENERATION_CACHE_VERSION,
        )
    elif cfg.generation.use_llm_chunk_rewriting and cfg.generation.llm_workers > 1:
        rows, rejects, failed_queries = render_chunks_grouped_rewrite(
            cfg=cfg,
            facts=fact_rows,
            ontology=ontology,
            cache=rewrite_cache,
            rewrite_cache_path=rewrite_cache_path,
            rng=rng,
            cache_version=REWRITE_CACHE_VERSION,
        )
    else:
        rows, rejects, failed_queries = _render_chunks_sequential(
            cfg=cfg,
            facts=fact_rows,
            ontology=ontology,
            generation_cache=generation_cache,
            generation_cache_path=local_cache_path,
            rewrite_cache=rewrite_cache,
            rewrite_cache_path=rewrite_cache_path,
            rng=rng,
        )

    chunks = pl.from_dicts(
        [row.model_dump(mode='python') for row in rows], infer_schema_length=None
    )
    write_parquet(paths, 'chunks', chunks)

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
            f'kept {len(chunks):,}/{len(facts):,} chunk rows'
        )
    return chunks


def _render_chunks_deterministic_parallel(
    cfg: ExperimentCfg,
    paths: MedicalDatasetGenPaths,
    facts: pl.DataFrame,
    ontology: MedicalOntology,
    output_path: Path,
) -> pl.DataFrame:
    cfg_dump = cfg.model_dump(mode='python')
    ontology_dump = ontology.model_dump(mode='python')
    n_batches = facts.select(pl.col('query_id').n_unique()).item()
    workers = max(1, os.cpu_count() or 1)
    writer: pq.ParquetWriter | None = None
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

            df = _chunk_rows_frame(rows)
            table = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            kept_rows += len(df)

    if writer is not None:
        writer.close()

    write_parquet(paths, 'generation_rejects', rejects_frame(rejects))
    if chunks_with_soft_warnings:
        print(
            f'[chunks] kept {chunks_with_soft_warnings:,} chunk(s) with soft warnings '
            f'({soft_warning_count:,} warnings total)'
        )
    if failed_queries:
        print(
            f'[chunks] dropped {len(failed_queries):,} query/queries after deterministic '
            f'validation failure; kept {kept_rows:,}/{len(facts):,} chunk rows'
        )
    print(f'[write] chunks: {kept_rows:,} rows -> {output_path}')
    return pl.DataFrame()


_DETERMINISTIC_WORKER_CFG: ExperimentCfg | None = None
_DETERMINISTIC_WORKER_ONTOLOGY: MedicalOntology | None = None


def _init_deterministic_worker(
    cfg_dump: dict[str, object], ontology_dump: dict[str, object]
) -> None:
    global _DETERMINISTIC_WORKER_CFG, _DETERMINISTIC_WORKER_ONTOLOGY
    _DETERMINISTIC_WORKER_CFG = ExperimentCfg.model_validate(cfg_dump)
    _DETERMINISTIC_WORKER_ONTOLOGY = MedicalOntology.model_validate(ontology_dump)


def _render_deterministic_chunk_batch(
    batch: tuple[int, list[dict[str, object]]],
) -> tuple[str, list[dict[str, object]], int, int, list[dict[str, object]], str | None]:
    if _DETERMINISTIC_WORKER_CFG is None or _DETERMINISTIC_WORKER_ONTOLOGY is None:
        raise RuntimeError('deterministic chunk worker was not initialized')

    start_index, fact_rows = batch
    first_fact = ClinicalFact.model_validate(fact_rows[0])
    rng = Random(_stable_seed(f'{_DETERMINISTIC_WORKER_CFG.global_.seed}:{first_fact.query_id}'))
    rows: list[dict[str, object]] = []
    rejects: list[dict[str, object]] = []
    soft_warning_count = 0
    chunks_with_soft_warnings = 0

    for offset, fact_row in enumerate(fact_rows):
        fact = ClinicalFact.model_validate(fact_row)
        draft_text = render_chunk_text_template(fact, _DETERMINISTIC_WORKER_ONTOLOGY, rng)
        state = new_chunk_state(
            draft_text,
            text_generation_source='fallback',
            llm_attempted=False,
            llm_rejected=False,
            validation=validate_chunk_text(draft_text, fact, _DETERMINISTIC_WORKER_ONTOLOGY),
        )
        try:
            row, _ = finalize_chunk_row(
                cfg=_DETERMINISTIC_WORKER_CFG,
                fact=fact,
                ontology=_DETERMINISTIC_WORKER_ONTOLOGY,
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


def _stable_seed(value: str) -> int:
    import hashlib

    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def _render_chunks_sequential(
    cfg: ExperimentCfg,
    facts: list[ClinicalFact],
    ontology: MedicalOntology,
    generation_cache: GenerationCache,
    generation_cache_path: Path,
    rewrite_cache: GenerationCache,
    rewrite_cache_path: Path,
    rng: Random,
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

        if cfg.generation.use_llm_chunk_generation:
            final_text, attempt_errors = generate_llm_chunk(cfg=cfg, fact=fact, ontology=ontology)
            if attempt_errors:
                rejects.append(reject_row(fact, '; '.join(attempt_errors), final_text))
                failed_queries.add(fact.query_id)
                print(
                    f'[chunks] dropping query {fact.query_id} after failed chunk {fact.fact_id}: '
                    + '; '.join(attempt_errors)
                )
                continue
            validation = validate_chunk_text(final_text, fact, ontology)
            state = new_chunk_state(
                final_text,
                text_generation_source='llm',
                llm_attempted=True,
                llm_rejected=False,
                validation=validation,
            )
        else:
            draft_text = render_chunk_text_template(fact, ontology, rng)
            cache_key: str | None = None
            if cfg.generation.use_llm_chunk_rewriting:
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
                        validation=validate_chunk_text(final_text, fact, ontology),
                    )
                    cache_key = None
                else:
                    final_text = rewritten_text
                    state = new_chunk_state(
                        final_text,
                        text_generation_source='llm',
                        llm_attempted=True,
                        llm_rejected=False,
                        validation=validate_chunk_text(final_text, fact, ontology),
                    )
                    cache_key = rewrite_key
            else:
                final_text = draft_text
                state = new_chunk_state(
                    final_text,
                    text_generation_source='fallback',
                    llm_attempted=False,
                    llm_rejected=False,
                    validation=validate_chunk_text(final_text, fact, ontology),
                )

        try:
            row, cache_entry = finalize_chunk_row(
                cfg=cfg,
                fact=fact,
                ontology=ontology,
                index=i,
                state=state,
                should_cache=cfg.generation.use_llm_chunk_generation
                or (
                    cfg.generation.use_llm_chunk_rewriting and state.text_generation_source == 'llm'
                ),
                cache_key=cache_key if not cfg.generation.use_llm_chunk_generation else None,
                cache_version=(
                    GENERATION_CACHE_VERSION
                    if cfg.generation.use_llm_chunk_generation
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
            if cfg.generation.use_llm_chunk_generation:
                append_generation_cache(generation_cache_path, cache_entry)
                remember_cache_entry(generation_cache, cache_entry)
            else:
                append_generation_cache(rewrite_cache_path, cache_entry)
                remember_cache_entry(rewrite_cache, cache_entry)
        rows.append(row)

    kept_rows = [row for row in rows if str(row['query_id']) not in failed_queries]
    return kept_rows, rejects, failed_queries


if __name__ == '__main__':
    from experiments.medical_dataset_gen.global_configs import (
        dump_effective_config,
        load_config_from_cli,
        paths_for,
        setup_logging,
    )

    cfg = load_config_from_cli()
    paths = paths_for(cfg)
    setup_logging(paths)
    dump_effective_config(cfg, paths)
    run_make_chunks(cfg, paths)
