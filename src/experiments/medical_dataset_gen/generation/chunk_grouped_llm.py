"""Grouped LLM chunk generation and rewriting.

This module deduplicates equivalent chunk-generation or chunk-rewrite jobs, runs
them sequentially or in parallel, and materializes the shared outputs back into
rows.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

from experiments.medical_dataset_gen.generation.chunk_cache import (
    GenerationCache,
    append_generation_cache,
    cached_chunk_state,
    cached_rewrite_chunk_state,
    chunk_generation_cache_key,
    chunk_rewrite_cache_key,
    remember_cache_entry,
)
from experiments.medical_dataset_gen.generation.chunk_rendering import (
    finalize_chunk_row,
    generate_llm_chunk,
    new_chunk_state,
    reject_row,
    render_canonical_chunk_text,
    rewrite_llm_chunk,
    row_from_state,
)
from experiments.medical_dataset_gen.generation.schemas import (
    ChunkRow,
    ClinicalFact,
    MedicalOntology,
)
from experiments.medical_dataset_gen.generation.text_templates import validate_chunk_text
from experiments.medical_dataset_gen.global_configs import ExperimentCfg


def render_chunks_grouped_llm(
    cfg: ExperimentCfg,
    facts: list[ClinicalFact],
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
    cache_version: int,
) -> tuple[list[ChunkRow], list[dict[str, object]], set[str]]:
    rows: list[ChunkRow | None] = [None] * len(facts)
    rejects: list[dict[str, object]] = []
    failed_queries: set[str] = set()
    missing_groups: dict[str, list[tuple[int, ClinicalFact]]] = {}
    exact_cache_hits = 0
    reusable_cache_hits = 0

    for idx, fact in enumerate(facts):
        cached = cached_chunk_state(cfg, fact, ontology, cache)
        if cached is not None:
            state, hit_kind = cached
            rows[idx] = row_from_state(idx, fact, state)
            if hit_kind == 'fact_id':
                exact_cache_hits += 1
            else:
                reusable_cache_hits += 1
            continue
        cache_key = chunk_generation_cache_key(cfg, fact)
        missing_groups.setdefault(cache_key, []).append((idx, fact))

    cache_hits = exact_cache_hits + reusable_cache_hits
    facts_to_generate = len(facts) - cache_hits
    duplicate_jobs_saved = max(0, facts_to_generate - len(missing_groups))
    print(
        f'[chunks] grouped LLM generation: facts={len(facts):,}, cache_hits={cache_hits:,} '
        f'(exact={exact_cache_hits:,}, reusable={reusable_cache_hits:,}), '
        f'unique_llm_jobs={len(missing_groups):,}, duplicate_jobs_saved={duplicate_jobs_saved:,}'
    )

    group_items = list(missing_groups.items())
    with tqdm(total=len(facts), desc='Rendering chunks', dynamic_ncols=True) as pbar:
        if cache_hits:
            pbar.update(cache_hits)
        if cfg.generation.llm_workers > 1:
            _generate_missing_groups_parallel(
                cfg=cfg,
                ontology=ontology,
                cache=cache,
                local_cache_path=local_cache_path,
                shared_cache_path=shared_cache_path,
                group_items=group_items,
                rows=rows,
                rejects=rejects,
                failed_queries=failed_queries,
                pbar=pbar,
                cache_version=cache_version,
            )
        else:
            _generate_missing_groups_sequential(
                cfg=cfg,
                ontology=ontology,
                cache=cache,
                local_cache_path=local_cache_path,
                shared_cache_path=shared_cache_path,
                group_items=group_items,
                rows=rows,
                rejects=rejects,
                failed_queries=failed_queries,
                pbar=pbar,
                cache_version=cache_version,
            )

    materialized_rows: list[ChunkRow] = [
        row for row in rows if row is not None and str(row['query_id']) not in failed_queries
    ]
    return materialized_rows, rejects, failed_queries


def render_chunks_grouped_rewrite(
    cfg: ExperimentCfg,
    facts: list[ClinicalFact],
    ontology: MedicalOntology,
    cache: GenerationCache,
    rewrite_cache_path: Path,
    cache_version: int,
) -> tuple[list[ChunkRow], list[dict[str, object]], set[str]]:
    rows: list[ChunkRow | None] = [None] * len(facts)
    rejects: list[dict[str, object]] = []
    failed_queries: set[str] = set()
    exact_cache_hits = 0
    reusable_cache_hits = 0
    unique_rewrite_jobs = 0
    duplicate_jobs_saved = 0
    next_idx = 0

    print(f'[chunks] grouped rewrite enabled with {cfg.generation.llm_workers} parallel workers')

    with (
        ThreadPoolExecutor(
            max_workers=cfg.generation.llm_workers,
            thread_name_prefix='mdg-rewrite',
        ) as executor,
        tqdm(total=len(facts), desc='Rendering chunks', dynamic_ncols=True) as pbar,
    ):
        pbar.refresh()
        while next_idx < len(facts):
            query_id = facts[next_idx].query_id
            query_end = next_idx + 1
            while query_end < len(facts) and facts[query_end].query_id == query_id:
                query_end += 1

            query_group = [(idx, facts[idx]) for idx in range(next_idx, query_end)]
            if query_id in failed_queries:
                pbar.update(len(query_group))
                next_idx = query_end
                continue

            group_stats = _render_rewrite_query_group(
                cfg=cfg,
                ontology=ontology,
                cache=cache,
                rewrite_cache_path=rewrite_cache_path,
                query_group=query_group,
                rows=rows,
                rejects=rejects,
                failed_queries=failed_queries,
                executor=executor,
                cache_version=cache_version,
                pbar=pbar,
            )
            exact_cache_hits += group_stats[0]
            reusable_cache_hits += group_stats[1]
            unique_rewrite_jobs += group_stats[2]
            duplicate_jobs_saved += group_stats[3]
            next_idx = query_end

    cache_hits = exact_cache_hits + reusable_cache_hits
    print(
        f'[chunks] grouped rewrite: facts={len(facts):,}, cache_hits={cache_hits:,} '
        f'(exact={exact_cache_hits:,}, reusable={reusable_cache_hits:,}), '
        f'unique_llm_jobs={unique_rewrite_jobs:,}, duplicate_jobs_saved={duplicate_jobs_saved:,}'
    )

    materialized_rows: list[ChunkRow] = [
        row for row in rows if row is not None and str(row['query_id']) not in failed_queries
    ]
    return materialized_rows, rejects, failed_queries


def _generate_missing_groups_sequential(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
    group_items: list[tuple[str, list[tuple[int, ClinicalFact]]]],
    rows: list[ChunkRow | None],
    rejects: list[dict[str, object]],
    failed_queries: set[str],
    pbar: tqdm,
    cache_version: int,
) -> None:
    for cache_key, group in group_items:
        active = _active_group(group, failed_queries)
        if not active:
            pbar.update(len(group))
            continue
        final_text, attempt_errors = generate_llm_chunk(
            cfg=cfg, fact=active[0][1], ontology=ontology
        )
        _materialize_generated_group(
            cfg=cfg,
            ontology=ontology,
            cache=cache,
            local_cache_path=local_cache_path,
            shared_cache_path=shared_cache_path,
            cache_key=cache_key,
            group=group,
            rows=rows,
            rejects=rejects,
            failed_queries=failed_queries,
            final_text=final_text,
            attempt_errors=attempt_errors,
            pbar=pbar,
            cache_version=cache_version,
        )


def _generate_missing_groups_parallel(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
    group_items: list[tuple[str, list[tuple[int, ClinicalFact]]]],
    rows: list[ChunkRow | None],
    rejects: list[dict[str, object]],
    failed_queries: set[str],
    pbar: tqdm,
    cache_version: int,
) -> None:
    pending: dict[
        Future[tuple[str, list[str]]],
        tuple[str, list[tuple[int, ClinicalFact]]],
    ] = {}
    next_idx = 0
    max_in_flight = max(cfg.generation.llm_workers * 2, cfg.generation.llm_workers)
    executor = ThreadPoolExecutor(
        max_workers=cfg.generation.llm_workers, thread_name_prefix='mdg-llm'
    )

    try:
        while next_idx < len(group_items) or pending:
            while next_idx < len(group_items) and len(pending) < max_in_flight:
                cache_key, group = group_items[next_idx]
                next_idx += 1
                active = _active_group(group, failed_queries)
                if not active:
                    pbar.update(len(group))
                    continue
                future = executor.submit(generate_llm_chunk, cfg, active[0][1], ontology)
                pending[future] = (cache_key, group)

            if not pending:
                continue

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                cache_key, group = pending.pop(future)
                final_text, attempt_errors = future.result()
                _materialize_generated_group(
                    cfg=cfg,
                    ontology=ontology,
                    cache=cache,
                    local_cache_path=local_cache_path,
                    shared_cache_path=shared_cache_path,
                    cache_key=cache_key,
                    group=group,
                    rows=rows,
                    rejects=rejects,
                    failed_queries=failed_queries,
                    final_text=final_text,
                    attempt_errors=attempt_errors,
                    pbar=pbar,
                    cache_version=cache_version,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _active_group(
    group: list[tuple[int, ClinicalFact]],
    failed_queries: set[str],
) -> list[tuple[int, ClinicalFact]]:
    return [(idx, fact) for idx, fact in group if fact.query_id not in failed_queries]


def _materialize_generated_group(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    cache: GenerationCache,
    local_cache_path: Path,
    shared_cache_path: Path,
    cache_key: str,
    group: list[tuple[int, ClinicalFact]],
    rows: list[ChunkRow | None],
    rejects: list[dict[str, object]],
    failed_queries: set[str],
    final_text: str,
    attempt_errors: list[str],
    pbar: tqdm,
    cache_version: int,
) -> None:
    active = _active_group(group, failed_queries)
    if not active:
        pbar.update(len(group))
        return

    if attempt_errors:
        reason = '; '.join(attempt_errors)
        for _, fact in active:
            rejects.append(reject_row(fact, reason, final_text))
            failed_queries.add(fact.query_id)
            print(
                f'[chunks] dropping query {fact.query_id} after failed chunk {fact.fact_id}: '
                + reason
            )
        pbar.update(len(group))
        return

    local_cache_written = cache_key in cache.by_reuse_key
    shared_cache_written = cache_key in cache.by_reuse_key
    for idx, fact in active:
        state = new_chunk_state(
            final_text,
            text_generation_source='llm',
            llm_attempted=True,
            llm_rejected=False,
            validation=validate_chunk_text(final_text, fact, ontology),
        )
        try:
            row, cache_entry = finalize_chunk_row(
                cfg=cfg,
                fact=fact,
                ontology=ontology,
                index=idx,
                state=state,
                should_cache=True,
                cache_version=cache_version,
                cache_key_fn=chunk_generation_cache_key,
            )
        except RuntimeError as exc:
            rejects.append(reject_row(fact, str(exc), final_text))
            failed_queries.add(fact.query_id)
            print(
                f'[chunks] dropping query {fact.query_id} after final validation failure in '
                f'{fact.fact_id}: {exc}'
            )
            continue

        if cache_entry is not None:
            remember_cache_entry(cache, cache_entry)
            if not local_cache_written:
                append_generation_cache(local_cache_path, cache_entry)
                local_cache_written = True
            if not shared_cache_written:
                append_generation_cache(shared_cache_path, cache_entry)
                shared_cache_written = True
        rows[idx] = row
    pbar.update(len(group))


def _render_rewrite_query_group(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    cache: GenerationCache,
    rewrite_cache_path: Path,
    query_group: list[tuple[int, ClinicalFact]],
    rows: list[ChunkRow | None],
    rejects: list[dict[str, object]],
    failed_queries: set[str],
    executor: ThreadPoolExecutor,
    cache_version: int,
    pbar: tqdm,
) -> tuple[int, int, int, int]:
    draft_text_by_index: dict[int, str] = {}
    rewrite_key_by_index: dict[int, str] = {}
    missing_groups: dict[str, list[tuple[int, ClinicalFact]]] = {}
    exact_cache_hits = 0
    reusable_cache_hits = 0

    for idx, fact in query_group:
        draft_text = render_canonical_chunk_text(fact, ontology)
        draft_text_by_index[idx] = draft_text
        cached = cached_rewrite_chunk_state(cfg, fact, ontology, cache, draft_text)
        if cached is not None:
            state, hit_kind = cached
            rows[idx] = row_from_state(idx, fact, state)
            pbar.update(1)
            if hit_kind == 'fact_id':
                exact_cache_hits += 1
            else:
                reusable_cache_hits += 1
            continue

        rewrite_key = chunk_rewrite_cache_key(cfg, fact, draft_text)
        rewrite_key_by_index[idx] = rewrite_key
        missing_groups.setdefault(rewrite_key, []).append((idx, fact))

    rewrite_results = _rewrite_group_results(
        cfg=cfg,
        ontology=ontology,
        executor=executor,
        missing_groups=missing_groups,
        draft_text_by_index=draft_text_by_index,
        pbar=pbar,
    )

    for idx, fact in query_group:
        if fact.query_id in failed_queries:
            break
        if rows[idx] is not None:
            continue

        draft_text = draft_text_by_index[idx]
        rewrite_key = rewrite_key_by_index[idx]
        rewritten_text, attempt_errors = rewrite_results[rewrite_key]
        if attempt_errors:
            print(
                f'[chunks] keeping deterministic template for {fact.query_id} after rewrite '
                f'failure in {fact.fact_id}: ' + '; '.join(attempt_errors)
            )
            state = new_chunk_state(
                draft_text,
                text_generation_source='fallback',
                llm_attempted=True,
                llm_rejected=True,
                validation=validate_chunk_text(draft_text, fact, ontology),
            )
            cache_key = None
        else:
            state = new_chunk_state(
                rewritten_text,
                text_generation_source='llm',
                llm_attempted=True,
                llm_rejected=False,
                validation=validate_chunk_text(rewritten_text, fact, ontology),
            )
            cache_key = rewrite_key

        try:
            row, cache_entry = finalize_chunk_row(
                cfg=cfg,
                fact=fact,
                ontology=ontology,
                index=idx,
                state=state,
                should_cache=state.text_generation_source == 'llm',
                cache_key=cache_key,
                cache_version=cache_version,
            )
        except RuntimeError as exc:
            rejects.append(reject_row(fact, str(exc), state.final_text))
            failed_queries.add(fact.query_id)
            print(
                f'[chunks] dropping query {fact.query_id} after final validation failure in '
                f'{fact.fact_id}: {exc}'
            )
            break

        if cache_entry is not None:
            append_generation_cache(rewrite_cache_path, cache_entry)
            remember_cache_entry(cache, cache_entry)
        rows[idx] = row

    unique_rewrite_jobs = len(missing_groups)
    duplicate_jobs_saved = sum(max(0, len(group) - 1) for group in missing_groups.values())
    return exact_cache_hits, reusable_cache_hits, unique_rewrite_jobs, duplicate_jobs_saved


def _rewrite_group_results(
    cfg: ExperimentCfg,
    ontology: MedicalOntology,
    executor: ThreadPoolExecutor,
    missing_groups: dict[str, list[tuple[int, ClinicalFact]]],
    draft_text_by_index: dict[int, str],
    pbar: tqdm,
) -> dict[str, tuple[str, list[str]]]:
    if not missing_groups:
        return {}

    pending: dict[Future[tuple[str, list[str]]], tuple[str, int]] = {}
    for rewrite_key, group in missing_groups.items():
        first_idx, first_fact = group[0]
        future = executor.submit(
            rewrite_llm_chunk,
            cfg,
            first_fact,
            ontology,
            draft_text_by_index[first_idx],
        )
        pending[future] = (rewrite_key, len(group))

    results: dict[str, tuple[str, list[str]]] = {}
    while pending:
        done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
        for future in done:
            rewrite_key, group_size = pending.pop(future)
            results[rewrite_key] = future.result()
            pbar.update(group_size)
    return results
