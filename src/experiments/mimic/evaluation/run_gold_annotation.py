"""
Answer Generation + Gold Annotation of chunks.

For each query + its candidate pool:
    1. Build facet vocabulary deterministically from the query's modifier list.
    2. MAP: per (batch, facet): extract key clinical facts from structurally eligible chunks,
       gated by a HARD structural prior (chunk hadm_id ∈ condition ∩ modifier hadm_ids).
       Each fact cites its source chunk_id.
    3. REDUCE: single LLM call per query: synthesize a unified comparative answer across
       all facets, citing which chunk IDs support each facet. Those cited IDs become the
       gold set stored in facets_json.

Output: gold_annotations.parquet
Columns: query_id, icd10_3char, condition_name, modifiers_json, query_text,
         facets_json, answer_text, n_facets, n_gold_chunks
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from experiments.mimic.chunking.schemas_chunking import AdmissionMetadataRow
from experiments.mimic.evaluation.candidate_pool import CandidatePool, CandidatePoolBuilder
from experiments.mimic.evaluation.run_evaluate import load_filtered_queries
from experiments.mimic.evaluation.schemas_evaluation import GoldAnnotationCfg
from experiments.mimic.global_configs import (
    duckdb_con,
    get_result_dir,
    get_table_path,
    global_cfg,
    read_parquet,
    setup_logging,
)
from experiments.mimic.queries.schemas_queries import QueryAspect, QueryRow
from experiments.mimic.utils.charlson import CHARLSON_LABELS_TO_STR
from experiments.mimic.utils.utils import (
    aspects_from_modifiers,
)
from helpers.ollama_client import generate, generate_json

gold_annotation_cfg = GoldAnnotationCfg.load()


def run_gold_annotation(
    con: DuckDBPyConnection = duckdb_con, cfg: GoldAnnotationCfg | None = None
) -> pl.DataFrame:
    global gold_annotation_cfg
    if cfg is not None:
        gold_annotation_cfg = cfg

    queries_df: pl.DataFrame = load_filtered_queries(global_cfg.embedding_model)
    hadm_id_to_metadata_str: dict[int, str] = _build_hadm_id_to_metadata_str()
    print(f'Loaded patient metadata for {len(hadm_id_to_metadata_str):,} admissions')

    out_path = get_table_path('gold_annotations')
    done_texts = set[str]()
    if out_path.exists():
        prev = pl.read_parquet(out_path)
        done_texts = set(prev['query_text'].to_list())
        print(
            f'Resuming: {len(done_texts)} queries already done, {len(queries_df) - len(done_texts)} remaining'
        )

    builder = CandidatePoolBuilder(con, embedding_model=global_cfg.embedding_model)
    result_df = annotate(queries_df, builder, hadm_id_to_metadata_str, done_texts)

    result_df.write_parquet(out_path)
    print(
        f'\nSaved {len(result_df):,} annotations to {out_path}\n'
        f'\tAvg gold chunks per query: {result_df["n_gold_chunks"].mean():.1f}'
    )
    return result_df


def annotate(
    queries_df: pl.DataFrame,
    builder: CandidatePoolBuilder,
    hadm_id_to_metadata_str: dict[int, str] | None,
    done_texts: set[str],
) -> pl.DataFrame:
    """Annotate all queries.
    Returns DataFrame with columns:
        query_id, icd10_3char, condition_name, modifiers_json, query_text,
        facets_json, answer_text, n_facets, n_gold_chunks
    """
    prompt_dump_dir = get_result_dir('gold_annotations') / '_prompt_dump'
    prompt_dump_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = get_table_path('gold_annotations', ext='jsonl')
    old_bs = gold_annotation_cfg.resume_batch_size or gold_annotation_cfg.batch_size
    prior_decisions = _load_prior_decisions(jsonl_path, old_bs)
    answers_jsonl_path = get_result_dir('gold_annotations') / 'gold_answers.jsonl'
    prior_answers = _load_prior_answers(answers_jsonl_path)

    total = len(queries_df)
    n_done = len(done_texts)
    completed_rows: list[dict] = []

    for i, row in enumerate(queries_df.iter_rows(named=True)):
        row = cast(QueryRow, row)
        condition_name = row.get('condition_name', '')
        if row['query_text'] in done_texts:
            continue

        n_done += 1
        print(
            f'\n{"=" * 60}\n  Query {n_done}/{total} (idx {i})\n  {row["query_text"]}\n{"=" * 60}'
        )

        modifiers_json: list[dict] = json.loads(row.get('modifiers_json', '') or '[]')
        if not modifiers_json:
            print('  [WARN] no modifiers_json for this query, skipping')
            continue
        aspects = aspects_from_modifiers(modifiers_json)
        print(f'  [aspects] {len(aspects)}: {[a.facet_label for a in aspects]}')

        condition_hadm_ids = builder.icd3_hadm_ids(row['icd10_3char'])
        aspect_hadm_sets: dict[str, set[int]] = {}
        for aspect, modifier in zip(aspects, modifiers_json, strict=True):
            aspect_hadm_sets[aspect.facet_label] = (
                builder.modifier_hadm_ids(modifier['text']) & condition_hadm_ids
            )

        work_pool = _build_stratified_pool(
            builder, row['query_text'], aspect_hadm_sets, row['icd10_3char']
        )

        facets, answer_text = annotate_query(
            query_text=row['query_text'],
            condition_name=condition_name,
            aspects=aspects,
            aspect_hadm_sets=aspect_hadm_sets,
            pool=work_pool,
            batch_size=gold_annotation_cfg.batch_size,
            hadm_id_to_metadata_str=hadm_id_to_metadata_str,
            prompt_dump_dir=prompt_dump_dir,
            query_idx=i,
            prior_decisions=prior_decisions,
            prior_answers=prior_answers,
        )

        all_gold_chunks = {cid for cids in facets.values() for cid in cids}

        completed_rows.append({
            'query_id': row['query_id'],
            'icd10_3char': row['icd10_3char'],
            'condition_name': condition_name,
            'modifiers_json': json.dumps(modifiers_json),
            'query_text': row['query_text'],
            'facets_json': json.dumps(facets),
            'answer_text': answer_text,
            'n_facets': len(facets),
            'n_gold_chunks': len(all_gold_chunks),
        })
    # end for i, row in enumerate(queries_df.iter_rows(named=True))

    out_path = get_table_path('gold_annotations')
    if completed_rows:
        new_df = pl.DataFrame(completed_rows)
        if out_path.exists():
            final_df = pl.concat([pl.read_parquet(out_path), new_df], how='diagonal_relaxed')
        else:
            final_df = new_df
        final_df.write_parquet(out_path)
        return final_df

    return pl.read_parquet(out_path) if out_path.exists() else pl.DataFrame()


def _build_stratified_pool(
    builder: CandidatePoolBuilder,
    query_text: str,
    aspect_hadm_sets: dict[str, set[int]],
    icd10_3char: str,
) -> CandidatePool:
    """Per-modifier direct fetch + cosine fill, capped at final_pool_n."""

    query_vec = builder.embed_query(query_text)
    modifier_pools: list[CandidatePool] = []

    for label, hadm_set in aspect_hadm_sets.items():
        mod_pool = builder.for_hadm_ids_cosine(
            query_vec, hadm_set, gold_annotation_cfg.min_per_modifier
        )
        print(
            f'  [pool] {label}: direct fetch {mod_pool.n} chunks from {len(hadm_set)} modifier+condition patients'
        )
        modifier_pools.append(mod_pool)

    cosine_pool = builder.for_query_cosine_condition(
        query_vec, icd10_3char, gold_annotation_cfg.wide_pool_n
    )
    print(f'  [pool] cosine condition pool: {cosine_pool.n} chunks')

    merged = CandidatePool.merge([*modifier_pools, cosine_pool])
    if merged.n > gold_annotation_cfg.final_pool_n:
        merged = merged.slice(np.arange(gold_annotation_cfg.final_pool_n, dtype=np.intp))

    print(f'  [pool] stratified: {merged.n} total')
    return merged


def annotate_query(
    query_text: str,
    condition_name: str,
    aspects: list[QueryAspect],
    aspect_hadm_sets: dict[str, set[int]],
    pool: CandidatePool,
    batch_size: int = 40,
    hadm_id_to_metadata_str: dict[int, str] | None = None,
    prompt_dump_dir: Path | None = None,
    query_idx: int = 0,
    prior_decisions: dict[tuple[int, int, str], list[dict]] | None = None,
    prior_answers: dict[int, str] | None = None,
) -> tuple[dict[str, list[str]], str]:
    """Map-reduce annotation for one query.
    Returns (facets_json_dict, answer_text).
    """
    n = pool.n
    n_batches = (n + batch_size - 1) // batch_size
    facts_per_facet: dict[str, list[dict]] = {a.facet_label: [] for a in aspects}
    jsonl_path = get_result_dir('gold_annotations') / 'gold_annotations.jsonl'
    answers_jsonl_path = get_result_dir('gold_annotations') / 'gold_answers.jsonl'

    # MAP phase - extract facts per (batch, facet)
    for batch_idx, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_ids = pool.chunk_ids[start:end]
        batch_texts = pool.texts[start:end]
        batch_sections = pool.section_names[start:end]
        batch_hadm_ids = pool.hadm_ids[start:end].tolist()
        batch_meta = (
            [hadm_id_to_metadata_str.get(h, '') for h in batch_hadm_ids]
            if hadm_id_to_metadata_str
            else None
        )

        for aspect in aspects:
            eligible_hadm_set = aspect_hadm_sets.get(aspect.facet_label, set())
            eligible_mask = [hid in eligible_hadm_set for hid in batch_hadm_ids]
            if not any(eligible_mask):
                print(
                    f'    [skip] b{batch_idx + 1}/{n_batches} {aspect.facet_label}: '
                    f'no chunks from condition && modifier patients in this batch (structural prior)'
                )
                continue

            if prior_decisions is not None:
                old_bs = gold_annotation_cfg.resume_batch_size or gold_annotation_cfg.batch_size
                sub_keys = [(query_idx, s, aspect.facet_label) for s in range(start, end, old_bs)]
                if all(k in prior_decisions for k in sub_keys):
                    cached = [d for k in sub_keys for d in prior_decisions[k]]
                    facts_per_facet[aspect.facet_label].extend(cached)
                    print(
                        f'    [resume] b{batch_idx + 1}/{n_batches} {aspect.facet_label}: {len(cached)} facts (cached)'
                    )
                    continue

            eligible_indices = [j for j, m in enumerate(eligible_mask) if m]
            eligible_ids = [batch_ids[j] for j in eligible_indices]
            eligible_texts = [batch_texts[j] for j in eligible_indices]
            eligible_sections = [batch_sections[j] for j in eligible_indices]
            eligible_meta = [batch_meta[j] for j in eligible_indices] if batch_meta else None

            facts = extract_facts_batch(
                query_text=query_text,
                condition_name=condition_name,
                aspect=aspect,
                chunk_ids=eligible_ids,
                texts=eligible_texts,
                batch_sections=eligible_sections,
                batch_meta=eligible_meta,
                batch_idx=batch_idx,
                n_batches=n_batches,
                prompt_dump_dir=prompt_dump_dir,
                query_idx=query_idx,
            )
            facts_per_facet[aspect.facet_label].extend(facts)

            with jsonl_path.open('a') as f:
                f.write(
                    json.dumps({
                        'query_idx': query_idx,
                        'batch_idx': batch_idx,
                        'facet_label': aspect.facet_label,
                        'decisions': facts,
                    })
                    + '\n'
                )

    # Build facets_json from MAP output: unique chunk_ids cited in extracted facts per facet
    pool_id_set = set(pool.chunk_ids)
    facets: dict[str, list[str]] = {}
    for aspect in aspects:
        cited = sorted({
            d['chunk_id']
            for d in facts_per_facet.get(aspect.facet_label, [])
            if d['chunk_id'] in pool_id_set
        })
        if cited:
            facets[aspect.facet_label] = cited

    all_gold = {cid for cids in facets.values() for cid in cids}
    print(
        f'  map: {len(aspects)} aspects → {len(facets)} non-empty facets, {len(all_gold)} gold chunks'
    )

    # REDUCE phase - synthesize unified answer from all extracted facts
    if prior_answers is not None and query_idx in prior_answers:
        answer_text = prior_answers[query_idx]
        print(f'  [resume] reduce q{query_idx}: answer cached ({len(answer_text)} chars)')
    else:
        answer_text = synthesize_answer(
            query_text=query_text,
            condition_name=condition_name,
            aspects=aspects,
            facts_per_facet=facts_per_facet,
            prompt_dump_dir=prompt_dump_dir,
            query_idx=query_idx,
        )
        with answers_jsonl_path.open('a') as f:
            f.write(
                json.dumps({
                    'query_idx': query_idx,
                    'query_text': query_text,
                    'answer_text': answer_text,
                })
                + '\n'
            )

    return facets, answer_text


def extract_facts_batch(
    query_text: str,
    condition_name: str,
    aspect: QueryAspect,
    chunk_ids: list[str],
    texts: list[str],
    batch_sections: list[str] | None = None,
    batch_meta: list[str] | None = None,
    batch_idx: int = 0,
    n_batches: int = 1,
    prompt_dump_dir: Path | None = None,
    query_idx: int = 0,
) -> list[dict]:
    """Extract clinical facts from one (aspect, chunk batch) - MAP step."""
    prompt = _build_map_prompt(
        query_text=query_text,
        condition_name=condition_name,
        aspect=aspect,
        chunk_ids=chunk_ids,
        texts=texts,
        batch_sections=batch_sections,
        batch_meta=batch_meta,
    )

    if prompt_dump_dir is not None:
        dump_path = (
            prompt_dump_dir / f'q{query_idx:03d}_b{batch_idx:03d}_{aspect.facet_label}_map.txt'
        )
        dump_path.write_text(
            f'=== SYSTEM ===\n{gold_annotation_cfg.fact_extract_system_prompt}\n\n=== USER ===\n{prompt}'
        )

    valid_ids = set(chunk_ids)

    try:
        result = generate_json(
            prompt,
            system=gold_annotation_cfg.fact_extract_system_prompt,
            model=gold_annotation_cfg.model,
            num_ctx=gold_annotation_cfg.num_ctx,
            num_predict=gold_annotation_cfg.num_predict,
            temperature=gold_annotation_cfg.temperature,
            top_p=gold_annotation_cfg.top_p,
            top_k=gold_annotation_cfg.top_k,
            think=gold_annotation_cfg.think,
            stream=gold_annotation_cfg.stream,
        )
    except Exception as e:
        print(
            f'[ERROR] extract_facts q{query_idx} b{batch_idx + 1}/{n_batches} {aspect.facet_label}: {e}'
        )
        return []

    raw: list = []
    if isinstance(result, list):
        raw = result
    elif isinstance(result, dict):
        raw = result.get('facts', []) or result.get('decisions', [])

    facts: list[dict] = []
    seen_ids: set[str] = set()
    n_dropped = 0
    for item in raw:
        if not isinstance(item, dict):
            n_dropped += 1
            continue
        chunk_id = item.get('chunk_id', '')
        fact = item.get('fact', '')
        if not chunk_id or chunk_id not in valid_ids or chunk_id in seen_ids or not fact:
            n_dropped += 1
            continue
        seen_ids.add(chunk_id)
        facts.append({'chunk_id': chunk_id, 'fact': str(fact)})

    print(
        f'    [map] b{batch_idx + 1}/{n_batches} {aspect.facet_label}: '
        f'{len(facts)} facts, {n_dropped} dropped (~{len(prompt) // 4} tokens)'
    )
    return facts


def synthesize_answer(
    query_text: str,
    condition_name: str,
    aspects: list[QueryAspect],
    facts_per_facet: dict[str, list[dict]],
    prompt_dump_dir: Path | None,
    query_idx: int,
) -> str:
    """Synthesize unified comparative answer from extracted facts - REDUCE step."""
    prompt = _build_reduce_prompt(
        query_text=query_text,
        condition_name=condition_name,
        aspects=aspects,
        facts_per_facet=facts_per_facet,
    )

    if prompt_dump_dir is not None:
        dump_path = prompt_dump_dir / f'q{query_idx:03d}_reduce.txt'
        dump_path.write_text(
            f'=== SYSTEM ===\n{gold_annotation_cfg.answer_system_prompt}\n\n=== USER ===\n{prompt}'
        )

    n_facts_total = sum(len(v) for v in facts_per_facet.values())
    print(f'  [reduce] synthesizing answer from {n_facts_total} extracted facts')

    try:
        return generate(
            prompt,
            system=gold_annotation_cfg.answer_system_prompt,
            model=gold_annotation_cfg.model,
            num_ctx=gold_annotation_cfg.num_ctx,
            num_predict=gold_annotation_cfg.num_predict,
            temperature=gold_annotation_cfg.temperature,
            top_p=gold_annotation_cfg.top_p,
            top_k=gold_annotation_cfg.top_k,
            think=gold_annotation_cfg.think,
            stream=gold_annotation_cfg.stream,
        )
    except Exception as e:
        print(f'[ERROR] synthesize_answer q{query_idx}: {e}')
        return ''


def _load_prior_decisions(
    jsonl_path: Path,
    resume_batch_size: int,
) -> dict[tuple[int, int, str], list[dict]]:
    """Load cached map entries keyed by (query_idx, chunk_start, facet_label).

    chunk_start = batch_idx * resume_batch_size - position-based key survives batch_size changes.
    """
    prior: dict[tuple[int, int, str], list[dict]] = {}
    if not jsonl_path.exists():
        return prior
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            chunk_start = entry['batch_idx'] * resume_batch_size
            key = (entry['query_idx'], chunk_start, entry['facet_label'])
            prior[key] = entry.get('decisions', [])

    print(f'[resume] loaded {len(prior)} cached map entries from jsonl')
    return prior


def _load_prior_answers(answers_jsonl_path: Path) -> dict[int, str]:
    """Load cached reduce answers keyed by query_idx."""
    prior: dict[int, str] = {}
    if not answers_jsonl_path.exists():
        return prior
    with answers_jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prior[entry['query_idx']] = entry.get('answer_text', '')
    print(f'[resume] loaded {len(prior)} cached reduce answers from jsonl')
    return prior


def _build_map_prompt(
    query_text: str,
    condition_name: str,
    aspect: QueryAspect,
    chunk_ids: list[str],
    texts: list[str],
    batch_sections: list[str] | None = None,
    batch_meta: list[str] | None = None,
) -> str:
    chunks_block = _format_chunk_batch(
        chunk_ids, texts, sections=batch_sections, meta_lines=batch_meta
    )
    return gold_annotation_cfg.fact_extract_template.format(
        query_text=query_text,
        condition_name=condition_name,
        facet_description=aspect.description,
        chunks_block=chunks_block,
    )


def _build_reduce_prompt(
    query_text: str,
    condition_name: str,
    aspects: list[QueryAspect],
    facts_per_facet: dict[str, list[dict]],
) -> str:
    subgroups_block = '\n\n'.join(
        f'=== SUBGROUP {i + 1} - {asp.description} ===\n'
        + (
            '\n'.join(
                f'- [{d["chunk_id"]}] {d["fact"]}' for d in facts_per_facet.get(asp.facet_label, [])
            )
            or '(no relevant facts found)'
        )
        for i, asp in enumerate(aspects)
    )
    return gold_annotation_cfg.answer_gen_template.format(
        query_text=query_text,
        condition_name=condition_name,
        subgroups_block=subgroups_block,
    )


def _format_chunk_batch(
    chunk_ids: list[str],
    texts: list[str],
    sections: list[str] | None = None,
    meta_lines: list[str] | None = None,
) -> str:
    parts = []
    for i, (cid, text) in enumerate(zip(chunk_ids, texts, strict=True)):
        section = sections[i] if sections else None
        meta = meta_lines[i] if meta_lines else None
        header = f'[CHUNK_ID: {cid}] [{section}]' if section else f'[CHUNK_ID: {cid}]'
        if meta:
            header += f'\n{meta}'
        parts.append(f'{header}\n{text}')
    return '\n---\n'.join(parts)


def _build_hadm_id_to_metadata_str() -> dict[int, str]:
    meta = read_parquet('admissions_metadata')

    lookup: dict[int, str] = {}
    for row in meta.iter_rows(named=True):
        row = cast(AdmissionMetadataRow, row)
        age = int(row['age']) if row.get('age') is not None else None  # type: ignore
        gender = 'F' if row.get('gender') == 'F' else 'M'
        age_str = f'age {age}' if age is not None else 'age unknown'

        comorbidities = [
            label
            for col, label in CHARLSON_LABELS_TO_STR.items()
            if row.get(col) and row[col] > 0  # type: ignore
        ]
        primary = row.get('primary_icd_description', '')

        parts = [f'{age_str}, {gender}']
        if primary:
            parts.append(f'primary dx: {primary}')
        if comorbidities:
            parts.append(f'comorbidities: {", ".join(comorbidities)}')

        lookup[row['hadm_id']] = 'Patient: ' + ' | '.join(parts)

    return lookup


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='queries')
    run_gold_annotation(cfg=GoldAnnotationCfg(**raw['gold_annotation']))
