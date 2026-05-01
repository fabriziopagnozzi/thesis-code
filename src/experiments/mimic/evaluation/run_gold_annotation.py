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

import polars as pl

from experiments.mimic.chunking.schemas_chunking import AdmissionMetadataRow
from experiments.mimic.global_configs import (
    get_result_dir,
    get_table_path,
    global_cfg,
    read_parquet,
    setup_logging,
)
from experiments.mimic.queries.schemas_queries import (
    QueryModifier,
    QueryRow,
)
from experiments.mimic.utils.charlson import CHARLSON_LABELS_TO_STR
from experiments.mimic.utils.utils import load_filtered_queries
from helpers.ollama_client import generate, generate_json

from ..utils.candidate_pools import ChunkPool, ChunkPoolBuilder
from .schemas_evaluation import GoldAnnotationCfg

gold_annotation_cfg = GoldAnnotationCfg.load()


def run_gold_annotation(cfg: GoldAnnotationCfg | None = None) -> pl.DataFrame:
    global gold_annotation_cfg
    if cfg is not None:
        gold_annotation_cfg = cfg

    queries_df: pl.DataFrame = load_filtered_queries(global_cfg.embedding_model)
    hadm_id_to_metadata_str: dict[int, str] = _build_hadm_id_to_metadata_str()

    out_path = get_table_path('gold_annotations')
    done_texts = set[str]()
    if out_path.exists():
        prev = pl.read_parquet(out_path)
        done_texts = set(prev['query_text'].to_list())
        print(
            f'Resuming: {len(done_texts)} queries already done, {len(queries_df) - len(done_texts)} remaining'
        )

    builder = ChunkPoolBuilder(model_name=global_cfg.embedding_model)
    result_df = annotate_all(queries_df, builder, hadm_id_to_metadata_str, done_texts)

    result_df.write_parquet(out_path)
    print(
        f'\nSaved {len(result_df):,} annotations to {out_path}\n'
        f'\tAvg gold chunks per query: {result_df["n_gold_chunks"].mean():.1f}'
    )
    return result_df


def annotate_all(
    queries_df: pl.DataFrame,
    builder: ChunkPoolBuilder,
    hadm_id_to_metadata_str: dict[int, str] | None,
    done_texts: set[str],
) -> pl.DataFrame:
    prompt_dump_dir = get_result_dir('gold_annotations') / '_prompt_dump'
    prompt_dump_dir.mkdir(parents=True, exist_ok=True)

    prior_decisions = GoldAnnotationResumer.load_prior_decisions()
    prior_answers = GoldAnnotationResumer.load_prior_answers()

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

        modifiers = QueryModifier.parse_list(row.get('modifiers_json', '') or '')
        if not modifiers:
            print('  [WARN] no modifiers_json for this query, skipping')
            continue
        print(f'\t[modifiers] {len(modifiers)}: {[m.comorb_label for m in modifiers]}')

        modifier_to_hadm_ids: dict[str, set[int]] = {}
        for modifier in modifiers:
            modifier_to_hadm_ids[modifier.comorb_label] = (
                builder.get_hadm_ids_by_condition_modifier(row['icd10_3char'], modifier)
            )

        facets, answer_text = annotate_query(
            pool=_build_gold_candidate_pool(builder, row['query_text'], modifier_to_hadm_ids),
            query_idx=i,
            query_text=row['query_text'],
            condition_name=condition_name,
            batch_size=gold_annotation_cfg.batch_size,
            modifiers=modifiers,
            modifier_to_hadm_ids=modifier_to_hadm_ids,
            hadm_id_to_metadata_str=hadm_id_to_metadata_str,
            prompt_dump_dir=prompt_dump_dir,
            prior_decisions=prior_decisions,
            prior_answers=prior_answers,
        )

        all_gold_chunks = {cid for cids in facets.values() for cid in cids}

        completed_rows.append({
            'query_id': row['query_id'],
            'icd10_3char': row['icd10_3char'],
            'condition_name': condition_name,
            'modifiers_json': row['modifiers_json'],
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


def annotate_query(
    query_text: str,
    condition_name: str,
    modifiers: list[QueryModifier],
    modifier_to_hadm_ids: dict[str, set[int]],
    pool: ChunkPool,
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
    facts_per_facet: dict[str, list[dict]] = {m.comorb_label: [] for m in modifiers}
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

        for modifier in modifiers:
            eligible_hadm_set = modifier_to_hadm_ids.get(modifier.comorb_label, set())
            eligible_mask = [hid in eligible_hadm_set for hid in batch_hadm_ids]
            if not any(eligible_mask):
                print(
                    f'\t  [skip] b{batch_idx + 1}/{n_batches} {modifier.comorb_label}: '
                    f'no chunks from condition && modifier patients in this batch (structural prior)'
                )
                continue

            if prior_decisions is not None:
                old_bs = gold_annotation_cfg.resume_batch_size or gold_annotation_cfg.batch_size
                aligned_start = (start // old_bs) * old_bs
                sub_keys = [
                    (query_idx, s, modifier.comorb_label) for s in range(aligned_start, end, old_bs)
                ]
                if all(k in prior_decisions for k in sub_keys):
                    cached = [d for k in sub_keys for d in prior_decisions[k]]
                    facts_per_facet[modifier.comorb_label].extend(cached)
                    print(
                        f'\t  [resume] b{batch_idx + 1}/{n_batches} {modifier.comorb_label}: {len(cached)} facts (cached)'
                    )
                    continue

            eligible_indices = [j for j, m in enumerate(eligible_mask) if m]
            eligible_ids = [batch_ids[j] for j in eligible_indices]
            eligible_texts = [batch_texts[j] for j in eligible_indices]
            eligible_sections = [batch_sections[j] for j in eligible_indices]
            eligible_meta = [batch_meta[j] for j in eligible_indices] if batch_meta else None

            facts = map_batch_extract_facts(
                query_text=query_text,
                condition_name=condition_name,
                modifier=modifier,
                chunk_ids=eligible_ids,
                texts=eligible_texts,
                batch_sections=eligible_sections,
                batch_meta=eligible_meta,
                batch_idx=batch_idx,
                n_batches=n_batches,
                prompt_dump_dir=prompt_dump_dir,
                query_idx=query_idx,
            )
            facts_per_facet[modifier.comorb_label].extend(facts)

            with jsonl_path.open('a') as f:
                f.write(
                    json.dumps({
                        'query_idx': query_idx,
                        'batch_idx': batch_idx,
                        'facet_label': modifier.comorb_label,
                        'decisions': facts,
                    })
                    + '\n'
                )

    # Build facets_json from MAP output: unique chunk_ids cited in extracted facts per facet
    pool_id_set = set(pool.chunk_ids)
    facets: dict[str, list[str]] = {}
    for modifier in modifiers:
        cited = sorted({
            d['chunk_id']
            for d in facts_per_facet.get(modifier.comorb_label, [])
            if d['chunk_id'] in pool_id_set
        })
        if cited:
            facets[modifier.comorb_label] = cited

    all_gold = {cid for cids in facets.values() for cid in cids}
    print(
        f'\tmap: {len(modifiers)} modifiers → {len(facets)} non-empty facets, {len(all_gold)} gold chunks'
    )

    # REDUCE phase - synthesize unified answer from all extracted facts
    if prior_answers is not None and query_idx in prior_answers:
        answer_text = prior_answers[query_idx]
        print(f'\t[resume] reduce q{query_idx}: answer cached ({len(answer_text)} chars)')
    else:
        answer_text = reduce_answer(
            query_text=query_text,
            condition_name=condition_name,
            modifiers=modifiers,
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


def map_batch_extract_facts(
    query_text: str,
    condition_name: str,
    modifier: QueryModifier,
    chunk_ids: list[str],
    texts: list[str],
    batch_sections: list[str] | None = None,
    batch_meta: list[str] | None = None,
    batch_idx: int = 0,
    n_batches: int = 1,
    prompt_dump_dir: Path | None = None,
    query_idx: int = 0,
) -> list[dict]:
    """Extract clinical facts from one (modifier, chunk batch) - MAP step."""
    prompt = _build_map_prompt(
        query_text=query_text,
        condition_name=condition_name,
        modifier=modifier,
        chunk_ids=chunk_ids,
        texts=texts,
        batch_sections=batch_sections,
        batch_meta=batch_meta,
    )

    if prompt_dump_dir is not None:
        dump_path = (
            prompt_dump_dir / f'q{query_idx:03d}_b{batch_idx:03d}_{modifier.comorb_label}_map.txt'
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
            f'[ERROR] extract_facts q{query_idx} b{batch_idx + 1}/{n_batches} {modifier.comorb_label}: {e}'
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
        f'\t  [map] b{batch_idx + 1}/{n_batches} {modifier.comorb_label}: '
        f'{len(facts)} facts, {n_dropped} dropped (~{len(prompt) // 4} tokens)'
    )
    return facts


def reduce_answer(
    query_text: str,
    condition_name: str,
    modifiers: list[QueryModifier],
    facts_per_facet: dict[str, list[dict]],
    prompt_dump_dir: Path | None,
    query_idx: int,
) -> str:
    """Synthesize unified comparative answer from extracted facts - REDUCE step."""
    prompt = _build_reduce_prompt(
        query_text=query_text,
        condition_name=condition_name,
        modifiers=modifiers,
        facts_per_facet=facts_per_facet,
    )

    if prompt_dump_dir is not None:
        dump_path = prompt_dump_dir / f'q{query_idx:03d}_reduce.txt'
        dump_path.write_text(
            f'=== SYSTEM ===\n{gold_annotation_cfg.answer_system_prompt}\n\n=== USER ===\n{prompt}'
        )

    n_facts_total = sum(len(v) for v in facts_per_facet.values())
    print(f'\t[reduce] synthesizing answer from {n_facts_total} extracted facts')

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


def _build_gold_candidate_pool(
    builder: ChunkPoolBuilder,
    query_text: str,
    modifier_to_hadm_ids: dict[str, set[int]],
) -> ChunkPool:
    """Top-N cosine chunks from union of (condition AND any_modifier) patients."""
    query_vec = builder.embed_query(query_text)
    union_hadm_ids: set[int] = set().union(*modifier_to_hadm_ids.values())
    print(
        f'\t[pool] union of {len(modifier_to_hadm_ids)} modifier sets: {len(union_hadm_ids)} unique patients'
    )
    pool = builder.topk_cosine_for_hadm_ids(
        query_vec, union_hadm_ids, gold_annotation_cfg.annotation_pool_n
    )
    print(f'\t[pool] top-{gold_annotation_cfg.annotation_pool_n} cosine: {pool.n} chunks')
    return pool


class GoldAnnotationResumer:
    annotations_jsonl_path: Path = get_table_path('gold_annotations', ext='jsonl')
    answers_jsonl_path: Path = get_table_path('gold_answers', ext='jsonl')
    old_batch_size = gold_annotation_cfg.resume_batch_size or gold_annotation_cfg.batch_size

    @classmethod
    def load_prior_decisions(cls) -> dict[tuple[int, int, str], list[dict]]:
        """Load cached map entries keyed by (query_idx, chunk_start, facet_label).

        chunk_start = batch_idx * resume_batch_size - position-based key survives batch_size changes.
        """
        prior: dict[tuple[int, int, str], list[dict]] = {}
        if not cls.annotations_jsonl_path.exists():
            return prior
        with cls.annotations_jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                chunk_start = entry['batch_idx'] * cls.old_batch_size
                key = (entry['query_idx'], chunk_start, entry['facet_label'])
                prior[key] = entry.get('decisions', [])

        print(f'[resume] loaded {len(prior)} cached map entries from jsonl')
        return prior

    @classmethod
    def load_prior_answers(cls) -> dict[int, str]:
        """Load cached reduce answers keyed by query_idx."""
        prior: dict[int, str] = {}
        if not cls.answers_jsonl_path.exists():
            return prior
        with cls.answers_jsonl_path.open() as f:
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
    modifier: QueryModifier,
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
        facet_description=modifier.descr,
        chunks_block=chunks_block,
    )


def _build_reduce_prompt(
    query_text: str,
    condition_name: str,
    modifiers: list[QueryModifier],
    facts_per_facet: dict[str, list[dict]],
) -> str:
    subgroups_block = '\n\n'.join(
        f'=== SUBGROUP {i + 1} - {m.descr} ===\n'
        + (
            '\n'.join(
                f'- [{d["chunk_id"]}] {d["fact"]}' for d in facts_per_facet.get(m.comorb_label, [])
            )
            or '(no relevant facts found)'
        )
        for i, m in enumerate(modifiers)
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

    raw = load_config_from_main(key='evaluation')
    run_gold_annotation(cfg=GoldAnnotationCfg(**raw['gold_annotation']))
