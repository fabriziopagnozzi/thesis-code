"""
Step 4.2: Gold facet annotation via two-phase pipeline.

For each query + its candidate pool:
  1. Build facet vocabulary deterministically from the query's modifier list.
  2. Per-aspect tagging (map: aspects x chunk-batches): yes/no decision + "reason" per chunk per aspect, gated by a HARD structural prior: a chunk is only shown to the LLM for aspect X if its hadm_id belongs to modifier X's hadm_id set.
  3. Reduce: union of relevant chunk_ids per aspect across batches.

Output: gold_annotations.parquet
"""

import json
import re
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from pydantic import BaseModel, field_validator

from experiments.mimic.configs import (
    EvaluateCfg,
    GoldAnnotationCfg,
    get_parquet_path,
    get_result_dir,
    global_cfg,
    setup_logging,
)
from experiments.mimic.duck_db_init import connect_mimic_duckdb
from experiments.mimic.phase_4_evaluation.candidate_pool import CandidatePool, CandidatePoolBuilder
from helpers.ollama_client import generate_json

gold_annotation_cfg = GoldAnnotationCfg.load()


class _Aspect(BaseModel):
    facet_label: str
    description: str

    @field_validator('facet_label')
    @classmethod
    def normalize_label(cls, v: str) -> str:
        return v.strip().lower().replace(' ', '_')


class _TagDecision(BaseModel):
    chunk_id: str
    reason: str = ''


def run_gold_annotation(
    con: duckdb.DuckDBPyConnection | None = None,
    cfg: GoldAnnotationCfg | None = None,
) -> pl.DataFrame:
    global gold_annotation_cfg
    if cfg is not None:
        gold_annotation_cfg = cfg
    if con is None:
        con = connect_mimic_duckdb()

    # Load filtered queries
    filtered_queries_path = get_parquet_path('divergence_stats')
    if filtered_queries_path.exists():
        all_queries = pl.read_parquet(filtered_queries_path)
        queries_df = all_queries.filter(pl.col('passes_filter'))
    else:
        queries_df = pl.read_parquet(get_parquet_path('queries'))

    # Load patient metadata for chunk context
    meta_path = get_parquet_path('admissions_metadata')
    patient_meta = (
        _build_patient_meta(meta_path, global_cfg.charlson_labels)
        if meta_path.exists()
        else None
    )
    if patient_meta:
        print(f'Loaded patient metadata for {len(patient_meta):,} admissions')

    # Resume from previous run if output exists
    out_path = get_parquet_path('gold_annotations')
    done_texts: set[str] = set()
    if out_path.exists():
        prev = pl.read_parquet(out_path)
        done_texts = set(prev['query_text'].to_list())
        print(
            f'Resuming: {len(done_texts)} queries already done, {len(queries_df) - len(done_texts)} remaining'
        )

    builder = CandidatePoolBuilder(con, cfg=EvaluateCfg.load(), device='cpu')
    result = annotate(queries_df, builder, patient_meta, done_texts)
    result.write_parquet(out_path)

    print(
        f'\nSaved {len(result):,} annotations to {out_path}\n'
        f'  Avg facets per query: {result["n_facets"].mean():.1f}\n'
        f'  Avg gold chunks per query: {result["n_gold_chunks"].mean():.1f}\n'
        f'  Queries with 0 facets: {result.filter(pl.col("n_facets") == 0).height}'
    )

    return result


def annotate(
    queries_df: pl.DataFrame,
    builder: CandidatePoolBuilder,
    patient_meta: dict[int, str] | None,
    done_texts: set[str],
) -> pl.DataFrame:
    """Annotate all queries.
    Returns DataFrame with columns:
        query_id, charlson_col, condition_name, modifiers_json, query_text,
        facets_json, n_facets, n_gold_chunks
    """
    out_path = get_parquet_path('gold_annotations')
    prompt_dump_dir = get_result_dir('gold_annotations') / '_prompt_dump'
    prompt_dump_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = get_result_dir('gold_annotations') / 'gold_annotations.jsonl'
    old_bs = gold_annotation_cfg.resume_batch_size or gold_annotation_cfg.batch_size
    prior_decisions = _load_prior_decisions(jsonl_path, old_bs)

    total = len(queries_df)
    n_done = len(done_texts)

    for i, row in enumerate(queries_df.iter_rows(named=True)):
        charlson_col = row['charlson_col']
        query_text = row['query_text']
        if query_text in done_texts:
            continue

        n_done += 1
        print(f'\n{"=" * 60}\n  Query {n_done}/{total} (idx {i})\n  {query_text}\n{"=" * 60}')

        modifiers_json: list[dict] = json.loads(row.get('modifiers_json', '') or '[]')
        if not modifiers_json:
            print('  [WARN] no modifiers_json for this query, skipping')
            continue
        aspects = aspects_from_modifiers(modifiers_json)
        print(f'  [aspects] {len(aspects)}: {[a.facet_label for a in aspects]}')

        # Precompute structural prior: hadm_id set per modifier-facet
        # Intersect with condition hadm_ids so gold chunks come from patients
        # who have BOTH the primary condition AND the modifier.
        condition_hadm_ids = builder.charlson_col_hadm_ids(charlson_col)
        aspect_hadm_sets: dict[str, set[int]] = {}
        for aspect, modifier in zip(aspects, modifiers_json, strict=True):
            aspect_hadm_sets[aspect.facet_label] = (
                builder.modifier_hadm_ids(modifier['text']) & condition_hadm_ids
            )

        query_vec = builder.embed_query(query_text)

        work_pool = _build_stratified_pool(builder, query_vec, aspect_hadm_sets, condition_hadm_ids)

        facets = annotate_query(
            query_text,
            aspects,
            aspect_hadm_sets,
            work_pool,
            batch_size=gold_annotation_cfg.batch_size,
            patient_meta=patient_meta,
            prompt_dump_dir=prompt_dump_dir,
            query_idx=i,
            prior_decisions=prior_decisions,
        )

        all_gold_chunks = set()
        for cids in facets.values():
            all_gold_chunks.update(cids)

        query_id = f'{charlson_col}_{i}'
        query_id = query_id.replace(' ', '_')[:120]

        new_row = pl.DataFrame(
            [
                {
                    'query_id': query_id,
                    'charlson_col': charlson_col,
                    'condition_name': row.get('condition_name', ''),
                    'modifiers_json': json.dumps(modifiers_json),
                    'query_text': query_text,
                    'facets_json': json.dumps(facets),
                    'n_facets': len(facets),
                    'n_gold_chunks': len(all_gold_chunks),
                }
            ]
        )

        # Append to parquet on disk after every query
        if out_path.exists():
            existing = pl.read_parquet(out_path)
            pl.concat([existing, new_row]).write_parquet(out_path)
        else:
            new_row.write_parquet(out_path)
    # end for i, row in enumerate(queries_df.iter_rows(named=True))

    return pl.read_parquet(out_path) if out_path.exists() else pl.DataFrame()


def _build_stratified_pool(
    builder: CandidatePoolBuilder,
    query_vec: np.ndarray,
    aspect_hadm_sets: dict[str, set[int]],
    condition_hadm_ids: set[int],
) -> CandidatePool:
    """Wide cosine pool restricted to the primary condition, with per-modifier quota.

    1. Fetch top-wide_pool_n by cosine similarity, then keep only chunks from
       condition patients (hard filter — non-condition chunks would be skipped by
       the per-facet prior anyway, so no information is lost).
    2. For each modifier, reserve up to min_per_modifier chunks from that modifier's hadm_id set.
    3. Fill remaining slots from the top of the cosine ranking.
    4. Cap at final_pool_n and return.
    """
    wide_pool_raw = builder.for_query_cosine(query_vec, n=gold_annotation_cfg.wide_pool_n)
    cond_indices = np.array(
        [i for i, h in enumerate(wide_pool_raw.hadm_ids.tolist()) if h in condition_hadm_ids],
        dtype=np.intp,
    )
    wide_pool = wide_pool_raw.slice(cond_indices)
    print(
        f'  [pool] condition filter: {wide_pool.n}/{wide_pool_raw.n} chunks from condition patients'
    )
    hadm_list = wide_pool.hadm_ids.tolist()

    guaranteed: set[int] = set()
    for label, hadm_set in aspect_hadm_sets.items():
        mod_indices = [i for i, h in enumerate(hadm_list) if h in hadm_set]
        take = min(gold_annotation_cfg.min_per_modifier, len(mod_indices))
        guaranteed.update(mod_indices[:take])
        print(f'  [pool] {label}: {len(mod_indices)} in wide pool, guaranteed {take}')

    remaining = gold_annotation_cfg.final_pool_n - len(guaranteed)
    fill = [i for i in range(wide_pool.n) if i not in guaranteed][: max(0, remaining)]

    keep = np.array(sorted(guaranteed | set(fill)), dtype=np.intp)
    pool = wide_pool.slice(keep)
    print(f'  [pool] stratified: {len(guaranteed)} guaranteed + {len(fill)} fill = {pool.n} total')
    return pool


def annotate_query(
    query_text: str,
    aspects: list[_Aspect],
    aspect_hadm_sets: dict[str, set[int]],
    pool: CandidatePool,
    batch_size: int = 40,
    patient_meta: dict[int, str] | None = None,
    prompt_dump_dir: Path | None = None,
    query_idx: int = 0,
    prior_decisions: dict[tuple[int, int, str], list[_TagDecision]] | None = None,
) -> dict[str, list[str]]:
    """Two-phase annotation for one query.
    Returns {facet_label: [chunk_id, ...]}, only non-empty facets.
    """
    n = pool.n
    n_batches = (n + batch_size - 1) // batch_size
    aspect_relevant: dict[str, set[str]] = {a.facet_label: set() for a in aspects}
    jsonl_path = get_result_dir('gold_annotations') / 'gold_annotations.jsonl'

    for batch_idx, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_ids = pool.chunk_ids[start:end]
        batch_texts = pool.texts[start:end]
        batch_sections = pool.section_names[start:end]
        batch_hadm_ids = pool.hadm_ids[start:end].tolist()
        batch_meta = [patient_meta.get(h, '') for h in batch_hadm_ids] if patient_meta else None

        for aspect in aspects:
            eligible_hadm_set = aspect_hadm_sets.get(aspect.facet_label, set())

            # HARD structural prior: only chunks whose hadm_id is in the modifier's set
            eligible_mask = [hid in eligible_hadm_set for hid in batch_hadm_ids]
            if not any(eligible_mask):
                continue  # skip LLM call, no eligible chunks for this aspect in this batch

            if prior_decisions is not None:
                old_bs = gold_annotation_cfg.resume_batch_size or gold_annotation_cfg.batch_size
                sub_keys = [(query_idx, s, aspect.facet_label) for s in range(start, end, old_bs)]
                if all(k in prior_decisions for k in sub_keys):
                    cached = [d for k in sub_keys for d in prior_decisions[k]]
                    for d in cached:
                        aspect_relevant[aspect.facet_label].add(d.chunk_id)
                    print(
                        f'    [resume] b{batch_idx + 1}/{n_batches} {aspect.facet_label}: {len(cached)} (cached)'
                    )
                    continue

            eligible_indices = [j for j, m in enumerate(eligible_mask) if m]
            eligible_ids = [batch_ids[j] for j in eligible_indices]
            eligible_texts = [batch_texts[j] for j in eligible_indices]
            eligible_sections = [batch_sections[j] for j in eligible_indices]
            eligible_meta = [batch_meta[j] for j in eligible_indices] if batch_meta else None

            decisions = tag_batch_for_aspect(
                query_text,
                aspect,
                eligible_ids,
                eligible_texts,
                batch_sections=eligible_sections,
                batch_meta=eligible_meta,
                batch_idx=batch_idx,
                n_batches=n_batches,
                prompt_dump_dir=prompt_dump_dir,
                query_idx=query_idx,
            )
            for d in decisions:
                aspect_relevant[aspect.facet_label].add(d.chunk_id)

            with jsonl_path.open('a') as f:
                f.write(
                    json.dumps(
                        {
                            'query_idx': query_idx,
                            'batch_idx': batch_idx,
                            'facet_label': aspect.facet_label,
                            'decisions': [d.model_dump() for d in decisions],
                        }
                    )
                    + '\n'
                )

    # Reduce - drop empty facets, sort chunk_ids
    facets = {label: sorted(cids) for label, cids in aspect_relevant.items() if cids}
    all_gold = {cid for cids in facets.values() for cid in cids}
    print(
        f'  reduce: {len(aspects)} aspects --> {len(facets)} non-empty facets, {len(all_gold)} gold chunks'
    )
    return facets


def tag_batch_for_aspect(
    query_text: str,
    aspect: _Aspect,
    chunk_ids: list[str],
    texts: list[str],
    batch_sections: list[str] | None = None,
    batch_meta: list[str] | None = None,
    batch_idx: int = 0,
    n_batches: int = 1,
    prompt_dump_dir: Path | None = None,
    query_idx: int = 0,
) -> list[_TagDecision]:
    """Yes/no tagging for one (aspect, structurally-eligible chunk sub-batch) pair."""
    chunks_block = _format_chunk_batch(
        chunk_ids, texts, sections=batch_sections, meta_lines=batch_meta
    )
    prompt = gold_annotation_cfg.tagging_user_template.format(
        query_text=query_text,
        facet_label=aspect.facet_label,
        facet_description=aspect.description,
        chunks_block=chunks_block,
    )
    prompt_chars = len(prompt)

    if prompt_dump_dir is not None:
        dump_path = prompt_dump_dir / f'q{query_idx:03d}_b{batch_idx:03d}_{aspect.facet_label}.txt'
        dump_path.write_text(
            f'=== SYSTEM ===\n{gold_annotation_cfg.tagging_system_prompt}\n\n=== USER ===\n{prompt}'
        )

    valid_ids = set(chunk_ids)

    try:
        result = generate_json(
            prompt,
            system=gold_annotation_cfg.tagging_system_prompt,
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
            f'[ERROR] tagging q{query_idx} b{batch_idx + 1}/{n_batches} {aspect.facet_label}: {e}'
        )
        return []

    raw: list = []
    if isinstance(result, dict):
        raw = result.get('decisions', [])
    elif isinstance(result, list):
        raw = result

    seen_ids: set[str] = set()
    decisions: list[_TagDecision] = []
    n_dropped = 0
    for item in raw:
        try:
            d = _TagDecision.model_validate(item)
        except Exception:
            n_dropped += 1
            continue
        if d.chunk_id not in valid_ids or d.chunk_id in seen_ids:
            n_dropped += 1
            continue
        seen_ids.add(d.chunk_id)
        decisions.append(d)

    print(
        f'    [tag] b{batch_idx + 1}/{n_batches} {aspect.facet_label}: '
        f'{len(decisions)} relevant, {n_dropped} dropped '
        f'(~{prompt_chars // 4} tokens)'
    )
    return decisions


def _load_prior_decisions(
    jsonl_path: Path,
    resume_batch_size: int,
) -> dict[tuple[int, int, str], list[_TagDecision]]:
    """Load completed entries keyed by (query_idx, chunk_start, facet_label).

    chunk_start = batch_idx * resume_batch_size — absolute pool position, independent of
    current batch_size so the cache survives a batch_size change mid-run.
    """
    prior: dict[tuple[int, int, str], list[_TagDecision]] = {}
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
            prior[key] = [_TagDecision.model_validate(d) for d in entry.get('decisions', [])]
    print(f'[resume] loaded {len(prior)} cached entries from jsonl')
    return prior


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


def _build_patient_meta(meta_path, charlson_labels: dict[str, str]) -> dict[int, str]:
    meta = pl.read_parquet(meta_path)

    lookup: dict[int, str] = {}
    for row in meta.iter_rows(named=True):
        age = int(row['age']) if row.get('age') is not None else None
        gender = 'F' if row.get('gender') == 'F' else 'M'
        age_str = f'age {age}' if age is not None else 'age unknown'

        comorbidities = [
            label for col, label in charlson_labels.items() if row.get(col) and row[col] > 0
        ]
        primary = row.get('primary_icd_description', '')

        parts = [f'{age_str}, {gender}']
        if primary:
            parts.append(f'primary dx: {primary}')
        if comorbidities:
            parts.append(f'comorbidities: {", ".join(comorbidities)}')

        lookup[row['hadm_id']] = 'Patient: ' + ' | '.join(parts)

    return lookup


def modifier_to_snake_label(text: str) -> str:
    s = re.sub(r'\([^)]*\)', '', text)
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s.lower())
    s = re.sub(r'\b(?:the|a|an)\b', '', s)
    tokens = [t for t in s.split() if t]
    return '_'.join(tokens[:6])


def aspects_from_modifiers(modifiers_json: list[dict]) -> list[_Aspect]:
    return [
        _Aspect(facet_label=modifier_to_snake_label(m['text']), description=m['text'])
        for m in modifiers_json
    ]


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=3)
    run_gold_annotation(cfg=GoldAnnotationCfg(**raw['gold_annotation']))
