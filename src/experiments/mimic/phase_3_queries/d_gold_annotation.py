"""
Step 4.2: Gold facet annotation via map-reduce LLM calls.

For each query + its candidate pool, annotates which chunks support which
facets (aspects of the answer). Uses ollama for local inference.

Map: batches of ~40 chunks → LLM extracts facts + facet labels + chunk citations
Reduce: merge facet labels across batches (deterministic, no LLM)

Output: gold_annotations.parquet
"""

import json

import numpy as np
import polars as pl
from tqdm import tqdm

from experiments.mimic.config_loader import load_phase_config
from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
)
from experiments.mimic.phase_4_evaluation.candidate_pool import CandidatePool, CandidatePoolBuilder
from helpers.ollama_client import generate_json

_cfg = load_phase_config(3)['gold_annotation']


def main():
    con = connect_mimic_duckdb()

    # Load filtered queries (only those that passed divergence filter)
    div_path = MIMIC_RESULTS_DIR / 'divergence_stats.parquet'
    if div_path.exists():
        all_queries = pl.read_parquet(div_path)
        queries_df = all_queries.filter(pl.col('passes_filter'))
        print(
            f'Loaded {len(queries_df):,} queries passing divergence filter '
            f'(from {len(all_queries):,} total)'
        )
    else:
        queries_df = pl.read_parquet(MIMIC_RESULTS_DIR / 'queries.parquet')
        print(f'No divergence_stats.parquet found, using all {len(queries_df):,} queries')

    builder = CandidatePoolBuilder(con, device='cuda')
    result = run_gold_annotation(queries_df, builder)

    out_path = MIMIC_RESULTS_DIR / 'gold_annotations.parquet'
    result.write_parquet(out_path)

    print(f'\nSaved {len(result):,} annotations to {out_path}')
    print(f'  Avg facets per query: {result["n_facets"].mean():.1f}')
    print(f'  Avg gold chunks per query: {result["n_gold_chunks"].mean():.1f}')
    print(f'  Queries with 0 facets: {result.filter(pl.col("n_facets") == 0).height}')

    if len(result) > 0:
        sample = result.row(0, named=True)
        print('\n--- Sample annotation ---')
        print(f'  Query: {sample["query_text"][:200]}')
        facets = json.loads(sample['facets_json'])
        for label, cids in list(facets.items())[:5]:
            print(f'  [{label}] → {len(cids)} chunks')


MAP_SYSTEM_PROMPT = """You are a clinical information analyst. Your task is to extract facts from discharge note excerpts that are relevant to answering a clinical question. Be exhaustive - tag every chunk that contains relevant information, not just the single best one."""

MAP_USER_TEMPLATE = """Question: {query_text}

Below are excerpts from clinical discharge notes. Each is prefixed with its ID.

{chunks_block}

For each fact in these excerpts that is relevant to answering the question:
1. State the fact concisely (one sentence)
2. List ALL chunk_id(s) that support this fact
3. Assign a short facet label - the clinical aspect this fact addresses (e.g. "anticoagulation_adjustment", "renal_dosing", "bp_target", "monitoring_frequency", "drug_choice")

Return a JSON array. Example format:
[{{"fact": "Heparin dose reduced to 10u/kg/hr due to CrCl < 30", "facet_label": "anticoagulation_adjustment", "chunk_ids": ["chunk_42", "chunk_87"]}}]

If no chunks in this batch are relevant to the question, return an empty array: []"""


def _format_chunk_batch(chunk_ids: list[str], texts: list[str]) -> str:
    parts = []
    for cid, text in zip(chunk_ids, texts, strict=True):
        parts.append(f'[CHUNK_ID: {cid}]\n{text}')
    return '\n---\n'.join(parts)


def annotate_batch(
    query_text: str,
    chunk_ids: list[str],
    texts: list[str],
    batch_idx: int = 0,
    n_batches: int = 1,
) -> list[dict]:
    """Run map-phase annotation on a single batch of chunks.

    Returns list of {fact, facet_label, chunk_ids} dicts.
    """
    chunks_block = _format_chunk_batch(chunk_ids, texts)
    prompt = MAP_USER_TEMPLATE.format(query_text=query_text, chunks_block=chunks_block)
    prompt_chars = len(prompt)

    valid_ids = set(chunk_ids)

    try:
        result = generate_json(
            prompt,
            system=MAP_SYSTEM_PROMPT,
            model=_cfg.get('model') or None,
            temperature=_cfg['temperature'],
            top_p=_cfg.get('top_p') or None,
            top_k=_cfg.get('top_k') or None,
            num_ctx=_cfg.get('num_ctx') or None,
            think=_cfg.get('think', False),
        )
    except Exception as e:
        print(f'    batch {batch_idx + 1}/{n_batches} FAILED ({len(chunk_ids)} chunks, ~{prompt_chars // 4} tokens): {e}')
        return []

    if not isinstance(result, list):
        # LLM sometimes wraps in {"annotations": [...]}
        if isinstance(result, dict):
            for key in ('annotations', 'facts', 'results'):
                if key in result and isinstance(result[key], list):
                    result = result[key]
                    break
            else:
                return []
        else:
            return []

    # Validate and clean results
    cleaned = []
    n_dropped = 0
    for item in result:
        if not isinstance(item, dict):
            n_dropped += 1
            continue
        fact = item.get('fact', '')
        label = item.get('facet_label', '')
        cids = item.get('chunk_ids', [])
        if not fact or not label or not cids:
            n_dropped += 1
            continue
        # Only keep chunk_ids that were actually in the batch
        cids = [c for c in cids if c in valid_ids]
        if not cids:
            n_dropped += 1
            continue
        cleaned.append(
            {
                'fact': fact.strip(),
                'facet_label': label.strip().lower().replace(' ', '_'),
                'chunk_ids': cids,
            }
        )

    facet_labels = {item['facet_label'] for item in cleaned}
    print(
        f'    batch {batch_idx + 1}/{n_batches}: '
        f'{len(cleaned)} facts, {len(facet_labels)} facets, '
        f'{n_dropped} dropped '
        f'({len(chunk_ids)} chunks, ~{prompt_chars // 4} tokens)'
    )
    return cleaned


def reduce_facets(all_batch_results: list[list[dict]]) -> dict[str, list[str]]:
    """Merge facet annotations across batches. Deterministic (no LLM).

    Groups by exact facet label, unions chunk_ids.
    """
    facet_to_chunks: dict[str, set[str]] = {}

    for batch in all_batch_results:
        for item in batch:
            label = item['facet_label']
            if label not in facet_to_chunks:
                facet_to_chunks[label] = set()
            facet_to_chunks[label].update(item['chunk_ids'])

    return {label: sorted(cids) for label, cids in facet_to_chunks.items()}


def annotate_query(
    query_text: str,
    pool: CandidatePool,
    batch_size: int = 40,
) -> dict[str, list[str]]:
    """Full map-reduce annotation for one query.

    Returns {facet_label: [chunk_id, ...]}.
    """
    n = pool.n
    n_batches = (n + batch_size - 1) // batch_size
    all_batch_results = []

    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_ids = pool.chunk_ids[start:end]
        batch_texts = pool.texts[start:end]

        batch_result = annotate_batch(
            query_text, batch_ids, batch_texts,
            batch_idx=i, n_batches=n_batches,
        )
        all_batch_results.append(batch_result)

    facets = reduce_facets(all_batch_results)
    total_facts = sum(len(b) for b in all_batch_results)
    all_gold = {cid for cids in facets.values() for cid in cids}
    print(f'    reduce: {total_facts} facts → {len(facets)} facets, {len(all_gold)} gold chunks')
    return facets


def run_gold_annotation(
    queries_df: pl.DataFrame,
    builder: CandidatePoolBuilder,
    prefilter_n: int | None = None,
    batch_size: int | None = None,
) -> pl.DataFrame:
    """Annotate all queries. Groups by icd10_3char for pool reuse.

    Returns DataFrame with columns:
        query_id, icd10_3char, query_text, facets_json, n_facets, n_gold_chunks
    """
    if prefilter_n is None:
        prefilter_n = _cfg['prefilter_n']
    if batch_size is None:
        batch_size = _cfg['batch_size']

    model_name = _cfg.get('model') or 'default'
    print('\n-- Gold annotation config --')
    print(f'  model={model_name}  temperature={_cfg["temperature"]}  '
          f'top_p={_cfg.get("top_p")}  top_k={_cfg.get("top_k")}  '
          f'num_ctx={_cfg.get("num_ctx")}  think={_cfg.get("think", False)}')
    print(f'  prefilter_n={prefilter_n}  batch_size={batch_size}')
    print(f'  queries={len(queries_df)}')
    print()

    condition_pools: dict[str, CandidatePool] = {}
    results = []

    for i, row in enumerate(
        tqdm(queries_df.iter_rows(named=True), total=len(queries_df), desc='Gold annotation')
    ):
        icd3 = row['icd10_3char']
        query_text = row['query_text']

        if icd3 not in condition_pools:
            condition_pools[icd3] = builder.for_condition(icd3)

        pool = condition_pools[icd3]
        query_vec = builder.embed_query(query_text)

        # Prefilter pool to top N by similarity (same as retrieval does)
        sim_to_query = pool.sim_to_query(query_vec)
        if pool.n > prefilter_n:  # type: ignore
            top_indices = np.argsort(sim_to_query)[::-1][:prefilter_n].copy()
            work_pool = pool.slice(top_indices)
        else:
            work_pool = pool

        n_batches = (work_pool.n + batch_size - 1) // batch_size  # type: ignore
        print(
            f'  [{i + 1}/{len(queries_df)}] {icd3} — '
            f'full_pool={pool.n}, work_pool={work_pool.n} chunks, '
            f'{n_batches} batches'
        )
        print(f'    query: {query_text[:120]}{"..." if len(query_text) > 120 else ""}')

        facets = annotate_query(query_text, work_pool, batch_size=batch_size)  # type: ignore

        all_gold_chunks = set()
        for cids in facets.values():
            all_gold_chunks.update(cids)

        query_id = f'{icd3}_{row.get("modifier_text", "")}_{row.get("persona", "")}_{i}'
        query_id = query_id.replace(' ', '_')[:120]

        results.append(
            {
                'query_id': query_id,
                'icd10_3char': icd3,
                'condition_name': row.get('condition_name', ''),
                'modifier_text': row.get('modifier_text', ''),
                'persona': row.get('persona', ''),
                'query_text': query_text,
                'facets_json': json.dumps(facets),
                'n_facets': len(facets),
                'n_gold_chunks': len(all_gold_chunks),
            }
        )

        # Flush after every query so progress survives crashes
        out_path = MIMIC_RESULTS_DIR / 'gold_annotations.parquet'
        pl.DataFrame(results).write_parquet(out_path)

    return pl.DataFrame(results)


if __name__ == '__main__':
    import argparse

    from experiments.mimic.config_loader import parse_config_arg

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.parse_args()

    _run_cfg = parse_config_arg(3)['gold_annotation']
    main()
