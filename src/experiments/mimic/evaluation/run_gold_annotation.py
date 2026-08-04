"""
Answer Generation + Gold Annotation of chunks.

For each query + its candidate pool:
    1. MAP: per (batch, facet): extract key clinical facts from structurally eligible chunks,
       gated by a HARD structural prior (chunk hadm_id ∈ condition ∩ modifier hadm_ids).
       Each fact cites its source chunk_id. Cited chunk_ids per facet → gold set (facets_json).
    2. REDUCE: single LLM call per query: synthesize a unified comparative answer across
       all facets from the MAP-extracted facts. Produces answer_text only.

Output: gold_annotations.parquet
Columns: query_id, icd10_3char, condition_name, modifiers_json, query_text,
         facets_json, answer_text, n_facets, n_gold_chunks
"""

import json
from pathlib import Path
from typing import Any, cast

import polars as pl

from experiments.mimic.global_configs import (
    get_result_dir,
    get_table_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.queries.schemas_queries import (
    QueryModifier,
    QueryModifierLabelId,
    QueryRow,
)
from experiments.mimic.utils.chunk_pools import ChunkPool, ChunkPoolBuilder
from experiments.mimic.utils.utils import load_filtered_queries
from helpers.ollama_client import generate, generate_json

from .schemas_evaluation import (
    AnnotationCacheKey,
    ExtractedFact,
    GoldAnnotationCfg,
)

gold_annotation_cfg = GoldAnnotationCfg.load()


def run_gold_annotation(cfg: GoldAnnotationCfg | None = None):
    global gold_annotation_cfg
    if cfg is not None:
        gold_annotation_cfg = cfg

    queries_df: pl.DataFrame = load_filtered_queries(global_cfg.embedding_model)
    persistence = GoldAnnotationPersistenceHandler(gold_annotation_cfg)

    out_path = get_table_path('gold_annotations')
    done_query_ids = set[int]()
    if out_path.exists():
        prev = pl.read_parquet(out_path)
        done_query_ids = set(prev['query_id'].to_list())
        print(
            f'Resuming: {len(done_query_ids)} queries already done, {len(queries_df) - len(done_query_ids)} remaining'
        )

    annotate_all(queries_df, done_query_ids, persistence, out_path)


def annotate_all(
    queries_df: pl.DataFrame,
    done_query_ids: set[int],
    persistence: GoldAnnotationPersistenceHandler,
    out_path: Path,
) -> None:
    total = len(queries_df)
    n_done = len(done_query_ids)

    pool_builder = ChunkPoolBuilder(model_name=global_cfg.embedding_model)
    prior_annotations = persistence.load_prior_annotations()
    prior_answers = persistence.load_prior_answers()

    for i, row in enumerate(queries_df.iter_rows(named=True)):
        row = cast(QueryRow, row)
        if row['query_id'] in done_query_ids:
            continue

        n_done += 1
        print(
            f'\n{"=" * 60}\n  Query {n_done}/{total} (query_id {row["query_id"]}, row {i})\n  {row["query_text"]}\n{"=" * 60}'
        )

        res = annotate_query(
            pool_builder=pool_builder,
            query_row=row,
            persistence=persistence,
            prior_annotations=prior_annotations,
            prior_answers=prior_answers,
        )
        if res is None:
            continue
        facets, answer_text = res
        if answer_text is None:
            continue

        new_row = pl.DataFrame(
            [
                {
                    'query_id': row['query_id'],
                    'icd10_3char': row['icd10_3char'],
                    'condition_name': row['condition_name'],
                    'modifiers_json': row['modifiers_json'],
                    'query_text': row['query_text'],
                    'facets_json': json.dumps(facets),
                    'answer_text': answer_text,
                    'n_facets': len(facets),
                    'n_gold_chunks': len({c_id for c_ids in facets.values() for c_id in c_ids}),
                }
            ]
        )
        existing = pl.read_parquet(out_path) if out_path.exists() else pl.DataFrame()
        pl.concat([existing, new_row], how='diagonal_relaxed').write_parquet(out_path)
        print(f'\t[saved] query_id={row["query_id"]} → {out_path.name} ({n_done} total)')


def annotate_query(
    pool_builder: ChunkPoolBuilder,
    query_row: QueryRow,
    persistence: GoldAnnotationPersistenceHandler,
    prior_annotations: dict[AnnotationCacheKey, list[ExtractedFact]],
    prior_answers: dict[int, str],
) -> tuple[dict[QueryModifierLabelId, list[str]], str | None] | None:
    """Map-reduce annotation for one query.
    Returns (facets_json_dict, answer_text).
    """
    query_id = query_row['query_id']

    modifiers = QueryModifier.parse_list(query_row.get('modifiers_json', ''))
    if not modifiers:
        print('\t[WARN] no modifiers_json for this query, skipping')
        return None

    print(f'\t[modifiers] {len(modifiers)}: {[m.label for m in modifiers]}')

    modifier_to_hadm_ids: dict[str, set[int]] = {}
    for modifier in modifiers:
        modifier_to_hadm_ids[modifier.label] = pool_builder.get_hadm_ids_by_condition_modifier(
            query_row['icd10_3char'], mod=modifier
        )

    pool = _build_gold_candidate_pool(pool_builder, query_row, modifiers)

    batch_size = gold_annotation_cfg.batch_size
    n_batches = (pool.n + batch_size - 1) // batch_size
    facts_per_facet: dict[QueryModifierLabelId, list[ExtractedFact]] = {
        m.label: [] for m in modifiers
    }

    # MAP phase - extract facts per (batch, facet)
    for batch_idx, (_start, batch_pool) in enumerate(pool.iter_batches(batch_size)):
        for modifier in modifiers:
            eligible_pool = batch_pool.filter_by_hadm_ids(
                modifier_to_hadm_ids.get(modifier.label, set())
            )
            if eligible_pool.n == 0:
                print(
                    f'\t\t[skip] b{batch_idx + 1}/{n_batches} {modifier.label}: '
                    f'no chunks from condition && modifier patients in this batch (structural prior)'
                )
                continue

            cache_key = persistence.annotation_cache_key(
                query_id=query_id,
                facet_label=modifier.label,
                chunk_ids=eligible_pool.chunk_ids,
            )
            if cache_key in prior_annotations:
                cached = prior_annotations[cache_key]
                facts_per_facet[modifier.label].extend(cached)
                print(
                    f'\t\t[resume] b{batch_idx + 1}/{n_batches} {modifier.label}: {len(cached)} facts (cached)'
                )
                continue

            facts = map_batch_extract_facts(
                query_text=query_row['query_text'],
                condition_name=query_row['condition_name'],
                modifier=modifier,
                chunks=eligible_pool,
                batch_idx=batch_idx,
                n_batches=n_batches,
                query_id=query_id,
                persistence=persistence,
            )
            persistence.append_annotation(
                {
                    'query_id': query_id,
                    'batch_idx': batch_idx,
                    'facet_label': modifier.label,
                    'chunk_ids': eligible_pool.chunk_ids,
                    'decisions': facts if facts is not None else '<LLM_ERROR>',
                }
            )
            if facts is not None:
                facts_per_facet[modifier.label].extend(facts)

    # Build facets_json from MAP output: unique chunk_ids cited in extracted facts per facet
    pool_id_set = set(pool.chunk_ids)
    aspects_to_chunk_ids: dict[QueryModifierLabelId, list[str]] = {}
    for modifier in modifiers:
        cited = sorted(
            {
                d['chunk_id']
                for d in facts_per_facet.get(modifier.label, [])
                if d['chunk_id'] in pool_id_set
            }
        )
        if cited:
            aspects_to_chunk_ids[modifier.label] = cited

    all_gold = {cid for cids in aspects_to_chunk_ids.values() for cid in cids}
    print(
        f'\tmap: {len(modifiers)} modifiers → {len(aspects_to_chunk_ids)} non-empty facets, {len(all_gold)} gold chunks'
    )

    # REDUCE phase - synthesize unified answer from all extracted facts
    if query_id in prior_answers:
        final_answer = prior_answers[query_id]
        print(f'\t[resume] reduce query_id={query_id}: answer cached ({len(final_answer)} chars)')
    else:
        final_answer = reduce_answer(
            query_text=query_row['query_text'],
            condition_name=query_row['condition_name'],
            modifiers=modifiers,
            facts_per_facet=facts_per_facet,
            query_id=query_id,
            persistence=persistence,
        )
        persistence.append_answer(
            {
                'query_id': query_id,
                'query_text': query_row['query_text'],
                'answer_text': final_answer if final_answer is not None else '<LLM_ERROR>',
            }
        )

    return aspects_to_chunk_ids, final_answer


def map_batch_extract_facts(
    query_text: str,
    condition_name: str,
    modifier: QueryModifier,
    chunks: ChunkPool,
    persistence: GoldAnnotationPersistenceHandler,
    batch_idx: int = 0,
    n_batches: int = 1,
    query_id: int = 0,
) -> list[ExtractedFact] | None:
    """Extract clinical facts from one (modifier, chunk batch) - MAP step."""
    prompt = _build_map_prompt(
        query_text=query_text,
        condition_name=condition_name,
        modifier=modifier,
        chunks=chunks,
    )
    if persistence is not None:
        persistence.dump_map_prompt(
            prompt, label=modifier.label, batch_idx=batch_idx, query_id=query_id
        )
    chunk_id_set = set(chunks.chunk_ids)

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
            f'[ERROR] extract_facts query_id={query_id} b{batch_idx + 1}/{n_batches} {modifier.label}: {e}'
        )
        return None

    raw: list = []
    if isinstance(result, list):
        raw = result
    elif isinstance(result, dict):
        raw = result.get('facts', []) or result.get('decisions', [])

    facts: list[ExtractedFact] = []
    seen_facts: set[tuple[str, str]] = set()
    n_dropped = 0
    for item in raw:
        if not isinstance(item, dict):
            n_dropped += 1
            continue
        chunk_id = str(item.get('chunk_id', '')).strip()
        fact = str(item.get('fact', '')).strip()
        fact_key = (chunk_id, _normalize_fact_for_dedupe(fact))
        if not chunk_id or chunk_id not in chunk_id_set or not fact or fact_key in seen_facts:
            n_dropped += 1
            continue
        seen_facts.add(fact_key)
        facts.append({'chunk_id': chunk_id, 'fact': fact})

    print(
        f'\t\t[map] b{batch_idx + 1}/{n_batches} {modifier.label}: '
        f'{len(facts)} facts, {n_dropped} dropped (~{len(prompt) // 4} tokens)'
    )
    return facts


def reduce_answer(
    query_text: str,
    condition_name: str,
    modifiers: list[QueryModifier],
    facts_per_facet: dict[QueryModifierLabelId, list[ExtractedFact]],
    query_id: int,
    persistence: GoldAnnotationPersistenceHandler,
) -> str | None:
    """Synthesize unified comparative answer from extracted facts - REDUCE step."""
    prompt = _build_reduce_prompt(
        query_text=query_text,
        condition_name=condition_name,
        modifiers=modifiers,
        facts_per_facet=facts_per_facet,
    )
    persistence.dump_reduce_prompt(prompt, query_id)

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
        print(f'[ERROR] synthesize_answer query_id={query_id}: {e}')
        return None


def _build_gold_candidate_pool(
    pool_builder: ChunkPoolBuilder, query_row: QueryRow, modifiers: list[QueryModifier]
) -> ChunkPool:
    """Top-k cosine per modifier, merged."""
    min_per_modifier = gold_annotation_cfg.min_per_modifier
    vec = pool_builder.embed_query(query_row['query_text'])

    modifier_pools: list[ChunkPool] = []
    modifier_pool_sizes: dict[QueryModifierLabelId, int] = {}
    for modifier in modifiers:
        modifier_pool = pool_builder.topk_cosine_for_modifiers(
            vec=vec,
            icd10_3char=query_row['icd10_3char'],
            modifiers=[modifier],
            k=min_per_modifier,
        )
        modifier_pools.append(modifier_pool)
        modifier_pool_sizes[modifier.label] = modifier_pool.n

    pool = ChunkPool.merge(modifier_pools)
    modifier_summary = '; '.join(f'{label}: {n}' for label, n in modifier_pool_sizes.items())
    print(f'\t[pool] min_per_modifier={min_per_modifier}: {pool.n} chunks; {modifier_summary}')

    return pool


def _build_map_prompt(
    query_text: str, condition_name: str, modifier: QueryModifier, chunks: ChunkPool
) -> str:
    chunks_block_parts = []
    for c_id, text, section, context_prefix in zip(
        chunks.chunk_ids,
        chunks.texts,
        chunks.section_names,
        chunks.contextual_prefixes,
        strict=True,
    ):
        header = f'[CHUNK_ID: {c_id}] [{section}]' if section else f'[CHUNK_ID: {c_id}]'
        if context_prefix:
            header += f'\n{context_prefix}'
        chunks_block_parts.append(f'{header}\n{text}')

    return gold_annotation_cfg.fact_extract_template.format(
        query_text=query_text,
        condition_name=condition_name,
        modifier_subgroup_text=(modifier.format_subgroup_info()),
        chunks_block='\n---\n'.join(chunks_block_parts),
    )


def _build_reduce_prompt(
    query_text: str,
    condition_name: str,
    modifiers: list[QueryModifier],
    facts_per_facet: dict[QueryModifierLabelId, list[ExtractedFact]],
) -> str:
    subgroups_block = '\n\n'.join(
        f'=== SUBGROUP {i + 1} - {(m.format_subgroup_info())} ===\n'
        + (
            '\n'.join(f'- [{d["chunk_id"]}] {d["fact"]}' for d in facts_per_facet.get(m.label, []))
            or '(no relevant facts found)'
        )
        for i, m in enumerate(modifiers)
    )
    return gold_annotation_cfg.answer_gen_template.format(
        query_text=query_text,
        condition_name=condition_name,
        subgroups_block=subgroups_block,
    )


def _normalize_fact_for_dedupe(fact: str) -> str:
    return ' '.join(fact.lower().split())


class GoldAnnotationPersistenceHandler:
    def __init__(self, cfg: GoldAnnotationCfg):
        self.cfg = cfg
        self.annotations_jsonl_path = get_table_path('gold_annotations', ext='jsonl')
        self.annotations_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.answers_jsonl_path = get_table_path('gold_answers', ext='jsonl')
        self.answers_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_dump_dir = (
            get_result_dir('gold_annotations') / '_prompt_dump'
            if cfg.dump_prompts is True
            else None
        )
        if self.prompt_dump_dir is not None:
            self.prompt_dump_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def annotation_cache_key(
        query_id: int, facet_label: str, chunk_ids: list[str]
    ) -> AnnotationCacheKey:
        return query_id, facet_label, tuple(chunk_ids)

    def load_prior_annotations(self) -> dict[AnnotationCacheKey, list[ExtractedFact]]:
        """Load cached MAP entries keyed by stable query_id, facet label, and exact prompt chunks."""
        prior: dict[AnnotationCacheKey, list[ExtractedFact]] = {}
        n_legacy = 0
        if not self.annotations_jsonl_path.exists():
            return prior
        with self.annotations_jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                decisions = entry.get('decisions', [])
                if decisions == '<LLM_ERROR>':
                    continue
                if 'query_id' not in entry or 'chunk_ids' not in entry:
                    n_legacy += 1
                    continue
                key = self.annotation_cache_key(
                    query_id=entry['query_id'],
                    facet_label=entry['facet_label'],
                    chunk_ids=entry['chunk_ids'],
                )
                prior[key] = self.dedupe_facts(decisions)

        legacy_msg = f', skipped {n_legacy} legacy positional entries' if n_legacy else ''
        print(f'[resume] loaded {len(prior)} cached map entries from jsonl{legacy_msg}')
        return prior

    def load_prior_answers(self) -> dict[int, str]:
        """Load cached REDUCE answers keyed by stable query_id."""
        prior: dict[int, str] = {}
        n_legacy = 0
        if not self.answers_jsonl_path.exists():
            return prior
        with self.answers_jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                ans = entry.get('answer_text', '')
                if ans == '<LLM_ERROR>':
                    continue
                if 'query_id' not in entry:
                    n_legacy += 1
                    continue
                prior[entry['query_id']] = ans
        legacy_msg = f', skipped {n_legacy} legacy positional entries' if n_legacy else ''
        print(f'[resume] loaded {len(prior)} cached reduce answers from jsonl{legacy_msg}')
        return prior

    def append_answer(self, ans: dict[str, int | str]) -> None:
        with self.answers_jsonl_path.open('a') as f:
            f.write(json.dumps(ans) + '\n')

    def append_annotation(self, ans: dict[str, Any]) -> None:
        with self.annotations_jsonl_path.open('a') as f:
            f.write(json.dumps(ans) + '\n')

    def dump_map_prompt(self, prompt: str, query_id: int, batch_idx: int, label: str) -> None:
        if self.prompt_dump_dir is not None:
            dump_path = (
                self.prompt_dump_dir / f'query_{query_id:03d}_b{batch_idx:03d}_{label}_map.txt'
            )
            dump_path.write_text(
                f'=== SYSTEM ===\n{self.cfg.fact_extract_system_prompt}\n\n=== USER ===\n{prompt}'
            )

    def dump_reduce_prompt(self, prompt: str, query_id: int) -> None:
        if self.prompt_dump_dir is not None:
            dump_path = self.prompt_dump_dir / f'query_{query_id:03d}_reduce.txt'
            dump_path.write_text(
                f'=== SYSTEM ===\n{self.cfg.answer_system_prompt}\n\n=== USER ===\n{prompt}'
            )

    @staticmethod
    def dedupe_facts(raw_facts: list[dict[str, Any]]) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        seen: set[tuple[str, str]] = set()
        for item in raw_facts:
            chunk_id = str(item.get('chunk_id', '')).strip()
            fact = str(item.get('fact', '')).strip()
            key = (chunk_id, _normalize_fact_for_dedupe(fact))
            if not chunk_id or not fact or key in seen:
                continue
            seen.add(key)
            facts.append({'chunk_id': chunk_id, 'fact': fact})
        return facts


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='evaluation')
    run_gold_annotation(cfg=GoldAnnotationCfg(**raw['gold_annotation']))
