"""
Step 4.2: Gold facet annotation via map-reduce LLM calls.

For each query + its candidate pool, annotates which chunks support which
facets (aspects of the answer). Uses ollama for local inference.

STEPS:
    Map: batches of N chunks --> LLM extracts facts + facet labels + chunk citations
    Reduce: merge facet labels across batches (deterministic)

Output: gold_annotations.parquet
"""

import json

import duckdb
import numpy as np
import polars as pl
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from experiments.mimic.configs import MIMIC_RESULTS_DIR, EvaluateCfg, GoldAnnotationCfg
from experiments.mimic.duck_db_init import (
    connect_mimic_duckdb,
)
from experiments.mimic.phase_4_evaluation.candidate_pool import CandidatePool, CandidatePoolBuilder
from helpers.ollama_client import generate_json

gold_annotation_cfg = GoldAnnotationCfg.load()


class _Annotation(BaseModel):
    fact: str
    facet_label: str
    chunk_ids: list[str]

    @field_validator('facet_label')
    @classmethod
    def normalize_label(cls, v: str) -> str:
        return v.strip().lower().replace(' ', '_').replace(' ', '_')

    @field_validator('fact')
    @classmethod
    def nonempty_fact(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('empty fact')
        return v


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
    filtered_queries_path = MIMIC_RESULTS_DIR / 'divergence_stats.parquet'
    if filtered_queries_path.exists():
        all_queries = pl.read_parquet(filtered_queries_path)
        queries_df = all_queries.filter(pl.col('passes_filter'))
    else:
        queries_df = pl.read_parquet(MIMIC_RESULTS_DIR / 'queries.parquet')

    # Load patient metadata for chunk context
    meta_path = MIMIC_RESULTS_DIR / 'admissions_metadata.parquet'
    patient_meta = _build_patient_meta(meta_path) if meta_path.exists() else None
    if patient_meta:
        print(f'Loaded patient metadata for {len(patient_meta):,} admissions')

    # Resume from previous run if output exists
    out_path = MIMIC_RESULTS_DIR / 'gold_annotations.parquet'
    done_texts: set[str] = set()
    if out_path.exists():
        prev = pl.read_parquet(out_path)
        done_texts = set(prev['query_text'].to_list())
        print(
            f'Resuming: {len(done_texts)} queries already done, {len(queries_df) - len(done_texts)} remaining'
        )

    builder = CandidatePoolBuilder(con, cfg=EvaluateCfg.load(), device='cuda')
    result = annotate(queries_df, builder, patient_meta, done_texts)

    out_path = MIMIC_RESULTS_DIR / 'gold_annotations.parquet'
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
        query_id, icd10_3char, query_text, facets_json, n_facets, n_gold_chunks
    """

    out_path = MIMIC_RESULTS_DIR / 'gold_annotations.parquet'

    for i, row in enumerate(
        tqdm(queries_df.iter_rows(named=True), total=len(queries_df), desc='Gold annotation')
    ):
        icd3 = row['icd10_3char']
        query_text = row['query_text']

        if query_text in done_texts:
            continue

        modifier_text = row.get('modifier_text')
        query_vec = builder.embed_query(query_text)

        # Pool is already stratified and prefiltered
        work_pool = builder.for_query_stratified(
            icd3,
            query_vec,
            prefilter_n=gold_annotation_cfg.prefilter_n,
            modifier_text=modifier_text,
        )

        # Sort by descending similarity so most relevant chunks are in
        # the first batches (shapes the accumulated facet vocabulary)
        sim_to_query = work_pool.sim_to_query(query_vec)
        sorted_indices = np.argsort(sim_to_query)[::-1]
        work_pool = work_pool.slice(sorted_indices.copy())

        facets = annotate_query(
            query_text,
            work_pool,
            batch_size=gold_annotation_cfg.batch_size,
            patient_meta=patient_meta,
        )  # type: ignore

        all_gold_chunks = set()
        for cids in facets.values():
            all_gold_chunks.update(cids)

        query_id = f'{icd3}_{row.get("modifier_text", "")}_{row.get("persona", "")}_{i}'
        query_id = query_id.replace(' ', '_')[:120]

        new_row = pl.DataFrame(
            [
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
            ]
        )

        # Append to parquet on disk after every query
        if out_path.exists():
            existing = pl.read_parquet(out_path)
            pl.concat([existing, new_row]).write_parquet(out_path)
        else:
            new_row.write_parquet(out_path)

    return pl.read_parquet(out_path) if out_path.exists() else pl.DataFrame()


def annotate_query(
    query_text: str,
    pool: CandidatePool,
    batch_size: int = 40,
    patient_meta: dict[int, str] | None = None,
) -> dict[str, list[str]]:
    """Full map-reduce annotation for one query.

    Returns {facet_label: [chunk_id, ...]}.
    """
    n = pool.n
    n_batches = (n + batch_size - 1) // batch_size
    all_batch_results = []
    accumulated_facets: set[str] = set()

    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_ids = pool.chunk_ids[start:end]
        batch_texts = pool.texts[start:end]
        batch_sections = pool.section_names[start:end]
        batch_hadm_ids = pool.hadm_ids[start:end].tolist()
        batch_meta = [patient_meta.get(h, '') for h in batch_hadm_ids] if patient_meta else None

        batch_result = annotate_batch(
            query_text,
            batch_ids,
            batch_texts,
            batch_sections=batch_sections,
            batch_meta=batch_meta,
            batch_idx=i,
            n_batches=n_batches,
            existing_facets=accumulated_facets,
        )
        all_batch_results.append(batch_result)
        accumulated_facets.update(item['facet_label'] for item in batch_result)

    facets = reduce_facets(all_batch_results)
    total_facts = sum(len(b) for b in all_batch_results)
    all_gold = {cid for cids in facets.values() for cid in cids}
    print(f'    reduce: {total_facts} facts --> {len(facets)} facets, {len(all_gold)} gold chunks')
    return facets


def annotate_batch(
    query_text: str,
    chunk_ids: list[str],
    texts: list[str],
    batch_sections: list[str] | None = None,
    batch_meta: list[str] | None = None,
    batch_idx: int = 0,
    n_batches: int = 1,
    existing_facets: set[str] | None = None,
) -> list[dict]:
    """Run map-phase annotation on a single batch of chunks.

    Returns list of {fact, facet_label, chunk_ids} dicts.
    """
    chunks_block = _format_chunk_batch(
        chunk_ids, texts, sections=batch_sections, meta_lines=batch_meta
    )
    prompt = gold_annotation_cfg.map_user_template.format(
        query_text=query_text, chunks_block=chunks_block
    )

    if existing_facets:
        facet_list = ', '.join(sorted(existing_facets))
        prompt += (
            f'\nThe following facet labels have already been identified in previous batches: '
            f'{facet_list}\n'
            f'Reuse these labels when a fact addresses the same clinical aspect. '
            f'Only create a new label if none of them fit.'
        )
    prompt_chars = len(prompt)

    valid_ids = set(chunk_ids)

    try:
        result = generate_json(
            prompt,
            system=gold_annotation_cfg.map_system_prompt,
            model=gold_annotation_cfg.model or None,
            temperature=gold_annotation_cfg.temperature,
            top_p=gold_annotation_cfg.top_p,
            top_k=gold_annotation_cfg.top_k,
            num_ctx=gold_annotation_cfg.num_ctx,
            num_predict=gold_annotation_cfg.num_predict,
            think=gold_annotation_cfg.think,
        )
    except Exception as e:
        print(
            f'[ERROR] batch {batch_idx + 1}/{n_batches} FAILED ({len(chunk_ids)} chunks, ~{prompt_chars // 4} tokens): {e}'
        )
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

    cleaned = []
    n_dropped = 0
    for item in result:
        try:
            ann = _Annotation.model_validate(item)
        except Exception:
            n_dropped += 1
            continue
        ann.chunk_ids = [c for c in ann.chunk_ids if c in valid_ids]
        if not ann.chunk_ids:
            n_dropped += 1
            continue
        cleaned.append(ann.model_dump())

    facet_labels = {item['facet_label'] for item in cleaned}
    print(
        f'[INFO] batch {batch_idx + 1}/{n_batches}: '
        f'{len(cleaned)} facts, {len(facet_labels)} facets, '
        f'{n_dropped} dropped '
        f'({len(chunk_ids)} chunks, ~{prompt_chars // 4} tokens)'
    )
    return cleaned


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


def _build_patient_meta(meta_path) -> dict[int, str]:
    meta = pl.read_parquet(meta_path)

    charlson_cols = {
        'myocardial_infarct': 'myocardial infarction',
        'congestive_heart_failure': 'congestive heart failure',
        'peripheral_vascular_disease': 'peripheral vascular disease',
        'cerebrovascular_disease': 'cerebrovascular disease',
        'chronic_pulmonary_disease': 'chronic pulmonary disease',
        'diabetes_without_cc': 'diabetes',
        'diabetes_with_cc': 'diabetes with complications',
        'renal_disease': 'renal disease',
        'mild_liver_disease': 'liver disease',
        'severe_liver_disease': 'severe liver disease',
        'malignant_cancer': 'cancer',
        'metastatic_solid_tumor': 'metastatic cancer',
    }

    lookup: dict[int, str] = {}
    for row in meta.iter_rows(named=True):
        age = int(row['age']) if row.get('age') is not None else None
        gender = 'F' if row.get('gender') == 'F' else 'M'
        age_str = f'age {age}' if age is not None else 'age unknown'

        comorbidities = [
            label for col, label in charlson_cols.items() if row.get(col) and row[col] > 0
        ]
        primary = row.get('primary_icd_description', '')

        parts = [f'{age_str}, {gender}']
        if primary:
            parts.append(f'primary dx: {primary}')
        if comorbidities:
            parts.append(f'comorbidities: {", ".join(comorbidities)}')

        lookup[row['hadm_id']] = 'Patient: ' + ' | '.join(parts)

    return lookup


def reduce_facets(all_batch_results: list[list[dict]]) -> dict[str, list[str]]:
    facet_to_chunks: dict[str, set[str]] = {}

    for batch in all_batch_results:
        for item in batch:
            label = item['facet_label']
            if label not in facet_to_chunks:
                facet_to_chunks[label] = set()
            facet_to_chunks[label].update(item['chunk_ids'])

    return {label: sorted(cids) for label, cids in facet_to_chunks.items()}


if __name__ == '__main__':
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=3)
    run_gold_annotation(cfg=GoldAnnotationCfg(**raw['gold_annotation']))
