"""
Step 3.1: Build grounded LLM prompts for query generation.

For each top condition (by frequency x comorbidity richness):
    1. Enumerate (condition, modifier) pairs - comorbidity axes from Charlson + demographic modifiers
    2. Sample real BHC chunks from the condition+modifier intersection
    3. Assemble a full prompt grounded in real clinical data

Output: queries_prompts.parquet
"""

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection

from experiments.mimic.configs import MIMIC_RESULTS_DIR, BuildQueryPromptsCfg, global_cfg
from experiments.mimic.duck_db_init import (
    connect_mimic_duckdb,
)

query_prompts_cfg = BuildQueryPromptsCfg.load()


def run_build_query_prompts(
    con: DuckDBPyConnection | None = None,
    cfg: BuildQueryPromptsCfg | None = None,
) -> pl.DataFrame:
    global query_prompts_cfg
    if cfg is not None:
        query_prompts_cfg = cfg

    if con is None:
        con = connect_mimic_duckdb()

    conditions = pl.read_parquet(MIMIC_RESULTS_DIR / 'conditions_stats.parquet')
    chunks = pl.read_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')
    metadata = pl.read_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')

    print(
        f'Loaded {len(conditions):,} conditions, {len(chunks):,} chunks, {len(metadata):,} admissions'
    )

    df = build_query_prompts(conditions, chunks, metadata, con)

    out_path = MIMIC_RESULTS_DIR / 'queries_prompts.parquet'
    df.write_parquet(out_path)

    return df


def build_query_prompts(
    conditions: pl.DataFrame,
    chunks: pl.DataFrame,
    metadata: pl.DataFrame,
    con,
) -> pl.DataFrame:
    filtered_chunks = chunks.filter(
        pl.col('section_name').is_in(query_prompts_cfg.high_value_sections)
    )
    chunks_by_hadm_id: dict[int, pl.DataFrame] = {
        key[0]: grp for (key, grp) in filtered_chunks.group_by('hadm_id')
    }
    meta_by_hadm_id: dict[int, dict] = {
        row['hadm_id']: row
        for row in metadata.select('hadm_id', 'age', 'gender', 'race')
        .unique(subset=['hadm_id'])
        .iter_rows(named=True)
    }

    print(f'Chunk index: {len(chunks_by_hadm_id):,} hadm_ids with high-value sections')

    results = []
    skipped = 0

    # OUTER LOOP: iterate over top N conditions from condition_stats (N defined in global_cfg)
    for condition_row in conditions.head(global_cfg.num_conditions).iter_rows(named=True):
        icd3 = condition_row['icd10_3char']
        cond_name = condition_row['condition_name'] or icd3
        cond_hadm_ids = _get_condition_hadm_ids(con, icd3)

        # FIND MODIFIERS
        # find dynamically the comborbidity modifier for current condition
        comorbidity_mods = _find_top_comorbidity_modifiers(
            con, icd3, n=query_prompts_cfg.max_modifiers
        )
        modifiers = [(disease_label, 'comorbidity') for _, disease_label, _ in comorbidity_mods]
        # use the hardocoded modifiers for demographics in the config
        modifiers.extend(
            [
                (demo_modifier, 'demographic')
                for demo_modifier in query_prompts_cfg.demographic_modifiers_text
            ]
        )

        # INNER LOOP 1: iterate over the modifiers
        for modifier_text, modifier_type in modifiers:
            candidate_hadm_ids = _get_modifier_hadm_ids(
                con, cond_hadm_ids, modifier_text, modifier_type
            )
            if not candidate_hadm_ids:
                continue

            # SAMPLE: use examples from the data to guide the LLM in the prompt
            seed = hash((icd3, modifier_text, modifier_type)) & 0xFFFFFFFF
            data_samples = _sample_grounding_chunks(
                chunks_by_hadm_id,
                meta_by_hadm_id,
                candidate_hadm_ids,
                seed=seed,
            )
            if not data_samples:
                continue

            full_prompt = query_prompts_cfg.prompt_template.format(
                condition=cond_name,
                modifier=modifier_text,
                chunks_block=_format_chunks_block(data_samples),
            )
            results.append(
                {
                    'icd10_3char': icd3,
                    'condition_name': cond_name,
                    'modifier_text': modifier_text,
                    'modifier_type': modifier_type,
                    'n_grounding_chunks': len(data_samples),
                    'n_intersection_admissions': len(candidate_hadm_ids),
                    'grounding_hadm_ids': [s['hadm_id'] for s in data_samples],
                    'full_prompt': full_prompt,
                }
            )
        # END INNER LOOP 1: iterate over the modifiers
    # END OUTER iterate over top N conditions (defined in global_cfg)

    print(f'Built {len(results):,} grounded prompts, skipped {skipped:,} (no intersection chunks)')
    return pl.DataFrame(results)


# -- Step 1: enumerate (condition, modifier) specs --
def _find_top_comorbidity_modifiers(
    con, condition_icd3: str, n: int = 3
) -> list[tuple[str, str, float]]:
    """Most common co-occurring Charlson categories for a condition. Returns (col, label, rate)."""
    charlson_cols = list(global_cfg.shared_queries_cfg.charlson_labels.keys())
    col_list = ', '.join(f'AVG(ch.{c}) AS {c}' for c in charlson_cols)

    rates = con.execute(f"""--sql
        SELECT {col_list}
        FROM mimiciv_hosp.diagnoses_icd di
        JOIN mimiciv_derived.charlson ch ON di.hadm_id = ch.hadm_id
        WHERE di.icd_version = 10 AND SUBSTR(di.icd_code, 1, 3) = '{condition_icd3}'
    """).fetchone()

    scored = [
        (col, global_cfg.shared_queries_cfg.charlson_labels[col], float(rate))
        for col, rate in zip(charlson_cols, rates, strict=True)
        if rate and rate > 0.05
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:n]


# -- Step 2: sample grounding chunks and assemble full prompts --
def _get_condition_hadm_ids(con, icd3: str) -> set[int]:
    rows = con.execute(f"""--sql
        SELECT DISTINCT diagnoses_icd.hadm_id
        FROM diagnoses_icd
        WHERE diagnoses_icd.icd_version = 10
        AND SUBSTR(diagnoses_icd.icd_code, 1, 3) = '{icd3}'
    """).fetchall()
    return {r[0] for r in rows}


def _get_modifier_hadm_ids(
    con, condition_hadm_ids: set[int], modifier_text: str, modifier_type: str
) -> set[int]:
    if modifier_type == 'comorbidity':
        return _filter_comorbidity(con, condition_hadm_ids, modifier_text)
    if modifier_type == 'demographic':
        return _filter_demographic(con, condition_hadm_ids, modifier_text)
    return set()


def _filter_comorbidity(con, condition_hadm_ids: set[int], modifier_text: str) -> set[int]:
    col = global_cfg.shared_queries_cfg.label_to_charlson_col.get(modifier_text)
    if col is None:
        return set()
    placeholders = ','.join(str(h) for h in condition_hadm_ids)

    rows = con.execute(f"""--sql
        SELECT charlson.hadm_id
        FROM charlson
        WHERE charlson.hadm_id IN ({placeholders})
        AND charlson.{col} > 0
    """).fetchall()
    return {r[0] for r in rows}


def _filter_demographic(con, condition_hadm_ids: set[int], modifier_text: str) -> set[int]:
    filt = query_prompts_cfg.demographic_filters.get(modifier_text)
    if filt is None:
        return set()
    _, op, val = filt
    placeholders = ','.join(str(h) for h in condition_hadm_ids)

    rows = con.execute(f"""--sql
        SELECT age.hadm_id
        FROM age
        WHERE age.hadm_id IN ({placeholders})
        AND age.age {op} {val}
    """).fetchall()
    return {r[0] for r in rows}


def _sample_grounding_chunks(
    chunks_by_hadm: dict[int, pl.DataFrame],
    meta_by_hadm: dict[int, dict],
    hadm_ids: set[int],
    seed: int = 42,
) -> list[dict]:
    candidates = _hadm_ids_with_bhc(chunks_by_hadm, hadm_ids)
    if not candidates:
        return []
    if len(candidates) > query_prompts_cfg.n_grounding_patients:
        rng = np.random.default_rng(seed)
        candidates = rng.choice(
            candidates, size=query_prompts_cfg.n_grounding_patients, replace=False
        ).tolist()
    samples = []
    for hid in candidates:
        samples.extend(_sample_patient(hid, chunks_by_hadm, meta_by_hadm))
    return samples


def _sample_patient(
    hadm_id: int, chunks_by_hadm: dict[int, pl.DataFrame], meta_by_hadm: dict[int, dict]
) -> list[dict]:
    group = chunks_by_hadm[hadm_id]
    bhc_row = (
        group.filter(pl.col('section_name') == 'BRIEF HOSPITAL COURSE')
        .sort('approx_tokens', descending=True)
        .row(0, named=True)
    )

    section_priority = {s: i for i, s in enumerate(query_prompts_cfg.high_value_sections)}
    supp = group.filter(pl.col('section_name') != 'BRIEF HOSPITAL COURSE')
    supp = (
        supp.with_columns(
            pl.col('section_name').replace_strict(section_priority, default=99).alias('_p')
        )
        .sort('_p')
        .head(1)
        .drop('_p')
    )

    meta = meta_by_hadm.get(hadm_id)
    age = meta['age'] if meta else None
    gender = meta['gender'] if meta else None
    race = meta['race'] if meta else None
    age_str = f'age {int(age)}' if age is not None else 'unknown age'
    gender_str = 'female' if gender == 'F' else 'male' if gender == 'M' else 'unknown sex'
    header_prefix = f'{age_str}, {gender_str}, {race or "unknown"}'

    rows = [bhc_row] + (supp.to_dicts() if not supp.is_empty() else [])
    return [
        {
            'header': f'{header_prefix} [{row["section_name"]}]',
            'text': row['text'],
            'hadm_id': hadm_id,
        }
        for row in rows
    ]


def _hadm_ids_with_bhc(chunks_by_hadm: dict[int, pl.DataFrame], hadm_ids: set[int]) -> list[int]:
    """hadm_ids from the set that have BHC chunks, sorted by longest BHC."""
    result = []
    for hid in hadm_ids:
        group = chunks_by_hadm.get(hid)
        if group is None:
            continue
        bhc = group.filter(pl.col('section_name') == 'BRIEF HOSPITAL COURSE')
        if not bhc.is_empty():
            result.append((hid, bhc['approx_tokens'].max()))
    result.sort(key=lambda x: x[1], reverse=True)
    return [hid for hid, _ in result]


def _format_chunks_block(samples: list[dict]) -> str:
    seen_hadm: dict[int, int] = {}
    parts = []
    for s in samples:
        hid = s['hadm_id']
        if hid not in seen_hadm:
            seen_hadm[hid] = len(seen_hadm) + 1
        parts.append(f'--- Patient {seen_hadm[hid]} ({s["header"]}) ---\n{s["text"]}')
    return '\n\n'.join(parts)


if __name__ == '__main__':
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=3)
    run_build_query_prompts(cfg=BuildQueryPromptsCfg(**raw['build_query_prompts']))
