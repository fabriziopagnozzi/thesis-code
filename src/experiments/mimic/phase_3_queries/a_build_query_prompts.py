"""
Step 3.1: Build grounded LLM prompts for query generation.

For each top Charlson-bucket condition (by n_admissions, from conditions_stats.parquet):
    1. Enumerate co-occurring Charlson comorbidities (excluding the condition's own bucket)
       + demographic modifiers. Build two modifier sets: [top_comorbidity, demographic] and
       [rare_comorbidity, demographic] - exactly 2 aspects each.
    2. Sample real BHC chunks from the condition's admissions, per-modifier quota.
    3. Assemble one full prompt per modifier set.

Output: queries_prompts.parquet - two rows per condition (top + rare comorbidity).
"""

import json

import polars as pl
from duckdb import DuckDBPyConnection

from experiments.mimic.configs import (
    BuildQueryPromptsCfg,
    get_parquet_path,
    global_cfg,
    setup_logging,
)
from experiments.mimic.duck_db_init import connect_mimic_duckdb

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

    conditions = pl.read_parquet(get_parquet_path('conditions_stats'))
    chunks = pl.read_parquet(get_parquet_path('chunks'))
    metadata = pl.read_parquet(get_parquet_path('admissions_metadata'))

    print(
        f'Loaded {len(conditions):,} conditions, {len(chunks):,} chunks, {len(metadata):,} admissions'
    )

    df = build_query_prompts(conditions, chunks, metadata, con)

    out_path = get_parquet_path('queries_prompts')
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

    for condition_row in conditions.head(global_cfg.num_conditions).iter_rows(named=True):
        charlson_col = condition_row['charlson_col']
        cond_name = condition_row['condition_name'] or charlson_col
        cond_hadm_ids = _get_charlson_hadm_ids(con, charlson_col)

        top_mods = json.loads(condition_row.get('top_comorbidity_mods_json') or '[]')
        charlson = [
            (m['label'], 'comorbidity') for m in top_mods[: query_prompts_cfg.max_modifiers]
        ]
        demographic = [
            (text, 'demographic') for text in query_prompts_cfg.demographic_modifiers_text
        ]

        if not charlson:
            skipped += 1
            continue

        # Two queries per condition: top + rare comorbidity, each paired with one demographic - exactly 2 aspects each
        demo = demographic[:1]
        modifier_sets: list[list[tuple[str, str]]] = [[charlson[0], *demo]]
        if len(charlson) >= 2:
            modifier_sets.append([charlson[-1], *demo])

        for modifiers in modifier_sets:
            data_samples = _sample_grounding_chunks_per_modifier(
                chunks_by_hadm_id,
                meta_by_hadm_id,
                cond_hadm_ids,
                modifiers,
                con,
                n=query_prompts_cfg.n_grounding_patients,
            )
            if not data_samples:
                continue

            modifier_list = '\n'.join(f'- {text}' for text, _ in modifiers)
            full_prompt = query_prompts_cfg.prompt_template.format(
                condition=cond_name,
                modifier_list=modifier_list,
                chunks_block=_format_chunks_block(data_samples),
            )
            results.append(
                {
                    'charlson_col': charlson_col,
                    'condition_name': cond_name,
                    'modifiers_json': json.dumps([{'text': t, 'type': ty} for t, ty in modifiers]),
                    'n_modifiers': len(modifiers),
                    'n_condition_admissions': len(cond_hadm_ids),
                    'n_grounding_chunks': len(data_samples),
                    'grounding_hadm_ids': list({s['hadm_id'] for s in data_samples}),
                    'full_prompt': full_prompt,
                }
            )

    print(
        f'Built {len(results):,} grounded prompts, skipped {skipped:,} conditions (no charlson modifiers)'
    )
    return pl.DataFrame(results)


# -- Step 2: sample grounding chunks --
def _get_charlson_hadm_ids(con, charlson_col: str) -> set[int]:
    rows = con.execute(f"""--sql
        SELECT DISTINCT hadm_id
        FROM mimiciv_derived.charlson
        WHERE {charlson_col} > 0
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


def _sample_grounding_chunks_per_modifier(
    chunks_by_hadm: dict[int, pl.DataFrame],
    meta_by_hadm: dict[int, dict],
    cond_hadm_ids: set[int],
    modifiers: list[tuple[str, str]],
    con,
    n: int,
) -> list[dict]:
    """Sample up to n patients, round-robin across modifiers for diversity."""
    seen_hadm: set[int] = set()
    samples: list[dict] = []

    for mod_text, mod_type in modifiers * n:
        if len(seen_hadm) >= n:
            break
        mod_hadm_ids = _get_modifier_hadm_ids(con, cond_hadm_ids, mod_text, mod_type)
        if not mod_hadm_ids:
            continue
        bhc_candidates = _hadm_ids_with_bhc(chunks_by_hadm, mod_hadm_ids - seen_hadm)
        if not bhc_candidates:
            continue
        chosen_hid = bhc_candidates[0]
        seen_hadm.add(chosen_hid)
        samples.extend(_sample_patient(chosen_hid, chunks_by_hadm, meta_by_hadm))

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
    setup_logging()
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=3)
    run_build_query_prompts(cfg=BuildQueryPromptsCfg(**raw['build_query_prompts']))
