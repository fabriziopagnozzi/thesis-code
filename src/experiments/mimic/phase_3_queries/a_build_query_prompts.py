"""
Step 3.1: Build grounded LLM prompts for query generation.

For each top condition (by frequency x comorbidity richness):
    1. Enumerate (modifier, persona) pairs - comorbidity axes from Charlson + demographic modifiers
    2. Sample real BHC chunks from the condition+modifier intersection
    3. Assemble a full prompt grounded in real clinical data

Output: queries_prompts.parquet
"""

import itertools

import numpy as np
import polars as pl
from experiments.mimic.config import N_CONDITIONS

from experiments.mimic.config_loader import load_config
from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
)

_cfg = load_config(phase=3)['build_query_prompts']


def main():
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
    print(
        f'\nSaved {len(df):,} prompts to {out_path}\n'
        f'  Conditions: {df["icd10_3char"].n_unique()}\n'
        f'  Avg grounding chunks: {df["n_grounding_chunks"].mean():.1f}\n'
        f'  Avg intersection admissions: {df["n_intersection_admissions"].mean():.0f}\n'
        f'\n--- Sample prompt (truncated) ---\n'
        f'{df["full_prompt"][0][:1500]}'
    )


PERSONAS: dict[str, str] = _cfg['personas']
CHARLSON_LABELS: dict[str, str] = _cfg['charlson_labels']

DEMOGRAPHIC_MODIFIERS: list[str] = [m['text'] for m in _cfg['demographic_modifiers']]

# -- Reverse maps used for filtering --
LABEL_TO_CHARLSON_COL = {v: k for k, v in CHARLSON_LABELS.items()}
DEMOGRAPHIC_FILTERS: dict[str, tuple] = {
    m['text']: (m['column'], m['op'], m['value']) for m in _cfg['demographic_modifiers']
}

N_GROUNDING_PATIENTS: int = _cfg['n_grounding_patients']
HIGH_VALUE_SECTIONS: list[str] = _cfg['high_value_sections']

PROMPT_TEMPLATE: str = _cfg['prompt_template']


def build_query_prompts(
    conditions: pl.DataFrame,
    chunks: pl.DataFrame,
    metadata: pl.DataFrame,
    con,
    max_modifiers: int | None = None,
) -> pl.DataFrame:
    """Build grounded query prompts for the top N_CONDITIONS conditions."""
    if max_modifiers is None:
        max_modifiers = int(_cfg['max_modifiers'])
    specs = _build_specs(conditions, con, max_modifiers)
    print(f'Built {len(specs):,} specs ({N_CONDITIONS} conditions)')

    chunk_index = _ChunkIndex(chunks, metadata)
    print(f'Chunk index: {len(chunk_index._by_hadm):,} hadm_ids with high-value sections')

    condition_hadm_cache: dict[str, set[int]] = {}
    modifier_hadm_cache: dict[tuple, set[int]] = {}
    samples_cache: dict[tuple, list[dict]] = {}

    results = []
    skipped = 0

    for spec in specs:
        icd3 = spec['icd10_3char']
        modifier_text = spec['modifier_text']
        modifier_type = spec['modifier_type']

        if icd3 not in condition_hadm_cache:
            condition_hadm_cache[icd3] = _get_condition_hadm_ids(con, icd3)

        cache_key = (icd3, modifier_text, modifier_type)
        if cache_key not in modifier_hadm_cache:
            modifier_hadm_cache[cache_key] = _get_modifier_hadm_ids(
                con, condition_hadm_cache[icd3], modifier_text, modifier_type
            )

        intersection = modifier_hadm_cache[cache_key]
        if not intersection:
            skipped += 1
            continue

        if cache_key not in samples_cache:
            samples_cache[cache_key] = _sample_grounding_chunks(
                chunk_index, intersection, seed=hash(cache_key) & 0xFFFFFFFF
            )
        samples = samples_cache[cache_key]

        if not samples:
            skipped += 1
            continue

        full_prompt = PROMPT_TEMPLATE.format(
            persona_prompt=spec['prompt'],
            condition=spec['condition_name'],
            modifier=modifier_text,
            chunks_block=_format_chunks_block(samples),
        )

        results.append(
            {
                'icd10_3char': icd3,
                'condition_name': spec['condition_name'],
                'modifier_text': modifier_text,
                'modifier_type': modifier_type,
                'persona': spec['persona'],
                'n_grounding_chunks': len(samples),
                'n_intersection_admissions': len(intersection),
                'grounding_hadm_ids': [s['hadm_id'] for s in samples],
                'full_prompt': full_prompt,
            }
        )

    print(f'Built {len(results):,} grounded prompts, skipped {skipped:,} (no intersection chunks)')
    return pl.DataFrame(results)


# -- Step 1: enumerate (condition, modifier, persona) specs --
def _find_top_comorbidity_modifiers(
    con, condition_icd3: str, n: int = 3
) -> list[tuple[str, str, float]]:
    """Most common co-occurring Charlson categories for a condition. Returns (col, label, rate)."""
    charlson_cols = list(CHARLSON_LABELS.keys())
    col_list = ', '.join(f'AVG(ch.{c}) AS {c}' for c in charlson_cols)

    rates = con.execute(f"""--sql
        SELECT {col_list}
        FROM mimiciv_hosp.diagnoses_icd di
        JOIN mimiciv_derived.charlson ch ON di.hadm_id = ch.hadm_id
        WHERE di.icd_version = 10 AND SUBSTR(di.icd_code, 1, 3) = '{condition_icd3}'
    """).fetchone()

    scored = [
        (col, CHARLSON_LABELS[col], float(rate))
        for col, rate in zip(charlson_cols, rates, strict=True)
        if rate and rate > 0.05
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:n]


def _build_specs(conditions: pl.DataFrame, con, max_modifiers: int = 3) -> list[dict]:
    specs = []
    for row in conditions.head(N_CONDITIONS).iter_rows(named=True):
        icd3 = row['icd10_3char']
        condition_name = row['condition_name'] or icd3

        comorbidity_mods = _find_top_comorbidity_modifiers(con, icd3, n=max_modifiers)
        modifiers = [(label, 'comorbidity') for _, label, _ in comorbidity_mods]
        modifiers.extend([(m, 'demographic') for m in DEMOGRAPHIC_MODIFIERS])

        for (modifier_text, modifier_type), persona_name in itertools.product(
            modifiers, PERSONAS.keys()
        ):
            specs.append(
                {
                    'icd10_3char': icd3,
                    'condition_name': condition_name,
                    'modifier_text': modifier_text,
                    'modifier_type': modifier_type,
                    'persona': persona_name,
                    'prompt': PERSONAS[persona_name].format(
                        condition=condition_name, modifier=modifier_text
                    ),
                }
            )
    return specs


# -- Step 2: sample grounding chunks and assemble full prompts --
def _get_condition_hadm_ids(con, icd3: str) -> set[int]:
    rows = con.execute(f"""--sql
        SELECT DISTINCT diagnoses_icd.hadm_id
        FROM diagnoses_icd
        WHERE diagnoses_icd.icd_version = 10
        AND SUBSTR(diagnoses_icd.icd_code, 1, 3) = '{icd3}'
    """).fetchall()
    return {r[0] for r in rows}


def _filter_comorbidity(con, condition_hadm_ids: set[int], modifier_text: str) -> set[int]:
    col = LABEL_TO_CHARLSON_COL.get(modifier_text)
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
    filt = DEMOGRAPHIC_FILTERS.get(modifier_text)
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


def _get_modifier_hadm_ids(
    con, condition_hadm_ids: set[int], modifier_text: str, modifier_type: str
) -> set[int]:
    if modifier_type == 'comorbidity':
        return _filter_comorbidity(con, condition_hadm_ids, modifier_text)
    if modifier_type == 'demographic':
        return _filter_demographic(con, condition_hadm_ids, modifier_text)
    return set()


class _ChunkIndex:
    """Pre-partitioned chunk index for fast hadm_id lookups."""

    def __init__(self, chunks: pl.DataFrame, metadata: pl.DataFrame):
        hv = chunks.filter(pl.col('section_name').is_in(HIGH_VALUE_SECTIONS))
        self._by_hadm: dict[int, pl.DataFrame] = {}
        for key, group in hv.group_by('hadm_id'):
            self._by_hadm[key[0]] = group
        self._meta: dict[int, dict] = {}
        for row in (
            metadata.select('hadm_id', 'age', 'gender', 'race')
            .unique(subset=['hadm_id'])
            .iter_rows(named=True)
        ):
            self._meta[row['hadm_id']] = row

    def hadm_ids_with_bhc(self, hadm_ids: set[int]) -> list[int]:
        """hadm_ids from the set that have BHC chunks, sorted by longest BHC."""
        result = []
        for hid in hadm_ids:
            group = self._by_hadm.get(hid)
            if group is None:
                continue
            bhc = group.filter(pl.col('section_name') == 'BRIEF HOSPITAL COURSE')
            if not bhc.is_empty():
                result.append((hid, bhc['approx_tokens'].max()))
        result.sort(key=lambda x: x[1], reverse=True)
        return [hid for hid, _ in result]

    def sample_patient(self, hadm_id: int) -> list[dict]:
        group = self._by_hadm[hadm_id]
        bhc_row = (
            group.filter(pl.col('section_name') == 'BRIEF HOSPITAL COURSE')
            .sort('approx_tokens', descending=True)
            .row(0, named=True)
        )

        section_priority = {s: i for i, s in enumerate(HIGH_VALUE_SECTIONS)}
        supp = group.filter(pl.col('section_name') != 'BRIEF HOSPITAL COURSE')
        supp = (
            supp.with_columns(
                pl.col('section_name').replace_strict(section_priority, default=99).alias('_p')
            )
            .sort('_p')
            .head(1)
            .drop('_p')
        )

        meta = self._meta.get(hadm_id)
        age = meta['age'] if meta else None
        gender = meta['gender'] if meta else None
        race = meta['race'] if meta else None
        age_str = f'age {int(age)}' if age is not None else 'unknown age'
        gender_str = 'female' if gender == 'F' else 'male' if gender == 'M' else 'unknown sex'
        header_prefix = f'{age_str}, {gender_str}, {race or "unknown"}'

        rows = [bhc_row] + (supp.to_dicts() if not supp.is_empty() else [])
        return [
            {
                'header': f'{header_prefix} [{row["section_name"] + (" > " + row["subsection_name"] if row.get("subsection_name") else "")}]',
                'text': row['text'],
                'hadm_id': hadm_id,
            }
            for row in rows
        ]


def _sample_grounding_chunks(
    chunk_index: _ChunkIndex, hadm_ids: set[int], seed: int = 42
) -> list[dict]:
    candidates = chunk_index.hadm_ids_with_bhc(hadm_ids)
    if not candidates:
        return []
    if len(candidates) > N_GROUNDING_PATIENTS:
        rng = np.random.default_rng(seed)
        candidates = rng.choice(candidates, size=N_GROUNDING_PATIENTS, replace=False).tolist()
    samples = []
    for hid in candidates:
        samples.extend(chunk_index.sample_patient(hid))
    return samples


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
    import argparse

    from experiments.mimic.config_loader import load_config_from_main

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.parse_args()

    _run_cfg = load_config_from_main(phase=3)['build_query_prompts']
    N_GROUNDING_PATIENTS = _run_cfg['n_grounding_patients']
    HIGH_VALUE_SECTIONS = _run_cfg['high_value_sections']
    CHARLSON_LABELS = _run_cfg['charlson_labels']
    DEMOGRAPHIC_MODIFIERS = [m['text'] for m in _run_cfg['demographic_modifiers']]
    DEMOGRAPHIC_FILTERS = {
        m['text']: (m['column'], m['op'], m['value']) for m in _run_cfg['demographic_modifiers']
    }
    PERSONAS = _run_cfg['personas']
    PROMPT_TEMPLATE = _run_cfg['prompt_template']
    LABEL_TO_CHARLSON_COL = {v: k for k, v in CHARLSON_LABELS.items()}

    main()
