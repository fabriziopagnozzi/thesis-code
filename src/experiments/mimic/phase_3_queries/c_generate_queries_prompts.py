"""
Step 2.3: Build grounded LLM prompts for query generation.

For each query spec (condition x modifier x persona), samples real BHC chunks
from the condition+modifier intersection and assembles a full prompt that
grounds the LLM in real clinical data.

Output: queries.parquet — ready to be fed to an LLM manually.
"""

import polars as pl

from experiments.mimic.duck_db_init import (
    MIMIC_RESULTS_DIR,
    connect_mimic_duckdb,
    run_sql_concept_script,
)
from experiments.mimic.phase_3_queries.b_query_templates import (
    CHARLSON_LABELS,
    DEMOGRAPHIC_MODIFIERS,
)

N_GROUNDING_PATIENTS = 4
# Sections ranked by QA value (from feasibility analysis)
HIGH_VALUE_SECTIONS = [
    'BRIEF HOSPITAL COURSE',
    'HISTORY OF PRESENT ILLNESS',
    'PERTINENT RESULTS',
    'DISCHARGE DIAGNOSIS',
    'DISCHARGE MEDICATIONS',
]

# -- Reverse map: human label → charlson column name --
LABEL_TO_CHARLSON_COL = {v: k for k, v in CHARLSON_LABELS.items()}

# -- Demographic modifier parsing --
DEMOGRAPHIC_FILTERS = {
    DEMOGRAPHIC_MODIFIERS[0]: ('age', '>', 75),  # elderly
    DEMOGRAPHIC_MODIFIERS[1]: ('age', '<', 40),  # young adult
}

PROMPT_TEMPLATE = """\
{persona_prompt}

Below are excerpts from real discharge summaries of patients admitted for \
{condition} where {modifier} was also present. Each excerpt is from the \
discharge notes, describing clinical management decisions, lab results, \
diagnoses, and medications.

{chunks_block}

Using these real cases as background context (but NOT overfitting to any \
specific patient), generate ONE clinical management question about \
{condition} that:
- Requires knowing how treatment, workup, or monitoring changes when \
{modifier} is present
- Is answerable by synthesizing patterns across multiple patient cases
- Focuses on a generalizable management decision (drug choice, dosing \
adjustments, contraindications, monitoring frequency, disposition)
- Requires multi-source evidence — no single patient note answers it fully

Return ONLY the question, nothing else."""


def _get_condition_hadm_ids(con, icd3: str) -> set[int]:
    rows = con.execute(f"""--sql
        SELECT DISTINCT diagnoses_icd.hadm_id
        FROM diagnoses_icd
        WHERE diagnoses_icd.icd_version = 10
        AND SUBSTR(diagnoses_icd.icd_code, 1, 3) = '{icd3}'
    """).fetchall()
    return {r[0] for r in rows}


def _filter_comorbidity_gt_0(con, condition_hadm_ids: set[int], modifier_text: str) -> set[int]:
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


def get_modifier_hadm_ids(
    con, condition_hadm_ids: set[int], modifier_text: str, modifier_type: str
) -> set[int]:
    if modifier_type == 'comorbidity':
        return _filter_comorbidity_gt_0(con, condition_hadm_ids, modifier_text)
    elif modifier_type == 'demographic':
        return _filter_demographic(con, condition_hadm_ids, modifier_text)
    return set()


def sample_grounding_chunks(
    all_chunks: pl.DataFrame,
    hadm_ids: set[int],
    metadata: pl.DataFrame,
    n_patients: int = N_GROUNDING_PATIENTS,
    seed: int = 42,
) -> list[dict]:
    """Sample multi-section chunks from the condition+modifier intersection.

    Picks n_patients distinct patients, then for each patient includes:
    - The longest BHC chunk (clinical reasoning, always present)
    - One supplementary high-value section if available
    """
    pool = all_chunks.filter(
        pl.col('hadm_id').is_in(hadm_ids) & pl.col('section_name').is_in(HIGH_VALUE_SECTIONS)
    )
    if pool.is_empty():
        return []

    bhc_pool = pool.filter(pl.col('section_name') == 'BRIEF HOSPITAL COURSE')
    if bhc_pool.is_empty():
        return []

    bhc_per_patient = bhc_pool.sort('approx_tokens', descending=True).unique(
        subset=['hadm_id'], keep='first'
    )
    if len(bhc_per_patient) > n_patients:
        bhc_per_patient = bhc_per_patient.sample(n_patients, seed=seed)

    selected_hadm_ids = set(bhc_per_patient['hadm_id'].to_list())

    supp_pool = pool.filter(
        pl.col('hadm_id').is_in(selected_hadm_ids)
        & (pl.col('section_name') != 'BRIEF HOSPITAL COURSE')
    )

    # Rank by section priority, keep one per patient
    section_priority = {s: i for i, s in enumerate(HIGH_VALUE_SECTIONS)}
    supp_pool = supp_pool.with_columns(
        pl.col('section_name').replace_strict(section_priority, default=99).alias('_priority')
    )
    supp_per_patient = (
        supp_pool.sort('_priority').unique(subset=['hadm_id'], keep='first').drop('_priority')
    )

    meta_sub = metadata.select(['hadm_id', 'age', 'gender', 'race']).unique(subset=['hadm_id'])

    samples = []
    for hadm_id in selected_hadm_ids:
        patient_chunks = []

        bhc_row = bhc_per_patient.filter(pl.col('hadm_id') == hadm_id).row(0, named=True)
        patient_chunks.append(bhc_row)

        other_chunks = supp_per_patient.filter(pl.col('hadm_id') == hadm_id)
        if not other_chunks.is_empty():
            patient_chunks.append(other_chunks.row(0, named=True))

        meta_row = meta_sub.filter(pl.col('hadm_id') == hadm_id)
        age = meta_row['age'][0] if not meta_row.is_empty() else None
        gender = meta_row['gender'][0] if not meta_row.is_empty() else None
        race = meta_row['race'][0] if not meta_row.is_empty() else None

        age_str = f'age {int(age)}' if age is not None else 'unknown age'
        gender_str = 'female' if gender == 'F' else 'male' if gender == 'M' else 'unknown sex'
        patient_header = f'{age_str}, {gender_str}, {race or "unknown"}'

        for chunk in patient_chunks:
            section = chunk['section_name']
            problem = chunk.get('subsection_name')
            label = f'{section} > {problem}' if problem else section
            samples.append(
                {
                    'header': f'{patient_header} [{label}]',
                    'text': chunk['text'],
                    'hadm_id': hadm_id,
                }
            )

    return samples


def format_chunks_block(samples: list[dict]) -> str:
    parts = []
    # Group by hadm_id so chunks from the same patient share a patient number
    seen_hadm: dict[int, int] = {}
    for s in samples:
        hid = s['hadm_id']
        if hid not in seen_hadm:
            seen_hadm[hid] = len(seen_hadm) + 1
        pid = seen_hadm[hid]
        parts.append(f'--- Patient {pid} ({s["header"]}) ---\n{s["text"]}')
    return '\n\n'.join(parts)


def build_grounded_prompts(
    specs: pl.DataFrame,
    chunks: pl.DataFrame,
    metadata: pl.DataFrame,
    con,
) -> pl.DataFrame:
    """For each query spec, sample grounding chunks and assemble the full prompt."""

    condition_hadm_cache: dict[str, set[int]] = {}
    modifier_hadm_cache: dict[tuple[str, str, str], set[int]] = {}

    results = []
    skipped = 0

    for row in specs.iter_rows(named=True):
        icd3 = row['icd10_3char']
        modifier_text = row['modifier_text']
        modifier_type = row['modifier_type']

        # Get condition hadm_ids (cached)
        if icd3 not in condition_hadm_cache:
            condition_hadm_cache[icd3] = _get_condition_hadm_ids(con, icd3)
        condition_hadm_ids = condition_hadm_cache[icd3]

        cache_key = (icd3, modifier_text, modifier_type)
        if cache_key not in modifier_hadm_cache:
            modifier_hadm_cache[cache_key] = get_modifier_hadm_ids(
                con, condition_hadm_ids, modifier_text, modifier_type
            )
        intersection_hadm_ids = modifier_hadm_cache[cache_key]

        if not intersection_hadm_ids:
            skipped += 1
            continue

        spec_seed = hash((icd3, modifier_text, modifier_type)) & 0xFFFFFFFF
        samples = sample_grounding_chunks(chunks, intersection_hadm_ids, metadata, seed=spec_seed)
        if not samples:
            skipped += 1
            continue

        chunks_block = format_chunks_block(samples)
        full_prompt = PROMPT_TEMPLATE.format(
            persona_prompt=row['prompt'],
            condition=row['condition_name'],
            modifier=modifier_text,
            chunks_block=chunks_block,
        )

        results.append(
            {
                'icd10_3char': icd3,
                'condition_name': row['condition_name'],
                'modifier_text': modifier_text,
                'modifier_type': modifier_type,
                'persona': row['persona'],
                'n_grounding_chunks': len(samples),
                'n_intersection_admissions': len(intersection_hadm_ids),
                'grounding_hadm_ids': [s['hadm_id'] for s in samples],
                'full_prompt': full_prompt,
            }
        )

    print(f'Built {len(results):,} grounded prompts, skipped {skipped:,} (no intersection chunks)')
    return pl.DataFrame(results)


def main():
    con = connect_mimic_duckdb()
    run_sql_concept_script(con, 'demographics/age.sql', 'comorbidity/charlson.sql')

    specs = pl.read_parquet(MIMIC_RESULTS_DIR / 'query_specs.parquet')
    chunks = pl.read_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')
    metadata = pl.read_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')

    print(f'Loaded {len(specs):,} specs, {len(chunks):,} chunks, {len(metadata):,} admissions')

    df = build_grounded_prompts(specs, chunks, metadata, con)

    out_path = MIMIC_RESULTS_DIR / 'queries_prompts.parquet'
    df.write_parquet(out_path)
    print(f'\nSaved {len(df):,} queries to {out_path}')
    print(f'  Conditions: {df["icd10_3char"].n_unique()}')
    print(f'  Avg grounding chunks: {df["n_grounding_chunks"].mean():.1f}')
    print(f'  Avg intersection admissions: {df["n_intersection_admissions"].mean():.0f}')

    print('\n--- Sample prompt (first row, truncated) ---')
    print(df['full_prompt'][0][:1500])


if __name__ == '__main__':
    main()
