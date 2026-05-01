"""
Step 3.1: Build grounded LLM prompts for query generation.

For each top ICD-10 3-char prefix condition (by n_admissions, from conditions_stats.parquet):
    1. Enumerate co-occurring Charlson comorbidities + demographic modifiers.
       Build two modifier sets: [top_comorbidity, demographic] and
       [rare_comorbidity, demographic], exactly 2 aspects each.
    2. Sample real BHC chunks from the condition's admissions, per-modifier quota.
    3. Assemble one full prompt per modifier set.

Output: queries_prompts.parquet, two rows per condition (top + rare comorbidity).
"""

import json
from typing import Literal, cast

import numpy as np
import polars as pl

from experiments.mimic.chunking.schemas_chunking import (
    AdmissionMetaSlimRow,
    ConditionStatsRow,
)
from experiments.mimic.global_configs import (
    get_table_path,
    global_cfg,
    read_parquet,
    setup_logging,
)
from experiments.mimic.queries.schemas_queries import (
    BuildQueryPromptsCfg,
    GroundingChunkSample,
    QueryModifier,
)
from experiments.mimic.utils.candidate_pools import ChunkPoolBuilder
from experiments.mimic.utils.charlson import ICD3_TO_CHARLSON_COLS

query_prompts_cfg = BuildQueryPromptsCfg.load()


def run_build_query_prompts(cfg: BuildQueryPromptsCfg | None = None) -> pl.DataFrame:
    global query_prompts_cfg
    if cfg is not None:
        query_prompts_cfg = cfg

    conditions = read_parquet('conditions_stats')
    chunks = read_parquet('chunks').filter(pl.col('section_name').is_in(global_cfg.sections_filter))
    metadata = read_parquet('admissions_metadata')

    print(
        f'Loaded {len(conditions):,} conditions, {len(chunks):,} chunks, {len(metadata):,} admissions'
    )

    df = build_query_prompts(conditions, chunks, metadata)
    out_path = get_table_path('queries_prompts')
    df.write_parquet(out_path)

    return df


def build_query_prompts(
    conditions: pl.DataFrame,
    chunks: pl.DataFrame,
    metadata: pl.DataFrame,
) -> pl.DataFrame:
    pool_builder = ChunkPoolBuilder(model_name=global_cfg.embedding_model)

    filtered_chunks = chunks.filter(
        pl.col('section_name').is_in(query_prompts_cfg.high_value_sections)
    )
    chunks_by_hadm_id: dict[int, pl.DataFrame] = {
        key[0]: grp for (key, grp) in filtered_chunks.group_by('hadm_id')
    }
    meta_by_hadm_id: dict[int, AdmissionMetaSlimRow] = {
        row['hadm_id']: cast(AdmissionMetaSlimRow, row)
        for row in metadata
        .select('hadm_id', 'age', 'gender', 'race')
        .unique(subset=['hadm_id'])
        .iter_rows(named=True)
    }

    print(f'Chunk index: {len(chunks_by_hadm_id):,} hadm_ids with high-value sections')

    selected_conditions = _select_conditions_stratified(
        conditions,
        num_conditions=global_cfg.num_conditions,
        min_adm=query_prompts_cfg.min_condition_admissions,
        max_adm=query_prompts_cfg.max_condition_admissions,
        n_strata=query_prompts_cfg.n_strata,
        scale=query_prompts_cfg.stratify_scale,
        seed=query_prompts_cfg.stratify_seed,
    )
    print(f'Stratified selection: {len(selected_conditions):,} conditions')

    results = []
    skipped = 0

    for row in selected_conditions.iter_rows(named=True):
        row = cast(ConditionStatsRow, row)
        stratum = row.get('stratum', 1)
        top_comorbidity_mods = json.loads(row.get('top_comorbidity_mods_json') or '[]')
        condition_hadm_ids = pool_builder.get_icd3_hadm_ids(row['icd10_3char'])
        excluded_cols = ICD3_TO_CHARLSON_COLS.get(row['icd10_3char'], frozenset())

        # NB label here is actually the charlson column DESCRIPTION, it was an old convention
        comorbidity_mod = [
            QueryModifier(descr=m['label'], type='comorbidity')
            for m in top_comorbidity_mods
            if m['col'] not in excluded_cols
        ][: query_prompts_cfg.max_comorbidity_modifiers]
        demographic_mod = [
            QueryModifier(descr=text, type='demographic')
            for text in query_prompts_cfg.demographic_modifiers_text
        ]

        if len(comorbidity_mod) == 0:
            skipped += 1
            continue

        # Two queries per condition: top + rare comorbidity, each paired with one demographic - exactly 2 aspects each
        demo = demographic_mod[:1]
        modifier_sets: list[list[QueryModifier]] = [[comorbidity_mod[0], *demo]]
        if len(comorbidity_mod) >= 2:
            modifier_sets.append([comorbidity_mod[-1], *demo])

        # POOL COMPISITION STATS
        modifier_stats: dict[QueryModifier, dict[str, int]] = {}
        n_condition_chunks = sum(
            len(chunks_by_hadm_id[h]) for h in condition_hadm_ids if h in chunks_by_hadm_id
        )
        unique_mods = set(comorbidity_mod + demographic_mod)

        for mod in unique_mods:
            mod_hadm_ids = pool_builder.get_hadm_ids_by_condition_modifier(row['icd10_3char'], mod)
            n_mod_chunks = sum(
                len(chunks_by_hadm_id[h]) for h in mod_hadm_ids if h in chunks_by_hadm_id
            )
            modifier_stats[mod] = {
                'n_admissions': len(mod_hadm_ids),
                'n_chunks': n_mod_chunks,
            }

        for modifiers in modifier_sets:
            # Skip if any modifier has too few admissions within this condition's pool
            if any(
                int(modifier_stats.get(mod).get('n_admissions', 0))
                < query_prompts_cfg.min_modifier_admissions
                for mod in modifiers
            ):
                skipped += 1
                continue

            data_samples = _sample_grounding_chunks_per_modifier(
                chunks_by_hadm_id,
                meta_by_hadm_id,
                row['icd10_3char'],
                modifiers,
                pool_builder,
                n=query_prompts_cfg.n_grounding_patients,
            )
            if not data_samples:
                continue

            modifier_list = '\n'.join(f'- {text}' for text, _ in modifiers)
            results.append({
                'icd10_3char': row['icd10_3char'],
                'condition_name': row['condition_name'],
                'stratum': stratum,
                'modifiers_json': json.dumps([{'text': t, 'type': ty} for t, ty in modifiers]),
                'n_modifiers': len(modifiers),
                'n_condition_admissions': len(condition_hadm_ids),
                'n_condition_chunks': n_condition_chunks,
                'modifier_stats_json': json.dumps({
                    mod.descr: modifier_stats[mod] for mod in modifiers
                }),
                'n_grounding_chunks': len(data_samples),
                'grounding_hadm_ids': list({s['hadm_id'] for s in data_samples}),
                'full_prompt': query_prompts_cfg.prompt_template.format(
                    condition=row['condition_name'],
                    modifier_list=modifier_list,
                    chunks_block=_format_chunks_block(data_samples),
                ),
            })
    # end for condition_row in selected_conditions.iter_rows(named=True):

    print(
        f'Built {len(results):,} grounded prompts, skipped {skipped:,} conditions '
        f'(no charlson modifiers or under min_modifier_admissions={query_prompts_cfg.min_modifier_admissions})'
    )
    return pl.DataFrame(results).with_row_index('query_id')


def _select_conditions_stratified(
    conditions: pl.DataFrame,
    num_conditions: int,
    min_adm: int | None,
    max_adm: int | None,
    n_strata: int,
    scale: Literal['linear', 'log'] = 'log',
    seed: int = 42,
) -> pl.DataFrame:
    """Stratified sample of conditions by n_admissions for pool-size diversity.

    Filter to [min_adm, max_adm], split the filtered range into n_strata log- or
    linear-spaced buckets, and sample num_conditions/n_strata from each bucket
    uniformly at random. Logs per-stratum counts.
    """
    filtered = conditions
    if min_adm is not None:
        filtered = filtered.filter(pl.col('n_admissions') >= min_adm)
    if max_adm is not None:
        filtered = filtered.filter(pl.col('n_admissions') <= max_adm)

    if filtered.is_empty():
        raise ValueError(
            f'No conditions in range [{min_adm}, {max_adm}]. '
            f'conditions_stats n_admissions span: [{conditions["n_admissions"].min()}, {conditions["n_admissions"].max()}]'
        )

    if n_strata <= 1:
        take = min(num_conditions, len(filtered))
        return filtered.sample(take, seed=seed).with_columns(pl.lit(1).alias('stratum'))

    lo = int(filtered['n_admissions'].min())  # type: ignore[arg-type]
    hi = int(filtered['n_admissions'].max())  # type: ignore[arg-type]
    if scale == 'log':
        edges = np.logspace(np.log10(max(lo, 1)), np.log10(hi), n_strata + 1)
    else:
        edges = np.linspace(lo, hi, n_strata + 1)
    edges[-1] = edges[-1] + 1  # make final edge inclusive via half-open intervals

    per_stratum, remainder = divmod(num_conditions, n_strata)
    parts: list[pl.DataFrame] = []
    for i in range(n_strata):
        bucket = filtered.filter(
            (pl.col('n_admissions') >= edges[i]) & (pl.col('n_admissions') < edges[i + 1])
        )
        quota = per_stratum + (1 if i < remainder else 0)
        take = min(quota, len(bucket))
        print(
            f'  [stratum {i + 1}/{n_strata}] n_admissions [{int(edges[i])}, {int(edges[i + 1])}): '
            f'{len(bucket):,} available, taking {take}'
        )
        if take > 0:
            parts.append(
                bucket.sample(take, seed=seed + i).with_columns(pl.lit(i + 1).alias('stratum'))
            )

    return pl.concat(parts) if parts else filtered.head(0)


def _sample_grounding_chunks_per_modifier(
    chunks_by_hadm: dict[int, pl.DataFrame],
    meta_by_hadm: dict[int, AdmissionMetaSlimRow],
    icd10_3char: str,
    modifiers: list[QueryModifier],
    pool_builder: ChunkPoolBuilder,
    n: int,
) -> list[GroundingChunkSample]:
    """Sample up to n patients, round-robin across modifiers for diversity."""
    seen_hadm: set[int] = set()
    samples: list[GroundingChunkSample] = []

    for mod in modifiers * n:
        if len(seen_hadm) >= n:
            break

        mod_hadm_ids = pool_builder.get_hadm_ids_by_condition_modifier(icd10_3char, mod)
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
    hadm_id: int,
    chunks_by_hadm: dict[int, pl.DataFrame],
    meta_by_hadm: dict[int, AdmissionMetaSlimRow],
) -> list[GroundingChunkSample]:
    group = chunks_by_hadm[hadm_id]
    bhc_row = (
        group
        .filter(pl.col('section_name') == 'BRIEF HOSPITAL COURSE')
        .sort('approx_tokens', descending=True)
        .row(0, named=True)
    )

    section_priority = {s: i for i, s in enumerate(query_prompts_cfg.high_value_sections)}
    supp = group.filter(pl.col('section_name') != 'BRIEF HOSPITAL COURSE')
    supp = (
        supp
        .with_columns(
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


def _format_chunks_block(samples: list[GroundingChunkSample]) -> str:
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
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='queries')
    run_build_query_prompts(cfg=BuildQueryPromptsCfg(**raw['build_query_prompts']))
