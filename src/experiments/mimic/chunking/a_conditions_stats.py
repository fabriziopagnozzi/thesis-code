import json

import duckdb
import polars as pl
from tqdm import tqdm

from experiments.mimic.configs import (
    ConditionsStatsCfg,
    get_table_path,
    setup_logging,
)
from experiments.mimic.utils.charlson import CHARLSON_LABELS_TO_STR, ICD3_TO_CHARLSON_COLS
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb
from experiments.mimic.utils.prompts_default import MimicDefaultPrompts
from helpers.ollama_client import generate_json

conditions_stats_cfg = ConditionsStatsCfg.load()


def run_conditions_stats(
    con: duckdb.DuckDBPyConnection | None = None, cfg: ConditionsStatsCfg | None = None
) -> pl.DataFrame:
    global conditions_stats_cfg
    if cfg is not None:
        conditions_stats_cfg = cfg
    if con is None:
        con = connect_mimic_duckdb()

    df = select_conditions(con=con, min_admissions=conditions_stats_cfg.min_admissions)
    out_path = get_table_path('conditions_stats')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f'\nSaved {len(df)} conditions to {out_path}')

    return df


def select_conditions(
    con: duckdb.DuckDBPyConnection,
    min_admissions: int,
) -> pl.DataFrame:
    """
    Returns a DataFrame with columns:
        icd10_3char, condition_name, n_admissions, mean_comorbidity_count, top_comorbidity_mods_json
    Sorted by n_admissions descending, filtered by min_admissions.
    """
    all_cols = list(CHARLSON_LABELS_TO_STR.keys())

    prefix_rows = con.execute(f"""--sql
        WITH condition_stats AS (
            SELECT
                LEFT(ud.unified_icd10, 3) AS icd10_3char,
                COUNT(DISTINCT ud.hadm_id) AS n_admissions
            FROM unified_diagnoses ud
            GROUP BY LEFT(ud.unified_icd10, 3)
            HAVING COUNT(DISTINCT ud.hadm_id) >= {min_admissions}
        )
        SELECT
            condition_stats.icd10_3char,
            icd_labels.long_title AS condition_name,
            condition_stats.n_admissions
        FROM condition_stats
        LEFT JOIN (
            SELECT LEFT(icd_code, 3) AS icd10_3char, STRING_AGG(DISTINCT long_title, ' | ') AS long_title
            FROM mimiciv_hosp.d_icd_diagnoses
            WHERE icd_version = 10
            GROUP BY LEFT(icd_code, 3)
        ) AS icd_labels ON condition_stats.icd10_3char = icd_labels.icd10_3char
        ORDER BY n_admissions DESC
    """).fetchall()

    code_to_info = coalesce_condition_names(prefix_rows)
    kept_rows = [
        (icd3, raw, n) for icd3, raw, n in prefix_rows if code_to_info.get(icd3, (None, True))[1]
    ]
    n_filtered = len(prefix_rows) - len(kept_rows)
    if n_filtered:
        print(f'Filtered out {n_filtered} administrative/generic codes by LLM')

    charlson_rates = con.execute(f"""--sql
        SELECT dedup.icd10_3char, {', '.join(f'AVG(c.{col}) AS {col}' for col in all_cols)}
        FROM (
            SELECT DISTINCT LEFT(unified_icd10, 3) AS icd10_3char, hadm_id
            FROM unified_diagnoses
            WHERE LEFT(unified_icd10, 3) IN ({', '.join(f"'{icd3}'" for icd3, _, _ in kept_rows)})
        ) dedup JOIN mimiciv_derived.charlson c ON dedup.hadm_id = c.hadm_id
        GROUP BY dedup.icd10_3char
    """).fetchall()

    rates_by_icd3: dict[str, tuple] = {row[0]: row[1:] for row in charlson_rates}

    rows = []
    for icd3, cond_name, n in kept_rows:
        rates = rates_by_icd3.get(icd3)
        if not rates:
            scored = []
            mean_comorbidity_count = 0.0
        else:
            mean_comorbidity_count = round(sum(float(r) for r in rates if r is not None), 2)
            excluded_cols = ICD3_TO_CHARLSON_COLS.get(icd3, frozenset())
            scored = [
                {'col': col, 'label': CHARLSON_LABELS_TO_STR[col], 'rate': round(float(r), 4)}
                for col, r in zip(all_cols, rates, strict=True)
                if r and r > 0.05 and col not in excluded_cols
            ]
            scored.sort(key=lambda x: x['rate'], reverse=True)

        rows.append(
            {
                'icd10_3char': icd3,
                'condition_name': code_to_info.get(icd3, (cond_name, True))[0],
                'n_admissions': n,
                'mean_comorbidity_count': mean_comorbidity_count,
                'top_comorbidity_mods_json': json.dumps(scored),
            }
        )

    df = pl.DataFrame(rows).sort('n_admissions', descending=True)
    print(
        f'ICD-3 conditions with >= {min_admissions} admissions: {len(df)}\n{df.select("icd10_3char", "n_admissions", "mean_comorbidity_count")}'
    )
    return df


def coalesce_condition_names(
    prefix_rows: list[tuple],
    batch_size: int = 3,
) -> dict[str, tuple[str, bool]]:
    """Call local LLM in batches; returns {code: (condition_name, keep)} per ICD-10 3-char code.
    Persists results incrementally to cond_filtering_jsonl_out_path and resumes from it on restart.
    """
    code_to_info: dict[str, tuple[str, bool]] = {}

    cond_filtering_jsonl_out_path = get_table_path('condition_filtering', ext='jsonl')
    if cond_filtering_jsonl_out_path.exists():
        with cond_filtering_jsonl_out_path.open() as f:
            for line in f:
                item = json.loads(line)
                code_to_info[item['code']] = (item['condition_name'], bool(item['keep']))
        print(
            f'Resumed: {len(code_to_info)} codes already processed from {cond_filtering_jsonl_out_path}'
        )

    batches = [prefix_rows[i : i + batch_size] for i in range(0, len(prefix_rows), batch_size)]

    with cond_filtering_jsonl_out_path.open('a') as jsonl_f:
        for batch in tqdm(batches, desc='Coalescing condition names', dynamic_ncols=True):
            payload = [
                {'code': icd3, 'titles': raw_titles.split(' | ')[:75]}
                for icd3, raw_titles, _ in batch
                if raw_titles and icd3 not in code_to_info
            ]
            if not payload:
                continue

            prompt = (
                'For each entry below, produce a concise clinical label and decide whether to keep it '
                'for a clinical QA benchmark (see system instructions).\n\n'
                f'Input:\n{json.dumps(payload, indent=2)}\n\n'
                'Output a JSON array: [{"code": "...", "condition_name": "...", "keep": true/false}, ...]'
            )
            print(prompt)
            result = generate_json(
                prompt,
                model=conditions_stats_cfg.cond_processing_llm,
                num_ctx=32_000,
                num_predict=-1,
                system=MimicDefaultPrompts.llm_conditions_cleaning_system,
                think=False,
                stream=False,
            )
            print(result)
            for item in result:
                code_to_info[item['code']] = (item['condition_name'], bool(item['keep']))
                jsonl_f.write(json.dumps(item) + '\n')
            jsonl_f.flush()

    return code_to_info


def coalesce_condition_names_mock(
    prefix_rows: list[tuple],
    batch_size: int = 10,
) -> dict[str, tuple[str, bool]]:
    """Mock for testing: skip LLM, keep all codes."""
    return {icd3: (raw_titles if raw_titles else icd3, True) for icd3, raw_titles, _ in prefix_rows}


if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(key='chunking')
    run_conditions_stats(cfg=ConditionsStatsCfg(**raw['conditions_stats']))
