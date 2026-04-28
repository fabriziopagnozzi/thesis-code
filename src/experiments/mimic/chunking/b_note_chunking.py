import os
import re
from dataclasses import dataclass
from multiprocessing import Pool

import duckdb
import polars as pl
from tokenizers import Tokenizer
from tqdm import tqdm

from experiments.mimic.configs import (
    EvaluateCfg,
    NoteChunkingCfg,
    get_table_path,
    global_cfg,
    read_parquet,
    setup_logging,
)
from experiments.mimic.utils.duck_db_init import connect_mimic_duckdb

chunking_cfg = NoteChunkingCfg.load()
evaluate_cfg = EvaluateCfg.load()


@dataclass
class MimicIVNoteChunk:
    text: str
    chunk_id: str
    note_id: str
    subject_id: int
    hadm_id: int
    section_name: str
    chief_complaint: str | None = None
    char_count: int = 0
    approx_tokens: int = 0
    is_split: bool = False


def run_note_chunking(
    con: duckdb.DuckDBPyConnection | None = None,
    cfg: NoteChunkingCfg | None = None,
) -> pl.DataFrame:
    global chunking_cfg, tokenizer
    if cfg is not None:
        chunking_cfg = cfg

    tokenizer = _load_tokenizer(evaluate_cfg.embedding_model)

    if con is None:
        con = connect_mimic_duckdb()
    out_path = get_table_path('chunks')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_all_notes(con)
    df.write_parquet(out_path)
    print(f'Saved to {out_path}')
    return df


def parse_all_notes(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    con.execute('SET arrow_large_buffer_size=true')

    df_top_codes = (  # noqa: F841
        read_parquet('conditions_stats')
        .head(global_cfg.num_conditions)
        .select(pl.col('icd10_3char').alias('code'))
    )

    # Pull into Polars (arrow) then convert to tuples for multiprocessing
    notes_df = con.execute("""--sql
        WITH target_admissions AS (
            SELECT DISTINCT ud.hadm_id
            FROM unified_diagnoses AS ud
            JOIN df_top_codes
              ON LEFT(ud.unified_icd10, 3) = df_top_codes.code
        )
        SELECT
            bhc.note_id,
            discharge.subject_id,
            discharge.hadm_id,
            bhc.input,
            discharge.text
        FROM target_admissions
        JOIN discharge ON target_admissions.hadm_id = discharge.hadm_id
        JOIN bhc ON discharge.note_id = bhc.note_id
    """).pl()

    notes_rows = notes_df.rows()

    with Pool(
        processes=min(os.cpu_count() or 1, len(notes_rows)),
        initializer=_worker_init,
        initargs=(evaluate_cfg.embedding_model, chunking_cfg),
    ) as pool:
        per_note = list(
            tqdm(
                pool.imap(_process_row, notes_rows, chunksize=32),
                total=len(notes_rows),
                desc='Parsing notes',
                dynamic_ncols=True,
            )
        )

    all_chunks = [chunk for note_chunks in per_note for chunk in note_chunks]
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = str(i)

    df = pl.DataFrame(
        all_chunks,
        schema_overrides={'char_count': pl.Int32, 'approx_tokens': pl.Int32},
    )

    print(
        f'Parsed {len(notes_rows):,} notes into {len(df):,} chunks\n'
        f'{df["section_name"].value_counts().sort("count", descending=True).head(15)}'
    )

    return df


def parse_note(
    note_id: str,
    subject_id: int,
    hadm_id: int,
    input_text: str,
    bhc_text: str | None,
) -> list[MimicIVNoteChunk]:
    sections = _parse_tagged_sections(input_text)
    chunks: list[MimicIVNoteChunk] = []
    chief_complaint: str | None = None
    seq = 0

    for section_name, section_text in sections:
        if section_name in chunking_cfg.skip_sections:
            continue
        if section_name in chunking_cfg.metadata_only_sections:
            chief_complaint = section_text
            continue
        if section_name not in chunking_cfg.keep_sections:
            continue

        sub_texts = _fixed_chunk(section_text)
        is_split = len(sub_texts) > 1
        for sub_text in sub_texts:
            seq += 1
            chunks.append(
                MimicIVNoteChunk(
                    text=sub_text,
                    chunk_id=f'{note_id}__{seq:03d}',
                    note_id=note_id,
                    subject_id=subject_id,
                    hadm_id=hadm_id,
                    section_name=section_name,
                    chief_complaint=chief_complaint,
                    char_count=len(sub_text),
                    approx_tokens=_count_tokens(sub_text),
                    is_split=is_split,
                )
            )

    # BHC from discharge.csv (newlines preserved, split on # markers)
    if bhc_text:
        for subtext in _split_bhc(bhc_text):
            sub_texts = _fixed_chunk(subtext)
            is_split = len(sub_texts) > 1
            for sub_text in sub_texts:
                seq += 1
                chunks.append(
                    MimicIVNoteChunk(
                        text=sub_text,
                        chunk_id=f'{note_id}__{seq:03d}',
                        note_id=note_id,
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        chief_complaint=chief_complaint,
                        section_name='BRIEF HOSPITAL COURSE',
                        char_count=len(sub_text),
                        approx_tokens=_count_tokens(sub_text),
                        is_split=is_split,
                    )
                )

    # Drop chunks that are still too short after merging, but keep all split tails
    chunks = [c for c in chunks if c.approx_tokens >= chunking_cfg.min_chunk_tokens or c.is_split]

    return chunks


def _worker_init(model_name: str, cfg: NoteChunkingCfg) -> None:
    global tokenizer, chunking_cfg
    chunking_cfg = cfg
    tokenizer = _load_tokenizer(model_name)


def _process_row(row: tuple) -> list[MimicIVNoteChunk]:
    note_id, subject_id, hadm_id, input_text, discharge_text = row
    bhc_text = _extract_bhc_from_discharge(discharge_text)
    return parse_note(note_id, subject_id, hadm_id, input_text, bhc_text)


def _parse_tagged_sections(text: str) -> list[tuple[str, str]]:
    """Split <TAG>-delimited input into (section_name, section_text) pairs."""
    matches = list(chunking_cfg.tag_re.finditer(text))
    if not matches:
        return [('_full_note', text.strip())]

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body))

    return sections


def _extract_bhc_from_discharge(discharge_text: str) -> str | None:
    """Extract the Brief Hospital Course section from the full discharge note (newlines preserved)."""
    start = re.search(r'Brief Hospital Course:', discharge_text, re.IGNORECASE)
    if not start:
        return None

    bhc_body = discharge_text[start.end() :]

    # Find the next major section header (all-caps line followed by colon)
    next_section = re.search(r'\n\s*[A-Z][A-Za-z ]+:\s*\n', bhc_body)
    if next_section:
        bhc_body = bhc_body[: next_section.start()]

    return bhc_body.strip() or None


def _split_bhc(text: str) -> list[str]:
    """Truncate at TRANSITIONAL ISSUES, split on # problem markers,
    then merge header-only fragments forward into the next sub-problem."""

    TRANSITIONAL_RE = re.compile(
        r'(?:(?<=\s)|^)\*{0,2}(?:TRANSITIONAL|TRANISTIONAL|TRANSITION)\s*ISSUES\b',
        re.IGNORECASE,
    )
    BHC_PROBLEM_RE = re.compile(r'\n\s*#\s*(?=[A-Za-z])')

    trans_match = TRANSITIONAL_RE.search(text)
    if trans_match:
        text = text[: trans_match.start()]

    parts = BHC_PROBLEM_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    # Merge short fragments (bare problem headers) into the next sub-problem
    merged: list[str] = []
    buf = ''
    for p in parts:
        if buf:
            p = buf + '\n' + p
            buf = ''
        if _count_tokens(p) < chunking_cfg.min_chunk_tokens:
            buf = p
        else:
            merged.append(p)
    if buf:
        if merged:
            merged[-1] = merged[-1] + '\n' + buf
        else:
            merged.append(buf)

    return merged


def _load_tokenizer(model_name: str) -> Tokenizer:
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    fast_tok = hf_tok._tokenizer
    fast_tok.no_padding()
    fast_tok.no_truncation()
    return fast_tok


def _count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def _fixed_chunk(text: str) -> list[str]:
    """Split text into overlapping token windows using the embedding tokenizer."""
    assert tokenizer is not None
    max_tok = chunking_cfg.max_tokens
    stride_tok = chunking_cfg.stride_tokens

    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) <= max_tok:
        return [text]

    step = max_tok - stride_tok
    windows: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_tok, len(ids))
        window = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()
        if window:
            windows.append(window)
        if end >= len(ids):
            break
        start += step
    return windows if windows else [text]


if __name__ == '__main__':
    setup_logging()
    run_note_chunking(cfg=NoteChunkingCfg.load())
