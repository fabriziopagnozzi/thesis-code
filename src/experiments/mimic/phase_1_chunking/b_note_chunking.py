import re
from dataclasses import asdict, dataclass

import duckdb
import polars as pl
from tokenizers import Tokenizer
from tqdm import tqdm

from experiments.mimic.configs import MIMIC_RESULTS_DIR, NoteChunkingCfg
from experiments.mimic.duck_db_init import connect_mimic_duckdb
from helpers.chunks_classes import MimicIVChunk

chunking_cfg = NoteChunkingCfg.load()

BHC_SUBPROBLEM_RE = re.compile(r'(?:(?<=\s)|^)#\s*(?=[A-Za-z]{2})')
TRANSITIONAL_RE = re.compile(
    r'(?:(?<=\s)|^)\*{0,2}(?:TRANSITIONAL|TRANISTIONAL|TRANSITION)\s*ISSUES\b',
    re.IGNORECASE,
)

CHUNK_SCHEMA = {
    'chunk_id': pl.Utf8,
    'note_id': pl.Utf8,
    'subject_id': pl.Int64,
    'hadm_id': pl.Int64,
    'section_name': pl.Utf8,
    'subsection_name': pl.Utf8,
    'text': pl.Utf8,
    'char_count': pl.Int32,
    'approx_tokens': pl.Int32,
}


@dataclass
class ParsedNote:
    chunks: list[MimicIVChunk]
    chief_complaint: str | None = None


def run_note_chunking(
    con: duckdb.DuckDBPyConnection | None = None,
    cfg: NoteChunkingCfg | None = None,
) -> pl.DataFrame | None:
    global chunking_cfg, tokenizer
    if cfg is not None:
        chunking_cfg = cfg

    tokenizer = _load_tokenizer(chunking_cfg.embedding_model)

    if con is None:
        con = connect_mimic_duckdb()
    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = MIMIC_RESULTS_DIR / 'chunks.parquet'

    df = parse_all_notes(con)
    df.write_parquet(out_path)
    print(f'Saved to {out_path}')


def parse_all_notes(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    rows = con.execute("""--sql
        SELECT DISTINCT
            bhc.note_id,
            discharge.subject_id,
            discharge.hadm_id,
            bhc.input,
            bhc.target
        FROM bhc
        JOIN discharge ON bhc.note_id = discharge.note_id
        JOIN diagnoses_icd ON discharge.hadm_id = diagnoses_icd.hadm_id
        JOIN conditions_stats ON SUBSTR(diagnoses_icd.icd_code, 1, 3) = conditions_stats.icd10_3char
        WHERE diagnoses_icd.icd_version = 10
    """).fetchall()

    all_dicts: list[dict] = []
    chief_complaints: dict[str, str] = {}

    for note_id, subject_id, hadm_id, input_text, target_text in tqdm(rows, desc='Parsing notes'):
        result = parse_note(note_id, subject_id, hadm_id, input_text, target_text)
        all_dicts.extend(asdict(c) for c in result.chunks)
        if result.chief_complaint:
            chief_complaints[note_id] = result.chief_complaint

    df = pl.DataFrame(all_dicts, schema=CHUNK_SCHEMA)

    cc_df = pl.DataFrame(
        {
            'note_id': list(chief_complaints.keys()),
            'chief_complaint': list(chief_complaints.values()),
        }
    )
    df = df.join(cc_df, on='note_id', how='left')

    print(
        f'Parsed {len(rows):,} notes into {len(df):,} chunks\n'
        f'{df["section_name"].value_counts().sort("count", descending=True).head(15)}'
    )

    return df


def parse_note(
    note_id: str, subject_id: int, hadm_id: int, input_text: str, target_text: str
) -> ParsedNote:
    sections = _parse_tagged_sections(input_text)
    chunks: list[MimicIVChunk] = []
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

        for sub_text in _fixed_chunk(section_text):
            seq += 1
            chunks.append(
                MimicIVChunk(
                    text=sub_text,
                    doc_title=note_id,
                    chunk_id=f'{note_id}__{seq:03d}',
                    note_id=note_id,
                    subject_id=subject_id,
                    hadm_id=hadm_id,
                    section_name=section_name,
                    char_count=len(sub_text),
                    approx_tokens=_count_tokens(sub_text),
                )
            )

    # BHC from target column
    if target_text:
        for subsection_name, subtext in _split_bhc(target_text):
            for sub_text in _fixed_chunk(subtext):
                seq += 1
                chunks.append(
                    MimicIVChunk(
                        text=sub_text,
                        doc_title=note_id,
                        chunk_id=f'{note_id}__{seq:03d}',
                        note_id=note_id,
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        section_name='BRIEF HOSPITAL COURSE',
                        subsection_name=subsection_name,
                        char_count=len(sub_text),
                        approx_tokens=_count_tokens(sub_text),
                    )
                )

    return ParsedNote(chunks=chunks, chief_complaint=chief_complaint)


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


def _split_bhc(text: str) -> list[tuple[str | None, str]]:
    """Split BHC target text by inline # problem markers.
    Returns list of (subsection_name_or_None, text).
    """
    # Truncate at TRANSITIONAL ISSUES
    trans_match = TRANSITIONAL_RE.search(text)
    if trans_match:
        text = text[: trans_match.start()]

    splits = list(BHC_SUBPROBLEM_RE.finditer(text))
    if not splits:
        stripped = text.strip()
        return [(None, stripped)] if stripped else []

    chunks: list[tuple[str | None, str]] = []

    # Preamble before first # marker
    preamble = text[: splits[0].start()].strip()
    if preamble:
        chunks.append((None, preamble))

    for i, m in enumerate(splits):
        start = m.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        # Extract label: text up to first period or end of body
        dot_pos = body.find('.')
        label = body[:dot_pos].strip().rstrip(':') if dot_pos > 0 else None
        chunks.append((label, body))

    return chunks


def _load_tokenizer(model_name: str) -> Tokenizer:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    fast_tok: Tokenizer = model.tokenizer._tokenizer
    del model
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
    from experiments.mimic.configs import load_config_from_main

    raw = load_config_from_main(phase=1)
    run_note_chunking(
        cfg=NoteChunkingCfg(**raw['note_chunking']),
    )
