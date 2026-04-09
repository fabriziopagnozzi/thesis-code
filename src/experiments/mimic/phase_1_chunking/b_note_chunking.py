import re
from dataclasses import asdict, dataclass

import duckdb
import polars as pl
from tokenizers import Tokenizer
from tqdm import tqdm

from experiments.mimic.configs import MIMIC_RESULTS_DIR, NoteChunkingCfg, global_cfg
from experiments.mimic.duck_db_init import connect_mimic_duckdb

chunking_cfg = NoteChunkingCfg.load()


@dataclass
class MimicIVChunk:
    text: str
    chunk_id: str
    note_id: str
    subject_id: int
    hadm_id: int
    section_name: str
    chief_complaint: str | None = None
    char_count: int = 0
    approx_tokens: int = 0


def run_note_chunking(
    con: duckdb.DuckDBPyConnection | None = None,
    cfg: NoteChunkingCfg | None = None,
) -> pl.DataFrame:
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
    return df


def parse_all_notes(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    top_icd3 = (
        pl.read_parquet(MIMIC_RESULTS_DIR / 'conditions_stats.parquet')
        .head(global_cfg.num_conditions)['icd10_3char']
        .to_list()
    )
    placeholders = ','.join(f"'{c}'" for c in top_icd3)

    rows = con.execute(f"""--sql
        SELECT DISTINCT
            bhc.note_id,
            discharge.subject_id,
            discharge.hadm_id,
            bhc.input,
            discharge.text
        FROM bhc
        JOIN discharge ON bhc.note_id = discharge.note_id
        JOIN diagnoses_icd ON discharge.hadm_id = diagnoses_icd.hadm_id
        WHERE diagnoses_icd.icd_version = 10
        AND SUBSTR(diagnoses_icd.icd_code, 1, 3) IN ({placeholders})
    """).fetchall()

    all_dicts: list[dict] = []

    for note_id, subject_id, hadm_id, input_text, discharge_text in tqdm(
        rows, desc='Parsing notes'
    ):
        bhc_text = _extract_bhc_from_discharge(discharge_text)
        all_dicts.extend(
            asdict(c) for c in parse_note(note_id, subject_id, hadm_id, input_text, bhc_text)
        )

    df = pl.DataFrame(
        all_dicts,
        schema={
            'chunk_id': pl.Utf8,
            'note_id': pl.Utf8,
            'subject_id': pl.Int64,
            'hadm_id': pl.Int64,
            'section_name': pl.Utf8,
            'chief_complaint': pl.Utf8,
            'text': pl.Utf8,
            'char_count': pl.Int32,
            'approx_tokens': pl.Int32,
        },
    )

    print(
        f'Parsed {len(rows):,} notes into {len(df):,} chunks\n'
        f'{df["section_name"].value_counts().sort("count", descending=True).head(15)}'
    )

    return df


def parse_note(
    note_id: str, subject_id: int, hadm_id: int, input_text: str, bhc_text: str | None
) -> list[MimicIVChunk]:
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
                    chunk_id=f'{note_id}__{seq:03d}',
                    note_id=note_id,
                    subject_id=subject_id,
                    hadm_id=hadm_id,
                    section_name=section_name,
                    chief_complaint=chief_complaint,
                    char_count=len(sub_text),
                    approx_tokens=_count_tokens(sub_text),
                )
            )

    # BHC from discharge.csv (newlines preserved, split on # markers)
    if bhc_text:
        for subtext in _split_bhc(bhc_text):
            for sub_text in _fixed_chunk(subtext):
                seq += 1
                chunks.append(
                    MimicIVChunk(
                        text=sub_text,
                        chunk_id=f'{note_id}__{seq:03d}',
                        note_id=note_id,
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        chief_complaint=chief_complaint,
                        section_name='BRIEF HOSPITAL COURSE',
                        char_count=len(sub_text),
                        approx_tokens=_count_tokens(sub_text),
                    )
                )

    return chunks


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


_TRANSITIONAL_RE = re.compile(
    r'(?:(?<=\s)|^)\*{0,2}(?:TRANSITIONAL|TRANISTIONAL|TRANSITION)\s*ISSUES\b',
    re.IGNORECASE,
)

_BHC_PROBLEM_RE = re.compile(r'\n\s*#\s*(?=[A-Za-z])')


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
    """Truncate at TRANSITIONAL ISSUES, then split on # problem markers."""
    trans_match = _TRANSITIONAL_RE.search(text)
    if trans_match:
        text = text[: trans_match.start()]

    parts = _BHC_PROBLEM_RE.split(text)
    result = [p.strip() for p in parts if p.strip()]
    return result


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
