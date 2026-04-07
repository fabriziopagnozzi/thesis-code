import re
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import polars as pl
from tqdm import tqdm

from experiments.mimic.config_loader import load_config
from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR, connect_mimic_duckdb
from helpers.chunks_classes import MimicIVChunk

_cfg = load_config(1)['note_chunking']
KEEP_SECTIONS = set(_cfg['keep_sections'])
SKIP_SECTIONS = set(_cfg['skip_sections'])
METADATA_ONLY_SECTIONS = set(_cfg['metadata_only_sections'])

ALL_SECTIONS = KEEP_SECTIONS | SKIP_SECTIONS | METADATA_ONLY_SECTIONS

TAG_RE = re.compile(rf'<({"|".join(re.escape(s) for s in ALL_SECTIONS)})>', re.IGNORECASE)

# -- BHC problem markers in the target column --
BHC_PROBLEM_RE = re.compile(r'^#\s*(.+)', re.MULTILINE)
BHC_SUBHEADER_RE = re.compile(
    r'^(SUMMARY|ACTIVE ISSUES|CHRONIC ISSUES|CHRONIC/RESOLVED ISSUES|RESOLVED ISSUES)\s*:?\s*=*\s*$',
    re.MULTILINE | re.IGNORECASE,
)
TRANSITIONAL_RE = re.compile(
    r'^[\s=]*\*{0,2}(?:TRANSITIONAL|TRANISTIONAL|TRANSITION)\s*ISSUES\s*:?\s*\*{0,2}\s*=*\s*$',
    re.MULTILINE | re.IGNORECASE,
)

CHUNK_SCHEMA = {
    'chunk_id': pl.Utf8,
    'note_id': pl.Utf8,
    'subject_id': pl.Int64,
    'hadm_id': pl.Int64,
    'section_name': pl.Utf8,
    'subsection_name': pl.Utf8,
    'bhc_category': pl.Utf8,
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
    cfg: dict | None = None,
    interactive: bool = False,
) -> pl.DataFrame | None:
    global _cfg, KEEP_SECTIONS, SKIP_SECTIONS, METADATA_ONLY_SECTIONS, ALL_SECTIONS, TAG_RE
    if cfg is not None:
        _cfg = cfg
        KEEP_SECTIONS = set(_cfg['keep_sections'])
        SKIP_SECTIONS = set(_cfg['skip_sections'])
        METADATA_ONLY_SECTIONS = set(_cfg['metadata_only_sections'])
        ALL_SECTIONS = KEEP_SECTIONS | SKIP_SECTIONS | METADATA_ONLY_SECTIONS
        TAG_RE = re.compile(rf'<({"|".join(re.escape(s) for s in ALL_SECTIONS)})>', re.IGNORECASE)

    if con is None:
        con = connect_mimic_duckdb()
    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if interactive:
        while True:
            try:
                raw = input('\nLimit (or q to quit): ').strip()
            except EOFError, KeyboardInterrupt:
                break
            if raw.lower() == 'q':
                break
            if not raw.isdigit():
                print('Enter a positive integer.')
                continue
            parse_all_notes(con, limit=int(raw))
        return None
    else:
        return parse_all_notes(con, output_path=MIMIC_RESULTS_DIR / 'chunks.parquet')


def parse_all_notes(
    con: duckdb.DuckDBPyConnection,
    output_path: Path | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    limit_clause = f'LIMIT {limit}' if limit is not None else ''

    rows = con.execute(f"""--sql
        SELECT DISTINCT
            bhc.note_id,
            discharge.subject_id,
            discharge.hadm_id,
            bhc.input,
            bhc.target
        FROM bhc
        JOIN discharge ON bhc.note_id = discharge.note_id
        JOIN diagnoses_icd ON discharge.hadm_id = diagnoses_icd.hadm_id
        JOIN conditions ON SUBSTR(diagnoses_icd.icd_code, 1, 3) = conditions.icd10_3char
        WHERE diagnoses_icd.icd_version = 10
        {limit_clause}
    """).fetchall()

    all_dicts: list[dict] = []
    chief_complaints: dict[str, str] = {}

    for note_id, subject_id, hadm_id, input_text, target_text in tqdm(rows, desc='Parsing notes'):
        result = parse_note(note_id, subject_id, hadm_id, input_text, target_text)
        all_dicts.extend(asdict(c) for c in result.chunks)
        if result.chief_complaint:
            chief_complaints[note_id] = result.chief_complaint

    df = pl.DataFrame(all_dicts, schema=CHUNK_SCHEMA)

    print(f'  chief_complaint found in {len(chief_complaints)}/{len(rows)} notes')

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

    if output_path:
        df.write_parquet(output_path)
        print(f'Saved to {output_path}')

    return df


def parse_note(
    note_id: str, subject_id: int, hadm_id: int, input_text: str, target_text: str
) -> ParsedNote:
    sections = _parse_tagged_sections(input_text)
    chunks: list[MimicIVChunk] = []
    chief_complaint: str | None = None
    seq = 0

    for section_name, section_text in sections:
        if section_name in SKIP_SECTIONS:
            continue
        if section_name in METADATA_ONLY_SECTIONS:
            chief_complaint = section_text
            continue
        if section_name not in KEEP_SECTIONS:
            continue

        seq += 1
        chunks.append(
            MimicIVChunk(
                text=section_text,
                doc_title=note_id,
                chunk_id=f'{note_id}__{seq:03d}',
                note_id=note_id,
                subject_id=subject_id,
                hadm_id=hadm_id,
                section_name=section_name,
                char_count=len(section_text),
                approx_tokens=len(section_text) // 4,
            )
        )

    # BHC from target column, split by # problems
    if target_text:
        bhc_chunks = _split_bhc_problems(target_text)
        for subsection_name, subtext, category in bhc_chunks:
            seq += 1
            chunks.append(
                MimicIVChunk(
                    text=subtext,
                    doc_title=note_id,
                    chunk_id=f'{note_id}__{seq:03d}',
                    note_id=note_id,
                    subject_id=subject_id,
                    hadm_id=hadm_id,
                    section_name='BRIEF HOSPITAL COURSE',
                    subsection_name=subsection_name,
                    bhc_category=category,
                    char_count=len(subtext),
                    approx_tokens=len(subtext) // 4,
                )
            )

    return ParsedNote(chunks=chunks, chief_complaint=chief_complaint)


def _parse_tagged_sections(text: str) -> list[tuple[str, str]]:
    """Split <TAG>-delimited input into (section_name, section_text) pairs."""
    matches = list(TAG_RE.finditer(text))
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


def _split_bhc_problems(text: str) -> list[tuple[str | None, str, str | None]]:
    """Split BHC target text by # problem markers.
    Returns list of (subsection_name_or_None, text, category_or_None).
    Category is the most recent subheader (ACTIVE ISSUES, CHRONIC ISSUES, etc).
    """
    # Truncate at TRANSITIONAL ISSUES (follow-up instructions, admin items)
    trans_match = TRANSITIONAL_RE.search(text)
    if trans_match:
        text = text[: trans_match.start()]

    # Clean up ====== decoration lines but keep subheader text
    text = re.sub(r'^=+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Build a list of events: subheader positions and # problem positions
    subheader_matches = list(BHC_SUBHEADER_RE.finditer(text))
    problem_matches = list(BHC_PROBLEM_RE.finditer(text))

    if not problem_matches:
        return [(None, text.strip(), None)]

    # Map each position to the most recent subheader category
    def _category_at(pos: int) -> str | None:
        cat = None
        for sh in subheader_matches:
            if sh.start() <= pos:
                cat = sh.group(1).strip().rstrip(':').title()
            else:
                break
        return cat

    clean_text = BHC_SUBHEADER_RE.sub('', text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    clean_matches = list(BHC_PROBLEM_RE.finditer(clean_text))
    if not clean_matches:
        return [(None, clean_text.strip(), None)]

    # Map each clean match back to a category using original positions
    categories: list[str | None] = []
    for cm in clean_matches:
        label = cm.group(1).strip()
        cat = None
        for om in problem_matches:
            if om.group(1).strip() == label:
                cat = _category_at(om.start())
                break
        categories.append(cat)

    chunks: list[tuple[str | None, str, str | None]] = []

    preamble = clean_text[: clean_matches[0].start()].strip()
    if preamble:
        chunks.append((None, preamble, _category_at(0)))

    for i, m in enumerate(clean_matches):
        raw_label = m.group(1).strip()
        label = re.split(r'[.\n]', raw_label)[0].strip().rstrip(':')
        start = m.start()
        end = clean_matches[i + 1].start() if i + 1 < len(clean_matches) else len(clean_text)
        body = clean_text[start:end].strip()
        body = re.sub(r'^#\s*', '', body, count=1)
        if body:
            chunks.append((label, body, categories[i]))

    return chunks


if __name__ == '__main__':
    import argparse

    from experiments.mimic.config_loader import load_config_from_main

    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    args, _ = parser.parse_known_args()

    run_note_chunking(
        cfg=load_config_from_main(phase=1)['note_chunking'],
        interactive=args.interactive,
    )
