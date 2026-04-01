import re
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import polars as pl
from tqdm import tqdm

from helpers.chunks_classes import MimicIVChunk

KNOWN_SECTIONS = {
    'Service',
    'Allergies',
    'Chief Complaint',
    'Major Surgical or Invasive Procedure',
    'History of Present Illness',
    'Past Medical History',
    'Social History',
    'Family History',
    'Physical Exam',
    'Pertinent Results',
    'Brief Hospital Course',
    'Medications on Admission',
    'Discharge Medications',
    'Discharge Diagnosis',
    'Discharge Disposition',
    'Discharge Condition',
    'Discharge Instructions',
    'Followup Instructions',
    'Facility',
    'Attending',
}
# Sections to discard: None-value (deidentified/content-free) + Low-value
SKIP_SECTIONS = {
    # None
    'Service',
    'Attending',
    'Social History',
    'Followup Instructions',
    'Facility',
    # Low
    'Major Surgical or Invasive Procedure',
    'Allergies',
    'Family History',
    'Medications on Admission',
    'Discharge Disposition',
    'Discharge Condition',
    'Discharge Instructions',
}
METADATA_ONLY_SECTIONS = {'Chief Complaint'}

CHUNK_SCHEMA = {
    'chunk_id': pl.Utf8,
    'note_id': pl.Utf8,
    'subject_id': pl.Int64,
    'hadm_id': pl.Int64,
    'section_name': pl.Utf8,
    'subsection_name': pl.Utf8,
    'chief_complaint': pl.Utf8,
    'text': pl.Utf8,
    'char_count': pl.Int32,
    'approx_tokens': pl.Int32,
}


# regex pattern
_header_pattern = '|'.join(re.escape(h) for h in KNOWN_SECTIONS)
SECTION_RE = re.compile(rf'^({_header_pattern})\s*:', re.MULTILINE | re.IGNORECASE)

# Within Brief Hospital Course: # problem markers
# TODO: understand if this split is good
BHC_PROBLEM_RE = re.compile(r'^#\s*(.+)', re.MULTILINE)
BHC_SUBHEADER_RE = re.compile(
    r'^(ACTIVE ISSUES|CHRONIC ISSUES|TRANSITIONAL ISSUES|TRANISTIONAL ISSUES)\s*:?',
    re.MULTILINE | re.IGNORECASE,
)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a discharge note into (section_name, section_text) pairs."""
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [('_full_note', text.strip())]

    sections: list[tuple[str, str]] = []

    # Text before the first recognized header (demographics block)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(('_preamble', preamble))

    for i, m in enumerate(matches):
        name = m.group(1)
        # Normalize casing to title case
        name = next((s for s in KNOWN_SECTIONS if s.lower() == name.lower()), name)
        start = m.end()  # after the colon
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body))

    return sections


def _split_bhc_problems(text: str) -> list[tuple[str | None, str]]:
    """Returns list of (subsection_name_or_None, text).
    If no # markers found, returns a single chunk with subsection_name=None.
    """
    matches = list(BHC_PROBLEM_RE.finditer(text))
    if not matches:
        return [(None, text.strip())]

    chunks: list[tuple[str | None, str]] = []

    # Text before first # marker
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append((None, preamble))

    for i, m in enumerate(matches):
        raw_label = m.group(1).strip()
        # The # line may contain text beyond the label (e.g., "# ASCITES. Now diuretic...")
        # Truncate at first period to get just the problem name
        label = re.split(r'[.\n]', raw_label)[0].strip().rstrip(':')
        # Everything after the # line (including the rest of the label line) is body text
        start = m.start()  # include the full # line in body for context
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        body = re.sub(r'^#\s*', '', body, count=1)
        if body:
            chunks.append((label, body))

    return chunks


@dataclass
class ParsedNote:
    chunks: list[MimicIVChunk]
    chief_complaint: str | None = None


def parse_note(note_id: str, subject_id: int, hadm_id: int, text: str) -> ParsedNote:
    sections = _split_sections(text)
    chunks: list[MimicIVChunk] = []
    chief_complaint: str | None = None
    seq = 0

    for section_name, section_text in sections:
        if section_name in SKIP_SECTIONS:
            continue
        if section_name in METADATA_ONLY_SECTIONS:
            chief_complaint = section_text
            continue

        if section_name == 'Brief Hospital Course':
            subchunks = _split_bhc_problems(section_text)
            for subsection_name, subtext in subchunks:
                seq += 1
                chunks.append(
                    MimicIVChunk(
                        text=subtext,
                        doc_title=note_id,
                        chunk_id=f'{note_id}__{seq:03d}',
                        note_id=note_id,
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        section_name=section_name,
                        subsection_name=subsection_name,
                        char_count=len(subtext),
                        approx_tokens=len(subtext) // 4,
                    )
                )
        else:
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

    return ParsedNote(chunks=chunks, chief_complaint=chief_complaint)


def parse_all_notes(
    con: duckdb.DuckDBPyConnection, output_path: Path | None = None
) -> pl.DataFrame:
    rows = con.execute("""--sql
        SELECT DISTINCT discharge.note_id, discharge.subject_id, discharge.hadm_id, discharge.text
        FROM discharge
        JOIN diagnoses_icd ON discharge.hadm_id = diagnoses_icd.hadm_id
        JOIN conditions ON SUBSTR(diagnoses_icd.icd_code, 1, 3) = conditions.icd10_3char
        WHERE diagnoses_icd.icd_version = 10
    """).fetchall()

    all_dicts: list[dict] = []
    chief_complaints: dict[str, str] = {}

    for note_id, subject_id, hadm_id, text in tqdm(rows, desc='Parsing notes'):
        result = parse_note(note_id, subject_id, hadm_id, text)
        all_dicts.extend(asdict(c) for c in result.chunks)
        if result.chief_complaint:
            chief_complaints[note_id] = result.chief_complaint

    df = pl.DataFrame(all_dicts, schema=CHUNK_SCHEMA)

    # Join chief_complaint onto each chunk via note_id
    cc_df = pl.DataFrame(
        {
            'note_id': list(chief_complaints.keys()),
            'chief_complaint': list(chief_complaints.values()),
        }
    )
    df = df.join(cc_df, on='note_id', how='left')

    print(f'Parsed {len(rows):,} notes into {len(df):,} chunks')
    print(df['section_name'].value_counts().sort('count', descending=True).head(15))

    if output_path:
        df.write_parquet(output_path)
        print(f'Saved to {output_path}')

    return df


if __name__ == '__main__':
    from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR, connect_mimic_duckdb

    con = connect_mimic_duckdb()
    MIMIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    parse_all_notes(con, output_path=MIMIC_RESULTS_DIR / 'chunks_raw.parquet')
