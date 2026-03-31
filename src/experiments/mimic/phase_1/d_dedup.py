"""Step 1.4: Deduplicate boilerplate chunks across same-patient admissions.

Sections like Past Medical History, Allergies, Family History, Social History are often copy-pasted across admissions for the same patient.
We keep only the most recent version per (subject_id, section_name, content_hash).
"""

import hashlib

import polars as pl

BOILERPLATE_SECTIONS = {
    'Past Medical History',
    'Allergies',
    'Family History',
    'Social History',
}


def _content_hash(text: str) -> str:
    normalized = ' '.join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def deduplicate(chunks: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate boilerplate chunks for the same patient.

    Deduplicates by (subject_id, section_name, content_hash), keeping only the chunk from the most recent admission (by hadm_id as proxy for time).
    """
    n_before = len(chunks)
    hashes = [_content_hash(t) for t in chunks['text'].to_list()]
    chunks = chunks.with_columns(pl.Series('content_hash', hashes))

    is_boilerplate = chunks['section_name'].is_in(list(BOILERPLATE_SECTIONS))
    bp = chunks.filter(is_boilerplate)
    non_bp = chunks.filter(~is_boilerplate)

    # keep only the row with the highest hadm_id per (subject_id, section, hash)
    bp_deduped = bp.sort('hadm_id', descending=True).unique(
        subset=['subject_id', 'section_name', 'content_hash'], keep='first'
    )

    result = pl.concat([non_bp, bp_deduped]).drop('content_hash')
    n_after = len(result)
    n_removed = n_before - n_after
    pct = n_removed / n_before * 100 if n_before > 0 else 0
    print(f'Deduplication: {n_before:,} -> {n_after:,} chunks ({n_removed:,} removed, {pct:.1f}%)')

    return result


if __name__ == '__main__':
    from experiments.mimic.duck_db_init import MIMIC_RESULTS_DIR

    chunks = pl.read_parquet(MIMIC_RESULTS_DIR / 'chunks_raw.parquet')
    metadata = pl.read_parquet(MIMIC_RESULTS_DIR / 'admissions_metadata.parquet')
    result = deduplicate(chunks, metadata)
    result.write_parquet(MIMIC_RESULTS_DIR / 'chunks.parquet')
    print(f'Saved to {MIMIC_RESULTS_DIR / "chunks.parquet"}')
