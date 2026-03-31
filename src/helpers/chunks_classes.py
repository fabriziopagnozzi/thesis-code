from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    doc_title: str       # HotpotQA: Wikipedia title; MIMIC: note_id
    doc_idx: int = -1    # position in the candidate pool; -1 until assigned at experiment load time


@dataclass
class HotpotChunk(Chunk):
    """Chunk with HotpotQA / MuSiQue gold annotations."""

    sentence_idx: int | None = None
    is_gold_doc: bool = False
    is_gold_fact: bool = False


@dataclass
class MimicIVChunk(Chunk):
    """Chunk with MIMIC-IV discharge note metadata."""

    chunk_id: str | None = None
    note_id: str | None = None
    subject_id: int | None = None
    hadm_id: int | None = None
    section_name: str | None = None
    subsection_name: str | None = None
    char_count: int = 0
    approx_tokens: int = 0
