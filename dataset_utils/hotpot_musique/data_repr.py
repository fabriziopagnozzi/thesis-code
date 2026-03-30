from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single candidate chunk for selection."""

    text: str
    doc_title: str
    doc_idx: int
    sentence_idx: int | None = None  # None for document-level chunks
    is_gold_doc: bool = False
    is_gold_fact: bool = False


@dataclass
class QARecord:
    """A single QA record with candidate chunks and gold annotations."""

    id: str
    question: str
    answer: str
    question_type: str
    chunks: list[Chunk] = field(default_factory=list)
    gold_doc_titles: set[str] = field(default_factory=set)
    gold_facts: list[tuple[str, int]] = field(default_factory=list)

    @property
    def n_gold_docs(self) -> int:
        return len(self.gold_doc_titles)

    @property
    def n_gold_facts(self) -> int:
        return len(self.gold_facts)


def _build_word_window_chunks(
    sentences: Sequence[str],
    doc_title: str,
    doc_idx: int,
    is_gold_doc: bool,
    gold_sent_indices: set[int],
    w: int,
) -> list[Chunk]:
    words: list[tuple[str, int]] = []
    for sentence_idx, sent in enumerate(sentences):
        for tok in sent.strip().split():
            words.append((tok, sentence_idx))

    if not words:
        return []

    stride = w // 2
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + w, len(words))
        window = words[start:end]

        text = ' '.join(tok for tok, _ in window)
        sent_indices_in_window = {si for _, si in window}
        has_gold = bool(sent_indices_in_window & gold_sent_indices)

        chunks.append(
            Chunk(
                text=text,
                doc_title=doc_title,
                doc_idx=doc_idx,
                sentence_idx=None,
                is_gold_doc=is_gold_doc,
                is_gold_fact=has_gold,
            )
        )

        if end >= len(words):
            break
        start += stride

    return chunks


def _build_token_window_chunks(
    sentences: Sequence[str],
    doc_title: str,
    doc_idx: int,
    is_gold_doc: bool,
    gold_sent_indices: set[int],
    chunk_tokens: int,
    stride: int,
) -> list[Chunk]:
    """
    TODO: plug in real tokenizer (e.g. HF AutoTokenizer).
    """
    tokenize = str.split

    tokens: list[tuple[str, int]] = []
    for sentence_idx, sent in enumerate(sentences):
        for tok in tokenize(sent.strip()):
            tokens.append((tok, sentence_idx))

    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        window = tokens[start:end]

        text = ' '.join(tok for tok, _ in window)
        sent_indices_in_window = {si for _, si in window}
        has_gold = bool(sent_indices_in_window & gold_sent_indices)

        chunks.append(
            Chunk(
                text=text,
                doc_title=doc_title,
                doc_idx=doc_idx,
                sentence_idx=None,
                is_gold_doc=is_gold_doc,
                is_gold_fact=has_gold,
            )
        )

        if end >= len(tokens):
            break
        start += stride

    return chunks


def _build_doc_chunks(
    sentences: Sequence[str],
    title: str,
    doc_idx: int,
    is_gold_doc: bool,
    gold_sent_indices: set[int],
    chunk_mode: str,
    chunk_tokens: int,
    stride: int,
    w: int,
) -> list[Chunk]:
    """Build chunks from a single document's sentences using the given mode."""
    if chunk_mode == 'sentence':
        return [
            Chunk(
                text=sent.strip(),
                doc_title=title,
                doc_idx=doc_idx,
                sentence_idx=sentence_idx,
                is_gold_doc=is_gold_doc,
                is_gold_fact=sentence_idx in gold_sent_indices,
            )
            for sentence_idx, sent in enumerate(sentences)
        ]
    elif chunk_mode == 'document':
        full_text = ' '.join(s.strip() for s in sentences)
        return [
            Chunk(
                text=full_text,
                doc_title=title,
                doc_idx=doc_idx,
                sentence_idx=None,
                is_gold_doc=is_gold_doc,
                is_gold_fact=bool(gold_sent_indices),
            )
        ]
    elif chunk_mode == 'word_window':
        return _build_word_window_chunks(
            sentences,
            title,
            doc_idx,
            is_gold_doc,
            gold_sent_indices,
            w=w,
        )
    elif chunk_mode == 'token_window':
        return _build_token_window_chunks(
            sentences,
            title,
            doc_idx,
            is_gold_doc,
            gold_sent_indices,
            chunk_tokens=chunk_tokens,
            stride=stride,
        )
    else:
        raise ValueError(f'Unknown chunk_mode: {chunk_mode!r}')


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]
