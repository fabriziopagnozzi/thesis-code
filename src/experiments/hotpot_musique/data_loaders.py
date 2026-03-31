import json
from collections.abc import Callable
from pathlib import Path

from helpers.chunks_classes import HotpotChunk

from .config import DatasetName
from .qa_processing import (
    QARecord,
    _build_doc_chunks,
    _split_sentences,
)


def _load_hotpot_format(
    path: str | Path,
    chunk_mode: str = 'sentence',
    max_docs: int | None = None,
    w: int = 100,
    chunk_tokens: int = 256,
    stride: int = 128,
) -> list[QARecord]:
    """HotpotQA or 2WikiMultihopQA.
    Each record has documents as (title, [sentence, ...]) pairs
    and supporting_facts as (title, sentence_idx) pairs.
    """
    with open(path) as f:
        raw = json.load(f)

    records: list[QARecord] = []
    for item in raw:
        gold_facts = [(title, sentence_idx) for title, sentence_idx in item['supporting_facts']]
        gold_doc_titles = {title for title, _ in gold_facts}

        context = item['context']
        if max_docs is not None:
            context = context[:max_docs]

        chunks: list[HotpotChunk] = []
        for doc_idx, (title, sentences) in enumerate(context):
            is_gold = title in gold_doc_titles
            gold_sents = {si for t, si in gold_facts if t == title}
            chunks.extend(
                _build_doc_chunks(
                    sentences,
                    title,
                    doc_idx,
                    is_gold,
                    gold_sents,
                    chunk_mode,
                    chunk_tokens,
                    stride,
                    w,
                )
            )

        records.append(
            QARecord(
                id=item['_id'],
                question=item['question'],
                answer=item['answer'],
                question_type=item.get('type', ''),
                chunks=chunks,
                gold_doc_titles=gold_doc_titles,
                gold_facts=gold_facts,
            )
        )

    return records


def _load_musique(
    path: str | Path,
    chunk_mode: str = 'sentence',
    max_docs: int | None = None,
    w: int = 100,
    chunk_tokens: int = 256,
    stride: int = 128,
) -> list[QARecord]:
    """Load MuSiQue answerable dev set.
    Each record has 20 paragraphs with is_supporting annotations.
    """
    records: list[QARecord] = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if not item.get('answerable', True):
                continue

            paragraphs = item['paragraphs']
            if max_docs is not None:
                paragraphs = paragraphs[:max_docs]

            gold_facts: list[tuple[str, int]] = []
            gold_doc_titles: set[str] = set()
            for para in paragraphs:
                if para['is_supporting']:
                    gold_doc_titles.add(para['title'])
                    gold_facts.append((para['title'], para['idx']))

            chunks: list[HotpotChunk] = []
            for doc_idx, para in enumerate(paragraphs):
                is_gold = para['is_supporting']
                sentences = _split_sentences(para['paragraph_text'])

                if chunk_mode == 'sentence':
                    gold_sents = {0} if is_gold else set()
                else:
                    gold_sents = set(range(len(sentences))) if is_gold else set()

                chunks.extend(
                    _build_doc_chunks(
                        sentences,
                        para['title'],
                        doc_idx,
                        is_gold,
                        gold_sents,
                        chunk_mode,
                        chunk_tokens,
                        stride,
                        w,
                    )
                )

            records.append(
                QARecord(
                    id=item['id'],
                    question=item['question'],
                    answer=item['answer'],
                    question_type=f'{item["id"].split("__")[0]}',
                    chunks=chunks,
                    gold_doc_titles=gold_doc_titles,
                    gold_facts=gold_facts,
                )
            )

    return records


LOADERS: dict[DatasetName, Callable] = {
    'hotpotqa_distractor': _load_hotpot_format,
    'musique': _load_musique,
    '2wikimultihopqa': _load_hotpot_format,
}


def load_dataset(
    dataset: DatasetName,
    path: str | Path,
    chunk_mode: str = 'sentence',
    max_docs: int | None = None,
    w: int = 100,
    chunk_tokens: int = 256,
    stride: int = 128,
) -> list[QARecord]:
    return LOADERS[dataset](
        path,
        chunk_mode=chunk_mode,
        max_docs=max_docs,
        w=w,
        chunk_tokens=chunk_tokens,
        stride=stride,
    )
