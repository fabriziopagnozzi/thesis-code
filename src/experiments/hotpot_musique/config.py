import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from helpers.query_algorithms import ScoringFunction

type DatasetName = Literal['hotpotqa_distractor', 'musique', '2wikimultihopqa']

type EmbeddingChunkMode = Literal['sentence', 'document', 'word_window', 'token_window']
"""
Chunking strategy
    "sentence"     - one chunk per sentence (default for multi-hop datasets)
    "document"     - concatenate all sentences per document into one chunk
    "word_window"  - sliding window of w words
    "token_window" - sliding window of chunk_tokens words with stride overlap
"""


@dataclass
class ExperimentConfig:
    dataset: DatasetName = 'hotpotqa_distractor'
    dataset_path: str = 'datasets/full-data/HotpotQA/hotpot_dev_distractor_v1.json'
    embedding_model: str = 'multi-qa-mpnet-base-cos-v1'

    scoring_functions: list[ScoringFunction] = field(
        default_factory=lambda: [
            'top_k',
            'mmr',
            'fac_loc',
        ]
    )
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])
    lambda_values: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.5, 0.7, 0.8])
    mmr_window: int | None = None
    theta: float | None = None  # sector_coverage

    chunk_mode: EmbeddingChunkMode = 'sentence'
    # token_window mode
    chunk_tokens: int = 256
    stride: int = 128
    t_max: int | None = None
    # word_window mode
    w: int = 100

    # Pool limits
    max_docs: int | None = None  # limit source docs per record
    max_cands: int | None = None  # limit total candidate chunks

    max_records: int | None = None  # limit records to read
    output_dir: str = 'results'
    batch_size: int = 256
    seed: int = 42
    device: Literal['cpu', 'gpu'] = 'cpu'

    def strategies_with_lambda(self) -> list[tuple[ScoringFunction, float | None]]:
        pairs: list[tuple[ScoringFunction, float | None]] = []
        for s in self.scoring_functions:
            if s in ('top_k', 'fps'):
                pairs.append((s, None))
            else:
                for lv in self.lambda_values:
                    pairs.append((s, lv))
        return pairs

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path) as f:
            d = json.load(f)
        return cls(**d)
