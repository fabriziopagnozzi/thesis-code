from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from experiments.medical_dataset_gen.evaluation.schemas import LightweightChunkRecord
from experiments.medical_dataset_gen.utils.global_schemas import EvaluationRerankerCfg

DENSE_RERANKER_STRATEGY = 'reranker'


@dataclass(slots=True)
class DenseReranker:
    model: Any
    batch_size: int
    prompt_name: str | None
    show_progress_bar: bool

    @classmethod
    def from_config(cls, cfg: EvaluationRerankerCfg) -> DenseReranker:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                'evaluation.use_reranker requires sentence-transformers to be installed'
            ) from exc

        prompt_name = 'clinical_retrieval' if cfg.prompt is not None else None
        prompts: dict[str, str] | None = (
            {'clinical_retrieval': cfg.prompt} if cfg.prompt is not None else None
        )
        model = CrossEncoder(
            cfg.model_name,
            device=cfg.device,
            prompts=prompts,
            default_prompt_name=prompt_name,
            trust_remote_code=cfg.trust_remote_code,
            max_length=cfg.max_length,
        )
        return cls(
            model=model,
            batch_size=int(cfg.batch_size),
            prompt_name=prompt_name,
            show_progress_bar=cfg.show_progress_bar,
        )

    def rank_indices(
        self,
        *,
        query_text: str,
        candidate_chunk_ids: list[str],
        chunk_by_id: Mapping[str, LightweightChunkRecord],
        top_k: int,
    ) -> NDArray[np.intp]:
        if top_k <= 0 or not candidate_chunk_ids:
            return np.array([], dtype=np.intp)

        documents = [
            chunk_by_id.get(chunk_id, LightweightChunkRecord()).text
            for chunk_id in candidate_chunk_ids
        ]
        if not any(document.strip() for document in documents):
            raise ValueError('reranker is enabled, but candidate chunk text is empty')

        rankings = cast(
            list[dict[str, int | float | str]],
            self.model.rank(
                query_text,
                documents,
                top_k=min(top_k, len(documents)),
                return_documents=False,
                prompt_name=self.prompt_name,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
            ),
        )
        return np.asarray([int(item['corpus_id']) for item in rankings], dtype=np.intp)
