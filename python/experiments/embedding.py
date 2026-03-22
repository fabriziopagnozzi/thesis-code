import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from experiments.config import EmbeddingModelName


class Embedder:
    def __init__(
        self,
        model_name: EmbeddingModelName,
        device: str = 'cpu',
        cache_dir: str | Path | None = None,
        batch_size: int = 256,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        assert dim is not None
        return dim

    def encode_record(
        self,
        record_id: str,
        query: str,
        chunk_texts: list[str],
        normalize: bool = True,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Encode query + chunks for a record, with optional disk cache.
        Returns (query_emb [dim], chunk_embs [n_chunks, dim]).
        """
        if self.cache_dir is not None:
            cached = self._load_cache(record_id)
            if cached is not None:
                return cached

        all_texts = [query, *chunk_texts]
        all_embs = self._model.encode(
            all_texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_emb = all_embs[0]
        chunk_embs = all_embs[1:]

        if self.cache_dir is not None:
            self._save_cache(record_id, query_emb, chunk_embs)

        return query_emb, chunk_embs

    def _cache_path(self, record_id: str) -> Path:
        assert self.cache_dir is not None
        model_slug = hashlib.md5(
            self.model_name.encode(), usedforsecurity=False
        ).hexdigest()[:8]
        d = self.cache_dir / model_slug
        d.mkdir(parents=True, exist_ok=True)
        return d / f'{record_id}.npz'

    def _load_cache(
        self, record_id: str
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
        p = self._cache_path(record_id)
        if p.exists():
            data = np.load(p)
            return data['query'], data['chunks']
        return None

    def _save_cache(
        self,
        record_id: str,
        query_emb: NDArray[np.float32],
        chunk_embs: NDArray[np.float32],
    ) -> None:
        p = self._cache_path(record_id)
        np.savez_compressed(p, query=query_emb, chunks=chunk_embs)
