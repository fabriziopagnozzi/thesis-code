from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class Embedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        device: str = 'cuda',
        devices: Sequence[str] | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        target_devices = [str(item) for item in (devices or []) if str(item).strip()]
        if len(target_devices) == 1:
            device = target_devices[0]

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt
        self.device = device
        self.devices = target_devices
        model_device = 'cpu' if len(target_devices) > 1 else device
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=model_device)
        if len(target_devices) > 1:
            print(f'[embedder] using multi-process data parallel devices: {target_devices}')
        self._pool = (
            self._model.start_multi_process_pool(target_devices=target_devices)
            if len(target_devices) > 1
            else None
        )

    @property
    def dim(self) -> int:
        dim = self._model.get_embedding_dimension()
        assert dim is not None
        return dim

    def embed_docs(self, texts: list[str], normalize: bool = True) -> NDArray[np.float32]:
        embs = self._model.encode_document(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
            pool=self._pool,
            prompt=self.document_prompt,
        )
        return np.asarray(embs, dtype=np.float32)

    def embed_query(self, query_text: str, normalize: bool = True) -> NDArray[np.float32]:
        if self._pool is not None:
            return self.embed_queries([query_text], normalize=normalize)[0]

        emb = self._model.encode_query(
            query_text,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
            prompt=self.query_prompt,
        )
        return np.asarray(emb, dtype=np.float32)

    def embed_queries(self, query_texts: list[str], normalize: bool = True) -> NDArray[np.float32]:
        embs = self._model.encode_query(
            query_texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
            prompt=self.query_prompt,
            pool=self._pool,
        )
        return np.asarray(embs, dtype=np.float32)

    def release(self) -> None:
        import gc

        import torch
        from sentence_transformers import SentenceTransformer

        if self._pool is not None:
            SentenceTransformer.stop_multi_process_pool(self._pool)
            self._pool = None
        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed_qa_record(
        self, query: str, chunk_texts: list[str], normalize: bool = True
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Encode query + chunks for a record.
        Returns (query_emb [dim], chunk_embs [n_chunks, dim]).
        """
        if self._pool is not None:
            return self.embed_query(query, normalize=normalize), self.embed_docs(
                chunk_texts, normalize=normalize
            )

        all_texts = [query, *chunk_texts]
        all_embs = self._model.encode(
            all_texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return all_embs[0], all_embs[1:]
