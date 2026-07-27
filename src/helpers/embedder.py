from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

type EmbeddingModelName = Literal[
    'multi-qa-mpnet-base-cos-v1',
    'BAAI/bge-m3',
    'Qwen/Qwen3-Embedding-0.6B',
    'Qwen/Qwen3-Embedding-4B',
    'Qwen/Qwen3-Embedding-8B',
    'jinaai/jina-embeddings-v5-text-small',
    'abhinand/MedEmbed-large-v0.1',
    'ncbi/MedCPT',
]

type EncodeMode = Literal[
    'encode_query_document',
    'plain_encode',
    'query_prompt_name_docs_plain',
    'jina_v5',
    'medcpt',
]


@dataclass(frozen=True)
class EmbeddingModelProfile:
    mode: EncodeMode
    trust_remote_code: bool = False
    query_prompt: str | None = None
    document_prompt: str | None = None
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    config_kwargs: dict[str, Any] = field(default_factory=dict)
    query_model_name: str | None = None
    document_model_name: str | None = None
    query_max_length: int = 64
    document_max_length: int = 512


MODEL_PROFILES: dict[str, EmbeddingModelProfile] = {
    'multi-qa-mpnet-base-cos-v1': EmbeddingModelProfile(mode='encode_query_document'),
    'BAAI/bge-m3': EmbeddingModelProfile(mode='plain_encode'),
    'Qwen/Qwen3-Embedding-0.6B': EmbeddingModelProfile(
        mode='query_prompt_name_docs_plain',
        query_prompt_name='query',
    ),
    'Qwen/Qwen3-Embedding-4B': EmbeddingModelProfile(
        mode='query_prompt_name_docs_plain',
        query_prompt_name='query',
    ),
    'Qwen/Qwen3-Embedding-8B': EmbeddingModelProfile(
        mode='query_prompt_name_docs_plain',
        query_prompt_name='query',
    ),
    'jinaai/jina-embeddings-v5-text-small': EmbeddingModelProfile(
        mode='jina_v5',
        trust_remote_code=True,
        query_prompt_name='query',
        document_prompt_name='document',
        model_kwargs={'default_task': 'retrieval'},
    ),
    'abhinand/MedEmbed-large-v0.1': EmbeddingModelProfile(mode='encode_query_document'),
    'ncbi/MedCPT': EmbeddingModelProfile(
        mode='medcpt',
        query_model_name='ncbi/MedCPT-Query-Encoder',
        document_model_name='ncbi/MedCPT-Article-Encoder',
    ),
}


class Embedder:
    def __init__(
        self,
        model_name: EmbeddingModelName,
        batch_size: int = 4,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        device: str = 'cuda',
        devices: Sequence[str] | None = None,
    ):
        target_devices = [str(item) for item in (devices or []) if str(item).strip()]
        if len(target_devices) == 1:
            device = target_devices[0]

        profile = MODEL_PROFILES[model_name]
        self.model_name = model_name
        self.batch_size = batch_size
        self.profile = profile
        self.query_prompt = query_prompt if query_prompt is not None else profile.query_prompt
        self.document_prompt = (
            document_prompt if document_prompt is not None else profile.document_prompt
        )
        self.device = device
        self.devices = target_devices
        self._dim: int | None = None
        self._pool = None
        self._query_tokenizer = None
        self._query_model = None
        self._document_tokenizer = None
        self._document_model = None

        if profile.mode == 'medcpt':
            if len(target_devices) > 1:
                raise ValueError('MedCPT backend does not support devices= multi-process encoding')
            self._load_medcpt()
        else:
            self._load_sentence_transformer()

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim

        if self.profile.mode == 'medcpt':
            dim = getattr(self._query_model.config, 'hidden_size', None)
        else:
            dim = self._model.get_embedding_dimension()

        if dim is None and self.profile.mode != 'medcpt':
            get_sentence_dim = getattr(self._model, 'get_embedding_dimension', None)
            if get_sentence_dim is not None:
                dim = get_sentence_dim()

        if dim is None:
            probe = self.embed_docs(['dimension probe'], normalize=False)
            dim = int(np.asarray(probe).shape[-1])

        self._dim = int(dim)
        return self._dim

    def embed_docs(self, texts: list[str], normalize: bool = True) -> NDArray[np.float32]:
        if self.profile.mode == 'medcpt':
            return self._embed_medcpt_docs(texts, normalize=normalize)
        if self.profile.mode == 'encode_query_document':
            embs = self._model.encode_document(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=True,
                convert_to_numpy=True,
                pool=self._pool,
                prompt=self.document_prompt,
            )
        elif self.profile.mode in {'plain_encode', 'query_prompt_name_docs_plain'}:
            embs = self._encode_sentence_transformer(
                texts,
                normalize=normalize,
                show_progress_bar=True,
                prompt=self.document_prompt,
            )
        elif self.profile.mode == 'jina_v5':
            embs = self._encode_sentence_transformer(
                texts,
                normalize=normalize,
                show_progress_bar=True,
                prompt=self.document_prompt,
                prompt_name=self.profile.document_prompt_name,
                task='retrieval',
            )
        else:
            raise ValueError(
                f'unsupported embedding encode mode for documents: {self.profile.mode}'
            )
        return np.asarray(embs, dtype=np.float32)

    def embed_query(self, query_text: str, normalize: bool = True) -> NDArray[np.float32]:
        return self.embed_queries([query_text], normalize=normalize)[0]

    def embed_queries(self, query_texts: list[str], normalize: bool = True) -> NDArray[np.float32]:
        if self.profile.mode == 'medcpt':
            return self._embed_medcpt_queries(query_texts, normalize=normalize)
        if self.profile.mode == 'encode_query_document':
            embs = self._model.encode_query(
                query_texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=True,
                convert_to_numpy=True,
                prompt=self.query_prompt,
                pool=self._pool,
            )
        elif self.profile.mode == 'plain_encode':
            embs = self._encode_sentence_transformer(
                query_texts,
                normalize=normalize,
                show_progress_bar=True,
                prompt=self.query_prompt,
            )
        elif self.profile.mode == 'query_prompt_name_docs_plain':
            embs = self._encode_sentence_transformer(
                query_texts,
                normalize=normalize,
                show_progress_bar=True,
                prompt=self.query_prompt,
                prompt_name=None
                if self.query_prompt is not None
                else self.profile.query_prompt_name,
            )
        elif self.profile.mode == 'jina_v5':
            embs = self._encode_sentence_transformer(
                query_texts,
                normalize=normalize,
                show_progress_bar=True,
                prompt=self.query_prompt,
                prompt_name=None
                if self.query_prompt is not None
                else self.profile.query_prompt_name,
                task='retrieval',
            )
        else:
            raise ValueError(f'unsupported embedding encode mode for queries: {self.profile.mode}')
        return np.asarray(embs, dtype=np.float32)

    def release(self) -> None:
        import gc

        import torch
        from sentence_transformers import SentenceTransformer

        if self._pool is not None:
            SentenceTransformer.stop_multi_process_pool(self._pool)
            self._pool = None
        if hasattr(self, '_model'):
            del self._model
        for attr_name in (
            '_query_model',
            '_query_tokenizer',
            '_document_model',
            '_document_tokenizer',
        ):
            if getattr(self, attr_name, None) is not None:
                setattr(self, attr_name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed_qa_record(
        self, query: str, chunk_texts: list[str], normalize: bool = True
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Encode query + chunks for a record.
        Returns (query_emb [dim], chunk_embs [n_chunks, dim]).
        """
        return self.embed_query(query, normalize=normalize), self.embed_docs(
            chunk_texts, normalize=normalize
        )

    def _load_sentence_transformer(self) -> None:
        from sentence_transformers import SentenceTransformer

        target_devices = self.devices
        model_device = 'cpu' if len(target_devices) > 1 else self.device
        self._model = SentenceTransformer(
            self.model_name,
            device=model_device,
            trust_remote_code=self.profile.trust_remote_code,
            model_kwargs=self.profile.model_kwargs or None,
            config_kwargs=self.profile.config_kwargs or None,
        )
        if len(target_devices) > 1:
            print(f'[embedder] using multi-process data parallel devices: {target_devices}')
        self._pool = (
            self._model.start_multi_process_pool(target_devices=target_devices)
            if len(target_devices) > 1
            else None
        )

    def _load_medcpt(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        query_model_name = self.profile.query_model_name
        document_model_name = self.profile.document_model_name
        if query_model_name is None or document_model_name is None:
            raise ValueError('MedCPT profile must define query_model_name and document_model_name')

        self._query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self._query_model = AutoModel.from_pretrained(query_model_name).to(self.device)
        self._query_model.eval()
        self._document_tokenizer = AutoTokenizer.from_pretrained(document_model_name)
        self._document_model = AutoModel.from_pretrained(document_model_name).to(self.device)
        self._document_model.eval()
        self._torch = torch

    def _encode_sentence_transformer(
        self,
        texts: list[str],
        *,
        normalize: bool,
        show_progress_bar: bool,
        prompt: str | None = None,
        prompt_name: str | None = None,
        task: str | None = None,
    ):
        kwargs: dict[str, Any] = {
            'batch_size': self.batch_size,
            'normalize_embeddings': normalize,
            'show_progress_bar': show_progress_bar,
            'convert_to_numpy': True,
        }
        if self._pool is not None:
            kwargs['pool'] = self._pool
        if prompt is not None:
            kwargs['prompt'] = prompt
        elif prompt_name is not None:
            kwargs['prompt_name'] = prompt_name
        if task is not None:
            kwargs['task'] = task
        return self._model.encode(texts, **kwargs)

    def _embed_medcpt_queries(
        self, query_texts: list[str], normalize: bool = True
    ) -> NDArray[np.float32]:
        return self._encode_medcpt(
            self._query_model,
            self._query_tokenizer,
            query_texts,
            max_length=self.profile.query_max_length,
            normalize=normalize,
        )

    def _embed_medcpt_docs(self, texts: list[str], normalize: bool = True) -> NDArray[np.float32]:
        article_inputs = [[text, ''] for text in texts]
        return self._encode_medcpt(
            self._document_model,
            self._document_tokenizer,
            article_inputs,
            max_length=self.profile.document_max_length,
            normalize=normalize,
        )

    def _encode_medcpt(
        self,
        model,
        tokenizer,
        texts,
        *,
        max_length: int,
        normalize: bool,
    ) -> NDArray[np.float32]:
        torch = self._torch
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                    max_length=max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                embs = model(**encoded).last_hidden_state[:, 0, :]
                if normalize:
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                chunks.append(embs.detach().cpu().numpy())
        if not chunks:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.asarray(np.concatenate(chunks, axis=0), dtype=np.float32)
