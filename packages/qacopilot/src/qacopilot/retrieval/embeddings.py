"""Embedding providers. Business logic depends on ``Embedder``, never on fastembed."""

import asyncio
from abc import ABC, abstractmethod

from pydantic import BaseModel

from qacopilot.config import Settings


class SparseVec(BaseModel):
    indices: list[int]
    values: list[float]


class Embedder(ABC):
    """Produces dense and sparse (keyword) vectors for hybrid search."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Size of the dense vectors."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        """Dense and sparse vectors for passages to be indexed."""

    @abstractmethod
    async def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        """Dense and sparse vectors for a search query."""


class FastEmbedEmbedder(Embedder):
    """Local CPU embeddings: a dense model plus BM25 sparse vectors. Models load on first use."""

    def __init__(self, settings: Settings | None = None) -> None:
        from fastembed import TextEmbedding

        self._settings = settings or Settings()
        self._dim = int(TextEmbedding.get_embedding_size(self._settings.embedding_model))
        self._dense: TextEmbedding | None = None
        self._sparse: object | None = None
        self._lock = asyncio.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        if self._dense is None:
            self._dense = TextEmbedding(self._settings.embedding_model)
        if self._sparse is None:
            self._sparse = SparseTextEmbedding(self._settings.sparse_model)

    def _embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        self._load()
        assert self._dense is not None and self._sparse is not None
        dense = [v.tolist() for v in self._dense.embed(texts)]
        sparse = [
            SparseVec(indices=s.indices.tolist(), values=s.values.tolist())
            for s in self._sparse.embed(texts)  # type: ignore[attr-defined]
        ]
        return dense, sparse

    def _embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        self._load()
        assert self._dense is not None and self._sparse is not None
        dense = next(iter(self._dense.query_embed(text))).tolist()
        s = next(iter(self._sparse.query_embed(text)))  # type: ignore[attr-defined]
        return dense, SparseVec(indices=s.indices.tolist(), values=s.values.tolist())

    async def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        if not texts:
            return [], []
        async with self._lock:  # onnx sessions are not meant to be shared across threads
            return await asyncio.to_thread(self._embed_documents, texts)

    async def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        async with self._lock:
            return await asyncio.to_thread(self._embed_query, text)
