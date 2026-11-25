"""Rerankers: score (query, passage) pairs more precisely than vector search can."""

import asyncio
from abc import ABC, abstractmethod

from qacopilot.config import Settings


class Reranker(ABC):
    @abstractmethod
    async def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage; higher is better and scores are comparable
        across queries (so a threshold can mean 'not in the documents')."""


class FastEmbedReranker(Reranker):
    """Local CPU cross-encoder. The model loads on first use."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._model_name = (settings or Settings()).reranker_model
        self._model: object | None = None
        self._lock = asyncio.Lock()

    def _score(self, query: str, passages: list[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self._model_name)
        return [float(s) for s in self._model.rerank(query, passages)]  # type: ignore[attr-defined]

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._score, query, passages)
