"""Deterministic offline embedder for tests: hashed bag-of-words, no model download."""

import hashlib
import math
import re

from qacopilot.retrieval.embeddings import Embedder, SparseVec
from qacopilot.retrieval.rerank import Reranker

DIM = 64


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bucket(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16)


class FakeEmbedder(Embedder):
    @property
    def dim(self) -> int:
        return DIM

    def _one(self, text: str) -> tuple[list[float], SparseVec]:
        dense = [0.0] * DIM
        counts: dict[int, float] = {}
        for tok in _tokens(text):
            h = _bucket(tok)
            dense[h % DIM] += 1.0
            counts[h % 100_000] = counts.get(h % 100_000, 0.0) + 1.0
        norm = math.sqrt(sum(x * x for x in dense)) or 1.0
        return [x / norm for x in dense], SparseVec(
            indices=list(counts), values=list(counts.values())
        )

    async def embed_documents(self, texts: list[str]) -> tuple[list[list[float]], list[SparseVec]]:
        pairs = [self._one(t) for t in texts]
        return [d for d, _ in pairs], [s for _, s in pairs]

    async def embed_query(self, text: str) -> tuple[list[float], SparseVec]:
        return self._one(text)


class FakeReranker(Reranker):
    """Score = number of query words that appear in the passage, minus a constant."""

    async def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_tokens(query))
        return [len(q & set(_tokens(p))) - 2.0 for p in passages]
