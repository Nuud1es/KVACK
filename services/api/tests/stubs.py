"""Offline stand-ins so API tests need no LLM, Qdrant server or model download."""

import hashlib
import math
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from qacopilot.llm import BaseLLMProvider, LLMResponse
from qacopilot.retrieval import Embedder, Reranker, SparseVec
from qacopilot.tracing import request_id_var

DIM = 64


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class StubEmbedder(Embedder):
    @property
    def dim(self) -> int:
        return DIM

    def _one(self, text: str) -> tuple[list[float], SparseVec]:
        dense = [0.0] * DIM
        counts: dict[int, float] = {}
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
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


class StubReranker(Reranker):
    async def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(_tokens(query))
        return [len(q & set(_tokens(p))) - 2.0 for p in passages]


class ScriptedLLM(BaseLLMProvider):
    def __init__(self, reply: str = "Answer [1].") -> None:
        self.reply = reply
        self.seen_request_ids: list[str | None] = []

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        self.seen_request_ids.append(request_id_var.get())
        return LLMResponse(content="Please arrange these calibrations before they lapse.")

    async def stream(self, messages: Sequence[Any], **kwargs: Any) -> AsyncIterator[str]:
        self.seen_request_ids.append(request_id_var.get())
        for word in self.reply.split(" "):
            yield word + " "

    async def health(self) -> bool:
        return True
