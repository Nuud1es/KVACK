"""Hybrid search, then (optionally) rerank and drop low-scoring hits."""

from qacopilot.ingest.models import DocType
from qacopilot.retrieval.rerank import Reranker
from qacopilot.retrieval.store import Hit, QdrantStore


class Retriever:
    def __init__(
        self,
        store: QdrantStore,
        reranker: Reranker | None = None,
        *,
        min_score: float | None = None,
    ) -> None:
        self._store = store
        self._reranker = reranker
        self._min_score = min_score

    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        candidates: int = 20,
        doc_type: DocType | None = None,
        doc_id: str | None = None,
    ) -> list[Hit]:
        """Best ``limit`` chunks for ``query``; empty if even the best one is below ``min_score``.

        ``min_score`` only applies with a reranker: fused scores are rank-based, not comparable
        between queries.
        """
        hits = await self._store.search(
            query, limit=candidates if self._reranker else limit, doc_type=doc_type, doc_id=doc_id
        )
        if not self._reranker:
            return hits[:limit]
        scores = await self._reranker.score(query, [h.chunk.text for h in hits])
        reranked = sorted(
            (
                Hit(chunk=h.chunk, score=s, fused_score=h.score)
                for h, s in zip(hits, scores, strict=True)
            ),
            key=lambda h: h.score,
            reverse=True,
        )
        # The cut-off decides whether the documents contain anything relevant at all, so it looks
        # at the best hit only. Judging each chunk would drop weak-scoring neighbours that hold the
        # actual fact (a "Limits" section scores low for a question that names the "Method").
        if self._min_score is not None and reranked and reranked[0].score < self._min_score:
            return []
        return reranked[:limit]
