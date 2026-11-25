"""Qdrant-backed chunk store with hybrid (dense + BM25, RRF-fused) search."""

from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from qacopilot.config import Settings
from qacopilot.ingest.chunking import Chunk
from qacopilot.ingest.models import DocType
from qacopilot.retrieval.embeddings import Embedder, SparseVec

DENSE = "dense"
SPARSE = "sparse"
_BATCH = 64


class Hit(BaseModel):
    chunk: Chunk
    score: float  # the score hits are ordered by (the reranker's, once reranked)
    fused_score: float | None = None  # the hybrid-search (RRF) score, kept after reranking


def _sparse(v: SparseVec) -> models.SparseVector:
    return models.SparseVector(indices=v.indices, values=v.values)


class QdrantStore:
    def __init__(
        self,
        embedder: Embedder,
        settings: Settings | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self._embedder = embedder
        self._settings = settings or Settings()
        self._client = client or AsyncQdrantClient(url=self._settings.qdrant_url)
        self._collection = self._settings.qdrant_collection

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            self._collection,
            vectors_config={
                DENSE: models.VectorParams(size=self._embedder.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in ("doc_id", "doc_type"):
            await self._client.create_payload_index(
                self._collection, field, models.PayloadSchemaType.KEYWORD
            )

    async def recreate_collection(self) -> None:
        """Drop and rebuild the collection (evals and re-indexing after a model change)."""
        if await self._client.collection_exists(self._collection):
            await self._client.delete_collection(self._collection)
        await self.ensure_collection()

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Embed and store chunks. Re-indexing a document replaces it (see ``replace_document``)."""
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            dense, sparse = await self._embedder.embed_documents([c.text for c in batch])
            await self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(
                        id=c.id,
                        vector={DENSE: d, SPARSE: _sparse(s)},
                        payload=c.model_dump(mode="json"),
                    )
                    for c, d, s in zip(batch, dense, sparse, strict=True)
                ],
            )

    async def delete_document(self, doc_id: str) -> None:
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
        )

    async def replace_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Index a document, removing any chunks left from an earlier version of it."""
        await self.delete_document(doc_id)
        await self.upsert(chunks)

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        doc_type: DocType | None = None,
        doc_id: str | None = None,
        candidates: int = 30,
    ) -> list[Hit]:
        dense, sparse = await self._embedder.embed_query(query)
        flt = _filter(doc_id=doc_id, doc_type=doc_type)
        result = await self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, limit=candidates, filter=flt),
                models.Prefetch(query=_sparse(sparse), using=SPARSE, limit=candidates, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [Hit(chunk=Chunk(**(p.payload or {})), score=p.score) for p in result.points]

    async def count(self) -> int:
        return (await self._client.count(self._collection)).count

    async def aclose(self) -> None:
        await self._client.close()


def _filter(doc_id: str | None = None, doc_type: DocType | None = None) -> models.Filter | None:
    must: list[Any] = []
    if doc_id:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    if doc_type:
        must.append(
            models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type.value))
        )
    return models.Filter(must=must) if must else None
