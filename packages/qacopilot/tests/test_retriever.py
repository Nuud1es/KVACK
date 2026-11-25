from collections.abc import AsyncIterator

import pytest
from fakes import FakeEmbedder, FakeReranker
from qdrant_client import AsyncQdrantClient

from qacopilot.config import Settings
from qacopilot.ingest import DocType, PageText, ParsedDocument, chunk_document
from qacopilot.retrieval import QdrantStore, Retriever


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source="t.pdf", pages=[PageText(page=1, text=text, method="text")])


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    s = QdrantStore(FakeEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
    await s.ensure_collection()
    for doc_id, body in {
        "a": "Probe TP-20944 is next due 2027-04-02.",
        "b": "Probe TP-20931 is next due 2027-03-14.",
        "c": "Canteen menu: soup and sandwiches.",
    }.items():
        await s.upsert(
            chunk_document(_doc(f"Cert {doc_id}\n\n1. Details\n{body}"), DocType.SOP, doc_id)
        )
    yield s
    await s.aclose()


async def test_without_reranker_returns_search_order(store: QdrantStore) -> None:
    hits = await Retriever(store).retrieve("probe TP-20944 due", limit=2)
    assert len(hits) == 2 and hits[0].fused_score is None


async def test_reranker_reorders_and_keeps_fused_score(store: QdrantStore) -> None:
    hits = await Retriever(store, FakeReranker()).retrieve("probe TP-20944 due 2027-04-02", limit=3)
    assert hits[0].chunk.doc_id == "a"
    assert hits[0].fused_score is not None
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


async def test_min_score_returns_nothing_for_out_of_scope_questions(store: QdrantStore) -> None:
    r = Retriever(store, FakeReranker(), min_score=0.0)
    assert await r.retrieve("fire evacuation assembly point") == []
    assert (await r.retrieve("probe TP-20944 due 2027-04-02"))[0].chunk.doc_id == "a"


async def test_the_cut_off_looks_at_the_best_hit_not_every_chunk(store: QdrantStore) -> None:
    """A weak-scoring chunk next to a strong one is kept: it often holds the actual fact."""
    r = Retriever(store, FakeReranker(), min_score=0.0)
    hits = await r.retrieve("probe TP-20944 due 2027-04-02", limit=3)
    scores = [h.score for h in hits]
    assert scores[0] > 0 and any(s < 0 for s in scores[1:])  # the others are below the bar...
    assert len(hits) == 3  # ...and still returned
