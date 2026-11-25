from collections.abc import AsyncIterator

import pytest
from fakes import FakeEmbedder
from qdrant_client import AsyncQdrantClient

from qacopilot.config import Settings
from qacopilot.ingest import DocType, PageText, ParsedDocument, chunk_document
from qacopilot.retrieval import QdrantStore


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source="t.pdf", pages=[PageText(page=1, text=text, method="text")])


SOP = _doc(
    "SOP-014 Cooking\n\n1. Release\nCooked duck must reach 75C core temperature before release."
)
SPEC = _doc(
    "Sauce specification\n\n1. Shelf life\nThe orange sauce keeps for 12 months in ambient storage."
)


@pytest.fixture
async def store() -> AsyncIterator[QdrantStore]:
    s = QdrantStore(FakeEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
    await s.ensure_collection()
    await s.upsert(chunk_document(SOP, DocType.SOP, "sop14"))
    await s.upsert(chunk_document(SPEC, DocType.SUPPLIER_SPEC, "spec1"))
    yield s
    await s.aclose()


async def test_search_finds_the_relevant_chunk_with_citation_fields(store: QdrantStore) -> None:
    hits = await store.search("what temperature must cooked duck reach", limit=2)
    top = hits[0].chunk
    assert top.doc_id == "sop14"
    assert top.page == 1
    assert top.section == "1. Release"
    assert hits[0].score >= hits[1].score


async def test_filters_by_doc_type_and_doc_id(store: QdrantStore) -> None:
    only_spec = await store.search("duck temperature", doc_type=DocType.SUPPLIER_SPEC)
    assert {h.chunk.doc_id for h in only_spec} == {"spec1"}
    only_sop = await store.search("shelf life", doc_id="sop14")
    assert {h.chunk.doc_id for h in only_sop} == {"sop14"}


async def test_replace_document_removes_stale_chunks(store: QdrantStore) -> None:
    before = await store.count()
    long = _doc("SOP-014 Cooking\n\n1. A\nalpha text\n\n2. B\nbeta text\n\n3. C\ngamma text")
    await store.replace_document("sop14", chunk_document(long, DocType.SOP, "sop14", max_chars=40))
    await store.replace_document("sop14", chunk_document(SOP, DocType.SOP, "sop14"))
    assert await store.count() == before
    assert not await store.search("alpha beta gamma", doc_id="sop14") or all(
        "alpha" not in h.chunk.text for h in await store.search("alpha", doc_id="sop14")
    )


async def test_delete_document(store: QdrantStore) -> None:
    await store.delete_document("spec1")
    assert await store.search("shelf life", doc_id="spec1") == []
