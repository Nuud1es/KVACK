"""Classification in the ingest pipeline: rules, the LLM fallback, a person's choice, and fixing a
wrong label afterwards."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pymupdf
import pytest
from fakes import FakeEmbedder
from qdrant_client import AsyncQdrantClient

from qacopilot.config import Settings
from qacopilot.ingest import DocType
from qacopilot.ingest.pipeline import ingest_pdf, reclassify_document
from qacopilot.llm import BaseLLMProvider, LLMResponse
from qacopilot.repositories import DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository
from qacopilot.retrieval import QdrantStore


def pdf(text: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    return bytes(doc.tobytes())


VAGUE = pdf("Notes about a product from a supplier, with no wording the rules know.\n" * 6)
CERT = pdf("CERTIFICATE OF CALIBRATION\nSerial Number: TP-1\nNext calibration due: 2027-01-01\n")


class Canned(BaseLLMProvider):
    def __init__(self, doc_type: str) -> None:
        self.doc_type, self.calls = doc_type, 0

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=json.dumps({"doc_type": self.doc_type, "reason": "it says so"}))

    async def health(self) -> bool:
        return True


@pytest.fixture
async def env() -> AsyncIterator[tuple[QdrantStore, DocumentRepository]]:
    try:
        async with temporary_repository() as repo:
            store = QdrantStore(FakeEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
            await store.ensure_collection()
            yield store, repo
            await store.aclose()
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


async def test_unsure_rules_hand_over_to_the_llm_and_the_result_says_so(env: Any) -> None:
    store, repo = env
    llm = Canned("supplier_spec")
    r = await ingest_pdf(VAGUE, "notes.pdf", store, repo=repo, llm=llm)
    assert (r.doc_type, r.classified_by, r.reason) == (DocType.SUPPLIER_SPEC, "llm", "it says so")
    stored = (await repo.list_documents())[0]
    assert (stored.doc_type, stored.classified_by) == ("supplier_spec", "llm")


async def test_clear_documents_do_not_use_the_llm(env: Any) -> None:
    store, repo = env
    llm = Canned("other")
    r = await ingest_pdf(CERT, "c.pdf", store, repo=repo, llm=llm)
    assert (r.doc_type, r.classified_by) == (DocType.CALIBRATION_CERT, "rules") and llm.calls == 0


async def test_a_persons_choice_beats_both(env: Any) -> None:
    store, repo = env
    llm = Canned("supplier_spec")
    r = await ingest_pdf(VAGUE, "n.pdf", store, repo=repo, llm=llm, doc_type=DocType.SOP)
    assert (r.doc_type, r.classified_by, r.confidence) == (DocType.SOP, "user", 1.0)
    assert llm.calls == 0


async def test_without_an_llm_an_unsure_document_stays_other(env: Any) -> None:
    store, repo = env
    r = await ingest_pdf(VAGUE, "n.pdf", store, repo=repo)
    assert (r.doc_type, r.classified_by) == (DocType.OTHER, "rules")


async def test_reclassifying_rebuilds_the_chunks_and_drops_the_old_extraction(env: Any) -> None:
    from qacopilot.extraction import CalibrationCert

    store, repo = env
    r = await ingest_pdf(CERT, "c.pdf", store, repo=repo)
    cert = CalibrationCert(
        instrument_serial="TP-1",
        calibration_date="2026-01-01",
        next_due="2027-01-01",
        result="pass",
    )  # type: ignore[arg-type]
    await repo.save_extraction(r.doc_id, r.doc_type, cert, prompt_version="t", model="t")
    assert await repo.get_extraction(r.doc_id) is not None

    chunks = await reclassify_document(r.doc_id, DocType.OTHER, store, repo)
    assert chunks == 1
    doc = (await repo.list_documents())[0]
    assert (doc.doc_type, doc.classified_by, doc.extracted) == ("other", "user", False)
    assert await repo.get_extraction(r.doc_id) is None
    assert await repo.certificates_due(10_000) == []
    hits = await store.search("calibration serial", doc_id=r.doc_id)
    assert hits and hits[0].chunk.doc_type is DocType.OTHER  # the index changed too


async def test_reclassifying_an_unknown_document_is_reported(env: Any) -> None:
    store, repo = env
    assert await reclassify_document("nope", DocType.SOP, store, repo) is None


async def test_a_disagreeing_llm_is_recorded_as_an_alternative_for_a_person_to_check(
    env: Any,
) -> None:
    store, repo = env
    weak = pdf("Date Reported: 1/1/2025\nBatch No: 12\nResult 5 mg/kg\n")
    r = await ingest_pdf(weak, "w.pdf", store, repo=repo, llm=Canned("supplier_spec"))
    assert (r.doc_type, r.classified_by, r.alternative) == (
        DocType.COA,
        "rules",
        DocType.SUPPLIER_SPEC,
    )
    stored = (await repo.list_documents())[0]
    assert (stored.doc_type, stored.alternative_type) == ("coa", "supplier_spec")

    await reclassify_document(r.doc_id, DocType.COA, store, repo)  # a person confirms the type
    assert (await repo.list_documents())[0].alternative_type is None  # the flag is cleared
