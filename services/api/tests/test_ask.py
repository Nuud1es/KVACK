import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel
from qacopilot.config import Settings
from qacopilot.repositories import DocumentRepository
from qacopilot.retrieval import QdrantStore, Retriever
from qdrant_client import AsyncQdrantClient
from stubs import ScriptedLLM, StubEmbedder, StubReranker

from qa_api.main import (
    app,
    get_llm,
    get_repository,
    get_repository_or_none,
    get_retriever,
    get_settings,
    get_store,
)
from qa_api.sse import PING, stream_events

SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples"


@pytest.fixture
async def client(repo: DocumentRepository) -> AsyncIterator[httpx.AsyncClient]:
    store = QdrantStore(StubEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
    await store.ensure_collection()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_repository_or_none] = lambda: repo
    app.dependency_overrides[get_retriever] = lambda: Retriever(
        store, StubReranker(), min_score=0.0
    )
    llm = ScriptedLLM("Cooked duck needs 75C [1].")
    app.state.test_llm = llm
    app.dependency_overrides[get_llm] = lambda: llm
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    app.dependency_overrides.clear()
    await store.aclose()


def _events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


async def _ingest(client: httpx.AsyncClient, name: str) -> httpx.Response:
    files = {"file": (name, (SAMPLES / name).read_bytes(), "application/pdf")}
    return await client.post("/ingest", files=files)


async def test_due_date_question_is_answered_from_the_certificates_table(
    client: httpx.AsyncClient, repo: DocumentRepository
) -> None:
    from datetime import date

    from qacopilot.extraction import CalibrationCert
    from qacopilot.ingest import DocType, PageText, ParsedDocument

    async def add(doc_id: str, calibrated: date, due: date, number: str) -> None:
        page = PageText(
            page=1, text=f"CERTIFICATE OF CALIBRATION {number} serial TP-20931", method="text"
        )
        await repo.save_document(
            doc_id,
            f"{doc_id}.pdf",
            DocType.CALIBRATION_CERT,
            ParsedDocument(source=doc_id, pages=[page]),
            1,
        )
        cert = CalibrationCert(
            instrument_serial="TP-20931",
            instrument_description="Digital probe thermometer",
            certificate_number=number,
            calibration_date=calibrated,
            next_due=due,
            result="pass",
        )
        await repo.save_extraction(
            doc_id, DocType.CALIBRATION_CERT, cert, prompt_version="v", model="m"
        )

    await add("stale", date(2026, 3, 14), date(2027, 3, 14), "CAL-2026-0412")
    await add("current", date(2026, 11, 10), date(2027, 11, 10), "CAL-2026-0640")

    r = await client.post("/ask", json={"question": "When is the probe certificate TP-20931 due?"})
    events = _events(r.text)
    done = events[-1][1]
    assert events[-1][0] == "done" and done["via"] == "certificates" and done["answered"]
    assert "2027-11-10" in done["answer"] and "superseded" in done["answer"]
    assert events[0][1]["citations"][0]["filename"] == "current.pdf"
    assert app.state.test_llm.seen_request_ids == []  # the model was never called

    # a question about something else still goes to document search
    r = await client.post("/ask", json={"question": "what core temperature must cooked duck reach"})
    assert [n for n, _ in _events(r.text)][0] in ("sources", "no_answer")


async def test_ingest_then_ask_streams_cited_answer(client: httpx.AsyncClient) -> None:
    r = await _ingest(client, "sop_cooking_temperature.pdf")
    assert r.status_code == 200
    assert r.json()["doc_type"] == "sop" and r.json()["chunks"] >= 1

    r = await client.post(
        "/ask", json={"question": "what core temperature must cooked duck reach before release"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert len(r.headers["x-request-id"]) == 32  # returned so the answer can be rated
    # The LLM call must run under that same id, or a rating could never find its trace.
    assert app.state.test_llm.seen_request_ids == [r.headers["x-request-id"]]
    events = _events(r.text)
    names = [n for n, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert "token" in names
    assert events[0][1]["citations"][0]["filename"] == "sop_cooking_temperature.pdf"
    assert events[-1][1]["cited"] == [1]


async def test_ask_with_nothing_indexed_says_no_answer(client: httpx.AsyncClient) -> None:
    r = await client.post("/ask", json={"question": "anything at all"})
    assert _events(r.text) == [
        ("no_answer", {"event": "no_answer", "reason": "no_relevant_documents"})
    ]


async def test_ask_validates_the_request(client: httpx.AsyncClient) -> None:
    assert (await client.post("/ask", json={"question": ""})).status_code == 422
    assert (await client.post("/ask", json={"question": "x", "limit": 99})).status_code == 422


async def test_ingest_rejects_non_pdf(client: httpx.AsyncClient) -> None:
    files = {"file": ("evil.pdf", b"not a pdf at all", "application/pdf")}
    assert (await client.post("/ingest", files=files)).status_code == 422


async def test_ingest_reports_unreadable_pdf_as_422(client: httpx.AsyncClient) -> None:
    files = {"file": ("broken.pdf", b"%PDF-1.4 garbage", "application/pdf")}
    assert (await client.post("/ingest", files=files)).status_code == 422


@pytest.fixture
def api_key() -> Iterator[None]:
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret")
    yield
    app.dependency_overrides.pop(get_settings, None)


@pytest.mark.usefixtures("api_key")
async def test_endpoints_require_the_api_key_when_configured(client: httpx.AsyncClient) -> None:
    assert (await client.post("/ask", json={"question": "x"})).status_code == 401
    ok = await client.post("/ask", json={"question": "x"}, headers={"x-api-key": "secret"})
    assert ok.status_code == 200
    assert (await _ingest(client, "sop_cooking_temperature.pdf")).status_code == 401


async def test_stream_sends_pings_while_quiet_and_reports_errors() -> None:
    class Tok(BaseModel):
        event: str = "token"

    async def slow() -> AsyncIterator[BaseModel]:
        await asyncio.sleep(0.25)
        yield Tok()
        raise RuntimeError("boom")

    out = [c async for c in stream_events(slow(), keepalive_s=0.1)]
    assert out.count(PING) >= 2
    assert out[-2].startswith("event: token")
    assert out[-1].startswith("event: error") and "boom" not in out[-1]


async def test_unreachable_llm_gets_a_specific_message() -> None:
    import openai

    async def down() -> AsyncIterator[BaseModel]:
        raise openai.APIConnectionError(request=httpx_request())
        yield  # pragma: no cover - makes this an async generator

    out = [c async for c in stream_events(down(), keepalive_s=0.1)]
    assert out[-1].startswith("event: error") and "could not be reached" in out[-1]
    # the client dispatches on the "event" field inside the data, so it must be there
    assert json.loads(out[-1].split("data: ", 1)[1])["event"] == "error"


def httpx_request() -> object:
    import httpx2  # the openai SDK's bundled httpx fork

    return httpx2.Request("POST", "http://llm.invalid/v1/chat/completions")


async def test_a_photo_is_accepted_and_read_by_ocr(client: httpx.AsyncClient) -> None:
    import pymupdf

    src = pymupdf.open()
    page = src.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 545, 400),
        "CERTIFICATE OF CALIBRATION\nSerial Number: ZX-9981\nNext calibration due: 2027-01-01",
        fontsize=14,
    )
    photo = src[0].get_pixmap(dpi=150).tobytes("jpeg")
    r = await client.post("/ingest", files={"file": ("phone.jpg", photo, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["doc_type"], body["pages"], body["ocr_pages"]) == ("calibration_cert", 1, 1)


async def test_other_file_types_are_still_rejected(client: httpx.AsyncClient) -> None:
    for name, data in [("a.exe", b"MZ\x90\x00"), ("a.gif", b"GIF89a"), ("a.txt", b"hello world")]:
        r = await client.post("/ingest", files={"file": (name, data, "application/octet-stream")})
        assert r.status_code == 422, name


async def test_ingest_reports_who_classified_the_document(client: httpx.AsyncClient) -> None:
    r = await _ingest(client, "sop_cooking_temperature.pdf")
    body = r.json()
    assert (body["doc_type"], body["classified_by"]) == ("sop", "rules")


async def test_a_wrong_type_can_be_corrected_and_the_document_is_re_chunked(
    client: httpx.AsyncClient,
) -> None:
    doc_id = (await _ingest(client, "sop_cooking_temperature.pdf")).json()["doc_id"]
    r = await client.post(f"/documents/{doc_id}/reclassify", json={"doc_type": "haccp"})
    assert r.status_code == 200 and r.json()["doc_type"] == "haccp"
    docs = (await client.get("/documents")).json()
    assert [(d["doc_type"], d["classified_by"]) for d in docs] == [("haccp", "user")]

    assert (
        await client.post(f"/documents/{doc_id}/reclassify", json={"doc_type": "nonsense"})
    ).status_code == 422
    assert (
        await client.post("/documents/nope/reclassify", json={"doc_type": "sop"})
    ).status_code == 404
