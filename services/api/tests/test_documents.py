import json
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from qacopilot.config import Settings
from qacopilot.extraction import PROMPT_VERSION as EXTRACT_PROMPT_VERSION
from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.llm import BaseLLMProvider, LLMResponse
from qacopilot.repositories import DocumentRepository
from qacopilot.retrieval import QdrantStore
from qdrant_client import AsyncQdrantClient
from stubs import StubEmbedder

from qa_api.main import app, get_llm, get_repository, get_settings, get_store

SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples"
TODAY = date.today()


class JsonLLM(BaseLLMProvider):
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self.replies.pop(0))

    async def health(self) -> bool:
        return True


def cert_json(serial: str, due: date) -> str:
    return json.dumps(
        {
            "instrument_serial": serial,
            "calibration_date": (due - timedelta(days=365)).isoformat(),
            "next_due": due.isoformat(),
            "result": "pass",
        }
    )


@pytest.fixture
async def client(repo: DocumentRepository) -> AsyncIterator[httpx.AsyncClient]:
    store = QdrantStore(StubEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
    await store.ensure_collection()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_repository] = lambda: repo
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    app.dependency_overrides.clear()
    await store.aclose()


async def _ingest(client: httpx.AsyncClient, name: str) -> str:
    files = {"file": (name, (SAMPLES / name).read_bytes(), "application/pdf")}
    r = await client.post("/ingest", files=files)
    assert r.status_code == 200, r.text
    return str(r.json()["doc_id"])


async def test_extract_stores_result_and_certificates_are_listed(
    client: httpx.AsyncClient,
) -> None:
    doc_id = await _ingest(client, "calibration_cert_probe.pdf")
    app.dependency_overrides[get_llm] = lambda: JsonLLM(
        cert_json("TP-20931", TODAY + timedelta(days=10))
    )
    r = await client.post(f"/documents/{doc_id}/extract")
    assert r.status_code == 200
    assert r.json()["data"]["instrument_serial"] == "TP-20931"

    stored = await client.get(f"/documents/{doc_id}/extraction")
    assert stored.json()["prompt_version"] == EXTRACT_PROMPT_VERSION  # whichever is current

    due = (await client.get("/certificates", params={"expiring_within": "30d"})).json()
    assert [(c["instrument_serial"], c["status"], c["days_left"]) for c in due] == [
        ("TP-20931", "due_soon", 10)
    ]
    assert (await client.get("/certificates", params={"expiring_within": "5"})).json() == []


async def test_certificates_validates_the_window(client: httpx.AsyncClient) -> None:
    for bad in ("soon", "-5d", "30 days", ""):
        r = await client.get("/certificates", params={"expiring_within": bad})
        assert r.status_code == 422, bad


async def test_extract_errors(client: httpx.AsyncClient) -> None:
    assert (await client.post("/documents/nope/extract")).status_code == 404
    sop = await _ingest(client, "sop_cooking_temperature.pdf")
    assert (await client.post(f"/documents/{sop}/extract")).status_code == 422  # no SOP schema
    cert = await _ingest(client, "calibration_cert_probe.pdf")
    app.dependency_overrides[get_llm] = lambda: JsonLLM("{}", "{}", "{}")
    assert (await client.post(f"/documents/{cert}/extract")).status_code == 502
    failed = (await client.get(f"/documents/{cert}/extraction")).json()
    assert (failed["status"], failed["done"], failed["failed"], failed["data"]) == (
        "failed",
        0,
        1,
        None,
    )


async def test_coa_check_flags_out_of_spec_results(client: httpx.AsyncClient) -> None:
    coa = await _ingest(client, "coa_duck_breast.pdf")
    spec = await _ingest(client, "supplier_spec_duck_breast.pdf")
    coa_json = {
        "batch": "B2611-07",
        "results": [
            {"test": "Total viable count", "result": "2.2E+06", "unit": "cfu/g"},
            {"test": "Enterobacteriaceae", "result": "<10", "unit": "cfu/g"},
            {"test": "Salmonella", "result": "Not detected in 25g"},
        ],
    }
    spec_json = {
        "product": "Duck breast fillet",
        "micro_limits": [
            {"test": "Total viable count", "limit": "1.0E+05 cfu/g"},
            {"test": "Enterobacteriaceae", "limit": "1.0E+03 cfu/g"},
            {"test": "Salmonella", "limit": "Not detected in 25g"},
        ],
    }
    app.dependency_overrides[get_llm] = lambda: JsonLLM(json.dumps(coa_json), json.dumps(spec_json))
    assert (await client.post(f"/documents/{coa}/extract")).status_code == 200
    assert (await client.post(f"/documents/{spec}/extract")).status_code == 200

    r = await client.get("/compliance/coa-check", params={"coa": coa, "spec": spec})
    body = r.json()
    assert body["overall"] == "fail"
    assert body["coa_doc_id"] == coa and body["spec_doc_id"] == spec
    verdicts = {row["test"]: row["verdict"] for row in body["rows"]}
    assert verdicts == {
        "Total viable count": "fail",
        "Enterobacteriaceae": "pass",
        "Salmonella": "pass",
    }

    swapped = await client.get("/compliance/coa-check", params={"coa": spec, "spec": coa})
    assert swapped.status_code == 422
    missing = await client.get("/compliance/coa-check", params={"coa": coa, "spec": "nope"})
    assert missing.status_code == 404


SPEC_JSON = {
    "product": "Chicken thigh fillet",
    "micro_limits": [{"test": "Total viable count", "limit": "1.0E+05 cfu/g"}],
}


async def _confirm_flagged(client: httpx.AsyncClient) -> None:
    """These tests feed partial extractions (no supplier, no storage): a person confirms them, so
    the verdict under test is not capped by the 'unconfirmed reading' rule."""
    for item in (await client.get("/review/documents")).json():
        url = f"/documents/{item['doc_id']}/extractions/{item['start_page']}/review"
        assert (await client.post(url, json={})).status_code == 200


async def _check_scanned_coa(client: httpx.AsyncClient, tvc: str) -> dict[str, object]:
    coa = await _ingest(client, "coa_chicken_b2701_scanned_skewed.pdf")  # read by OCR
    spec = await _ingest(client, "supplier_spec_duck_breast.pdf")
    coa_json = {"batch": "B2701-02", "results": [{"test": "Total viable count", "result": tvc}]}
    app.dependency_overrides[get_llm] = lambda: JsonLLM(json.dumps(coa_json), json.dumps(SPEC_JSON))
    extracted = await client.post(f"/documents/{coa}/extract")
    assert extracted.json()["ocr_pages"] == 1  # the UI uses this to warn
    await client.post(f"/documents/{spec}/extract")
    await _confirm_flagged(client)
    return dict(
        (await client.get("/compliance/coa-check", params={"coa": coa, "spec": spec})).json()
    )


async def test_a_result_built_on_ocr_text_is_never_a_clean_pass(client: httpx.AsyncClient) -> None:
    body = await _check_scanned_coa(client, "4.1E+03")  # well inside the limit
    assert body["rows"][0]["verdict"] == "pass"  # the figure itself is fine...
    assert body["overall"] == "review"  # ...but it came from OCR, so a person must confirm it
    assert "read by OCR" in body["warnings"][0]


async def test_a_failing_result_on_ocr_text_still_fails(client: httpx.AsyncClient) -> None:
    body = await _check_scanned_coa(client, "4.1E+07")
    assert body["overall"] == "fail"
    assert len(body["warnings"]) == 1


async def test_text_pdfs_carry_no_ocr_warning(client: httpx.AsyncClient) -> None:
    coa = await _ingest(client, "coa_duck_breast.pdf")
    app.dependency_overrides[get_llm] = lambda: JsonLLM(
        json.dumps({"batch": "B1", "results": [{"test": "Total viable count", "result": "100"}]})
    )
    out = await client.post(f"/documents/{coa}/extract")
    assert out.json()["ocr_pages"] == 0


def bundle_pdf(*serials: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for serial in serials:
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 545, 400),
            "CERTIFICATE OF CALIBRATION\nInstrument: Probe\n"
            f"Serial Number: {serial}\nTraceable to national standards (UKAS)\n",
            fontsize=12,
        )
    return bytes(doc.tobytes())


async def _upload(client: httpx.AsyncClient, name: str, data: bytes) -> str:
    r = await client.post("/ingest", files={"file": (name, data, "application/pdf")})
    assert r.status_code == 200, r.text
    return str(r.json()["doc_id"])


async def _wait_for(client: httpx.AsyncClient, doc_id: str, status: str) -> dict[str, object]:
    import asyncio

    for _ in range(100):
        body = (await client.get(f"/documents/{doc_id}/extraction")).json()
        if body["status"] == status:
            return dict(body)
        await asyncio.sleep(0.05)
    raise AssertionError(f"never reached {status}: {body}")


async def test_a_bundle_extracts_in_the_background_with_progress(client: httpx.AsyncClient) -> None:
    doc_id = await _upload(client, "bundle.pdf", bundle_pdf("TP-1001", "TP-1002", "TP-1003"))
    app.dependency_overrides[get_llm] = lambda: JsonLLM(
        cert_json("TP-1001", TODAY + timedelta(days=10)),
        cert_json("TP-1002", TODAY + timedelta(days=20)),
        cert_json("TP-1003", TODAY + timedelta(days=300)),
    )
    r = await client.post(f"/documents/{doc_id}/extract")
    assert r.status_code == 202  # accepted: it keeps working after the response
    assert (r.json()["status"], r.json()["total"]) == ("running", 3)

    done = await _wait_for(client, doc_id, "done")
    assert (done["total"], done["done"], done["failed"]) == (3, 3, 0)
    serials = [s["data"]["instrument_serial"] for s in done["segments"]]  # type: ignore[index]
    assert serials == ["TP-1001", "TP-1002", "TP-1003"]
    assert [s["start_page"] for s in done["segments"]] == [1, 2, 3]  # type: ignore[attr-defined]

    due = (await client.get("/certificates", params={"expiring_within": "60d"})).json()
    assert [(c["instrument_serial"], c["start_page"]) for c in due] == [
        ("TP-1001", 1),
        ("TP-1002", 2),
    ]


async def test_a_bundle_with_one_unreadable_certificate_still_keeps_the_rest(
    client: httpx.AsyncClient,
) -> None:
    doc_id = await _upload(client, "bundle.pdf", bundle_pdf("TP-2001", "TP-2002"))
    app.dependency_overrides[get_llm] = lambda: JsonLLM(
        cert_json("TP-2001", TODAY + timedelta(days=5)), "junk", "junk", "junk"
    )
    await client.post(f"/documents/{doc_id}/extract")
    done = await _wait_for(client, doc_id, "done")
    assert (done["done"], done["failed"]) == (1, 1)
    assert "pages 2-2" in str(done["error"])


async def test_odd_dates_in_an_extracted_certificate_are_flagged(client: httpx.AsyncClient) -> None:
    doc_id = await _upload(client, "bundle.pdf", bundle_pdf("TP-3001", "TP-3002"))
    odd = json.dumps(
        {"instrument_serial": "TP-3002", "calibration_date": "2026-01-01",
         "next_due": "2029-01-01", "result": "pass"}
    )  # fmt: skip
    app.dependency_overrides[get_llm] = lambda: JsonLLM(
        cert_json("TP-3001", TODAY + timedelta(days=9)), odd
    )
    await client.post(f"/documents/{doc_id}/extract")
    done = await _wait_for(client, doc_id, "done")
    warnings = [s["warnings"] for s in done["segments"]]  # type: ignore[attr-defined]
    assert warnings[0] == [] and "interval" in warnings[1][0]


async def test_only_one_extraction_of_a_document_runs_at_a_time(client: httpx.AsyncClient) -> None:
    import asyncio

    gate = asyncio.Event()

    class Slow(JsonLLM):
        async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
            await gate.wait()
            return await super().chat(messages)

    doc_id = await _upload(client, "bundle.pdf", bundle_pdf("TP-4001", "TP-4002"))
    app.dependency_overrides[get_llm] = lambda: Slow(
        cert_json("TP-4001", TODAY + timedelta(days=9)),
        cert_json("TP-4002", TODAY + timedelta(days=9)),
    )
    assert (await client.post(f"/documents/{doc_id}/extract")).status_code == 202
    assert (await client.post(f"/documents/{doc_id}/extract")).status_code == 409
    gate.set()
    await _wait_for(client, doc_id, "done")
    again = await client.post(f"/documents/{doc_id}/extract")
    assert again.status_code in (200, 202)  # again is fine
    # let a re-run finish: the fixture drops the schema afterwards, and a job still writing to
    # it deadlocks the DROP SCHEMA
    await _wait_for(client, doc_id, "done")


async def test_a_restart_marks_an_interrupted_extraction_failed(
    client: httpx.AsyncClient, repo: DocumentRepository
) -> None:
    doc_id = await _upload(client, "bundle.pdf", bundle_pdf("TP-5001", "TP-5002"))
    await repo.start_job(doc_id, 2)  # as if the process died mid-run
    assert (await client.get(f"/documents/{doc_id}/extraction")).json()["status"] == "running"
    assert await repo.fail_stale_jobs() == 1
    after = (await client.get(f"/documents/{doc_id}/extraction")).json()
    assert after["status"] == "failed" and "restart" in after["error"]


def _limited_coa(batch: str, report: str, tvc: str) -> str:
    return json.dumps(
        {
            "product": "Duck breast fillet",
            "batch": batch,
            "report_date": report,
            "results": [
                {"test": "Total viable count", "result": tvc, "limit": "< 1.0E+05 cfu/g"},
                {"test": "Salmonella", "result": "Not detected in 25g"},
            ],
        }
    )


async def _extract(client: httpx.AsyncClient, doc: str, reply: str) -> None:
    app.dependency_overrides[get_llm] = lambda: JsonLLM(reply)
    assert (await client.post(f"/documents/{doc}/extract")).status_code == 200
    await _confirm_flagged(client)


async def test_coa_check_finds_the_spec_and_the_revision_in_force(
    client: httpx.AsyncClient,
) -> None:
    v1 = await _ingest(client, "supplier_spec_duck_breast.pdf")
    v2 = await _ingest(
        client, "supplier_spec_duck_breast_v2.pdf"
    )  # "Revision 2 ... Effective: 2026-10-01"
    limits = [
        {"test": "Total viable count", "limit": "1.0E+05 cfu/g"},
        {"test": "Salmonella", "limit": "Not detected in 25g"},
    ]
    spec = {"supplier": "Example Poultry Ltd", "product": "Duck breast fillet skin on"}
    await _extract(client, v1, json.dumps({**spec, "micro_limits": limits}))
    await _extract(
        client, v2, json.dumps({**spec, "micro_limits": []})
    )  # rev 2 only says "unchanged"

    new = await _ingest(client, "coa_duck_breast_b2613.pdf")
    await _extract(client, new, _limited_coa("B2613-01", "2026-11-05", "2.4E+05"))
    body = (await client.get("/compliance/coa-check", params={"coa": new})).json()
    assert body["pairing"] == "matched" and body["spec_doc_id"] == v2
    assert body["spec_revision"] == 2 and body["spec_effective"] == "2026-10-01"
    # revision 2 says "Microbiological limits are unchanged", so revision 1's limits apply
    assert body["overall"] == "fail" and body["limits_from"] == "supplier_spec_duck_breast.pdf"
    assert {r["limit_source"] for r in body["rows"]} == {"spec"}
    assert any("limits are unchanged" in w for w in body["warnings"])

    old = await _ingest(client, "coa_duck_breast.pdf")  # dated before revision 2 took effect
    await _extract(client, old, _limited_coa("B2611-07", "2026-09-20", "3.0E+03"))
    body = (await client.get("/compliance/coa-check", params={"coa": old})).json()
    assert body["spec_doc_id"] == v1 and body["overall"] == "pass"
    assert any("before revision 2 took effect" in w for w in body["warnings"])

    # naming the spec still works, and is reported as given
    given = (await client.get("/compliance/coa-check", params={"coa": old, "spec": v2})).json()
    assert given["pairing"] == "given" and given["spec_doc_id"] == v2


async def test_coa_check_without_a_matching_or_with_several_specs(
    client: httpx.AsyncClient,
) -> None:
    coa = await _ingest(client, "coa_duck_breast.pdf")
    await _extract(client, coa, _limited_coa("B2611-07", "2026-09-20", "3.0E+03"))
    none = await client.get("/compliance/coa-check", params={"coa": coa})
    assert none.status_code == 404 and "No extracted supplier spec" in none.json()["detail"]

    a = await _ingest(client, "supplier_spec_duck_breast.pdf")
    b = await _ingest(client, "supplier_spec_stuffing.pdf")
    await _extract(
        client,
        a,
        json.dumps(
            {"supplier": "Example Poultry Ltd", "product": "Duck breast fillet", "micro_limits": []}
        ),
    )
    await _extract(
        client,
        b,
        json.dumps(
            {"supplier": "Another Ltd", "product": "Duck breast fillet", "micro_limits": []}
        ),
    )
    many = await client.get("/compliance/coa-check", params={"coa": coa})
    assert many.status_code == 409 and "Candidates:" in many.json()["detail"]


async def _poor_scan_cert(client: httpx.AsyncClient, repo: DocumentRepository) -> str:
    """A certificate whose single page was read by OCR with low confidence."""
    from qacopilot.extraction import CalibrationCert
    from qacopilot.ingest import DocType, PageText, ParsedDocument

    page = PageText(
        page=1, text="CERTIFICATE ... Next due: 2029-03-14", method="ocr", confidence=58.0
    )
    doc = ParsedDocument(source="scan.pdf", pages=[page])
    await repo.save_document("scan1", "scan.pdf", DocType.CALIBRATION_CERT, doc, 1)
    cert = CalibrationCert(
        instrument_serial="TP-7",
        calibration_date=date(2026, 3, 14),
        next_due=date(2029, 3, 14),  # OCR read 2027 as 2029
        result="pass",
    )
    await repo.save_extraction(
        "scan1", DocType.CALIBRATION_CERT, cert, prompt_version="v", model="m"
    )
    return "scan1"


async def test_a_poor_scan_goes_to_the_review_queue_and_a_correction_clears_it(
    client: httpx.AsyncClient, repo: DocumentRepository
) -> None:
    doc = await _poor_scan_cert(client, repo)
    queue = (await client.get("/certificates/review")).json()
    assert [(i["doc_id"], i["instrument_serial"]) for i in queue] == [(doc, "TP-7")]
    assert "poor scan" in queue[0]["reason"] and queue[0]["min_ocr_confidence"] == 58.0

    seg = (await client.get(f"/documents/{doc}/extraction")).json()["segments"][0]
    assert seg["needs_review"] and "poor scan" in seg["review_reason"]

    bad = await client.post(
        f"/documents/{doc}/certificates/1/review", json={"corrections": {"next_due": "soon"}}
    )
    assert bad.status_code == 422
    locked = await client.post(
        f"/documents/{doc}/certificates/1/review", json={"corrections": {"result": "fail"}}
    )
    assert locked.status_code == 422 and "cannot be corrected" in locked.json()["detail"]
    assert (await client.post("/documents/nope/certificates/1/review", json={})).status_code == 404

    ok = await client.post(
        f"/documents/{doc}/certificates/1/review",
        json={"corrections": {"next_due": "2027-03-14"}, "note": "checked the paper copy"},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["needs_review"] is False and body["next_due"] == "2027-03-14"
    assert body["verified_by"] == "dev"  # with no keys configured, every caller is "dev"
    assert (await client.get("/certificates/review")).json() == []
    seg = (await client.get(f"/documents/{doc}/extraction")).json()["segments"][0]
    assert (
        not seg["needs_review"]
        and seg["data"]["next_due"] == "2027-03-14"
        and seg["verified_by"] == "dev"
    )

    (event,) = (await client.get(f"/documents/{doc}/certificates/1/reviews")).json()
    assert event["action"] == "correct" and event["before"]["next_due"] == "2029-03-14"
    assert event["note"] == "checked the paper copy"


@pytest.fixture
def api_key() -> Iterator[None]:
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="secret")
    yield
    app.dependency_overrides.pop(get_settings, None)


@pytest.mark.usefixtures("api_key")
async def test_review_endpoints_require_the_api_key(client: httpx.AsyncClient) -> None:
    assert (await client.get("/certificates/review")).status_code == 401
    assert (await client.post("/documents/x/certificates/1/review", json={})).status_code == 401
    assert (await client.get("/documents/x/certificates/1/reviews")).status_code == 401


async def test_review_queue_and_confirm_for_a_report(
    client: httpx.AsyncClient, repo: DocumentRepository
) -> None:
    from qacopilot.extraction import CoA, TestResult

    text = "Product: Duck\nBatch: B1\nTotal viable count 100\nSalmonella Not detected\n"
    await repo.save_document(
        "c1",
        "coa.pdf",
        DocType.COA,
        ParsedDocument(source="x", pages=[PageText(page=1, text=text, method="text")]),
        1,
    )
    await repo.save_extraction(  # the Salmonella row was skipped
        "c1",
        DocType.COA,
        CoA(
            product="Duck",
            batch="B1",
            results=[TestResult(test="Total viable count", result="100")],
        ),
        prompt_version="v",
        model="m",
    )
    queue = (await client.get("/review/documents")).json()
    assert [i["doc_id"] for i in queue] == ["c1"] and "Salmonella" in queue[0]["reason"]
    bad = await client.post(
        "/documents/c1/extractions/1/review", json={"corrections": {"results": []}}
    )
    assert bad.status_code == 422
    ok = await client.post("/documents/c1/extractions/1/review", json={"note": "checked"})
    assert ok.status_code == 200 and ok.json()["verified_by"] == "dev"
    assert (await client.get("/review/documents")).json() == []
    assert (await client.post("/documents/zz/extractions/1/review", json={})).status_code == 404


async def test_export_specs_returns_a_csv_and_refuses_what_it_cannot_export(
    client: httpx.AsyncClient,
) -> None:
    ok = await client.post("/export/specs", json={"query": {"intent": "count"}})
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("text/csv")
    assert 'filename="specs-' in ok.headers["content-disposition"]
    assert ok.content.startswith(b"\xef\xbb\xbfNAV code,Product,Customer")  # BOM for Excel
    compare = await client.post(
        "/export/specs",
        json={"query": {"intent": "compare", "filter": {"product": "x"}, "compare_with": "y"}},
    )
    assert compare.status_code == 422 and "comparison" in compare.json()["detail"]
    unknown = await client.post(
        "/export/specs", json={"query": {"intent": "count", "filter": {"customer": "Nobody"}}}
    )
    assert unknown.status_code == 422 and "No customer called" in unknown.json()["detail"]
