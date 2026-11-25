"""Extracting every certificate of a bundle: progress, partial failure, storage, reruns."""

import json
from collections.abc import AsyncIterator, Sequence
from datetime import date, timedelta
from typing import Any

import pytest

from qacopilot.extraction import (
    CalibrationCert,
    UnknownDocument,
    UnsupportedDocType,
    certificate_warnings,
    plan_segments,
    run_extraction,
)
from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.llm import BaseLLMProvider, LLMResponse
from qacopilot.repositories import DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository

TODAY = date.today()


def page_text(serial: str, title: bool = True) -> str:
    head = "CERTIFICATE OF CALIBRATION\n" if title else "Calibration report\n"
    return (
        f"{head}Instrument: Probe\nSerial Number: {serial}\n"
        "Traceable to national standards (UKAS)\n"
    )


def reply(serial: str, due_in: int = 100) -> str:
    due = TODAY + timedelta(days=due_in)
    return json.dumps(
        {
            "instrument_serial": serial,
            "calibration_date": (due - timedelta(days=365)).isoformat(),
            "next_due": due.isoformat(),
            "result": "pass",
        }
    )


class Scripted(BaseLLMProvider):
    """Answers each extraction from a script, in order; an Exception entry means 'give junk'."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        self.prompts.append(messages[-1]["content"])
        return LLMResponse(content=self.replies.pop(0) if self.replies else "{}")

    async def health(self) -> bool:
        return True


@pytest.fixture
async def repo() -> AsyncIterator[DocumentRepository]:
    try:
        async with temporary_repository() as r:
            yield r
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


async def store(repo: DocumentRepository, doc_id: str, *pages: str) -> None:
    parsed = ParsedDocument(
        source=f"{doc_id}.pdf",
        pages=[PageText(page=i, text=t, method="text") for i, t in enumerate(pages, start=1)],
    )
    await repo.save_document(doc_id, f"{doc_id}.pdf", DocType.CALIBRATION_CERT, parsed, len(pages))


async def test_every_certificate_in_a_bundle_is_extracted_and_stored(
    repo: DocumentRepository,
) -> None:
    await store(
        repo,
        "b",
        page_text("TP-1001"),
        page_text("TP-1002"),
        "continuation page",
        page_text("SC-77"),
    )
    llm = Scripted(reply("TP-1001", 10), reply("TP-1002", 20), reply("SC-77", 30))
    run = await run_extraction(llm, repo, "b", model="m")
    assert (run.total, run.done, run.failed) == (3, 3, 0)

    rows = await repo.get_extractions("b")
    assert [(r.start_page, r.end_page, r.data["instrument_serial"]) for r in rows] == [
        (1, 1, "TP-1001"), (2, 3, "TP-1002"), (4, 4, "SC-77"),
    ]  # fmt: skip
    assert "continuation page" in llm.prompts[1] and "SC-77" not in llm.prompts[1]  # own pages only
    due = await repo.certificates_due(40)
    assert [(c.instrument_serial, c.start_page) for c in due] == [
        ("TP-1001", 1), ("TP-1002", 2), ("SC-77", 4),
    ]  # fmt: skip
    job = await repo.get_job("b")
    assert job is not None and (job.status, job.total, job.done, job.failed) == ("done", 3, 3, 0)


async def test_one_unreadable_certificate_does_not_lose_the_others(
    repo: DocumentRepository,
) -> None:
    await store(repo, "b", page_text("TP-1001"), page_text("TP-1002"), page_text("TP-1003"))
    llm = Scripted(reply("TP-1001"), "not json", "not json", "not json", reply("TP-1003"))
    run = await run_extraction(llm, repo, "b", model="m")
    assert (run.done, run.failed) == (2, 1)
    assert [r.data["instrument_serial"] for r in await repo.get_extractions("b")] == [
        "TP-1001",
        "TP-1003",
    ]
    job = await repo.get_job("b")
    assert (
        job is not None
        and job.status == "done"
        and job.failed == 1
        and "pages 2-2" in (job.error or "")
    )


async def test_a_bundle_where_nothing_can_be_read_is_a_failed_job(repo: DocumentRepository) -> None:
    await store(repo, "b", page_text("TP-1001"), page_text("TP-1002"))
    run = await run_extraction(Scripted(), repo, "b", model="m")
    assert (run.done, run.failed) == (0, 2)
    job = await repo.get_job("b")
    assert job is not None and job.status == "failed"


async def test_a_rerun_replaces_results_and_drops_certificates_no_longer_found(
    repo: DocumentRepository,
) -> None:
    await store(repo, "b", page_text("TP-1001"), page_text("TP-1002"))
    await run_extraction(Scripted(reply("TP-1001"), reply("TP-1002")), repo, "b", model="m")
    assert len(await repo.get_extractions("b")) == 2

    await store(repo, "b", page_text("TP-1001"))  # the document is re-read as a single certificate
    await run_extraction(Scripted(reply("TP-1001", 5)), repo, "b", model="m")
    rows = await repo.get_extractions("b")
    assert [r.start_page for r in rows] == [1]
    assert [c.instrument_serial for c in await repo.certificates_due(10)] == ["TP-1001"]


async def test_progress_is_reported_after_each_certificate(repo: DocumentRepository) -> None:
    await store(repo, "b", page_text("TP-1001"), page_text("TP-1002"))
    seen: list[tuple[int, int]] = []

    async def note(run: Any) -> None:
        job = await repo.get_job("b")
        assert job is not None
        seen.append((run.done, job.done))  # the database agrees with the callback

    await run_extraction(
        Scripted(reply("TP-1001"), reply("TP-1002")), repo, "b", model="m", on_progress=note
    )
    assert seen == [(1, 1), (2, 2)]


async def test_a_single_certificate_document_is_one_segment(repo: DocumentRepository) -> None:
    await store(repo, "one", page_text("TP-1001"))
    doc_type, segments = await plan_segments(repo, "one")
    assert doc_type is DocType.CALIBRATION_CERT and len(segments) == 1


async def test_unknown_and_unsupported_documents_are_reported(repo: DocumentRepository) -> None:
    with pytest.raises(UnknownDocument):
        await plan_segments(repo, "nope")
    sop = ParsedDocument(source="s.pdf", pages=[PageText(page=1, text="SOP", method="text")])
    await repo.save_document("sop", "s.pdf", DocType.SOP, sop, 1)
    with pytest.raises(UnsupportedDocType):
        await plan_segments(repo, "sop")


async def test_stale_jobs_from_a_dead_process_are_marked_failed_at_startup(
    repo: DocumentRepository,
) -> None:
    await store(repo, "b", page_text("TP-1001"))
    await repo.start_job("b", 5)
    assert await repo.fail_stale_jobs() == 1
    job = await repo.get_job("b")
    assert job is not None and job.status == "failed" and "restart" in (job.error or "")
    assert await repo.fail_stale_jobs() == 0  # finished jobs are left alone


def cert(**over: Any) -> CalibrationCert:
    base: dict[str, Any] = {
        "instrument_serial": "TP-1", "calibration_date": date(2026, 1, 10),
        "next_due": date(2027, 1, 10), "result": "pass",
    }  # fmt: skip
    return CalibrationCert(**{**base, **over})


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({}, []),
        ({"next_due": None}, ["No next-due date was found"]),
        ({"next_due": date(2025, 1, 10)}, ["The next-due date is not after the calibration date"]),
        (
            {"next_due": date(2028, 1, 10)},
            ["The calibration interval is 730 days (expected about a year)"],
        ),
        (
            {"next_due": date(2026, 7, 10)},
            ["The calibration interval is 181 days (expected about a year)"],
        ),
    ],
)
def test_implausible_dates_are_flagged(over: dict[str, Any], expected: list[str]) -> None:
    assert certificate_warnings(cert(**over)) == expected


class Gauge(BaseLLMProvider):
    """Records how many extractions are in flight at once; replies out of order."""

    def __init__(self, serials: list[str]) -> None:
        self.serials = serials
        self.in_flight = 0
        self.peak = 0

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        import asyncio

        text = messages[-1]["content"]
        serial = next(s for s in self.serials if s in text)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        # earlier certificates take longer, so completion order differs from page order
        await asyncio.sleep(0.05 * (len(self.serials) - self.serials.index(serial)))
        self.in_flight -= 1
        return LLMResponse(content=reply(serial))

    async def health(self) -> bool:
        return True


@pytest.mark.parametrize(("concurrency", "expected_peak"), [(1, 1), (3, 3), (10, 4)])
async def test_certificates_can_be_read_in_parallel_up_to_the_limit(
    repo: DocumentRepository, concurrency: int, expected_peak: int
) -> None:
    serials = ["TP-1001", "TP-1002", "TP-1003", "TP-1004"]
    await store(repo, "b", *(page_text(s) for s in serials))
    llm = Gauge(serials)
    run = await run_extraction(llm, repo, "b", model="m", concurrency=concurrency)
    assert (run.done, run.failed) == (4, 0) and llm.peak == expected_peak
    rows = await repo.get_extractions("b")  # stored by page whatever order they finished in
    assert [(r.start_page, r.data["instrument_serial"]) for r in rows] == list(
        enumerate(serials, start=1)
    )
    job = await repo.get_job("b")
    assert job is not None and (job.done, job.status) == (4, "done")


async def test_an_unexpected_failure_cancels_the_remaining_certificates(
    repo: DocumentRepository,
) -> None:
    import asyncio

    started: list[str] = []

    class Breaks(BaseLLMProvider):
        async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
            started.append(messages[-1]["content"][-40:])
            if len(started) == 1:
                raise ConnectionError("the LLM server went away")
            await asyncio.sleep(5)  # would run for ages if it were not cancelled
            return LLMResponse(content="{}")

        async def health(self) -> bool:
            return False

    await store(repo, "b", *(page_text(f"TP-10{n}") for n in range(1, 5)))
    t0 = asyncio.get_running_loop().time()
    with pytest.raises(ConnectionError):
        await run_extraction(Breaks(), repo, "b", model="m", concurrency=2)
    assert asyncio.get_running_loop().time() - t0 < 2  # the queued ones did not run out their sleep
    job = await repo.get_job("b")
    assert job is not None and job.status == "failed" and "ConnectionError" in (job.error or "")
