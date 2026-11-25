"""'Does batch X meet spec?' is answered by running the check, with report and spec cited."""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from qacopilot.compliance import CheckFailure, run_coa_check
from qacopilot.ingest.models import DocType
from qacopilot.rag import answer_from_coa, answer_with_lookup, is_coa_question
from qacopilot.rag.answer import DoneEvent, SourcesEvent, TokenEvent
from qacopilot.repositories import ExtractionRecord

NOW = datetime(2026, 11, 1, tzinfo=UTC)

LIMITS = [
    {"test": "Total viable count", "limit": "< 1.0E+05 cfu/g"},
    {"test": "Enterobacteriaceae", "limit": "< 1.0E+03 cfu/g"},
    {"test": "Salmonella", "limit": "Not detected in 25g"},
]


def spec_data(
    limits: list[dict[str, str]] | None = None, product: str = "Duck breast fillet skin on"
) -> dict[str, Any]:
    return {
        "supplier": "Example Poultry Ltd",
        "product": product,
        "product_code": "4051234",
        "micro_limits": LIMITS if limits is None else limits,
    }


def coa_data(
    batch: str, tvc: str, report: str = "2026-11-05", product: str = "Duck breast fillet"
) -> dict[str, Any]:
    return {
        "product": product,
        "batch": batch,
        "report_date": report,
        "results": [
            {"test": "Total viable count", "result": tvc, "unit": "cfu/g", "limit": None},
            {"test": "Enterobacteriaceae", "result": "4.0E+02", "unit": "cfu/g", "limit": None},
            {"test": "Salmonella", "result": "Not detected in 25g", "unit": None, "limit": None},
        ],
    }


def record(doc_id: str, data: dict[str, Any], text: str = "", ocr: int = 0) -> ExtractionRecord:
    return ExtractionRecord(
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        data=data,
        text=text,
        ingested_at=NOW,
        ocr_pages=ocr,
    )


class FakeSource:
    def __init__(self, coas: list[ExtractionRecord], specs: list[ExtractionRecord]) -> None:
        self.coas, self.specs = coas, specs

    def _all(self) -> dict[str, tuple[str, ExtractionRecord]]:
        return {
            **{r.doc_id: (DocType.COA.value, r) for r in self.coas},
            **{r.doc_id: (DocType.SUPPLIER_SPEC.value, r) for r in self.specs},
        }

    async def get_extraction(self, doc_id: str) -> Any:
        found = self._all().get(doc_id)
        return SimpleNamespace(doc_type=found[0], data=found[1].data) if found else None

    async def extraction_records(
        self, doc_type: DocType, *, with_text_pages: int = 0
    ) -> list[ExtractionRecord]:
        return self.coas if doc_type is DocType.COA else self.specs

    async def get_document(self, doc_id: str) -> Any:
        found = self._all().get(doc_id)
        return (
            SimpleNamespace(ocr_pages=found[1].ocr_pages, filename=found[1].filename)
            if found
            else None
        )

    async def poor_ocr_pages(self, doc_id: str) -> int:
        return 0

    async def get_page(self, doc_id: str, page: int) -> Any:
        return (None, SimpleNamespace(text=f"page text of {doc_id}"))


def source(*coas: ExtractionRecord, specs: list[ExtractionRecord] | None = None) -> FakeSource:
    return FakeSource(list(coas), specs if specs is not None else [record("spec1", spec_data())])


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Does batch B2613-01 meet spec?", True),
        ("Is B2613-01 within limits?", True),
        ("Did the CoA for the duck breast pass?", True),
        ("When is TP-20931 due?", False),
        ("What is the shelf life of the duck breast?", False),
        ("Does the SOP pass the audit?", False),
    ],
)
def test_is_coa_question(question: str, expected: bool) -> None:
    assert is_coa_question(question) is expected


async def test_a_failing_batch_names_the_failed_result_and_cites_report_and_spec() -> None:
    out = await answer_from_coa(
        "Does batch B2613-01 meet spec?", source(record("c1", coa_data("B2613-01", "2.4E+05")))
    )
    assert out is not None
    sources, done = out[0], out[-1]
    assert isinstance(sources, SourcesEvent) and [c.filename for c in sources.citations] == [
        "c1.pdf",
        "spec1.pdf",
    ]
    assert isinstance(done, DoneEvent) and done.via == "coa_check" and done.cited == [1, 2]
    assert (
        "FAIL" in done.answer and "Total viable count" in done.answer and "2.4E+05" in done.answer
    )
    assert "2 result(s) within limits" in done.answer  # the passing ones are summarised, not listed


async def test_a_clean_batch_passes() -> None:
    out = await answer_from_coa(
        "Is B2612-03 within spec?", source(record("c1", coa_data("B2612-03", "3.0E+03")))
    )
    assert out is not None and ": PASS" in out[-1].answer  # type: ignore[union-attr]


async def test_an_unknown_batch_is_left_to_search() -> None:
    assert (
        await answer_from_coa(
            "Does batch B9999-01 meet spec?", source(record("c1", coa_data("B2613-01", "1")))
        )
        is None
    )
    assert await answer_from_coa("Does batch B2613-01 meet spec?", source()) is None


async def test_no_matching_spec_is_reported_not_guessed() -> None:
    out = await answer_from_coa(
        "Does batch B2613-01 meet spec?",
        source(
            record("c1", coa_data("B2613-01", "1")),
            specs=[record("g", spec_data(product="Beef gravy"))],
        ),
    )
    assert out is not None and "could not be checked" in out[-1].answer  # type: ignore[union-attr]
    assert "No extracted supplier spec matches" in out[-1].answer  # type: ignore[union-attr]


async def test_a_report_dated_before_a_new_revision_is_judged_against_the_old_one() -> None:
    v1 = record("v1", spec_data(), text="PRODUCT SPECIFICATION")
    v2 = record("v2", spec_data(limits=[]), text="Revision 2. Effective: 2026-10-01")
    v2.ingested_at = NOW.replace(day=2)
    out = await answer_from_coa(
        "Does batch B2611-07 meet spec?",
        source(record("c1", coa_data("B2611-07", "3.0E+03", report="2026-09-20")), specs=[v1, v2]),
    )
    assert out is not None
    answer = out[-1].answer  # type: ignore[union-attr]
    assert "v1.pdf" in answer and "before revision 2 took effect" in answer


async def test_a_revision_that_says_limits_are_unchanged_borrows_them_from_the_earlier_one() -> (
    None
):
    v1 = record("v1", spec_data(), text="PRODUCT SPECIFICATION")
    v2 = record(
        "v2",
        spec_data(limits=[]),
        text="Revision 2. Effective: 2026-10-01. Microbiological limits are unchanged.",
    )
    v2.ingested_at = NOW.replace(day=2)
    out = await answer_from_coa(
        "Does batch B2613-01 meet spec?",
        source(record("c1", coa_data("B2613-01", "2.4E+05")), specs=[v1, v2]),
    )
    assert out is not None
    answer = out[-1].answer  # type: ignore[union-attr]
    assert "v2.pdf" in answer and "Total viable count" in answer and "FAIL" in answer
    assert "limits from v1.pdf" in answer and "were used" in answer
    # without the "unchanged" statement nothing is borrowed: the report's own limits are used
    v2.text = "Revision 2. Effective: 2026-10-01."
    out = await answer_from_coa(
        "Does batch B2613-01 meet spec?",
        source(record("c1", coa_data("B2613-01", "2.4E+05")), specs=[v1, v2]),
    )
    assert out is not None and "lists no microbiological limits" in out[-1].answer  # type: ignore[union-attr]


async def test_an_ocr_report_cannot_be_a_clean_pass() -> None:
    out = await answer_from_coa(
        "Is B2612-03 within spec?", source(record("c1", coa_data("B2612-03", "3.0E+03"), ocr=1))
    )
    assert out is not None
    assert ": REVIEW" in out[-1].answer and "read by OCR" in out[-1].answer  # type: ignore[union-attr]


async def test_run_coa_check_errors_carry_a_status() -> None:
    with pytest.raises(CheckFailure) as e:
        await run_coa_check(source(), "missing")
    assert e.value.status == 404
    ambiguous = source(
        record("c1", coa_data("B1", "1")),
        specs=[record("a", spec_data()), record("b", {**spec_data(), "supplier": "Other Ltd"})],
    )
    with pytest.raises(CheckFailure) as e2:
        await run_coa_check(ambiguous, "c1")
    assert e2.value.status == 409 and len(e2.value.candidates) == 2 and "a.pdf" in e2.value.message


async def search() -> Any:
    yield TokenEvent(text="from search")


async def test_answer_with_lookup_routes_coa_questions_and_falls_back() -> None:
    src = source(record("c1", coa_data("B2613-01", "2.4E+05")))
    got = [e async for e in answer_with_lookup("Does batch B2613-01 meet spec?", src, search())]
    assert isinstance(got[-1], DoneEvent) and got[-1].via == "coa_check"
    got = [e async for e in answer_with_lookup("Does batch B0000-00 meet spec?", src, search())]
    assert [type(e) for e in got] == [TokenEvent]


_ = date  # (kept: dates appear in the data above as strings)
