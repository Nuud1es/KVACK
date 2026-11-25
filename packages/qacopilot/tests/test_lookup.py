"""Due-date questions are answered from the certificates table, not from search results."""

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from qacopilot.rag import answer_from_certificates, answer_with_lookup, is_due_question
from qacopilot.rag.answer import DoneEvent, NoAnswerEvent, SourcesEvent, TokenEvent
from qacopilot.repositories import InstrumentCertificate, OlderCertificate

TODAY = date(2026, 11, 19)


def cert(
    serial: str,
    description: str | None,
    due: str | None,
    *,
    number: str = "CAL-1",
    calibrated: str = "2026-03-14",
    older: list[OlderCertificate] | None = None,
) -> InstrumentCertificate:
    return InstrumentCertificate(
        doc_id=f"doc-{serial}",
        filename=f"{serial}.pdf",
        instrument_serial=serial,
        instrument_description=description,
        certificate_number=number,
        calibration_date=date.fromisoformat(calibrated),
        next_due=date.fromisoformat(due) if due else None,
        days_left=(date.fromisoformat(due) - TODAY).days if due else None,
        older=older or [],
    )


class FakeSource:
    def __init__(self, certs: list[InstrumentCertificate], unread: int = 0) -> None:
        self.certs = certs
        self.unread = unread

    async def latest_certificates(self) -> list[InstrumentCertificate]:
        return self.certs

    async def unextracted_calibration_documents(self) -> int:
        return self.unread

    async def get_page(self, doc_id: str, page: int) -> Any:
        return (None, SimpleNamespace(text=f"CERTIFICATE OF CALIBRATION for {doc_id}"))


PROBE = cert(
    "TP-20931",
    "Digital probe thermometer",
    "2027-11-10",
    number="CAL-2026-0640",
    calibrated="2026-11-10",
    older=[
        OlderCertificate(
            certificate_number="CAL-2026-0412",
            calibration_date=date(2026, 3, 14),
            next_due=date(2027, 3, 14),
        )
    ],
)
PROBE2 = cert("TP-20944", "Digital probe thermometer", "2027-04-02", number="CAL-2026-0455")
XRAY = cert("XR-30918", "X-ray inspection unit, line 4", "2027-09-02", number="CAL-2026-0588")
SCALE = cert("SC-1001", "Platform scale", "2026-11-01", number="CAL-2026-0300")
CERTS = [SCALE, PROBE, PROBE2, XRAY]  # soonest first, like the repository returns


@pytest.mark.parametrize(
    "question,expected",
    [
        ("When is the probe calibration certificate due for renewal?", True),
        ("Which certificates expire next month?", True),
        ("When does TP-20931 expire?", True),
        ("when does tp-20931 expire", True),
        ("What core temperature must cooked duck reach?", False),
        ("What is the shelf life?", False),
        ("Which probe is used on line 4?", False),
    ],
)
def test_is_due_question(question: str, expected: bool) -> None:
    assert is_due_question(question) is expected


async def events(question: str, source: FakeSource) -> list[Any] | None:
    return await answer_from_certificates(question, source, today=TODAY)


async def test_a_serial_gets_its_latest_certificate_and_names_the_superseded_one() -> None:
    out = await events("When is TP-20931 due for recalibration?", FakeSource(CERTS))
    assert out is not None
    done = out[-1]
    assert isinstance(done, DoneEvent) and done.via == "certificates" and done.cited == [1]
    assert "2027-11-10" in done.answer and "CAL-2026-0640" in done.answer
    # the old certificate is mentioned only as superseded, and its date is not offered as the answer
    assert "superseded" in done.answer and "CAL-2026-0412" in done.answer
    assert "due 2027-03-14, in" not in done.answer
    sources = out[0]
    assert isinstance(sources, SourcesEvent)
    assert sources.citations[0].filename == "TP-20931.pdf"
    assert "CERTIFICATE OF CALIBRATION" in sources.citations[0].snippet


async def test_serial_matching_ignores_punctuation_and_case() -> None:
    out = await events("when is tp 20931 due", FakeSource(CERTS))
    assert out is not None and "TP-20931" in out[-1].answer  # type: ignore[union-attr]


async def test_an_ambiguous_kind_lists_every_match_instead_of_picking_one() -> None:
    out = await events("When is the probe certificate due?", FakeSource(CERTS))
    assert out is not None
    done = out[-1]
    assert isinstance(done, DoneEvent) and done.cited == [1, 2]
    assert "TP-20931" in done.answer and "TP-20944" in done.answer
    assert "Several instruments match" in done.answer
    assert "XR-30918" not in done.answer


async def test_xray_is_found_by_its_description() -> None:
    out = await events("When does the x-ray calibration expire?", FakeSource(CERTS))
    assert out is not None and "XR-30918" in out[-1].answer  # type: ignore[union-attr]


async def test_a_list_question_uses_the_window_and_reports_overdue() -> None:
    out = await events("Which certificates are due in the next 400 days?", FakeSource(CERTS))
    assert out is not None
    answer = out[-1].answer  # type: ignore[union-attr]
    assert "OVERDUE by 18 days" in answer  # the scale
    assert "You asked for 400 days; reminders look at most 365 days ahead." in answer
    assert "TP-20931" in answer


async def test_a_list_question_with_nothing_due_says_so_without_citations() -> None:
    out = await events("Which certificates expire in the next 7 days?", FakeSource([PROBE]))
    assert out is not None
    done = out[-1]
    assert isinstance(done, DoneEvent) and done.answered and done.cited == []
    assert "No calibration certificates are overdue or due within 7 days" in done.answer


@pytest.mark.parametrize(
    "question",
    [
        "When does the BRC certificate expire?",  # another kind of certificate
        "Which certificates does the audit need?",
        "When is the freezer certificate due?",  # not an instrument we hold
        "When is the certificate due?",  # no instrument, not a list question
    ],
)
async def test_questions_it_cannot_place_are_left_to_search(question: str) -> None:
    assert await events(question, FakeSource([PROBE, XRAY])) is None


async def test_a_window_is_not_mistaken_for_a_serial() -> None:
    # "next 400" must not trigger the unknown-serial guard; it is a list question with a window
    out = await events("Which certificates are due in the next 400 days?", FakeSource(CERTS))
    assert out is not None and "OVERDUE" in out[-1].answer  # type: ignore[union-attr]


async def test_an_unknown_serial_is_not_answered_with_other_instruments() -> None:
    # "the probe" would match both probes, but the serial asked about does not exist
    assert await events("When is probe TP-99999 due for calibration?", FakeSource(CERTS)) is None


async def test_a_certificate_number_finds_its_instrument() -> None:
    out = await events("When does certificate CAL-2026-0640 expire?", FakeSource(CERTS))
    assert out is not None and "TP-20931" in out[-1].answer  # type: ignore[union-attr]


async def test_no_extracted_certificates_means_search() -> None:
    assert await events("When is TP-20931 due?", FakeSource([])) is None


async def test_unread_documents_are_disclosed() -> None:
    out = await events("When is TP-20931 due?", FakeSource(CERTS, unread=2))
    assert out is not None and "2 calibration document(s) have not been read yet" in (
        out[-1].answer  # type: ignore[union-attr]
    )


async def test_a_certificate_with_no_due_date_says_so() -> None:
    out = await events("When is TP-1001 due?", FakeSource([cert("TP-1001", "Probe", None)]))
    assert out is not None and "states no due date" in out[-1].answer  # type: ignore[union-attr]


async def search_events() -> Any:
    yield SourcesEvent(citations=[])
    yield TokenEvent(text="from search")
    yield NoAnswerEvent(reason="not_in_documents")


async def test_answer_with_lookup_prefers_the_table_and_falls_back() -> None:
    got = [
        e
        async for e in answer_with_lookup(
            "When is TP-20931 due?", FakeSource(CERTS), search_events()
        )
    ]
    assert isinstance(got[-1], DoneEvent) and got[-1].via == "certificates"

    got = [e async for e in answer_with_lookup("shelf life?", FakeSource(CERTS), search_events())]
    assert [type(e) for e in got] == [SourcesEvent, TokenEvent, NoAnswerEvent]

    got = [e async for e in answer_with_lookup("When is TP-20931 due?", None, search_events())]
    assert [type(e) for e in got] == [SourcesEvent, TokenEvent, NoAnswerEvent]


async def test_a_database_failure_falls_back_to_search() -> None:
    class Broken(FakeSource):
        async def latest_certificates(self) -> list[InstrumentCertificate]:
            raise RuntimeError("db down")

    got = [
        e async for e in answer_with_lookup("When is TP-20931 due?", Broken([]), search_events())
    ]
    assert [type(e) for e in got] == [SourcesEvent, TokenEvent, NoAnswerEvent]


async def test_an_unverified_reading_is_marked_in_the_answer() -> None:
    poor = PROBE.model_copy(
        update={"unverified": True, "review_reason": "1 page(s) are a poor scan"}
    )
    out = await events("When is TP-20931 due for recalibration?", FakeSource([poor]))
    assert out is not None
    assert "UNVERIFIED" in out[-1].answer  # type: ignore[union-attr]
    clean = await events("When is TP-20931 due for recalibration?", FakeSource([PROBE]))
    assert clean is not None and "UNVERIFIED" not in clean[-1].answer  # type: ignore[union-attr]
