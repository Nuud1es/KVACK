"""Repository tests against a real PostgreSQL, in a throwaway schema (skipped if unreachable)."""

from collections.abc import AsyncIterator
from datetime import date

import pytest

from qacopilot.extraction import CalibrationCert, CoA, TestResult
from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.repositories import DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository

TODAY = date(2026, 11, 18)


@pytest.fixture
async def repo() -> AsyncIterator[DocumentRepository]:
    try:
        async with temporary_repository() as r:
            yield r
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


def _doc(*pages: str) -> ParsedDocument:
    return ParsedDocument(
        source="x.pdf",
        pages=[PageText(page=i, text=t, method="text") for i, t in enumerate(pages, start=1)],
    )


def _cert(serial: str, calibrated: str, due: str | None) -> CalibrationCert:
    return CalibrationCert(
        instrument_serial=serial,
        calibration_date=date.fromisoformat(calibrated),
        next_due=date.fromisoformat(due) if due else None,
        result="pass",
    )


def _printed(cert: CalibrationCert) -> str:
    """What a certificate page would show, so the values are grounded in its text."""
    return (
        f"CERTIFICATE OF CALIBRATION\nSerial Number: {cert.instrument_serial}\n"
        f"Date of calibration: {cert.calibration_date}\n"
        + (f"Next calibration due: {cert.next_due}\n" if cert.next_due else "")
    )


async def _add(repo: DocumentRepository, doc_id: str, cert: CalibrationCert) -> None:
    await repo.save_document(
        doc_id, f"{doc_id}.pdf", DocType.CALIBRATION_CERT, _doc(_printed(cert)), 1
    )
    await repo.save_extraction(
        doc_id, DocType.CALIBRATION_CERT, cert, prompt_version="v", model="m"
    )


async def test_document_and_pages_round_trip_and_resave_replaces(repo: DocumentRepository) -> None:
    await repo.save_document("d1", "a.pdf", DocType.SOP, _doc("one", "two"), chunks=3)
    parsed = await repo.load_parsed("d1")
    assert parsed is not None and [p.text for p in parsed.pages] == ["one", "two"]
    doc = await repo.get_document("d1")
    assert doc is not None and (doc.doc_type, doc.pages, doc.chunks) == ("sop", 2, 3)

    await repo.save_document("d1", "a.pdf", DocType.SOP, _doc("only"), chunks=1)
    parsed = await repo.load_parsed("d1")
    assert parsed is not None and [p.text for p in parsed.pages] == ["only"]
    assert await repo.load_parsed("missing") is None


async def test_coa_extraction_is_stored_as_json_without_a_cert_row(
    repo: DocumentRepository,
) -> None:
    await repo.save_document("c1", "coa.pdf", DocType.COA, _doc("x"), 1)
    coa = CoA(batch="B1", results=[TestResult(test="TVC", result="<10")])
    await repo.save_extraction("c1", DocType.COA, coa, prompt_version="extract_v3", model="m")
    row = await repo.get_extraction("c1")
    assert row is not None and row.data["batch"] == "B1" and row.prompt_version == "extract_v3"
    assert await repo.certificates_due(10_000, today=TODAY) == []


async def test_certificates_due_lists_overdue_and_soon_in_order(repo: DocumentRepository) -> None:
    await _add(repo, "a", _cert("S-OVERDUE", "2025-01-01", "2026-11-01"))
    await _add(repo, "b", _cert("S-SOON", "2025-12-01", "2026-12-10"))
    await _add(repo, "c", _cert("S-EDGE", "2025-12-18", "2026-12-18"))  # exactly 30 days
    await _add(repo, "d", _cert("S-LATER", "2026-06-01", "2027-06-01"))
    await _add(repo, "e", _cert("S-NODUE", "2026-06-01", None))

    due = await repo.certificates_due(30, today=TODAY)
    assert [(c.instrument_serial, c.status, c.days_left) for c in due] == [
        ("S-OVERDUE", "overdue", -17),
        ("S-SOON", "due_soon", 22),
        ("S-EDGE", "due_soon", 30),
    ]
    assert [c.instrument_serial for c in await repo.certificates_due(0, today=TODAY)] == [
        "S-OVERDUE"
    ]


async def test_only_the_latest_certificate_per_instrument_counts(repo: DocumentRepository) -> None:
    await _add(repo, "old", _cert("TP-1", "2025-03-01", "2026-03-01"))  # long expired...
    await _add(repo, "new", _cert("TP-1", "2026-03-05", "2027-03-05"))  # ...but recalibrated
    await _add(repo, "other", _cert("TP-2", "2025-12-01", "2026-12-01"))
    due = await repo.certificates_due(60, today=TODAY)
    assert [c.instrument_serial for c in due] == ["TP-2"]


async def test_reextraction_replaces_the_certificate_row(repo: DocumentRepository) -> None:
    await _add(repo, "x", _cert("TP-9", "2026-01-01", "2026-12-01"))
    await _add(repo, "x", _cert("TP-9", "2026-01-01", "2028-01-01"))  # corrected by a re-run
    assert await repo.certificates_due(60, today=TODAY) == []


async def test_ocr_confidence_is_stored_and_weak_pages_are_counted(
    repo: DocumentRepository,
) -> None:
    pages = [
        PageText(page=1, text="good scan", method="ocr", confidence=92.0),
        PageText(page=2, text="poor scan", method="ocr", confidence=61.5),
        PageText(page=3, text="poorer", method="ocr", confidence=40.0),
        PageText(page=4, text="text layer", method="text"),
    ]
    doc = ParsedDocument(source="s.pdf", pages=pages)
    await repo.save_document("d", "s.pdf", DocType.CALIBRATION_CERT, doc, 4)
    loaded = await repo.load_parsed("d")
    assert loaded is not None
    assert [p.confidence for p in loaded.pages] == [92.0, 61.5, 40.0, None]
    assert await repo.poor_ocr_pages("d") == 2
    (summary,) = await repo.list_documents()
    assert (summary.ocr_pages, summary.poor_ocr_pages) == (3, 2)


async def test_latest_certificates_picks_one_per_instrument_and_lists_older(
    repo: DocumentRepository,
) -> None:
    await _add(repo, "old", _cert("TP-1", "2026-03-14", "2027-03-14"))
    await _add(repo, "new", _cert("TP-1", "2026-11-10", "2027-11-10"))
    await _add(
        repo, "old_scan", _cert("TP-1", "2026-03-14", "2027-03-14")
    )  # same certificate again
    await _add(repo, "new_scan", _cert("TP-1", "2026-11-10", "2027-11-10"))  # a copy of the latest
    await _add(repo, "other", _cert("XR-9", "2026-09-02", None))
    got = await repo.latest_certificates(today=TODAY)
    assert [c.instrument_serial for c in got] == ["TP-1", "XR-9"]  # no due date sorts last
    tp = got[0]
    assert (tp.doc_id, tp.next_due, tp.days_left) == ("new", date(2027, 11, 10), 357)
    assert [o.next_due for o in tp.older] == [date(2027, 3, 14)]
    assert got[1].days_left is None and tp.filename == "new.pdf"


async def test_unextracted_calibration_documents_counts_only_unread_ones(
    repo: DocumentRepository,
) -> None:
    await _add(repo, "read", _cert("TP-1", "2026-03-14", "2027-03-14"))
    await repo.save_document("unread", "u.pdf", DocType.CALIBRATION_CERT, _doc("c"), 1)
    await repo.save_document("sop", "s.pdf", DocType.SOP, _doc("c"), 1)
    assert await repo.unextracted_calibration_documents() == 1


async def test_extraction_records_return_one_type_with_leading_page_text(
    repo: DocumentRepository,
) -> None:
    from qacopilot.extraction import SupplierSpec

    await repo.save_document(
        "s1", "s1.pdf", DocType.SUPPLIER_SPEC, _doc("Revision 2", "p2", "p3", "p4"), 1
    )
    await repo.save_extraction(
        "s1",
        DocType.SUPPLIER_SPEC,
        SupplierSpec(product="Duck", micro_limits=[]),
        prompt_version="v",
        model="m",
    )
    await _add(repo, "c1", _cert("TP-1", "2026-03-14", "2027-03-14"))
    got = await repo.extraction_records(DocType.SUPPLIER_SPEC, with_text_pages=3)
    assert [r.doc_id for r in got] == ["s1"]
    assert (
        got[0].data["product"] == "Duck" and got[0].text == "Revision 2\np2\np3"
    )  # first 3 pages only
    assert (await repo.extraction_records(DocType.SUPPLIER_SPEC))[0].text == ""


# --- review of unverified readings ------------------------------------------------------------


def _scan(confidence: float | None, text: str = "c") -> ParsedDocument:
    return ParsedDocument(
        source="s.pdf",
        pages=[PageText(page=1, text=text, method="ocr", confidence=confidence)],
    )


async def _add_scan(
    repo: DocumentRepository, doc_id: str, cert: CalibrationCert, confidence: float | None
) -> None:
    await repo.save_document(
        doc_id, f"{doc_id}.pdf", DocType.CALIBRATION_CERT, _scan(confidence, _printed(cert)), 1
    )
    await repo.save_extraction(
        doc_id, DocType.CALIBRATION_CERT, cert, prompt_version="v", model="m"
    )


async def test_a_poor_scan_or_an_odd_interval_is_flagged_and_a_clean_one_is_not(
    repo: DocumentRepository,
) -> None:
    await _add_scan(repo, "poor", _cert("TP-1", "2026-03-14", "2027-03-14"), 61.0)
    await _add_scan(repo, "good", _cert("TP-2", "2026-03-14", "2027-03-14"), 91.0)
    await _add(
        repo, "typed", _cert("TP-3", "2026-03-14", "2027-03-14")
    )  # text layer: no confidence
    await _add(repo, "odd", _cert("TP-4", "2026-03-14", "2029-03-14"))  # a 3-year interval
    queue = {i.instrument_serial: i for i in await repo.certificates_to_review()}
    assert set(queue) == {"TP-1", "TP-4"}
    assert "poor scan" in (queue["TP-1"].reason or "") and queue["TP-1"].min_ocr_confidence == 61.0
    assert "interval" in (queue["TP-4"].reason or "")


async def test_unverified_readings_are_marked_in_due_lists(repo: DocumentRepository) -> None:
    await _add_scan(repo, "poor", _cert("TP-1", "2025-12-01", "2026-12-01"), 60.0)
    await _add(repo, "clean", _cert("TP-2", "2025-12-02", "2026-12-02"))
    due = {c.instrument_serial: c for c in await repo.certificates_due(60, today=TODAY)}
    assert due["TP-1"].unverified and "poor scan" in (due["TP-1"].review_reason or "")
    assert not due["TP-2"].unverified
    latest = {c.instrument_serial: c for c in await repo.latest_certificates(today=TODAY)}
    assert latest["TP-1"].unverified and not latest["TP-2"].unverified


async def test_confirming_clears_the_flag_and_is_recorded(repo: DocumentRepository) -> None:
    await _add_scan(repo, "poor", _cert("TP-1", "2026-03-14", "2027-03-14"), 60.0)
    item = await repo.review_certificate("poor", 1, "alice", note="checked against the paper")
    assert item is not None and not item.needs_review and item.verified_by == "alice"
    assert await repo.certificates_to_review() == []
    (event,) = await repo.review_history("poor", 1)
    assert (event.action, event.reviewed_by, event.note) == (
        "confirm",
        "alice",
        "checked against the paper",
    )
    assert event.before == event.after
    due = await repo.certificates_due(400, today=TODAY)
    assert due and not due[0].unverified


async def test_correcting_fixes_the_stored_values_everywhere_and_keeps_the_before(
    repo: DocumentRepository,
) -> None:
    await _add_scan(
        repo, "poor", _cert("TP-1", "2026-03-14", "2029-03-14"), 55.0
    )  # OCR: 2027 -> 2029
    item = await repo.review_certificate(
        "poor", 1, "bob", corrections={"next_due": "2027-03-14", "instrument_serial": "TP-1A"}
    )
    assert (
        item is not None
        and item.next_due == date(2027, 3, 14)
        and item.instrument_serial == "TP-1A"
    )
    # the stored extraction (what the UI and the check read) and the typed row agree
    ext = await repo.get_extraction("poor")
    assert (
        ext is not None
        and ext.data["next_due"] == "2027-03-14"
        and ext.data["instrument_serial"] == "TP-1A"
    )
    latest = await repo.latest_certificates(today=TODAY)
    assert [(c.instrument_serial, c.next_due) for c in latest] == [("TP-1A", date(2027, 3, 14))]
    (event,) = await repo.review_history("poor", 1)
    assert event.action == "correct"
    assert event.before["next_due"] == "2029-03-14" and event.after["next_due"] == "2027-03-14"


async def test_bad_corrections_are_refused_and_change_nothing(repo: DocumentRepository) -> None:
    await _add_scan(repo, "poor", _cert("TP-1", "2026-03-14", "2027-03-14"), 55.0)
    with pytest.raises(ValueError, match="cannot be corrected"):
        await repo.review_certificate("poor", 1, "bob", corrections={"result": "fail"})
    with pytest.raises(ValueError, match="valid certificate"):
        await repo.review_certificate("poor", 1, "bob", corrections={"next_due": "not a date"})
    with pytest.raises(ValueError, match="valid certificate"):
        await repo.review_certificate("poor", 1, "bob", corrections={"instrument_serial": ""})
    assert (
        len(await repo.certificates_to_review()) == 1 and await repo.review_history("poor", 1) == []
    )
    assert await repo.review_certificate("nope", 1, "bob") is None


async def test_re_extraction_keeps_a_confirmation_only_if_the_reading_is_unchanged(
    repo: DocumentRepository,
) -> None:
    cert = _cert("TP-1", "2026-03-14", "2027-03-14")
    await _add_scan(repo, "poor", cert, 60.0)
    await repo.review_certificate("poor", 1, "alice")

    await repo.save_extraction(
        "poor", DocType.CALIBRATION_CERT, cert, prompt_version="v", model="m"
    )
    assert await repo.certificates_to_review() == []  # identical re-read: the confirmation stands

    changed = _cert("TP-1", "2026-03-14", "2027-03-15")
    await repo.save_extraction(
        "poor", DocType.CALIBRATION_CERT, changed, prompt_version="v", model="m"
    )
    (item,) = await repo.certificates_to_review()
    assert item.verified_by is None and "differs from what alice confirmed" in (item.reason or "")
    assert len(await repo.review_history("poor", 1)) == 1  # the earlier review is not lost


async def test_values_the_page_does_not_print_are_flagged_even_when_the_scan_looks_fine(
    repo: DocumentRepository,
) -> None:
    """A model can invent a serial and dates from garbled text; OCR confidence cannot tell."""
    cert = _cert("TP-1", "2026-03-14", "2027-03-14")
    await repo.save_document(
        "made_up", "made_up.pdf", DocType.CALIBRATION_CERT, _scan(90.0, "x7# 0o ~~ illegible ~~"), 1
    )
    await repo.save_extraction(
        "made_up", DocType.CALIBRATION_CERT, cert, prompt_version="v", model="m"
    )
    (item,) = await repo.certificates_to_review()
    assert "serial 'TP-1' does not appear" in (item.reason or "")
    assert "calibration date (2026-03-14) does not appear" in (item.reason or "")
    assert "poor scan" not in (item.reason or "")  # the OCR itself looked confident
