"""Reports and specs that look incomplete go to the review queue, like poor certificate scans."""

from collections.abc import AsyncIterator
from datetime import date

import pytest

from qacopilot.compliance import run_coa_check
from qacopilot.extraction import CoA, SupplierSpec, TestResult
from qacopilot.extraction.checks import coa_problems, spec_problems
from qacopilot.extraction.schemas import MicroLimit
from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.repositories import DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository

COA_TEXT = (
    "CERTIFICATE OF ANALYSIS\nProduct: Duck breast fillet   Batch: B2611-07\n"
    "Report date: 2026-03-02\n"
    "Total viable count   2.2E+03 cfu/g   < 1.0E+05\n"
    "Enterobacteriaceae   <10 cfu/g       < 1.0E+03\n"
    "Salmonella           Not detected in 25g\n"
)
SPEC_TEXT = (
    "SUPPLIER SPECIFICATION\nSupplier: Acme Foods\nProduct: Duck breast fillet\n"
    "Shelf life: 14 days from pack date\nStorage: 0 to 4 C\n"
    "Microbiological limits\nSalmonella  Not detected in 25g\nEnterobacteriaceae  < 1.0E+03\n"
)


def _coa(**changes: object) -> CoA:
    base = CoA(
        product="Duck breast fillet",
        batch="B2611-07",
        report_date=date(2026, 3, 2),
        results=[
            TestResult(test="Total viable count", result="2.2E+03", limit="< 1.0E+05"),
            TestResult(test="Enterobacteriaceae", result="<10"),
            TestResult(test="Salmonella", result="Not detected in 25g"),
        ],
    )
    return base.model_copy(update=changes)


def _spec(**changes: object) -> SupplierSpec:
    base = SupplierSpec(
        supplier="Acme Foods",
        product="Duck breast fillet",
        shelf_life="14 days from pack date",
        storage="0 to 4 C",
        micro_limits=[
            MicroLimit(test="Salmonella", limit="Not detected in 25g"),
            MicroLimit(test="Enterobacteriaceae", limit="< 1.0E+03"),
        ],
    )
    return base.model_copy(update=changes)


def test_a_complete_report_and_spec_raise_nothing() -> None:
    assert coa_problems(_coa(), COA_TEXT) == []
    assert spec_problems(_spec(), SPEC_TEXT) == []


def test_a_skipped_test_row_is_flagged() -> None:
    skipped = _coa(results=_coa().results[:2])  # the model dropped the Salmonella row
    (problem,) = coa_problems(skipped, COA_TEXT)
    assert "Salmonella is printed on the page but no result was read" in problem


def test_invented_values_are_flagged() -> None:
    assert any("batch" in p for p in coa_problems(_coa(batch="Z9999-99"), COA_TEXT))
    wrong = _coa(results=[TestResult(test="Salmonella", result="Detected 77")])
    assert any("Detected 77" in p for p in coa_problems(wrong, COA_TEXT))
    assert any(
        "report date" in p for p in coa_problems(_coa(report_date=date(2025, 1, 1)), COA_TEXT)
    )


def test_no_results_and_a_missed_date_are_flagged_but_a_report_without_a_date_is_not() -> None:
    assert "No test results were read" in coa_problems(_coa(results=[]), COA_TEXT)
    missed = coa_problems(_coa(report_date=None), COA_TEXT)
    assert any("prints a date but no report date" in p for p in missed)
    undated = COA_TEXT.replace("Report date: 2026-03-02\n", "")
    assert coa_problems(_coa(report_date=None), undated) == []  # nothing to read: not a misread


def test_a_spec_missing_its_limits_or_fields_is_flagged() -> None:
    no_limits = spec_problems(_spec(micro_limits=[]), SPEC_TEXT)
    assert any("Salmonella" in p and "no limit" in p for p in no_limits)
    assert any("Shelf life" in p for p in spec_problems(_spec(shelf_life=None), SPEC_TEXT))
    assert any("Storage" in p for p in spec_problems(_spec(storage=None), SPEC_TEXT))
    assert any("supplier" in p.lower() for p in spec_problems(_spec(supplier=None), SPEC_TEXT))


@pytest.fixture
async def repo() -> AsyncIterator[DocumentRepository]:
    try:
        async with temporary_repository() as r:
            yield r
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source="x.pdf", pages=[PageText(page=1, text=text, method="text")])


async def _save(repo: DocumentRepository, doc_id: str, result: CoA | SupplierSpec) -> None:
    kind = DocType.COA if isinstance(result, CoA) else DocType.SUPPLIER_SPEC
    text = COA_TEXT if isinstance(result, CoA) else SPEC_TEXT
    await repo.save_document(doc_id, f"{doc_id}.pdf", kind, _doc(text), 1)
    await repo.save_extraction(doc_id, kind, result, prompt_version="v", model="m")


async def test_a_complete_reading_is_not_queued_but_an_incomplete_one_is(
    repo: DocumentRepository,
) -> None:
    await _save(repo, "good", _coa())
    await _save(repo, "bad", _coa(results=_coa().results[:2]))
    queue = await repo.extractions_to_review()
    assert [i.doc_id for i in queue] == ["bad"]
    assert queue[0].reason and "Salmonella" in queue[0].reason
    assert set(queue[0].fields) == {"product", "batch", "report_date", "lab"}
    assert (await repo.review_flags("bad"))[1][0] is True


async def test_confirming_clears_the_flag_and_is_recorded(repo: DocumentRepository) -> None:
    await _save(repo, "bad", _spec(micro_limits=[]))
    item = await repo.review_extraction("bad", 1, "alice", note="no limits in the original")
    assert item is not None and item.verified_by == "alice"
    assert await repo.extractions_to_review() == []
    (event,) = await repo.review_history("bad", 1)
    assert event.action == "confirm" and event.reviewed_by == "alice"


async def test_a_correction_changes_the_stored_reading_and_keeps_the_old_value(
    repo: DocumentRepository,
) -> None:
    await _save(repo, "bad", _spec(supplier=None))
    await repo.review_extraction("bad", 1, "bob", corrections={"supplier": "Acme Foods Ltd"})
    row = await repo.get_extraction("bad")
    assert row is not None and row.data["supplier"] == "Acme Foods Ltd"
    (event,) = await repo.review_history("bad", 1)
    assert event.action == "correct"
    assert event.before["supplier"] is None and event.after["supplier"] == "Acme Foods Ltd"


async def test_bad_corrections_are_refused(repo: DocumentRepository) -> None:
    await _save(repo, "bad", _coa(results=[]))
    with pytest.raises(ValueError, match="cannot be corrected"):
        await repo.review_extraction("bad", 1, "bob", corrections={"results": []})
    with pytest.raises(ValueError, match="Not valid"):
        await repo.review_extraction("bad", 1, "bob", corrections={"batch": ""})
    with pytest.raises(ValueError, match="Not valid"):
        await repo.review_extraction("bad", 1, "bob", corrections={"report_date": "tomorrow"})
    assert len(await repo.extractions_to_review()) == 1
    assert await repo.review_extraction("nope", 1, "bob") is None


async def test_re_extraction_keeps_a_confirmation_only_if_the_reading_is_unchanged(
    repo: DocumentRepository,
) -> None:
    incomplete = _coa(results=_coa().results[:2])
    await _save(repo, "d", incomplete)
    await repo.review_extraction("d", 1, "alice")
    await _save(repo, "d", incomplete)
    assert await repo.extractions_to_review() == []  # the same reading: the confirmation stands
    await _save(repo, "d", _coa(results=_coa().results[:1], lab="Other Lab"))
    (item,) = await repo.extractions_to_review()
    assert item.reason and "differs from what alice confirmed" in item.reason


async def test_a_coa_check_on_an_unconfirmed_reading_cannot_pass(repo: DocumentRepository) -> None:
    await _save(repo, "coa", _coa(results=_coa().results[:2]))
    await _save(repo, "spec", _spec())
    flagged = await run_coa_check(repo, "coa", "spec")  # type: ignore[arg-type]
    assert flagged.overall != "pass"
    assert any("nobody has confirmed" in w for w in flagged.warnings)
    await repo.review_extraction("coa", 1, "alice")
    cleared = await run_coa_check(repo, "coa", "spec")  # type: ignore[arg-type]
    assert not any("nobody has confirmed" in w for w in cleared.warnings)
