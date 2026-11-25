import json
from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from qacopilot.compliance import Verdict, cap_for_unverified_source, check_coa_against_spec, overall
from qacopilot.extraction import (
    CalibrationCert,
    CoA,
    MicroLimit,
    SupplierSpec,
    TestResult,
    UnsupportedDocType,
    extract_document,
)
from qacopilot.extraction.schemas import ISO
from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.llm import BaseLLMProvider, LLMResponse, StructuredOutputError


class JsonLLM(BaseLLMProvider):
    """Replies with canned content, one per call, and records the prompts."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[Sequence[Any]] = []

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        self.prompts.append(messages)
        return LLMResponse(content=self.replies.pop(0))

    async def health(self) -> bool:
        return True


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source="t.pdf", pages=[PageText(page=1, text=text, method="text")])


CERT = {
    "instrument_serial": "TP-20931",
    "calibration_date": "2026-03-14",
    "next_due": "2027-03-14",
    "result": "pass",
}


async def test_extracts_a_calibration_cert_with_dates_parsed() -> None:
    llm = JsonLLM("```json\n" + json.dumps(CERT) + "\n```")
    cert = await extract_document(
        llm, _doc("CERTIFICATE OF CALIBRATION ..."), DocType.CALIBRATION_CERT
    )
    assert isinstance(cert, CalibrationCert)
    assert cert.instrument_serial == "TP-20931"
    assert cert.next_due == date(2027, 3, 14)
    # the document text reached the model, inside the user message
    assert "CERTIFICATE OF CALIBRATION" in llm.prompts[0][-1]["content"]


async def test_retries_with_the_validation_error_when_the_date_format_is_wrong() -> None:
    bad = {**CERT, "calibration_date": "14 March 2026"}
    llm = JsonLLM(json.dumps(bad), json.dumps(CERT))
    cert = await extract_document(llm, _doc("x"), DocType.CALIBRATION_CERT)
    assert isinstance(cert, CalibrationCert) and cert.calibration_date == date(2026, 3, 14)
    assert "calibration_date" in llm.prompts[1][-1]["content"]  # error fed back to the model


async def test_gives_up_with_a_clear_error() -> None:
    llm = JsonLLM(*["{}"] * 3)
    with pytest.raises(StructuredOutputError):
        await extract_document(llm, _doc("x"), DocType.CALIBRATION_CERT)


async def test_unsupported_type() -> None:
    with pytest.raises(UnsupportedDocType):
        await extract_document(JsonLLM(), _doc("x"), DocType.SOP)


def test_every_date_field_asks_for_iso_format() -> None:
    for model in (CalibrationCert, CoA, SupplierSpec):
        for name, field in model.model_fields.items():
            if "date" in str(field.annotation) or name.endswith(("_date", "_due")):
                assert ISO in (field.description or ""), f"{model.__name__}.{name}"


def _coa(*rows: tuple[str, str, str | None]) -> CoA:
    return CoA(batch="B1", results=[TestResult(test=t, result=r, limit=lim) for t, r, lim in rows])


SPEC = SupplierSpec(
    product="Duck",
    micro_limits=[
        MicroLimit(test="Total viable count", limit="1.0E+05 cfu/g"),
        MicroLimit(test="Enterobacteriaceae", limit="1.0E+03 cfu/g"),
        MicroLimit(test="Salmonella", limit="Not detected in 25g"),
        MicroLimit(test="Listeria monocytogenes", limit="Not detected in 25g"),
    ],
)


def test_coa_check_matches_aliases_and_flags_out_of_spec() -> None:
    coa = _coa(
        ("TVC", "2.2E+05", None),  # alias of Total viable count; over 1.0E+05
        ("Entero", "<10", None),
        ("Salmonella spp", "Not detected in 25g", None),
    )
    rows = {r.test: r for r in check_coa_against_spec(coa, SPEC)}
    assert rows["TVC"].verdict is Verdict.FAIL and rows["TVC"].limit_source == "spec"
    assert rows["Entero"].verdict is Verdict.PASS
    assert rows["Salmonella spp"].verdict is Verdict.PASS
    # a spec limit with no result on the report is surfaced, not silently passed
    listeria = rows["Listeria monocytogenes"]
    assert listeria.verdict is Verdict.REVIEW and listeria.result == ""
    assert overall(rows.values()) is Verdict.FAIL


def test_coa_check_uses_the_report_limit_only_when_the_spec_has_none() -> None:
    coa = _coa(("Yeasts", "50", "< 100 cfu/g"), ("Moulds", "50", None))
    rows = {r.test: r for r in check_coa_against_spec(coa, SupplierSpec(product="x"))}
    assert rows["Yeasts"].verdict is Verdict.PASS and rows["Yeasts"].limit_source == "coa"
    assert rows["Moulds"].verdict is Verdict.REVIEW and rows["Moulds"].limit_source == "none"


def test_unreadable_limit_goes_to_review() -> None:
    spec = SupplierSpec(product="x", micro_limits=[MicroLimit(test="TVC", limit="see HACCP")])
    (row,) = check_coa_against_spec(_coa(("TVC", "10", None)), spec)
    assert row.verdict is Verdict.REVIEW and "see HACCP" in row.reason


def test_all_passing_is_pass() -> None:
    rows = check_coa_against_spec(_coa(("TVC", "100", None)), SPEC)
    assert overall(r for r in rows if r.result) is Verdict.PASS


def test_a_spec_limit_missing_from_the_report_stops_the_overall_pass() -> None:
    coa = _coa(
        ("TVC", "100", None), ("Entero", "<10", None), ("Salmonella", "Not detected", None)
    )  # no Listeria result, but the spec has a Listeria limit
    assert overall(check_coa_against_spec(coa, SPEC)) is Verdict.REVIEW


async def test_an_empty_serial_is_rejected_and_the_model_is_asked_again() -> None:
    empty = {**CERT, "instrument_serial": ""}
    llm = JsonLLM(json.dumps(empty), json.dumps(CERT))
    cert = await extract_document(llm, _doc("x"), DocType.CALIBRATION_CERT)
    assert isinstance(cert, CalibrationCert) and cert.instrument_serial == "TP-20931"
    assert "instrument_serial" in llm.prompts[1][-1]["content"]


def test_text_read_by_ocr_can_fail_but_never_cleanly_pass() -> None:
    assert cap_for_unverified_source(Verdict.PASS) is Verdict.REVIEW
    assert cap_for_unverified_source(Verdict.REVIEW) is Verdict.REVIEW
    assert cap_for_unverified_source(Verdict.FAIL) is Verdict.FAIL


SPEC_TEXT = "PRODUCT SPECIFICATION\nSupplier: Example Poultry Ltd\nProduct: Duck breast fillet\n"


async def test_a_supplier_the_model_skipped_is_read_from_the_header() -> None:
    llm = JsonLLM(json.dumps({"product": "Duck breast fillet"}))  # supplier left out
    spec = await extract_document(llm, _doc(SPEC_TEXT), DocType.SUPPLIER_SPEC)
    assert isinstance(spec, SupplierSpec) and spec.supplier == "Example Poultry Ltd"


async def test_what_the_model_found_is_never_overwritten_by_the_header() -> None:
    llm = JsonLLM(json.dumps({"product": "Duck", "supplier": "Model's own reading"}))
    spec = await extract_document(llm, _doc(SPEC_TEXT), DocType.SUPPLIER_SPEC)
    assert isinstance(spec, SupplierSpec) and spec.supplier == "Model's own reading"


async def test_without_a_labelled_header_the_field_stays_empty() -> None:
    llm = JsonLLM(json.dumps({"product": "Duck breast fillet"}))
    spec = await extract_document(
        llm, _doc("Example Poultry Ltd\nDuck breast fillet spec"), DocType.SUPPLIER_SPEC
    )
    assert isinstance(spec, SupplierSpec) and spec.supplier is None


async def test_a_coa_product_is_filled_without_taking_the_batch_on_the_same_line() -> None:
    text = "CERTIFICATE OF ANALYSIS\nProduct: Duck breast fillet   Batch: B2611-07\n"
    llm = JsonLLM(json.dumps({"batch": "B2611-07", "results": []}))
    coa = await extract_document(llm, _doc(text), DocType.COA)
    assert isinstance(coa, CoA) and coa.product == "Duck breast fillet" and coa.batch == "B2611-07"


async def test_certificates_are_not_touched_by_the_fallback() -> None:
    text = "CERTIFICATE OF CALIBRATION\nSupplier: Someone\nProduct: Thing\n"
    cert = await extract_document(JsonLLM(json.dumps(CERT)), _doc(text), DocType.CALIBRATION_CERT)
    assert isinstance(cert, CalibrationCert) and cert.calibrated_by is None


async def test_a_next_due_the_model_skipped_is_read_from_its_label() -> None:
    reply = {**CERT, "next_due": None}
    text = (
        "CERTIFICATE OF CALIBRATION\nDate of calibration: 14 March 2026\n"
        "Next calibration due: 14/03/2027\n"
    )
    cert = await extract_document(JsonLLM(json.dumps(reply)), _doc(text), DocType.CALIBRATION_CERT)
    assert isinstance(cert, CalibrationCert) and cert.next_due == date(2027, 3, 14)


async def test_the_models_own_next_due_is_never_overwritten() -> None:
    text = "Next calibration due: 01/01/2030\n"
    cert = await extract_document(JsonLLM(json.dumps(CERT)), _doc(text), DocType.CALIBRATION_CERT)
    assert isinstance(cert, CalibrationCert) and cert.next_due == date(2027, 3, 14)


async def test_a_labelled_date_before_the_calibration_date_is_a_misread_and_is_ignored() -> None:
    reply = {**CERT, "next_due": None}  # calibrated 2026-03-14
    text = "Next calibration due: 2025-01-01\n"
    cert = await extract_document(JsonLLM(json.dumps(reply)), _doc(text), DocType.CALIBRATION_CERT)
    assert isinstance(cert, CalibrationCert) and cert.next_due is None


def test_ungrounded_fields_catches_values_the_page_does_not_print() -> None:
    from datetime import date

    from qacopilot.extraction import CalibrationCert
    from qacopilot.extraction.checks import ungrounded_fields

    def cert(serial: str, cal: date, due: date | None) -> CalibrationCert:
        return CalibrationCert(
            instrument_serial=serial, calibration_date=cal, next_due=due, result="pass"
        )

    text = "Serial No: TP - 20931\nCalibrated 14 March 2026\nNext due 14/03/2027"
    ok = cert("TP-20931", date(2026, 3, 14), date(2027, 3, 14))
    assert ungrounded_fields(ok, text) == []  # spacing, punctuation and date formats do not matter
    assert (
        ungrounded_fields(cert("TP-20931", date(2026, 3, 14), None), text) == []
    )  # no due date: nothing to ground

    bad = ungrounded_fields(cert("12345", date(2025, 5, 12), date(2026, 5, 12)), text)
    assert len(bad) == 3
    assert "serial '12345'" in bad[0] and "calibration date (2025-05-12)" in bad[1]
    # a misread digit is caught too: 2029 for 2027
    assert any(
        "next-due date (2029-03-14)" in m
        for m in ungrounded_fields(cert("TP-20931", date(2026, 3, 14), date(2029, 3, 14)), text)
    )
