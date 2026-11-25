import pytest

from qacopilot.ingest import (
    DocType,
    PageText,
    ParsedDocument,
    find_serial,
    segment_document,
    split_certificates,
)


def cert(serial: str, title: bool = True, extra: str = "") -> str:
    head = "CERTIFICATE OF CALIBRATION\n" if title else "Example Calibration Ltd\n"
    return (
        f"{head}Instrument: Digital probe thermometer\nSerial Number: {serial}\n"
        f"Date of calibration: 2026-03-14\nNext calibration due: 2027-03-14\n"
        f"Traceable to national standards (UKAS). Result: PASS\n{extra}"
    )


def doc(*pages: str) -> ParsedDocument:
    return ParsedDocument(
        source="bundle.pdf",
        pages=[PageText(page=i, text=t, method="text") for i, t in enumerate(pages, start=1)],
    )


def ranges(d: ParsedDocument) -> list[tuple[int, int]]:
    return [(s.start_page, s.end_page) for s in split_certificates(d)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Serial Number: TP-20931", "TP-20931"),
        ("Serial No. 10416236", "10416236"),
        ("Serial: AB-12/34", "AB-12/34"),
        ("S/N QPB55-30-29464D", "QPB55-30-29464D"),
        ("Machine Serial Number\n\nEQMD-220-DEV-065789", "EQMD-220-DEV-065789"),
        ("serial number: ab-6070n", "AB-6070N"),
        ("Serial Number: Date\nSerial Number: ZX-9981", "ZX-9981"),  # a label is not a value
    ],
)
def test_find_serial(text: str, expected: str) -> None:
    assert find_serial(text) == expected


@pytest.mark.parametrize(
    "text", ["", "no serial anywhere", "Serial Number: abc", "Serial Number: 1-2"]
)
def test_find_serial_rejects_non_serials(text: str) -> None:
    assert find_serial(text) is None  # needs a digit and 4+ characters


def test_one_certificate_is_one_segment() -> None:
    assert ranges(doc(cert("TP-1001"))) == [(1, 1)]


def test_each_new_serial_starts_a_new_certificate() -> None:
    d = doc(cert("TP-1001"), cert("TP-1002"), cert("SC-7712"))
    assert ranges(d) == [(1, 1), (2, 2), (3, 3)]
    assert [s.serial for s in split_certificates(d)] == ["TP-1001", "TP-1002", "SC-7712"]


def test_a_second_page_repeating_the_header_without_a_serial_stays_with_its_certificate() -> None:
    page2 = (
        "CERTIFICATE OF CALIBRATION\nCertificate Number: CAL-1\nCalibration engineer: A. Person\n"
    )
    assert ranges(doc(cert("TP-1001"), page2, cert("TP-1002"))) == [(1, 2), (3, 3)]


def test_a_page_repeating_the_same_serial_is_a_continuation() -> None:
    survey = cert("TP-1001", extra="Radiation survey results follow.")
    assert ranges(doc(cert("TP-1001"), survey, cert("TP-1002"))) == [(1, 2), (3, 3)]


def test_certificates_without_a_title_are_still_split_by_serial() -> None:
    d = doc(
        cert("KL-5001", title=False), cert("KL-5002", title=False), cert("KL-5003", title=False)
    )
    assert ranges(d) == [(1, 1), (2, 2), (3, 3)]


def test_a_blank_or_unreadable_page_joins_the_previous_certificate() -> None:
    assert ranges(doc(cert("TP-1001"), "", "  \n", cert("TP-1002"))) == [(1, 3), (4, 4)]


def test_a_non_certificate_page_with_another_serial_does_not_split() -> None:
    invoice = "SALES INVOICE\nSerial Number: INV-99881\n1.00 Item 4.10\nVAT 20%\n"
    assert ranges(doc(cert("TP-1001"), invoice)) == [(1, 2)]


def test_other_document_types_are_never_split() -> None:
    d = doc("SOP-014\nSerial Number: AB-1234", "More text\nSerial Number: CD-5678")
    assert ranges_for(d, DocType.SOP) == [(1, 2)]
    assert ranges_for(d, DocType.COA) == [(1, 2)]


def ranges_for(d: ParsedDocument, t: DocType) -> list[tuple[int, int]]:
    return [(s.start_page, s.end_page) for s in split_certificates(d, t)]


def test_an_empty_document_gives_one_segment() -> None:
    assert [(s.start_page, s.end_page) for s in split_certificates(doc())] == [(1, 1)]


def test_segment_document_returns_only_its_pages_with_original_numbers() -> None:
    d = doc(cert("TP-1001"), cert("TP-1002"), "continuation", cert("TP-1003"))
    first, second, third = split_certificates(d)
    part = segment_document(d, second)
    assert [p.page for p in part.pages] == [2, 3] and part.source == "bundle.pdf"
    assert [p.page for p in segment_document(d, third).pages] == [4]
    assert first.pages == range(1, 2)


FULL = "x " * 400  # makes a page as long as a real certificate (the splitter compares lengths)


def full_cert(serial: str | None = None, number: str | None = None, title: bool = True) -> str:
    head = "CERTIFICATE OF CALIBRATION\n" if title else "Calibration record\n"
    ids = (f"Serial Number: {serial}\n" if serial else "") + (
        f"Certificate Number: {number}\n" if number else ""
    )
    body = (
        "Instrument: probe thermometer. Traceable to national standards (UKAS). "
        "Next calibration due"
    )
    return f"{head}{ids}{body} next year.\n{FULL}"


def test_a_different_certificate_number_splits_even_when_no_serial_can_be_read() -> None:
    d = doc(
        full_cert(number="CAL-1001"), full_cert(number="CAL-1002"), full_cert(number="CAL-1003")
    )
    assert ranges(d) == [(1, 1), (2, 2), (3, 3)]


def test_the_same_certificate_number_on_the_next_page_is_a_continuation() -> None:
    d = doc(
        full_cert(number="CAL-1001"), full_cert(number="CAL-1001"), full_cert(number="CAL-1002")
    )
    assert ranges(d) == [(1, 2), (3, 3)]


def test_full_length_pages_with_unreadable_identifiers_are_split_not_glued() -> None:
    """Real bundles often print identifiers in forms we cannot parse. Gluing two certificates
    together would lose one silently; over-splitting only fails visibly."""
    d = doc(full_cert(), full_cert(), full_cert(title=False))
    assert ranges(d) == [(1, 1), (2, 2), (3, 3)]


def test_a_short_page_after_a_full_certificate_is_a_continuation_even_with_a_title() -> None:
    header_only = (
        "CERTIFICATE OF CALIBRATION\nCalibration engineer: A. Person\nSigned: 2026-01-01\n"
    )
    d = doc(full_cert(serial="TP-1001"), header_only, full_cert(serial="TP-1002"))
    assert ranges(d) == [(1, 2), (3, 3)]


def test_a_short_titled_page_with_nothing_to_attach_to_starts_its_own_certificate() -> None:
    d = doc(full_cert(), "CERTIFICATE OF CALIBRATION\nInstrument: small probe\n")
    assert ranges(d) == [(1, 1), (2, 2)]
