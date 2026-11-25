"""Plausibility checks on an extracted certificate. They catch the usual misreads (OCR turning a
2026 into 2028, a swapped day and month) and mirror the toolkit's ``check_info``."""

import re

from qacopilot.extraction.dates import find_dates
from qacopilot.extraction.schemas import CalibrationCert, CoA, SupplierSpec

TYPICAL_INTERVAL_DAYS = (330, 400)  # about a year


def certificate_warnings(cert: CalibrationCert) -> list[str]:
    warnings: list[str] = []
    if cert.next_due is None:
        warnings.append("No next-due date was found")
        return warnings
    days = (cert.next_due - cert.calibration_date).days
    if days <= 0:
        warnings.append("The next-due date is not after the calibration date")
    elif not TYPICAL_INTERVAL_DAYS[0] <= days <= TYPICAL_INTERVAL_DAYS[1]:
        warnings.append(f"The calibration interval is {days} days (expected about a year)")
    return warnings


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ungrounded_fields(cert: CalibrationCert, text: str) -> list[str]:
    """Values the model reported that are not printed on the page: it may have made them up.

    From a blurred or garbled scan a model can return a plausible serial and date that appear
    nowhere in the text (OCR confidence does not notice: it measures how sure the OCR was, not
    whether the words make sense). The serial must appear in the page text (ignoring spaces and
    punctuation) and each date must be one the page prints, in any common format.
    """
    problems: list[str] = []
    if _alnum(cert.instrument_serial) not in _alnum(text):
        problems.append(f"the serial {cert.instrument_serial!r} does not appear in the page text")
    printed = find_dates(text)
    for label, value in (("calibration", cert.calibration_date), ("next-due", cert.next_due)):
        if value is not None and value not in printed:
            problems.append(f"the {label} date ({value}) does not appear in the page text")
    return problems


# --- reports and specs -------------------------------------------------------------------------

# Tests a quality team looks for. A name that is printed on the page but has no result (report) or
# no limit (spec) in the reading is the usual sign the model skipped a row.
_TESTS: dict[str, re.Pattern[str]] = {
    "Total viable count": re.compile(
        r"total\s+viable\s+count|\btvc\b|aerobic\s+(?:plate|colony)\s+count", re.I
    ),
    "Enterobacteriaceae": re.compile(r"enterobacteri", re.I),
    "Salmonella": re.compile(r"salmonell", re.I),
    "Listeria": re.compile(r"listeria", re.I),
    "E. coli": re.compile(r"\be\.?\s?coli\b|escherichia", re.I),
    "Campylobacter": re.compile(r"campylobacter", re.I),
    "Staphylococcus": re.compile(r"staphylococc|\bs\.\s?aureus", re.I),
    "Yeasts and moulds": re.compile(r"\byeasts?\b|\bmou?lds?\b", re.I),
}


def _tests_in(text: str) -> set[str]:
    return {name for name, pattern in _TESTS.items() if pattern.search(text)}


def _missing(on_page: set[str], read_text: str, noun: str) -> list[str]:
    read = _tests_in(read_text)
    return [
        f"{name} is printed on the page but no {noun} was read for it"
        for name in sorted(on_page - read)
    ]


def coa_problems(coa: CoA, text: str) -> list[str]:
    """Reasons a person should look at a report's reading: rows or values that seem missing."""
    problems: list[str] = []
    if not coa.results:
        problems.append("No test results were read")
    problems += _missing(_tests_in(text), " ".join(r.test for r in coa.results), "result")
    page = _alnum(text)
    if _alnum(coa.batch) not in page:
        problems.append(f"the batch {coa.batch!r} does not appear in the page text")
    for row in coa.results:
        if _alnum(row.result) not in page:
            problems.append(f"the result {row.result!r} for {row.test} does not appear in the page")
    printed = find_dates(text)
    if coa.report_date is None:
        if printed:  # a report that prints no date at all is not a misread
            problems.append("The page prints a date but no report date was read")
    elif coa.report_date not in printed:
        problems.append(f"the report date ({coa.report_date}) does not appear in the page text")
    if not coa.product:
        problems.append("No product name was found")
    return problems


def spec_problems(spec: SupplierSpec, text: str) -> list[str]:
    """Reasons a person should look at a spec's reading."""
    problems: list[str] = []
    if not spec.supplier:
        problems.append("No supplier name was found")
    if _alnum(spec.product) not in _alnum(text):
        problems.append(f"the product {spec.product!r} does not appear in the page text")
    problems += _missing(_tests_in(text), " ".join(m.test for m in spec.micro_limits), "limit")
    lowered = text.lower()
    if spec.shelf_life is None and "shelf life" in lowered:
        problems.append("Shelf life is mentioned on the page but none was read")
    if spec.storage is None and "storage" in lowered:
        problems.append("Storage is mentioned on the page but none was read")
    return problems
