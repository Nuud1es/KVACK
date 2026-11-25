"""Layouts seen in real documents, reproduced with invented text (no real content is stored).

Each case here was a real failure: a document classified wrongly, split into hundreds of tiny
chunks, or crashed OCR.
"""

import io

import pymupdf
import pytest
from PIL import Image

from qacopilot.ingest import DocType, PageText, ParsedDocument, chunk_document, classify_text
from qacopilot.ingest import pdf as pdfmod

CASES = {
    # A lab "test certificate": no phrase "certificate of analysis" anywhere.
    "lab_test_certificate": (
        DocType.COA,
        "TEST CERTIFICATE\nExample Laboratories Ltd, 1 Lab Road\n"
        "Date Analysis Started: 15/02/2025\n"
        "Date Reported: 23/02/2025\nCertificate Number: T123456-1 Final\nSample Description: Rice\n"
        "Test   Result   Units\nMoisture   12.1   %\n",
    ),
    # An ingredient supplier's one-page "specification data sheet", read by OCR (short).
    "specification_data_sheet": (
        DocType.SUPPLIER_SPEC,
        "EXAMPLE SPICES\nEffective : 11.03.2015\nSupersedes: Nil\nSPECIFICATION DATA SHEET\n"
        "PRODUCT : Cracked pepper\nPacking: laminated paper bags\n",
    ),
    # A customer's manual whose contents list "product specification", allergens, nutrition...
    "customer_manual_with_busy_contents": (
        DocType.CUSTOMER_COP,
        "Example Foodservice\nInspired by you\nSupplier Technical Manual\n\n"
        "Contents\n1.1 Product specification and allergens\n1.2 Packaging and net weight\n"
        "1.3 Nutritional information\n1.4 Shelf life and storage\n2.1 Supplier audits\n",
    ),
    # An invoice with decimal prices at the start of lines and one "shall" in the terms.
    "invoice": (
        DocType.OTHER,
        "SALES INVOICE\nInvoice No 100245\n1.00 Duck breast 12.50\n2.00 Gravy 4.10\n"
        "3.00 Stuffing 2.20\n"
        "Terms: payment shall be made within 30 days.\nVAT 20%\n",
    ),
}


@pytest.mark.parametrize(("name", "expected_text"), [(k, v) for k, v in CASES.items()])
def test_real_layouts_are_classified(name: str, expected_text: tuple[DocType, str]) -> None:
    expected, text = expected_text
    assert classify_text(text, f"{name}.pdf").doc_type is expected


def test_a_long_document_is_judged_by_its_start_not_by_every_topic_it_mentions() -> None:
    filler = "General information about our company and its values. " * 120  # > 4000 characters
    text = (
        "Example Company Handbook\n" + filler + "\nCertificate of Analysis, TVC, HACCP, calibration"
    )
    assert classify_text(text, "handbook.pdf").doc_type is DocType.OTHER


def test_a_clause_that_mentions_another_documents_name_is_not_a_title() -> None:
    text = (
        "Example Retailer Code of Practice for Suppliers\n"
        "1.1 Suppliers shall hold current BRCGS certification at grade A or AA.\n"
        "1.2 Suppliers shall be audited every 12 months.\n3.1 Chilled goods shall arrive at 5C.\n"
    )
    assert classify_text(text, "x.pdf").doc_type is DocType.CUSTOMER_COP


def test_table_label_cells_do_not_become_hundreds_of_headings() -> None:
    """A spec header table (NAV CODE / PLU / VERSION ...) once gave 53 chunks for 3 pages."""
    cells = [
        ("NAV CODE", "4000001"),
        ("PLU", "1234"),
        ("VERSION", "7"),
        ("BRAND", "OWN"),
        ("SITE", "REDGRAVE"),
        ("SEASON", "ALL"),
    ] * 12
    labels = "\n".join(f"{name}\n{value}" for name, value in cells)
    doc = ParsedDocument(source="t", pages=[PageText(page=1, text=labels, method="text")])
    chunks = chunk_document(doc, DocType.SUPPLIER_SPEC, "t")
    assert len(chunks) <= 3
    assert "NAV CODE" in chunks[0].text  # the label words are still searchable


def test_real_numbered_sections_still_split() -> None:
    body = "This section has enough text to be a real section on its own. " * 3
    doc = ParsedDocument(
        source="t",
        pages=[
            PageText(page=1, text=f"Title\n\n1. Scope\n{body}\n\n2. Method\n{body}", method="text")
        ],
    )
    sections = [c.section for c in chunk_document(doc, DocType.SOP, "t")]
    assert sections == ["1. Scope", "2. Method"]


def test_a_huge_page_is_rendered_no_larger_than_the_cap() -> None:
    """A 211-megapixel phone photo crashed OCR with a decompression-bomb error."""
    doc = pymupdf.open()
    page = doc.new_page(width=20000, height=20000)  # ~277 inches square
    png = pdfmod._render(page)
    side = max(Image.open(io.BytesIO(png)).size)
    assert side <= pdfmod.MAX_OCR_SIDE_PX + 1


def test_an_ordinary_page_is_still_rendered_at_full_ocr_resolution() -> None:
    doc = pymupdf.open()
    page = doc.new_page()  # A4
    side = max(Image.open(io.BytesIO(pdfmod._render(page))).size)
    assert 3400 <= side <= 3600  # 11.7 inches at 300 dpi


def test_a_photo_is_read_like_a_scan() -> None:
    src = pymupdf.open()
    page = src.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 545, 400),
        "CERTIFICATE OF CALIBRATION\nSerial Number: ZX-9981",
        fontsize=14,
    )
    png = src[0].get_pixmap(dpi=150).tobytes("png")
    assert pdfmod.sniff_kind(png) == "png"
    doc = pdfmod.extract_pages(png)
    assert [p.method for p in doc.pages] == ["ocr"]
    assert "ZX-9981" in doc.full_text


def test_unknown_bytes_are_not_mistaken_for_a_supported_file() -> None:
    assert pdfmod.sniff_kind(b"hello") is None
    assert pdfmod.sniff_kind(b"%PDF-1.7 ...") == "pdf"
    assert pdfmod.sniff_kind(b"\xff\xd8\xff\xe0 jpeg") == "jpeg"


def test_ocr_runs_one_thread_per_tesseract_process() -> None:
    import os

    assert os.environ.get("OMP_THREAD_LIMIT") == "1"


def test_process_flow_steps_stay_together_instead_of_one_chunk_per_step() -> None:
    """A one-page HACCP flow chart (numbered boxes) once became 26 chunks of about 90 characters."""
    steps = [
        "Receive raw material", "Inspect and sample", "Store chilled", "Transfer to department",
        "Weigh and portion", "Cook", "Probe core temperature", "Chill", "Pack", "Metal detect",
        "Label and code", "Dispatch",
    ]  # fmt: skip
    text = "HACCP process flow - Example product\n\n" + "\n".join(
        f"{n}. {step}" for n, step in enumerate(steps, start=1)
    )
    doc = ParsedDocument(source="t", pages=[PageText(page=1, text=text, method="text")])
    chunks = chunk_document(doc, DocType.HACCP, "t")
    assert len(chunks) == 1
    assert "6. Cook" in chunks[0].text and "12. Dispatch" in chunks[0].text  # nothing lost


def test_a_short_numbered_section_between_real_ones_is_still_a_section() -> None:
    body = "This section has enough text to be a real section on its own. " * 2
    text = f"Title\n\n1. Scope\n{body}\n\n2. Records\nUse form GF-12.\n\n3. Method\n{body}"
    doc = ParsedDocument(source="t", pages=[PageText(page=1, text=text, method="text")])
    sections = [c.section for c in chunk_document(doc, DocType.SOP, "t")]
    assert sections == ["1. Scope", "2. Records", "3. Method"]


def test_two_short_numbered_sections_in_a_row_are_not_mistaken_for_a_flow() -> None:
    body = "This section has enough text to be a real section on its own. " * 2
    text = (
        f"Title\n\n1. Scope\n{body}\n\n2. Records\nUse GF-12.\n3. Review\nAnnual.\n\n"
        f"4. Method\n{body}"
    )
    doc = ParsedDocument(source="t", pages=[PageText(page=1, text=text, method="text")])
    sections = [c.section for c in chunk_document(doc, DocType.SOP, "t")]
    assert "2. Records" in sections and "3. Review" in sections


def test_a_diagram_page_of_short_fragments_is_packed_by_size_not_split_per_label() -> None:
    """A real one-page flow chart (196 lines averaging 12 characters) gave 26 tiny chunks."""
    labels = ["Receive", "CCP1", "75C", "Yes", "No", "Chill", "Pack", "94.", "Hold", "Release"]
    lines = [f"{labels[i % len(labels)]} {i}" if i % 3 else labels[i % len(labels)].upper()
             for i in range(160)]  # fmt: skip
    doc = ParsedDocument(source="t", pages=[PageText(page=1, text="\n".join(lines), method="text")])
    chunks = chunk_document(doc, DocType.HACCP, "t")
    assert len(chunks) <= 4
    joined = "\n".join(c.text for c in chunks)
    assert all(line in joined for line in lines[:: len(lines) // 8])  # nothing was dropped


def test_ordinary_prose_pages_are_not_mistaken_for_diagrams() -> None:
    sentence = "The probe must be checked against the reference before every shift starts."
    text = "\n".join(f"{n}. Check {n}\n{sentence}" for n in range(1, 30))
    doc = ParsedDocument(source="t", pages=[PageText(page=1, text="Title\n" + text, method="text")])
    assert any(c.section for c in chunk_document(doc, DocType.SOP, "t"))  # headings still used


def test_ocr_pages_carry_a_confidence_and_text_pages_do_not() -> None:
    src = pymupdf.open()
    page = src.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 545, 400),
        "CERTIFICATE OF CALIBRATION\nSerial Number: ZX-9981",
        fontsize=14,
    )
    png = src[0].get_pixmap(dpi=150).tobytes("png")
    scanned = pdfmod.extract_pages(png).pages[0]
    assert scanned.method == "ocr" and scanned.confidence is not None and scanned.confidence > 75

    text_doc = pymupdf.open()
    text_doc.new_page().insert_textbox(
        pymupdf.Rect(50, 50, 545, 400), "A page with a real text layer, long enough.", fontsize=12
    )
    assert pdfmod.extract_pages(bytes(text_doc.tobytes())).pages[0].confidence is None


def test_a_badly_degraded_scan_gets_a_visibly_lower_confidence() -> None:
    import random

    from PIL import ImageFilter

    from qacopilot.ingest.models import POOR_OCR_CONFIDENCE

    src = pymupdf.open()
    page = src.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 545, 700),
        "CERTIFICATE OF CALIBRATION\nSerial Number: ZX-9981\nDate of calibration: 14 March 2026\n"
        * 6,
        fontsize=11,
    )
    clean = Image.open(
        io.BytesIO(src[0].get_pixmap(dpi=200, colorspace=pymupdf.csGRAY).tobytes("png"))
    )
    rng = random.Random(1)
    noisy = clean.filter(ImageFilter.GaussianBlur(2.2))
    px = noisy.load()
    for _ in range(int(noisy.width * noisy.height * 0.15)):  # speckle, like a bad photocopy
        px[rng.randrange(noisy.width), rng.randrange(noisy.height)] = rng.choice((0, 255))
    out = io.BytesIO()
    noisy.save(out, "PNG")
    good = pdfmod._read(clean, "eng")[1]
    bad = pdfmod._read(noisy, "eng")[1]
    assert good is not None and good > POOR_OCR_CONFIDENCE
    assert bad is None or bad < good - 10  # clearly worse than the clean render
