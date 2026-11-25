from pathlib import Path

import pytest

from qacopilot.ingest import DocType, classify_text, extract_pages

SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples"

EXPECTED = {
    "calibration_cert_probe.pdf": DocType.CALIBRATION_CERT,
    "coa_duck_breast.pdf": DocType.COA,
    "supplier_spec_sauce.pdf": DocType.SUPPLIER_SPEC,
    "sop_cooking_temperature.pdf": DocType.SOP,
    "brcgs_certificate.pdf": DocType.AUDIT_CERT,
    "unrelated_memo.pdf": DocType.OTHER,
    "calibration_cert_probe_2.pdf": DocType.CALIBRATION_CERT,
    "calibration_cert_scale.pdf": DocType.CALIBRATION_CERT,
    "calibration_cert_metal_detector.pdf": DocType.CALIBRATION_CERT,
    "coa_duck_breast_b2612.pdf": DocType.COA,
    "supplier_spec_duck_breast.pdf": DocType.SUPPLIER_SPEC,
    "supplier_spec_gravy.pdf": DocType.SUPPLIER_SPEC,
    "calibration_cert_probe_renewal.pdf": DocType.CALIBRATION_CERT,
    "calibration_cert_xray.pdf": DocType.CALIBRATION_CERT,
    "coa_duck_breast_b2613.pdf": DocType.COA,
    "supplier_spec_duck_breast_v2.pdf": DocType.SUPPLIER_SPEC,
    "supplier_spec_plum_sauce.pdf": DocType.SUPPLIER_SPEC,
    "supplier_spec_stuffing.pdf": DocType.SUPPLIER_SPEC,
    "customer_cop_micro_appendix.pdf": DocType.CUSTOMER_COP,
    "sop_cleaning_verification.pdf": DocType.SOP,
    "sop_metal_detection.pdf": DocType.SOP,
    "haccp_cooked_duck.pdf": DocType.HACCP,
    "customer_cop_retailer.pdf": DocType.CUSTOMER_COP,
}


@pytest.mark.parametrize(("name", "expected"), EXPECTED.items())
def test_classifies_text_pdfs(name: str, expected: DocType) -> None:
    doc = extract_pages(SAMPLES / name)
    assert all(p.method == "text" for p in doc.pages)
    assert classify_text(doc.full_text, name).doc_type is expected


def test_scanned_pdf_falls_back_to_ocr() -> None:
    doc = extract_pages(SAMPLES / "calibration_cert_probe_scanned.pdf")
    assert [p.method for p in doc.pages] == ["ocr"]
    assert "TP-20931" in doc.full_text
    assert classify_text(doc.full_text).doc_type is DocType.CALIBRATION_CERT


def test_mixed_document_is_handled_per_page() -> None:
    doc = extract_pages(SAMPLES / "mixed_text_and_scanned.pdf")
    assert [p.method for p in doc.pages] == ["text", "ocr"]


def test_ocr_can_be_disabled() -> None:
    doc = extract_pages(SAMPLES / "calibration_cert_probe_scanned.pdf", ocr=False)
    assert doc.pages[0].text == ""


def test_accepts_bytes() -> None:
    data = (SAMPLES / "coa_duck_breast.pdf").read_bytes()
    assert "CERTIFICATE OF ANALYSIS" in extract_pages(data).full_text


def _scan_variant(name: str, degrees: float, *, dpi: int = 200, quality: int = 60) -> bytes:
    """The sample scan, rotated and re-compressed: what a skewed or sideways scan looks like."""
    import io

    import pymupdf
    from PIL import Image

    src = pymupdf.open(SAMPLES / name)
    pix = src[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).rotate(degrees, expand=True, fillcolor=255)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    doc = pymupdf.open()
    page = doc.new_page(width=img.width * 72 / dpi, height=img.height * 72 / dpi)
    page.insert_image(page.rect, stream=buf.getvalue())
    return bytes(doc.tobytes())


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_sideways_and_upside_down_scans_are_read(degrees: int) -> None:
    text = extract_pages(_scan_variant("calibration_cert_probe_scanned.pdf", degrees)).full_text
    assert "TP-20931" in text and "2027-03-14" in text
    assert classify_text(text).doc_type is DocType.CALIBRATION_CERT


def test_a_slightly_skewed_low_quality_scan_is_still_read() -> None:
    data = _scan_variant("calibration_cert_probe_scanned.pdf", 3, dpi=150, quality=35)
    assert "TP-20931" in extract_pages(data).full_text


def test_a_blank_scan_does_not_crash_orientation_detection() -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    assert extract_pages(bytes(doc.tobytes())).pages[0].text == ""


@pytest.mark.parametrize(
    ("name", "doc_type", "needle"),
    [
        ("calibration_cert_balance_rotated_scan.pdf", DocType.CALIBRATION_CERT, "BL-44017"),
        ("coa_chicken_b2701_scanned_skewed.pdf", DocType.COA, "B2701-02"),
    ],
)
def test_adversarial_scan_only_samples_are_read_and_classified(
    name: str, doc_type: DocType, needle: str
) -> None:
    doc = extract_pages(SAMPLES / name)
    assert [p.method for p in doc.pages] == ["ocr"]
    assert needle in doc.full_text
    assert classify_text(doc.full_text, name).doc_type is doc_type
