"""Split a PDF that holds several certificates into one segment per certificate.

Real calibration PDFs are mostly bundles (one file had 24 certificate pages). A different serial
or certificate number starts a new certificate (as in the toolkit's calibration reader); the same
one is a continuation; short pages with nothing identifying them (a repeated header, a signature
page) belong to the certificate before. Where identifiers cannot be read the rule splits rather
than glues, because a glued certificate is lost silently while an over-split one fails visibly.
"""

import re
from dataclasses import dataclass

from qacopilot.ingest.classifier import classify_text, has_title_for
from qacopilot.ingest.models import DocType, ParsedDocument

# "Serial No: TP-20931", "Serial number 10416236", "S/N AB-1234". The value must contain a digit
# (so a following label such as "Date" is not mistaken for one) and be 4-24 characters.
_SERIAL = re.compile(
    r"(?:serial\s*(?:no\.?|number|num\.?|#)?|s/n)\s*[:#.\-]?\s*([A-Z0-9][A-Z0-9\-/]{3,23})",
    re.IGNORECASE,
)


# "Certificate No: 296382", "Certificate Number 26 0534 CH" (spaces: only the first token is read).
_CERT_NO = re.compile(
    r"certificate\s*(?:no\.?|number|num\.?|ref(?:erence)?\.?|#)\s*[:#.\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,23})",
    re.IGNORECASE,
)


# A page shorter than this fraction of the certificate's first page, with nothing identifying it
# as a new certificate, is a continuation (a repeated header, a signature page, a blank).
SHORT_PAGE_RATIO = 0.6


@dataclass
class Segment:
    start_page: int  # 1-based, inclusive
    end_page: int
    serial: str | None = None
    cert_no: str | None = None

    @property
    def pages(self) -> range:
        return range(self.start_page, self.end_page + 1)


def find_serial(text: str) -> str | None:
    """The first plausible serial number on the page, normalised, or None."""
    for match in _SERIAL.finditer(text):
        value = match.group(1).strip("-/.").upper()
        if any(ch.isdigit() for ch in value):
            return value
    return None


def find_certificate_number(text: str) -> str | None:
    """The certificate's own number, normalised, or None. It must contain a digit."""
    for match in _CERT_NO.finditer(text):
        value = match.group(1).strip("-/.").upper()
        if any(ch.isdigit() for ch in value):
            return value
    return None


def split_certificates(
    doc: ParsedDocument, doc_type: DocType = DocType.CALIBRATION_CERT
) -> list[Segment]:
    """One segment per certificate. Documents of other types are a single segment."""
    if doc_type is not DocType.CALIBRATION_CERT or not doc.pages:
        return [Segment(1, len(doc.pages) or 1)]
    segments: list[Segment] = []
    first_len = 0  # length of the first page of the current certificate
    for page in doc.pages:
        serial = find_serial(page.text)
        cert_no = find_certificate_number(page.text)
        looks_like_cert = classify_text(page.text).doc_type is DocType.CALIBRATION_CERT
        titled = has_title_for(page.text, DocType.CALIBRATION_CERT)
        current = segments[-1] if segments else None
        if current is None:
            segments.append(Segment(page.page, page.page, serial, cert_no))
            first_len = len(page.text.strip())
            continue
        # Identity: the serial and the certificate number. A field counts only if both the page
        # and the certificate so far carry it; "different" if any such field disagrees.
        pairs = [(serial, current.serial), (cert_no, current.cert_no)]
        shared = [(a, b) for a, b in pairs if a is not None and b is not None]
        different = any(a != b for a, b in shared)
        same = bool(shared) and not different
        short = len(page.text.strip()) < SHORT_PAGE_RATIO * first_len
        if different and looks_like_cert:
            new = True  # another instrument, or another certificate for the same one
        elif same:
            new = False  # e.g. a survey sheet that repeats the certificate's serial
        elif looks_like_cert and not short:
            # A full-length certificate-like page with nothing tying it to the one before. Real
            # bundles often print identifiers in forms we cannot read, so being unable to find a
            # serial is not evidence of a continuation. Splitting too finely fails visibly (one
            # certificate "could not be read"); gluing two certificates together would lose one
            # silently, so when in doubt we split.
            new = True
        elif looks_like_cert and titled and current.serial is None and current.cert_no is None:
            new = True  # a short page with its own title, and nothing to attach it to
        else:
            new = False  # short, repeated header, blank or unreadable: belongs to the one before
        if new:
            segments.append(Segment(page.page, page.page, serial, cert_no))
            first_len = len(page.text.strip())
        else:
            current.end_page = page.page
            current.serial = current.serial or serial
            current.cert_no = current.cert_no or cert_no
    return segments


def segment_document(doc: ParsedDocument, segment: Segment) -> ParsedDocument:
    """The pages of one segment as a document of its own (for extraction)."""
    pages = [p for p in doc.pages if segment.start_page <= p.page <= segment.end_page]
    return ParsedDocument(source=doc.source, pages=pages)
