"""Text extraction from PDFs and photos, with OCR for scanned pages.

Scanned pages are the slow part (several seconds each), so they are read in parallel, in small
batches to bound memory. Page images are rendered no larger than ``MAX_OCR_SIDE_PX``: a phone
photo can be hundreds of megapixels, which is both pointless for OCR and a way to exhaust memory.
"""

import io
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output

from qacopilot.ingest.models import PageText, ParsedDocument

# A page with fewer extractable characters than this is treated as scanned.
MIN_TEXT_CHARS = 30
OCR_DPI = 300
# A4 at 300 dpi is 3508 x 2480; anything bigger is shrunk to this on its longest side.
MAX_OCR_SIDE_PX = 4500
# Tesseract's orientation confidence runs to about 15; below this the guess is not trusted.
OSD_MIN_CONFIDENCE = 2.0
OCR_WORKERS = max(1, min(4, os.cpu_count() or 1))
# Tesseract uses every core by default, so several of them at once fight over the same cores and
# are slower than one (measured: 9 pages in 32 s with 4 workers, 19 s with 1). One thread each
# makes 4 workers about 2x faster than one. Only the tesseract child processes read this.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

_IMAGE_MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG": "png"}


def sniff_kind(data: bytes) -> str | None:
    """ "pdf", "jpeg", "png", or None if the bytes are none of the formats we read."""
    if data.startswith(b"%PDF"):
        return "pdf"
    return next((kind for magic, kind in _IMAGE_MAGIC.items() if data.startswith(magic)), None)


def _upright(image: Image.Image) -> Image.Image:
    """Turn a sideways or upside-down scan the right way up (scanners and phones do this a lot;
    reading it as-is gives garbage such as 'SSVd :nsay'). If orientation cannot be detected,
    e.g. a page with almost no text, the image is left alone."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
    except Exception:
        return image
    angle = int(osd["rotate"])  # degrees clockwise needed to be upright
    if angle and float(osd["orientation_conf"]) >= OSD_MIN_CONFIDENCE:
        return image.rotate(-angle, expand=True)  # PIL rotates counter-clockwise
    return image


def _render(page: pymupdf.Page) -> bytes:
    """The page as a greyscale PNG at OCR resolution, capped in size."""
    longest_inches = max(page.rect.width, page.rect.height) / 72
    # the size cap always wins, even for a page that is metres across
    dpi = max(1, int(min(OCR_DPI, MAX_OCR_SIDE_PX / max(longest_inches, 0.01))))
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return bytes(pix.tobytes("png"))


def _read(image: Image.Image, lang: str) -> tuple[str, float | None]:
    """Text and mean word confidence from a single Tesseract pass.

    The text is rebuilt from the word table (lines in order, a blank line between paragraphs),
    so the confidence costs nothing extra.
    """
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        if not str(word).strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(str(word))
        if float(data["conf"][i]) >= 0:
            confidences.append(float(data["conf"][i]))
    out: list[str] = []
    previous: tuple[int, int] | None = None
    for (block, par, _line), words in lines.items():
        if previous is not None and previous != (block, par):
            out.append("")
        out.append(" ".join(words))
        previous = (block, par)
    confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
    return "\n".join(out).strip(), confidence


def _ocr_png(png: bytes, lang: str) -> tuple[str, float | None]:
    return _read(_upright(Image.open(io.BytesIO(png))), lang)


def _ocr_pages(pngs: Sequence[bytes], lang: str) -> list[tuple[str, float | None]]:
    if len(pngs) == 1:
        return [_ocr_png(pngs[0], lang)]
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:  # each call is its own process
        return list(pool.map(lambda p: _ocr_png(p, lang), pngs))


def extract_pages(
    source: str | Path | bytes,
    *,
    ocr: bool = True,
    min_chars: int = MIN_TEXT_CHARS,
    lang: str = "eng",
) -> ParsedDocument:
    """Read every page's text layer; OCR pages that have (almost) none.

    Accepts a PDF or a JPEG/PNG photo (which has no text layer, so it is always OCR'd). Scanned
    pages are handled per page, so a mixed document works. With ``ocr=False`` a scanned page
    comes back as empty ``text`` rather than failing.
    """
    if isinstance(source, bytes):
        kind = sniff_kind(source) or "pdf"
        doc = pymupdf.open(stream=source, filetype=kind)
        name = "<bytes>"
    else:
        doc = pymupdf.open(source)
        name = Path(source).name
    pages: list[PageText | None] = []
    with doc:
        batch: list[tuple[int, bytes]] = []  # (index into pages, rendered image)

        def flush() -> None:
            if not batch:
                return
            results = _ocr_pages([png for _, png in batch], lang)
            for (idx, _), (text, confidence) in zip(batch, results, strict=True):
                pages[idx] = PageText(page=idx + 1, text=text, method="ocr", confidence=confidence)
            batch.clear()

        for i in range(doc.page_count):
            page: pymupdf.Page = doc[i]
            text = str(page.get_text("text")).strip()
            if len(text) >= min_chars or not ocr:
                pages.append(PageText(page=i + 1, text=text, method="text"))
                continue
            pages.append(None)  # filled in by OCR
            batch.append((i, _render(page)))
            if len(batch) >= OCR_WORKERS * 2:  # bound memory: at most a few page images at once
                flush()
        flush()
    return ParsedDocument(source=name, pages=[p for p in pages if p is not None])
