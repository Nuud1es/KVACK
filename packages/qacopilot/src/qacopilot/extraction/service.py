"""LLM-backed extraction of structured fields from a parsed document."""

from pydantic import BaseModel

from qacopilot.extraction.dates import NEXT_DUE, find_labelled_date
from qacopilot.extraction.prompts import EXTRACT_SYSTEM, MAX_CHARS
from qacopilot.extraction.schemas import CalibrationCert, CoA, SupplierSpec
from qacopilot.ingest.chunking import header_fields
from qacopilot.ingest.models import DocType, ParsedDocument
from qacopilot.llm import BaseLLMProvider

Extracted = CalibrationCert | CoA | SupplierSpec

SCHEMAS: dict[DocType, type[BaseModel]] = {
    DocType.CALIBRATION_CERT: CalibrationCert,
    DocType.COA: CoA,
    DocType.SUPPLIER_SPEC: SupplierSpec,
}


class UnsupportedDocType(ValueError):
    """No extraction schema exists for this document type."""


def _document_text(doc: ParsedDocument) -> str:
    text = "\n\n".join(f"--- page {p.page} ---\n{p.text}" for p in doc.pages)
    return text[:MAX_CHARS]


async def extract_document(
    llm: BaseLLMProvider, doc: ParsedDocument, doc_type: DocType, *, thinking: bool = False
) -> Extracted:
    """Extract the fields for ``doc_type``; raises ``StructuredOutputError`` if the model
    cannot produce valid output after its retries, ``UnsupportedDocType`` for other types."""
    schema = SCHEMAS.get(doc_type)
    if schema is None:
        raise UnsupportedDocType(doc_type.value)
    result = await llm.extract(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Document:\n\n{_document_text(doc)}"},
        ],
        schema,
        thinking=thinking,
    )
    assert isinstance(result, CalibrationCert | CoA | SupplierSpec)
    return _fill_from_text(result, doc)


_MAX_HEADER_VALUE = 120


def _fill_from_text(result: Extracted, doc: ParsedDocument) -> Extracted:
    """Fill fields the model left empty but the document prints with a clear label.

    * A supplier or product from a ``Supplier:`` / ``Product:`` header line.
    * A certificate's next-due date from a label such as "Next calibration due". A missing one
      would hide the instrument from expiry reminders, so it is worth a safety net.

    A model answering without reasoning often skips optional fields that are plainly printed
    ("Supplier: Example Poultry Ltd"). The chunker already reads those header lines with a plain
    pattern, so this is code reading text, not a guess. It only fills a blank: whatever the model
    found is kept, and a document with no labelled header is left alone.
    """
    header = header_fields(doc)
    if isinstance(result, CalibrationCert) and result.next_due is None:
        found = find_labelled_date("\n".join(p.text for p in doc.pages), NEXT_DUE)
        # only a date after the calibration date can be a next-due date; anything else is a misread
        if found is not None and found > result.calibration_date:
            return result.model_copy(update={"next_due": found})

    def usable(key: str) -> str | None:
        value = header.get(key)
        return value if value and len(value) <= _MAX_HEADER_VALUE else None

    if isinstance(result, SupplierSpec) and not result.supplier:
        if supplier := usable("supplier"):
            return result.model_copy(update={"supplier": supplier})
    if isinstance(result, CoA) and not result.product:
        if product := usable("product") or usable("product_name"):
            return result.model_copy(update={"product": product})
    return result
