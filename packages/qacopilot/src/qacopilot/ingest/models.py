"""Plain data types shared by the ingestion steps."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class DocType(StrEnum):
    CALIBRATION_CERT = "calibration_cert"
    COA = "coa"
    SUPPLIER_SPEC = "supplier_spec"
    FACT_SPEC = "fact_spec"  # Gressingham's own product spec (packaging table, raw materials)
    AUDIT_CERT = "audit_cert"
    SOP = "sop"
    HACCP = "haccp"
    CUSTOMER_COP = "customer_cop"
    OTHER = "other"


# Below this mean word confidence (0-100) an OCR'd page is a poor scan: digits and dates are likely
# to be misread, so what is extracted from it needs a person's eye.
POOR_OCR_CONFIDENCE = 75.0


class PageText(BaseModel):
    page: int  # 1-based
    text: str
    method: Literal["text", "ocr"]
    confidence: float | None = None  # OCR only: mean word confidence, 0-100


class ParsedDocument(BaseModel):
    source: str
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class Classification(BaseModel):
    doc_type: DocType
    confidence: float  # 0..1; share of the winning type's rules that matched
    matched: list[str]
    source: Literal["rules", "llm", "user"] = "rules"  # who decided
    reason: str | None = None  # the LLM's one-line explanation, when it decided or disagreed
    # A type the LLM suggested instead of the rules' unconfident answer. The rules' answer is kept
    # (a specific keyword match is not silently overruled); this just flags it for a person.
    alternative: DocType | None = None
    # How far ahead of the next-best type the winner was (score points), and which type that was.
    # None when no type scored enough. A small margin means the answer is close to a coin flip
    # even if the winner's own confidence looks high.
    margin: float | None = None
    runner_up: DocType | None = None
