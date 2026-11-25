"""Run a CoA-vs-spec check end to end: find the spec, compare in code, flag unverified inputs."""

from datetime import date
from typing import Protocol

from pydantic import BaseModel

from qacopilot.compliance.check import (
    CheckRow,
    cap_for_unverified_source,
    check_coa_against_spec,
    overall,
)
from qacopilot.compliance.limits import Verdict
from qacopilot.compliance.pairing import Pairing, SpecCandidate, candidate_from_record, pair_spec
from qacopilot.extraction.schemas import CoA, SupplierSpec
from qacopilot.ingest.models import DocType
from qacopilot.repositories.documents import ExtractionRecord

SPEC_TEXT_PAGES = 3


class CheckSource(Protocol):
    async def get_extraction(self, doc_id: str) -> "ExtractionLike | None": ...

    async def extraction_records(
        self, doc_type: DocType, *, with_text_pages: int = 0
    ) -> list[ExtractionRecord]: ...

    async def get_document(self, doc_id: str) -> "DocumentLike | None": ...

    async def poor_ocr_pages(self, doc_id: str) -> int: ...


class ExtractionLike(Protocol):
    doc_type: str
    data: dict[str, object]


class DocumentLike(Protocol):
    ocr_pages: int


class CoACheck(BaseModel):
    coa_doc_id: str
    spec_doc_id: str
    overall: Verdict
    rows: list[CheckRow]
    warnings: list[str] = []  # OCR, odd pairings: a person should look before relying on this
    # how the spec was chosen: "given" (caller named it) or "matched" (found by product)
    pairing: str = "given"
    spec_filename: str | None = None
    spec_revision: int | None = None
    spec_effective: str | None = None
    # a revision that says its limits are unchanged does not repeat them: where they came from
    limits_from: str | None = None


class CheckFailure(Exception):
    """The check could not run; ``status`` is the HTTP-style reason, ``candidates`` any options."""

    def __init__(self, status: int, message: str, candidates: list[SpecCandidate] | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.candidates = candidates or []


def _earlier_with_limits(pairing: Pairing) -> SpecCandidate | None:
    """The newest older revision of the matched spec that lists micro limits."""
    chosen = pairing.spec
    older = [
        c
        for c in pairing.candidates
        if chosen is not None
        and c.doc_id != chosen.doc_id
        and c.micro_limits > 0
        and (c.revision or 0, c.effective or date.min)
        <= (chosen.revision or 0, chosen.effective or date.max)
    ]
    return max(
        older, key=lambda c: (c.revision or 0, c.effective or date.min, c.ingested_at), default=None
    )


async def _ocr_warning(source: CheckSource, label: str, doc_id: str) -> str | None:
    document = await source.get_document(doc_id)
    if document is None or document.ocr_pages <= 0:
        return None
    poor = await source.poor_ocr_pages(doc_id)
    return (
        f"The {label} was read by OCR ({document.ocr_pages} page(s)"
        + (f", {poor} of them a poor scan" if poor else "")
        + "): check the figures against the original before relying on this result."
    )


async def run_coa_check(source: CheckSource, coa_id: str, spec_id: str | None = None) -> CoACheck:
    """Check one CoA against a spec: the one named, or else the one that matches its product."""
    coa_row = await source.get_extraction(coa_id)
    if coa_row is None:
        raise CheckFailure(404, "The report must be extracted first")
    if coa_row.doc_type != DocType.COA.value:
        raise CheckFailure(422, "Expected a certificate of analysis")
    coa = CoA.model_validate(coa_row.data)

    notes: list[str] = []
    pairing: Pairing | None = None
    if spec_id is None:
        records = await source.extraction_records(
            DocType.SUPPLIER_SPEC, with_text_pages=SPEC_TEXT_PAGES
        )
        candidates = [c for r in records if (c := candidate_from_record(r)) is not None]
        pairing = pair_spec(coa, candidates)
        if pairing.status == "no_spec":
            raise CheckFailure(404, " ".join(pairing.notes) or "No matching spec")
        if pairing.status == "ambiguous" or pairing.spec is None:
            names = ", ".join(c.filename for c in pairing.candidates)
            raise CheckFailure(
                409, " ".join(pairing.notes) + f" Candidates: {names}.", pairing.candidates
            )
        spec_id = pairing.spec.doc_id
        notes = list(pairing.notes)

    spec_row = await source.get_extraction(spec_id)
    if spec_row is None:
        raise CheckFailure(404, "The spec must be extracted first")
    if spec_row.doc_type != DocType.SUPPLIER_SPEC.value:
        raise CheckFailure(422, "Expected a supplier spec")
    spec = SupplierSpec.model_validate(spec_row.data)

    limits_from: str | None = None
    if pairing is not None and pairing.spec is not None and not spec.micro_limits:
        earlier = _earlier_with_limits(pairing)
        if pairing.spec.limits_unchanged and earlier is not None:
            earlier_row = await source.get_extraction(earlier.doc_id)
            if earlier_row is not None:
                spec = spec.model_copy(
                    update={
                        "micro_limits": SupplierSpec.model_validate(earlier_row.data).micro_limits
                    }
                )
                limits_from = earlier.filename
                notes.append(
                    f"{pairing.spec.filename} says its microbiological limits are unchanged "
                    f"and does not repeat them, so the limits from {earlier.filename}"
                    + (f" (revision {earlier.revision})" if earlier.revision else "")
                    + " were used."
                )
    rows = check_coa_against_spec(coa, spec)
    verdict = overall(rows)  # a spec limit missing from the report already counts as 'review'
    warnings = notes
    if not spec.micro_limits:
        warnings.append(
            "This spec lists no microbiological limits, so the limits printed on the report "
            "were used where there were any."
        )
    for label, doc_id in (("report", coa_id), ("specification", spec_id)):
        if (w := await _ocr_warning(source, label, doc_id)) is not None:
            warnings.append(w)
            verdict = cap_for_unverified_source(verdict)
    for label, row in (("report", coa_row), ("specification", spec_row)):
        if getattr(row, "needs_review", False):  # read incompletely and not yet confirmed
            why = getattr(row, "review_reason", None)
            warnings.append(
                f"The {label} has a reading nobody has confirmed ({why}): check it against the "
                "original before relying on this result."
            )
            verdict = cap_for_unverified_source(verdict)
    chosen = pairing.spec if pairing is not None else None
    return CoACheck(
        coa_doc_id=coa_id,
        spec_doc_id=spec_id,
        overall=verdict,
        rows=rows,
        warnings=warnings,
        pairing="matched" if pairing is not None else "given",
        spec_filename=chosen.filename if chosen else None,
        spec_revision=chosen.revision if chosen else None,
        spec_effective=chosen.effective.isoformat() if chosen and chosen.effective else None,
        limits_from=limits_from,
    )
