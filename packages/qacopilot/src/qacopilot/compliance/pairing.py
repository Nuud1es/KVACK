"""Find the supplier spec that applies to a certificate of analysis. Pure code, no LLM.

A CoA names a product and a batch, not a spec. The spec to judge it against is the one for the
same product, and when a spec has several revisions, the revision in force on the report date.
Anything ambiguous is reported as such: judging a batch against the wrong spec gives a confident
wrong verdict, so asking a person is better than guessing.
"""

import re
from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal, Protocol

from pydantic import BaseModel

from qacopilot.extraction.dates import find_labelled_date
from qacopilot.extraction.schemas import CoA, SupplierSpec

_REVISION = re.compile(r"\b(?:revision|rev\.?|version|issue)\s*[:#.]?\s*(\d{1,3})\b", re.IGNORECASE)
_EFFECTIVE = re.compile(
    r"\b(?:effective(?:\s+from|\s+date)?|valid\s+from|date\s+of\s+issue|issue\s+date)\b", re.I
)
# "Microbiological limits are unchanged": the revision does not repeat the table.
_LIMITS_UNCHANGED = re.compile(
    r"\b(?:micro\w*\s+)?limits?\s+(?:are\s+|remain\s+)?(?:unchanged|not\s+changed)\b", re.I
)
# OCR sometimes runs the next label into the product ("Chicken thigh fillet Batch: B2701-02").
_LABEL_RUN_ON = re.compile(r"\s+\b(?:batch|lot|date|report)\b.*$", re.IGNORECASE)
_STOP = frozenset({"and", "with", "the", "of", "in", "for"})
MIN_SIMILARITY = 0.5


class SpecRecordLike(Protocol):
    doc_id: str
    filename: str
    data: dict[str, object]
    text: str
    ingested_at: datetime


class SpecCandidate(BaseModel):
    doc_id: str
    filename: str
    supplier: str | None
    product: str
    product_code: str | None
    revision: int | None  # read from the spec's own text, not the schema
    effective: date | None
    micro_limits: int
    limits_unchanged: bool = False  # the revision says its micro limits are as before
    ingested_at: datetime


class Pairing(BaseModel):
    status: Literal["paired", "ambiguous", "no_spec"]
    spec: SpecCandidate | None = None
    # paired: every revision of the matched spec; ambiguous: the competing specs
    candidates: list[SpecCandidate] = []
    notes: list[str] = []


def parse_revision(text: str) -> int | None:
    m = _REVISION.search(text)
    return int(m.group(1)) if m else None


def parse_effective(text: str) -> date | None:
    return find_labelled_date(text, _EFFECTIVE, window=30)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def clean_product(name: str) -> str:
    return _LABEL_RUN_ON.sub("", name).strip()


def product_words(name: str) -> frozenset[str]:
    return frozenset(
        _stem(w) for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _STOP and len(w) > 1
    )


def similarity(coa_product: str, spec_product: str) -> float:
    """0 unless every word of the CoA's product is in the spec's; then Jaccard overlap."""
    a, b = product_words(clean_product(coa_product)), product_words(spec_product)
    if not a or not b or not a <= b:
        return 0.0
    return len(a) / len(b)


def candidate_from_record(record: SpecRecordLike) -> SpecCandidate | None:
    try:
        spec = SupplierSpec.model_validate(record.data)
    except Exception:  # an extraction that no longer fits the schema is not usable
        return None
    return SpecCandidate(
        doc_id=record.doc_id,
        filename=record.filename,
        supplier=spec.supplier,
        product=spec.product,
        product_code=spec.product_code,
        revision=parse_revision(record.text),
        effective=parse_effective(record.text),
        micro_limits=len(spec.micro_limits),
        limits_unchanged=bool(_LIMITS_UNCHANGED.search(record.text)),
        ingested_at=record.ingested_at,
    )


def _identity(c: SpecCandidate) -> tuple[str, frozenset[str], str]:
    return ((c.supplier or "").strip().lower(), product_words(c.product), c.product_code or "")


def _order(c: SpecCandidate) -> tuple[int, date, datetime]:
    return (c.revision or 0, c.effective or date.min, c.ingested_at)


def pair_spec(coa: CoA, specs: Sequence[SpecCandidate]) -> Pairing:
    """The spec (and revision) a CoA should be judged against, or why that is not clear."""
    if not coa.product:
        return Pairing(
            status="no_spec", notes=["The report names no product, so no spec can be chosen."]
        )
    scored = [(similarity(coa.product, s.product), s) for s in specs]
    scored = [(score, s) for score, s in scored if score >= MIN_SIMILARITY]
    if not scored:
        return Pairing(
            status="no_spec",
            notes=[
                f"No extracted supplier spec matches the product {clean_product(coa.product)!r}."
            ],
        )
    best = max(score for score, _ in scored)
    top = [s for score, s in scored if score == best]
    groups: dict[tuple[str, frozenset[str], str], list[SpecCandidate]] = {}
    for s in top:
        groups.setdefault(_identity(s), []).append(s)
    if len(groups) > 1:
        latest = [max(g, key=_order) for g in groups.values()]
        return Pairing(
            status="ambiguous",
            candidates=latest,
            notes=[
                f"{len(latest)} different specs match {coa.product!r} equally well "
                "(different supplier, code or product wording); choose one."
            ],
        )

    revisions = sorted(next(iter(groups.values())), key=_order)
    newest = revisions[-1]
    chosen = newest
    notes: list[str] = []
    if coa.report_date is not None:
        in_force = [r for r in revisions if r.effective is None or r.effective <= coa.report_date]
        if in_force:
            chosen = in_force[-1]
        else:  # the report predates every revision we hold
            chosen = revisions[0]
            notes.append(
                f"The report ({coa.report_date}) is dated before the earliest spec held; "
                "check which revision applied."
            )
        if chosen is not newest:
            notes.append(
                f"The report is dated {coa.report_date}, before revision "
                f"{newest.revision or '?'} took effect"
                + (f" ({newest.effective})" if newest.effective else "")
                + f"; it was checked against {chosen.filename}."
            )
    elif len(revisions) > 1:
        notes.append(
            f"The report has no date, so the newest of {len(revisions)} revisions was used "
            f"({chosen.filename})."
        )
    same = [r for r in revisions if _order(r)[:2] == _order(chosen)[:2] and r is not chosen]
    if same:
        notes.append(
            "More than one file holds the same spec revision; the most recently added was used."
        )
    return Pairing(status="paired", spec=chosen, candidates=revisions, notes=notes)
