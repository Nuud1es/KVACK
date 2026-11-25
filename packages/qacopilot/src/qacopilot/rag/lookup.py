"""Answer "when is X due?" from the certificates table instead of from search results.

Search cannot tell which of several near-identical certificates is the current one, so it can
cite a superseded date with confidence. The extracted certificates table already knows the
latest certificate per instrument (see ``DocumentRepository.latest_certificates``), so these
questions are answered from it, deterministically, with the source page cited. The LLM is not
involved: the facts are dates and serial numbers, which code reads better than a model does.

When a question is not clearly about calibration due dates, or names an instrument that is not in
the table, nothing here answers it and the caller falls back to document search.
"""

import logging
import re
from collections.abc import AsyncIterator
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol

from qacopilot.rag.answer import AskEvent, Citation, DoneEvent, SourcesEvent, TokenEvent
from qacopilot.timewindow import parse_window_days, window_cap_note

if TYPE_CHECKING:
    from qacopilot.repositories.documents import InstrumentCertificate

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 240
MAX_LISTED = 10

_DUE = re.compile(
    r"\b(due|expir\w*|renew\w*|recalibrat\w*|valid\s+(?:until|to)|next\s+cal\w*)\b",
    re.IGNORECASE,
)
_CALIBRATION = re.compile(
    r"\b(calibrat\w*|certificates?|certs?|probes?|thermometers?|scales?|balances?|detectors?"
    r"|x-?rays?|instruments?|equipment|serial)\b",
    re.IGNORECASE,
)
# An instrument serial such as TP-20931 or XR 30918 (the table decides whether it is real).
# Case-sensitive on purpose: "next 400" is not a serial, "TP 20931" and "tp-20931" are.
_SERIAL_LIKE = re.compile(r"\b(?:[A-Z]{1,4}[- ]?\d{3,}|[A-Za-z]{1,4}-\d{3,})\b")
# Looser, for deciding whether a question is about a due date at all ("when is tp 20931 due").
_SERIAL_LIKE_LOOSE = re.compile(r"\b[a-z]{1,4}[- ]?\d{3,}\b", re.IGNORECASE)
# Other kinds of certificate or document: these are not calibration due dates.
_OTHER_DOCS = re.compile(
    r"\b(brc\w*|audit\w*|iso|haccp|specs?|specifications?|coa|analysis|customer|supplier"
    r"|insurance|licen[cs]e|training)\b",
    re.IGNORECASE,
)
_AGGREGATE_WORD = re.compile(r"\b(which|what|list|all|any|show|how\s+many)\b", re.IGNORECASE)
_AGGREGATE_NOUN = re.compile(
    r"\b(certificates|certs|instruments|calibrations|probes|thermometers|scales|equipment)\b",
    re.IGNORECASE,
)
# Words in a description that do not say what kind of instrument it is.
_GENERIC = frozenset({"digital", "unit", "line", "inspection", "machine", "system", "the", "and"})


class CertificateSource(Protocol):
    """The repository methods this module needs (so tests can use a small fake)."""

    async def latest_certificates(self) -> "list[InstrumentCertificate]": ...

    async def unextracted_calibration_documents(self) -> int: ...

    async def get_page(self, doc_id: str, page: int) -> Any: ...


def is_due_question(question: str) -> bool:
    """A question about when calibration certificates or instruments are due."""
    return bool(
        _DUE.search(question)
        and (_CALIBRATION.search(question) or _SERIAL_LIKE_LOOSE.search(question))
    )


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3} - _GENERIC


def select_certificates(
    question: str, certs: "list[InstrumentCertificate]", *, today: date
) -> "tuple[list[InstrumentCertificate], str] | None":
    """The instruments the question is about, and how they were chosen; None = not ours.

    Order: a serial number named in the question; then instrument words ("probe", "scale");
    then a list question ("which certificates expire next month"). Anything else is left to
    document search rather than guessed.
    """
    q_alnum = _alnum(question)
    by_serial = [
        c
        for c in certs
        if len(_alnum(c.instrument_serial)) >= 4 and _alnum(c.instrument_serial) in q_alnum
    ]
    if by_serial:
        return by_serial, "serial"

    # A certificate number (CAL-2026-0412) names that certificate's instrument.
    by_number = [
        c
        for c in certs
        if len(_alnum(c.certificate_number or "")) >= 6
        and _alnum(c.certificate_number or "") in q_alnum
    ]
    if by_number:
        return by_number, "serial"
    # A serial-like reference we do not hold (TP-99999) must not fall through to "the probes":
    # listing other instruments' dates would answer a question that was not asked.
    if _SERIAL_LIKE.search(question):
        return None

    q_words = _words(question)
    by_kind = [
        c for c in certs if c.instrument_description and q_words & _words(c.instrument_description)
    ]
    if by_kind:
        return by_kind, "kind"

    if (
        _AGGREGATE_WORD.search(question)
        and _AGGREGATE_NOUN.search(question)
        and not _OTHER_DOCS.search(question)
    ):
        window = parse_window_days(question)
        return [c for c in certs if c.days_left is not None and c.days_left <= window], "window"
    return None


def _when(c: "InstrumentCertificate") -> str:
    if c.next_due is None or c.days_left is None:
        return "the certificate states no due date"
    if c.days_left < 0:
        return f"OVERDUE by {-c.days_left} days (was due {c.next_due.isoformat()})"
    if c.days_left == 0:
        return f"due today ({c.next_due.isoformat()})"
    return f"due {c.next_due.isoformat()}, in {c.days_left} days"


def _line(n: int, c: "InstrumentCertificate") -> str:
    name = (
        f"{c.instrument_serial} ({c.instrument_description})"
        if c.instrument_description
        else (c.instrument_serial)
    )
    cert = f"certificate {c.certificate_number}, " if c.certificate_number else ""
    calibrated = c.calibration_date.isoformat()
    text = f"{name}: next calibration {_when(c)} ({cert}calibrated {calibrated}) [{n}]."
    if c.unverified:
        text += (
            " UNVERIFIED: this reading is from a poor scan or has odd dates and no one has "
            "confirmed it; check it on the Documents page."
        )
    if c.older:
        older = ", ".join(
            f"{o.certificate_number or 'undated certificate'}"
            + (f" (due {o.next_due.isoformat()})" if o.next_due else "")
            for o in c.older
        )
        text += f" Earlier certificate(s) now superseded: {older}."
    return text


async def _citation(source: CertificateSource, n: int, c: "InstrumentCertificate") -> Citation:
    snippet = ""
    found = await source.get_page(c.doc_id, c.start_page)
    if found is not None:
        snippet = found[1].text.strip()[:SNIPPET_CHARS]
    return Citation(
        n=n,
        doc_id=c.doc_id,
        filename=c.filename,
        page=c.start_page,
        score=1.0,
        snippet=snippet or f"{c.instrument_serial}, next due {c.next_due}",
    )


async def answer_from_certificates(
    question: str, source: CertificateSource, *, today: date | None = None
) -> list[AskEvent] | None:
    """Events answering a due-date question from the table, or None to use document search."""
    if not is_due_question(question):
        return None
    today = today or date.today()
    # The repository computes days_left against its own "today"; recompute for a pinned date.
    certs = [
        c.model_copy(update={"days_left": (c.next_due - today).days if c.next_due else None})
        for c in await source.latest_certificates()
    ]
    if not certs:
        return None
    chosen = select_certificates(question, certs, today=today)
    if chosen is None:
        return None
    matched, how = chosen

    unread = await source.unextracted_calibration_documents()
    note = (
        f"Note: {unread} calibration document(s) have not been read yet (use Extract on the "
        "Documents page), so they are not included."
        if unread
        else ""
    )

    if not matched:  # a list question where nothing is due
        window = parse_window_days(question)
        text = " ".join(
            p
            for p in (
                f"No calibration certificates are overdue or due within {window} days.",
                window_cap_note(question),
                note,
            )
            if p
        )
        return [
            TokenEvent(text=text),
            DoneEvent(answer=text, answered=True, cited=[], via="certificates"),
        ]

    shown = matched[:MAX_LISTED]
    citations = [await _citation(source, i, c) for i, c in enumerate(shown, start=1)]
    lines = [_line(i, c) for i, c in enumerate(shown, start=1)]
    if how == "window":
        window = parse_window_days(question)
        header = (
            f"{len(matched)} instrument(s) are overdue or due within {window} days (soonest first):"
        )
        if cap := window_cap_note(question):
            header += f" {cap}"
        lines = [header, *lines]
    elif len(shown) > 1:
        lines = ["Several instruments match; each is shown with its latest certificate:", *lines]
    if len(matched) > len(shown):
        lines.append(f"...and {len(matched) - len(shown)} more.")
    if note:
        lines.append(note)
    text = "\n".join(lines)
    return [
        SourcesEvent(citations=citations),
        TokenEvent(text=text),
        DoneEvent(
            answer=text, answered=True, cited=list(range(1, len(shown) + 1)), via="certificates"
        ),
    ]


async def answer_with_lookup(
    question: str,
    source: Any,
    fallback: AsyncIterator[AskEvent],
    llm: Any = None,
    context: Any = None,
) -> AsyncIterator[AskEvent]:
    """A table or check answer if there is one, else the document-search events.

    ``source`` is a ``DocumentRepository`` (or None without a database). Certificate due dates
    come from the certificates table, "does batch X meet spec" from the CoA check and packaging
    questions from the Fact Spec packaging table; anything
    else, and any database problem, falls back to search so Ask keeps working. ``context`` is the
    query behind the previous answer, so "give me all from Tesco" refines it.
    """
    events: list[AskEvent] | None = None
    if source is not None:
        try:
            events = await answer_from_certificates(question, source)
            if events is None:
                from qacopilot.rag.coa_lookup import answer_from_coa

                events = await answer_from_coa(question, source)
            if events is None:
                from qacopilot.rag.specquery import answer_from_specquery

                events = await answer_from_specquery(question, source, llm, context)
            if events is None:
                from qacopilot.rag.packaging import answer_from_packaging

                events = await answer_from_packaging(question, source)
            if events is None:
                from qacopilot.rag.factspec_lookup import answer_from_factspec

                events = await answer_from_factspec(question, source)
            if events is None:
                from qacopilot.rag.specquery import answer_vague

                events = await answer_vague(question, source)
            if events is None and await source.fact_spec_count() > 0:
                from qacopilot.rag.specquery import GUARD_MESSAGE, get_vocab, looks_like_aggregate

                if looks_like_aggregate(question, await get_vocab(source)):
                    events = [
                        TokenEvent(text=GUARD_MESSAGE),
                        DoneEvent(answer=GUARD_MESSAGE, answered=True, cited=[], via="fact_spec"),
                    ]
        except Exception:
            logger.exception("table lookup failed; using document search")
            events = None
    if events is not None:
        await fallback.aclose()  # type: ignore[attr-defined]
        for e in events:
            yield e
        return
    async for e in fallback:
        yield e
