"""Answer "does batch X meet its spec?" by running the CoA-vs-spec check, not by reading chunks.

Whether a result is inside a limit is arithmetic on printed values (``<10``, ``2.2E+03``), and
which spec applies is a matching problem. Both are done in code; the LLM is not involved. A
question that names no batch we hold, or whose report is not extracted, is left to search.
"""

import logging
import re
from typing import Any, Protocol

from qacopilot.compliance import CheckFailure, CoACheck, Verdict, run_coa_check
from qacopilot.compliance.service import CheckSource
from qacopilot.ingest.models import DocType
from qacopilot.rag.answer import AskEvent, Citation, DoneEvent, SourcesEvent, TokenEvent
from qacopilot.repositories.documents import ExtractionRecord

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 240
MAX_BATCHES = 5
MAX_ROWS_SHOWN = 8

_JUDGE = re.compile(
    r"\b(meets?|met|compl(?:y|ies|iant|iance)|within|pass(?:es|ed)?|fail(?:s|ed)?|conform\w*"
    r"|accept\w*|in\s+spec|out\s+of\s+spec|spec|limits?)\b",
    re.IGNORECASE,
)
_ABOUT_BATCH = re.compile(
    r"\b(batch|lot|coa|certificate\s+of\s+analysis|analysis)\b", re.IGNORECASE
)
# A batch reference such as B2613-01 (letters optional, digits required).
_BATCH_TOKEN = re.compile(r"\b[a-z]{0,3}\d{3,}(?:[-/]\d+)?\b", re.IGNORECASE)


class CoASource(CheckSource, Protocol):
    async def get_page(self, doc_id: str, page: int) -> Any: ...


def is_coa_question(question: str) -> bool:
    has_batch_ref = bool(_BATCH_TOKEN.search(question))
    return bool(_JUDGE.search(question) and (has_batch_ref or _ABOUT_BATCH.search(question)))


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def batches_in(question: str) -> list[str]:
    return [t for t in (_alnum(m.group(0)) for m in _BATCH_TOKEN.finditer(question)) if len(t) >= 4]


def match_reports(question: str, records: list[ExtractionRecord]) -> list[ExtractionRecord]:
    """CoAs whose batch number is named in the question (newest document first, one per batch)."""
    wanted = batches_in(question)
    seen: set[str] = set()
    out: list[ExtractionRecord] = []
    for r in records:
        batch = _alnum(str(r.data.get("batch", "")))
        if batch and batch not in seen and any(w == batch or w in batch for w in wanted):
            seen.add(batch)
            out.append(r)
    return out


def _snippet(found: Any) -> str:
    return found[1].text.strip()[:SNIPPET_CHARS] if found is not None else ""


async def _citation(source: CoASource, n: int, doc_id: str, filename: str) -> Citation:
    return Citation(
        n=n,
        doc_id=doc_id,
        filename=filename,
        page=1,
        score=1.0,
        snippet=_snippet(await source.get_page(doc_id, 1)) or filename,
    )


def _verdict_word(v: Verdict) -> str:
    return {"pass": "PASS", "fail": "FAIL", "review": "REVIEW"}[v.value]


def _describe(
    check: CoACheck, batch: str, product: str | None, cites: tuple[int, int]
) -> list[str]:
    spec = check.spec_filename or "the spec you named"
    rev = f", revision {check.spec_revision}" if check.spec_revision else ""
    what = f"Batch {batch}" + (f" ({product})" if product else "")
    head = f"{what} against {spec}{rev}: {_verdict_word(check.overall)} [{cites[0]}][{cites[1]}]."
    lines = [head]
    problems = [r for r in check.rows if r.verdict is not Verdict.PASS]
    for r in problems[:MAX_ROWS_SHOWN]:
        limit = f" vs limit {r.limit}" if r.limit else ""
        got = r.result or "not tested"
        lines.append(f"  - {r.test}: {got}{limit} -> {_verdict_word(r.verdict)} ({r.reason})")
    if len(problems) > MAX_ROWS_SHOWN:
        lines.append(f"  - ...and {len(problems) - MAX_ROWS_SHOWN} more.")
    passed = len(check.rows) - len(problems)
    if passed:
        lines.append(f"  {passed} result(s) within limits.")
    lines.extend(f"  Note: {w}" for w in check.warnings)
    return lines


async def answer_from_coa(question: str, source: CoASource) -> list[AskEvent] | None:
    """Events giving the check result for the batch(es) named, or None to use search."""
    if not is_coa_question(question) or not batches_in(question):
        return None
    reports = match_reports(question, await source.extraction_records(DocType.COA))
    if not reports:
        return None

    citations: list[Citation] = []
    lines: list[str] = []
    for report in reports[:MAX_BATCHES]:
        batch = str(report.data.get("batch", report.doc_id))
        product = report.data.get("product")
        n = len(citations) + 1
        try:
            check = await run_coa_check(source, report.doc_id)
        except CheckFailure as exc:
            citations.append(await _citation(source, n, report.doc_id, report.filename))
            lines.append(f"Batch {batch} could not be checked: {exc.message} [{n}]")
            continue
        spec_doc = await source.get_document(check.spec_doc_id)
        spec_name = check.spec_filename or (spec_doc.filename if spec_doc else check.spec_doc_id)  # type: ignore[attr-defined]
        citations.append(await _citation(source, n, report.doc_id, report.filename))
        citations.append(await _citation(source, n + 1, check.spec_doc_id, spec_name))
        lines.extend(
            _describe(check, batch, product if isinstance(product, str) else None, (n, n + 1))
        )
    text = "\n".join(lines)
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
    return [
        SourcesEvent(citations=citations),
        TokenEvent(text=text),
        DoneEvent(answer=text, answered=True, cited=cited, via="coa_check"),
    ]
