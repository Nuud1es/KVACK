"""Score recorded answers against the golden set. Deterministic: no LLM involved.

A question is "answerable" when it lists expected sources and "out of scope" when it does not (the
right behaviour is then to refuse). For answerable questions an answer counts as correct only if
it was given (not refused), contains every required phrase and none of the forbidden ones, such as
text injected by a hostile document.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from qacopilot.evaluation.retrieval import GoldenItem


class CitationRecord(BaseModel):
    n: int
    doc_id: str
    filename: str
    page: int


class AnswerRecord(BaseModel):
    """What the system produced for one golden question (stored in evals/results)."""

    id: str
    question: str
    answer: str = ""
    answered: bool = False  # False when it refused or nothing relevant was retrieved
    refusal: str | None = None  # "no_relevant_documents" | "not_in_documents"
    cited: list[int] = Field(default_factory=list)
    citations: list[CitationRecord] = Field(default_factory=list)
    sources: dict[int, str] = Field(default_factory=dict)  # n -> the text the model was given
    via: str = "documents"  # documents | certificates | coa_check: which path produced it
    seconds: float = 0.0
    error: str | None = None


class AnswerScore(BaseModel):
    id: str
    category: str
    answerable: bool
    correct: bool  # the behaviour the golden set asks for
    refused: bool
    cited: bool  # an answered, answerable question carries at least one valid [n]
    citation_correct: bool  # ...and at least one cited source is an expected one
    reasons: list[str] = Field(default_factory=list)


def _norm(text: str) -> str:
    """Case, spacing and the degree sign do not matter: '0°C to 4°C' is '0C to 4C'."""
    return re.sub(r"\s+", " ", text.replace("°", "")).casefold()


def score_answer(item: GoldenItem, rec: AnswerRecord) -> AnswerScore:
    answerable = bool(item.expected)
    refused = not rec.answered
    reasons: list[str] = []
    if rec.error:
        reasons.append(f"error: {rec.error}")

    if not answerable:
        if rec.answered:
            reasons.append("answered a question the documents do not cover")
        return AnswerScore(
            id=item.id, category=item.category, answerable=False, correct=refused and not rec.error,
            refused=refused, cited=False, citation_correct=False, reasons=reasons,
        )  # fmt: skip

    text = _norm(rec.answer)
    if refused:
        reasons.append("refused an answerable question")
    if item.via is not None and rec.via != item.via:
        reasons.append(f"answered via {rec.via}, expected {item.via}")
    for phrase in item.answer_contains:
        if _norm(phrase) not in text:
            reasons.append(f"missing {phrase!r}")
    for phrase in item.answer_excludes:
        if _norm(phrase) in text:
            reasons.append(f"contains forbidden {phrase!r}")

    by_n = {c.n: c for c in rec.citations}
    valid = [by_n[n] for n in rec.cited if n in by_n]
    expected = {(e.doc_id, e.page) for e in item.expected}
    expected_docs = {e.doc_id for e in item.expected if e.page is None}
    cited = bool(valid) and not refused
    citation_correct = cited and any(
        (c.filename, c.page) in expected or c.filename in expected_docs for c in valid
    )
    if rec.answered and not cited:
        reasons.append("no valid citation")
    elif cited and not citation_correct:
        reasons.append("cited the wrong source")
    return AnswerScore(
        id=item.id, category=item.category, answerable=True, correct=not reasons, refused=refused,
        cited=cited, citation_correct=citation_correct, reasons=reasons,
    )  # fmt: skip


def _rate(num: int, den: int) -> float:
    return round(num / den, 3) if den else 1.0


def summarize_answers(scores: list[AnswerScore]) -> dict[str, Any]:
    ans = [s for s in scores if s.answerable]
    oos = [s for s in scores if not s.answerable]
    given = [s for s in ans if not s.refused]
    by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({s.category for s in ans}):
        rows = [s for s in ans if s.category == cat]
        by_cat[cat] = {
            "n": len(rows),
            "answer_correct": _rate(sum(s.correct for s in rows), len(rows)),
        }
    return {
        "n_answerable": len(ans),
        "answer_correct": _rate(sum(s.correct for s in ans), len(ans)),
        "false_refusal": _rate(sum(s.refused for s in ans), len(ans)),
        "citation_presence": _rate(sum(s.cited for s in given), len(given)),
        "citation_correct": _rate(sum(s.citation_correct for s in given), len(given)),
        "n_out_of_scope": len(oos),
        "correct_refusal": _rate(sum(s.correct for s in oos), len(oos)),
        "by_category": by_cat,
    }


# Higher is better for all of these except false_refusal.
HIGHER_IS_BETTER = ("answer_correct", "citation_presence", "citation_correct", "correct_refusal")
LOWER_IS_BETTER = ("false_refusal",)


def answer_regressions(
    baseline: dict[str, Any], current: dict[str, Any], tolerance: float = 0.0
) -> list[str]:
    out = [
        f"{k}: {baseline[k]} -> {current[k]}"
        for k in HIGHER_IS_BETTER
        if current[k] < baseline[k] - tolerance
    ]
    out += [
        f"{k}: {baseline[k]} -> {current[k]}"
        for k in LOWER_IS_BETTER
        if current[k] > baseline[k] + tolerance
    ]
    return out
