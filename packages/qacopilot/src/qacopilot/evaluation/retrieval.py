"""Retrieval evaluation: golden questions, per-question scoring, summaries and regression checks.

A golden item lists the (document, page) pairs that answer it. A hit counts if its document
matches and, when the expectation names a page, the page matches too. Items with no expected
pairs are out-of-scope questions whose right answer is "not in the documents"; they are not part
of the hit rates, but their top score is recorded so a "not found" threshold can be chosen.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from qacopilot.retrieval.store import Hit

Category = Literal[
    "lookup",
    "near_duplicate",
    "table",
    "clause",
    "scanned",
    "adversarial",
    "out_of_scope",
    "due_date",  # answered from the certificates table
    "coa_check",  # answered by the CoA-vs-spec check
]
KS = (1, 3, 5)
TRACKED = ("doc_hit@1", "doc_hit@3", "doc_hit@5", "page_hit@1", "page_hit@3", "mrr")


class Expected(BaseModel):
    doc_id: str
    page: int | None = None


class GoldenItem(BaseModel):
    id: str
    question: str
    category: Category
    expected: list[Expected] = Field(default_factory=list)  # any-of; empty = out of scope
    answer_contains: list[str] = Field(default_factory=list)  # all must appear in the answer
    answer_excludes: list[str] = Field(default_factory=list)  # none may appear (e.g. injected text)
    # Which path must answer it; None = either. A table/check question answered by search is a
    # routing regression even when the text happens to be right.
    via: Literal["documents", "certificates", "coa_check"] | None = None


class ItemResult(BaseModel):
    id: str
    category: Category
    doc_rank: int | None  # 1-based rank of the first hit on an expected document
    page_rank: int | None  # ... on an expected document and page
    top_doc: str | None
    top_score: float | None


def load_golden(path: str | Path) -> list[GoldenItem]:
    return [GoldenItem(**row) for row in json.loads(Path(path).read_text(encoding="utf-8"))]


def score_item(item: GoldenItem, hits: list[Hit]) -> ItemResult:
    doc_rank = page_rank = None
    for rank, hit in enumerate(hits, start=1):
        for exp in item.expected:
            if hit.chunk.doc_id != exp.doc_id:
                continue
            doc_rank = doc_rank or rank
            if exp.page is None or exp.page == hit.chunk.page:
                page_rank = page_rank or rank
    return ItemResult(
        id=item.id,
        category=item.category,
        doc_rank=doc_rank,
        page_rank=page_rank,
        top_doc=hits[0].chunk.doc_id if hits else None,
        top_score=hits[0].score if hits else None,
    )


def _rate(ranks: list[int | None], k: int) -> float:
    return (
        round(sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 3) if ranks else 0.0
    )


def _metrics(results: list[ItemResult]) -> dict[str, float]:
    doc = [r.doc_rank for r in results]
    page = [r.page_rank for r in results]
    out = {f"doc_hit@{k}": _rate(doc, k) for k in KS}
    out["page_hit@1"] = _rate(page, 1)
    out["page_hit@3"] = _rate(page, 3)
    out["mrr"] = round(sum(1 / r for r in doc if r) / len(doc), 3) if doc else 0.0
    return out


def summarize(results: list[ItemResult]) -> dict[str, Any]:
    in_scope = [r for r in results if r.category != "out_of_scope"]
    by_cat: dict[str, list[ItemResult]] = defaultdict(list)
    for r in in_scope:
        by_cat[r.category].append(r)
    oos = [r for r in results if r.category == "out_of_scope"]
    in_scores = [r.top_score for r in in_scope if r.top_score is not None]
    oos_scores = [r.top_score for r in oos if r.top_score is not None]
    return {
        "n_in_scope": len(in_scope),
        **_metrics(in_scope),
        "by_category": {c: {"n": len(rs), **_metrics(rs)} for c, rs in sorted(by_cat.items())},
        "out_of_scope": {
            "n": len(oos),
            "top_scores": sorted(round(s, 3) for s in oos_scores),
            "in_scope_top_scores_min": round(min(in_scores), 3) if in_scores else None,
        },
    }


def regressions(
    baseline: dict[str, Any], current: dict[str, Any], tolerance: float = 0.0
) -> list[str]:
    """Metrics in ``current`` that are lower than ``baseline`` by more than ``tolerance``."""
    out = []
    for key in TRACKED:
        if current[key] < baseline[key] - tolerance:
            out.append(f"{key}: {baseline[key]} -> {current[key]}")
    return out
