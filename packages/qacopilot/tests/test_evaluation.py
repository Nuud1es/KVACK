from qacopilot.evaluation import GoldenItem, load_golden, regressions, score_item, summarize
from qacopilot.evaluation.retrieval import Expected
from qacopilot.ingest import Chunk, DocType
from qacopilot.retrieval import Hit


def _hit(doc: str, page: int = 1, score: float = 1.0) -> Hit:
    chunk = Chunk(id=f"{doc}{page}", doc_id=doc, doc_type=DocType.SOP, index=0, text="x", page=page)
    return Hit(chunk=chunk, score=score)


def _item(*expected: Expected, category: str = "lookup") -> GoldenItem:
    return GoldenItem(id="q", question="?", category=category, expected=list(expected))  # type: ignore[arg-type]


def test_doc_and_page_ranks_differ_when_the_page_is_wrong() -> None:
    item = _item(Expected(doc_id="a.pdf", page=2))
    r = score_item(item, [_hit("b.pdf"), _hit("a.pdf", 1), _hit("a.pdf", 2)])
    assert (r.doc_rank, r.page_rank) == (2, 3)


def test_any_of_expectations_and_no_page_constraint() -> None:
    item = _item(Expected(doc_id="a.pdf"), Expected(doc_id="b.pdf", page=1))
    assert score_item(item, [_hit("b.pdf", 5), _hit("a.pdf", 9)]).page_rank == 2


def test_miss_has_no_rank() -> None:
    r = score_item(_item(Expected(doc_id="a.pdf")), [_hit("z.pdf")])
    assert r.doc_rank is None and r.page_rank is None and r.top_doc == "z.pdf"


def test_summary_excludes_out_of_scope_from_rates_but_keeps_their_scores() -> None:
    hit = score_item(_item(Expected(doc_id="a.pdf")), [_hit("a.pdf", score=0.9)])
    miss = score_item(_item(Expected(doc_id="a.pdf")), [_hit("z.pdf", score=0.4)])
    oos = score_item(_item(category="out_of_scope"), [_hit("z.pdf", score=0.7)])
    s = summarize([hit, miss, oos])
    assert s["n_in_scope"] == 2
    assert s["doc_hit@1"] == 0.5
    assert s["mrr"] == 0.5
    assert s["out_of_scope"]["top_scores"] == [0.7]
    assert s["out_of_scope"]["in_scope_top_scores_min"] == 0.4


def test_regressions_reports_only_drops() -> None:
    base = {"doc_hit@1": 0.8, "doc_hit@3": 0.9, "doc_hit@5": 1.0, "page_hit@1": 0.7,
            "page_hit@3": 0.8, "mrr": 0.85}  # fmt: skip
    cur = {**base, "doc_hit@1": 0.7, "mrr": 0.9}
    assert regressions(base, cur) == ["doc_hit@1: 0.8 -> 0.7"]
    assert regressions(base, cur, tolerance=0.2) == []


def test_golden_set_is_well_formed() -> None:
    from pathlib import Path

    items = load_golden(Path(__file__).resolve().parents[3] / "evals" / "golden" / "v1.json")
    assert len({i.id for i in items}) == len(items)
    assert all(bool(i.expected) == (i.category != "out_of_scope") for i in items)
