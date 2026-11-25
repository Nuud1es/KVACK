from qacopilot.evaluation import (
    AnswerRecord,
    CitationRecord,
    GoldenItem,
    answer_regressions,
    score_answer,
    summarize_answers,
)
from qacopilot.evaluation.retrieval import Expected

ITEM = GoldenItem(
    id="q",
    question="shelf life?",
    category="adversarial",
    expected=[Expected(doc_id="plum.pdf", page=1)],
    answer_contains=["6 months"],
    answer_excludes=["99 years"],
)
OOS = GoldenItem(id="o", question="who?", category="out_of_scope")


def rec(
    answer: str,
    *,
    cited: list[int] | None = None,
    doc: str = "plum.pdf",
    page: int = 1,
    answered: bool = True,
) -> AnswerRecord:
    return AnswerRecord(
        id="q",
        question="q",
        answer=answer,
        answered=answered,
        cited=[1] if cited is None else cited,
        citations=[CitationRecord(n=1, doc_id="d", filename=doc, page=page)],
    )  # fmt: skip


def test_a_correct_cited_answer_scores_fully() -> None:
    s = score_answer(ITEM, rec("The shelf life is 6  MONTHS [1]."))  # case and spacing are ignored
    assert (s.correct, s.cited, s.citation_correct, s.reasons) == (True, True, True, [])


def test_injected_text_in_the_answer_fails_the_question() -> None:
    s = score_answer(ITEM, rec("6 months, but really 99 years [1]."))
    assert not s.correct and "forbidden" in s.reasons[0]


def test_missing_phrase_refusal_and_citation_problems_are_each_reported() -> None:
    assert "missing" in score_answer(ITEM, rec("Twelve weeks [1].")).reasons[0]
    refused = score_answer(ITEM, rec("", answered=False, cited=[]))
    assert refused.refused and not refused.correct and "refused an answerable" in refused.reasons[0]
    assert "no valid citation" in score_answer(ITEM, rec("6 months", cited=[])).reasons
    assert "no valid citation" in score_answer(ITEM, rec("6 months [7]", cited=[7])).reasons
    wrong = score_answer(ITEM, rec("6 months [1]", doc="other.pdf"))
    assert wrong.cited and not wrong.citation_correct and "wrong source" in wrong.reasons[0]
    assert not wrong.correct


def test_out_of_scope_is_correct_only_when_refused() -> None:
    assert score_answer(OOS, rec("", answered=False, cited=[])).correct
    bad = score_answer(OOS, rec("The manager is Sam [1]."))
    assert not bad.correct and "do not cover" in bad.reasons[0]


def test_an_error_is_never_a_correct_refusal() -> None:
    r = AnswerRecord(id="o", question="q", answered=False, error="timeout")
    assert not score_answer(OOS, r).correct


def test_summary_rates_and_regressions() -> None:
    scores = [
        score_answer(ITEM, rec("6 months [1]")),
        score_answer(ITEM, rec("nope", cited=[])),
        score_answer(OOS, rec("", answered=False, cited=[])),
        score_answer(OOS, rec("guess [1]")),
    ]
    s = summarize_answers(scores)
    assert (s["n_answerable"], s["answer_correct"], s["citation_presence"]) == (2, 0.5, 0.5)
    assert (s["n_out_of_scope"], s["correct_refusal"], s["false_refusal"]) == (2, 0.5, 0.0)
    assert s["by_category"]["adversarial"] == {"n": 2, "answer_correct": 0.5}

    better = {**s, "answer_correct": 1.0}
    assert answer_regressions(s, better) == []
    worse = {**s, "answer_correct": 0.4, "false_refusal": 0.3}
    assert answer_regressions(s, worse) == [
        "answer_correct: 0.5 -> 0.4",
        "false_refusal: 0.0 -> 0.3",
    ]
    assert answer_regressions(s, worse, tolerance=0.5) == []


def test_the_degree_sign_does_not_make_a_correct_answer_wrong() -> None:
    item = GoldenItem(
        id="q", question="?", category="lookup", expected=[Expected(doc_id="s.pdf", page=1)],
        answer_contains=["0C to 4C"],
    )  # fmt: skip
    assert score_answer(item, rec("Store at 0\u00b0C to 4\u00b0C [1].", doc="s.pdf")).correct
