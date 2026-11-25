import json
from collections.abc import Sequence
from typing import Any

from qacopilot.evaluation.answers import AnswerRecord
from qacopilot.evaluation.judge import JudgedAnswer, judge_answer, summarize_judgements
from qacopilot.llm import BaseLLMProvider, LLMResponse


class CannedJudge(BaseLLMProvider):
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[Sequence[Any]] = []

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        self.prompts.append(messages)
        return LLMResponse(content=self.replies.pop(0))

    async def health(self) -> bool:
        return True


REC = AnswerRecord(
    id="q1", question="What is the shelf life?", answer="14 days [1].", answered=True, cited=[1],
    sources={1: "Shelf life: 14 days from pack date."},
)  # fmt: skip


async def test_judge_reads_question_sources_and_answer_but_not_the_golden_answer() -> None:
    llm = CannedJudge(json.dumps({"faithful": True, "relevant": True}))
    j = await judge_answer(llm, REC)
    assert (j.faithful, j.relevant, j.error) == (True, True, None)
    # extract() prepends its own JSON-schema instruction; ours are the last two messages
    system, user = (m["content"] for m in llm.prompts[0][-2:])
    assert "strict reviewer" in system
    assert "What is the shelf life?" in user and "[1]\nShelf life: 14 days" in user
    assert "14 days [1]." in user


async def test_unsupported_claims_are_reported() -> None:
    reply = json.dumps(
        {"faithful": False, "relevant": True, "unsupported": ["stored for 99 years"]}
    )
    j = await judge_answer(CannedJudge(reply), REC)
    assert j.faithful is False and j.unsupported == ["stored for 99 years"]


async def test_a_judge_that_cannot_answer_is_an_error_not_a_pass() -> None:
    j = await judge_answer(CannedJudge("{}", "{}", "{}"), REC)
    assert j.faithful is None and j.error is not None
    s = summarize_judgements([j, JudgedAnswer(id="b", faithful=True, relevant=True)])
    assert (s["n_judged"], s["n_judge_errors"], s["faithfulness"]) == (2, 1, 1.0)


def test_summary_of_nothing_is_zero_not_a_crash() -> None:
    assert summarize_judgements([])["faithfulness"] == 0.0
