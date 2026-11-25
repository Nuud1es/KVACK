from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder, FakeReranker
from qdrant_client import AsyncQdrantClient

from qacopilot.config import Settings
from qacopilot.ingest import DocType
from qacopilot.ingest.pipeline import ingest_pdf
from qacopilot.llm import BaseLLMProvider, LLMResponse
from qacopilot.rag import DoneEvent, NoAnswerEvent, SourcesEvent, TokenEvent, answer_question
from qacopilot.retrieval import QdrantStore, Retriever

SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples"


class ScriptedLLM(BaseLLMProvider):
    """Streams a canned reply and records the prompt it was given."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[Any] = []
        self.calls = 0

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, messages: Sequence[Any], **kwargs: Any) -> AsyncIterator[str]:
        self.calls += 1
        self.messages = list(messages)
        for word in self.reply.split(" "):
            yield word + " "

    async def health(self) -> bool:
        return True


@pytest.fixture
async def retriever() -> AsyncIterator[Retriever]:
    store = QdrantStore(FakeEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
    await store.ensure_collection()
    name = "sop_cooking_temperature.pdf"
    await ingest_pdf((SAMPLES / name).read_bytes(), name, store)
    yield Retriever(store, FakeReranker(), min_score=0.0)
    await store.aclose()


async def _run(question: str, retriever: Retriever, llm: ScriptedLLM) -> list[Any]:
    return [e async for e in answer_question(question, retriever, llm)]


async def test_answer_streams_sources_tokens_then_done(retriever: Retriever) -> None:
    llm = ScriptedLLM("Cooked duck must reach 75C core before release [1].")
    events = await _run(
        "what core temperature must cooked duck reach before release", retriever, llm
    )

    assert isinstance(events[0], SourcesEvent)
    first = events[0].citations[0]
    assert (first.filename, first.page, first.n) == ("sop_cooking_temperature.pdf", 1, 1)
    assert any(isinstance(e, TokenEvent) for e in events)
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.answered and done.cited == [1]
    # the prompt holds the numbered source and the question
    user = llm.messages[1]["content"]
    assert "[1] (sop_cooking_temperature.pdf, page 1)" in user and "Question:" in user


async def test_no_relevant_documents_skips_the_llm(retriever: Retriever) -> None:
    llm = ScriptedLLM("should never be used")
    events = await _run("fire evacuation assembly point", retriever, llm)
    assert [type(e) for e in events] == [NoAnswerEvent]
    assert events[0].reason == "no_relevant_documents"
    assert llm.calls == 0


async def test_model_saying_not_found_is_reported(retriever: Retriever) -> None:
    llm = ScriptedLLM("NOT_IN_DOCUMENTS")
    events = await _run("what core temperature must cooked duck reach", retriever, llm)
    assert isinstance(events[-2], NoAnswerEvent) and events[-2].reason == "not_in_documents"
    assert isinstance(events[-1], DoneEvent) and not events[-1].answered


async def test_a_reply_that_cites_nothing_real_is_not_shown_as_an_answer(
    retriever: Retriever,
) -> None:
    """[9] points at a source that was never given, so it is as good as no citation."""
    events = await _run(
        "what core temperature must cooked duck reach", retriever, ScriptedLLM("It is 75C [9].")
    )
    assert isinstance(events[-2], NoAnswerEvent) and events[-2].reason == "uncited"
    done = events[-1]
    assert isinstance(done, DoneEvent) and not done.answered and done.cited == []


@pytest.mark.parametrize(
    "reply",
    [
        "The protein content is not provided in the given sources.",
        "The documents do not contain this information.",
        "There is no information about that.",
        "I cannot find it in the sources.",
    ],
)
async def test_a_refusal_in_the_models_own_words_counts_as_a_refusal(
    retriever: Retriever, reply: str
) -> None:
    events = await _run(
        "what core temperature must cooked duck reach", retriever, ScriptedLLM(reply)
    )
    assert isinstance(events[-2], NoAnswerEvent) and events[-2].reason == "not_in_documents"
    assert isinstance(events[-1], DoneEvent) and not events[-1].answered


@pytest.mark.parametrize(
    "reply",
    [
        "CAL-2026-9999 does not exist in the provided documents. [1]",
        "That batch is not in the documents. [1]",
        "There is no such certificate on file [1].",
    ],
)
async def test_a_refusal_is_a_refusal_even_when_it_still_cites_a_source(
    retriever: Retriever, reply: str
) -> None:
    events = await _run(
        "what core temperature must cooked duck reach", retriever, ScriptedLLM(reply)
    )
    assert isinstance(events[-2], NoAnswerEvent) and events[-2].reason == "not_in_documents"
    assert isinstance(events[-1], DoneEvent) and not events[-1].answered and events[-1].cited == []


async def test_a_cited_answer_that_mentions_a_limit_is_not_mistaken_for_a_refusal(
    retriever: Retriever,
) -> None:
    reply = "The limit is not exceeded when cooked to 75C [1]."  # contains 'not' but cites a source
    events = await _run(
        "what core temperature must cooked duck reach", retriever, ScriptedLLM(reply)
    )
    assert isinstance(events[-1], DoneEvent) and events[-1].answered and events[-1].cited == [1]


async def test_ingest_pipeline_classifies_ocrs_and_reports() -> None:
    store = QdrantStore(FakeEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
    await store.ensure_collection()
    data = (SAMPLES / "calibration_cert_probe_scanned.pdf").read_bytes()
    result = await ingest_pdf(data, "../../etc/scan.pdf", store)
    assert result.filename == "scan.pdf"  # path stripped
    assert result.doc_type is DocType.CALIBRATION_CERT
    assert (result.pages, result.ocr_pages, result.chunks) == (1, 1, 1)
    again = await ingest_pdf(data, "scan.pdf", store)
    assert again.doc_id == result.doc_id and await store.count() == 1  # replaced, not duplicated
    overridden = await ingest_pdf(data, "scan.pdf", store, doc_type=DocType.OTHER)
    assert overridden.doc_type is DocType.OTHER and overridden.confidence == 1.0
    await store.aclose()
