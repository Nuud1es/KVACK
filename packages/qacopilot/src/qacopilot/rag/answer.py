"""Answer a question from retrieved chunks, streaming the reply with citations."""

import re
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

from qacopilot.ingest.models import DocType
from qacopilot.llm import BaseLLMProvider
from qacopilot.rag.prompts import ANSWER_REMINDER, ANSWER_SYSTEM, NOT_FOUND, PROMPT_VERSION
from qacopilot.retrieval import Hit, Retriever

_CITE = re.compile(r"\[(\d+)\]")
# A refusal in the model's own words ("The protein content is not provided in the given sources.")
_REFUSAL = re.compile(
    r"not\s+(?:provided|stated|mentioned|specified|found|included|available)"
    r"|not\s+in\s+the\s+(?:given\s+|provided\s+)?(?:sources|documents)"
    r"|(?:does|do)\s+not\s+(?:contain|provide|include|mention|specify)|no\s+information"
    r"|cannot\s+(?:find|be\s+determined)",
    re.IGNORECASE,
)
# Unmistakable refusals, honoured even when the model still tacks on a [n] marker ("X does not
# exist in the provided documents. [1]"). Kept narrow: a cited answer may legitimately contain
# words like "not provided", so the broader pattern above is used only for uncited replies.
_EXPLICIT_REFUSAL = re.compile(
    r"does\s+not\s+exist|not\s+(?:present\s+)?in\s+the\s+(?:given\s+|provided\s+|indexed\s+)?"
    r"(?:sources|documents)|no\s+such\s+(?:certificate|document|batch|record)",
    re.IGNORECASE,
)
SNIPPET_CHARS = 240


class Citation(BaseModel):
    n: int  # the [n] marker the model uses
    doc_id: str
    filename: str
    page: int
    section: str | None = None
    clause: str | None = None
    score: float
    snippet: str


class SourcesEvent(BaseModel):
    event: Literal["sources"] = "sources"
    citations: list[Citation]


class TokenEvent(BaseModel):
    event: Literal["token"] = "token"
    text: str


class NoAnswerEvent(BaseModel):
    event: Literal["no_answer"] = "no_answer"
    # uncited: the model replied without citing a source, so the reply is not shown as an answer
    reason: Literal["no_relevant_documents", "not_in_documents", "uncited"]


class ExportRef(BaseModel):
    """How to download every row of a table answer: the validated query and how many rows."""

    query: dict[str, Any]
    count: int


class DoneEvent(BaseModel):
    event: Literal["done"] = "done"
    answer: str
    answered: bool  # False when the model said the sources do not contain the answer
    cited: list[int]  # [n] markers actually used; empty with answered=True means uncited
    prompt_version: str = PROMPT_VERSION
    # where the answer came from: search results read by the LLM, the certificates table,
    # the CoA-vs-spec check, or the Fact Spec packaging table (computed in code, with no LLM)
    via: Literal["documents", "certificates", "coa_check", "packaging", "fact_spec"] = "documents"
    export: ExportRef | None = None  # set when the answer is a list or count of specs


AskEvent = SourcesEvent | TokenEvent | NoAnswerEvent | DoneEvent


def _citation(n: int, hit: Hit) -> Citation:
    c = hit.chunk
    body = c.text.split("\n\n", 1)[-1]
    return Citation(
        n=n,
        doc_id=c.doc_id,
        filename=c.metadata.get("filename", c.doc_id),
        page=c.page,
        section=c.section,
        clause=c.clause,
        score=hit.score,
        snippet=body[:SNIPPET_CHARS],
    )


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    sources = "\n\n".join(
        f"[{i}] ({h.chunk.metadata.get('filename', h.chunk.doc_id)}, page {h.chunk.page})\n"
        f"{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"Sources:\n\n{sources}\n\nQuestion: {question}\n\n{ANSWER_REMINDER}",
        },
    ]


async def answer_question(
    question: str,
    retriever: Retriever,
    llm: BaseLLMProvider,
    *,
    limit: int = 5,
    doc_type: DocType | None = None,
    max_tokens: int = 4096,
    capture: list[Hit] | None = None,
    doc_id: str | None = None,
) -> AsyncIterator[AskEvent]:
    """Yield ``sources``, then ``token`` events, then ``done`` (or just ``no_answer``).

    With nothing relevant retrieved the LLM is not called at all. ``capture``, if given, receives
    the retrieved hits (used by the evals to record exactly what the model was shown). ``doc_id``
    limits the search to one document (used when the question names a particular Fact Spec).
    """
    hits = await retriever.retrieve(question, limit=limit, doc_type=doc_type, doc_id=doc_id)
    if capture is not None:
        capture.extend(hits)
    if not hits:
        yield NoAnswerEvent(reason="no_relevant_documents")
        return
    yield SourcesEvent(citations=[_citation(i, h) for i, h in enumerate(hits, start=1)])

    parts: list[str] = []
    async for delta in llm.stream(build_messages(question, hits), max_tokens=max_tokens):
        parts.append(delta)
        yield TokenEvent(text=delta)

    answer = "".join(parts).strip()
    cited = sorted({int(m) for m in _CITE.findall(answer) if 1 <= int(m) <= len(hits)})
    # An answer is only shown if it cites a source we actually gave. Without one it is either a
    # refusal in the model's own words or a claim with nothing behind it; neither is an answer.
    explicit = bool(_EXPLICIT_REFUSAL.search(answer))
    if NOT_FOUND in answer or not cited or explicit:
        refused = NOT_FOUND in answer or explicit or bool(_REFUSAL.search(answer))
        yield NoAnswerEvent(reason="not_in_documents" if refused else "uncited")
        yield DoneEvent(answer=answer, answered=False, cited=[])
        return
    yield DoneEvent(answer=answer, answered=True, cited=cited)
