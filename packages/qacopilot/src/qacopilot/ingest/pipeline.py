"""End to end: PDF bytes -> text (OCR if needed) -> classify -> chunk -> index."""

import asyncio
import hashlib
from pathlib import PurePath
from typing import TYPE_CHECKING

from pydantic import BaseModel

from qacopilot.ingest.chunking import chunk_document
from qacopilot.ingest.llm_classifier import classify_document
from qacopilot.ingest.models import Classification, DocType, ParsedDocument
from qacopilot.ingest.pdf import extract_pages
from qacopilot.llm import BaseLLMProvider
from qacopilot.retrieval import QdrantStore

if TYPE_CHECKING:
    from qacopilot.repositories import DocumentRepository


class IngestResult(BaseModel):
    doc_id: str
    filename: str
    doc_type: DocType
    confidence: float
    classified_by: str  # rules | llm | user
    reason: str | None = None  # the LLM's explanation, when it decided or disagreed
    alternative: DocType | None = None  # what the LLM suggested instead: a person should check
    pages: int
    ocr_pages: int
    chunks: int


async def ingest_pdf(
    data: bytes,
    filename: str,
    store: QdrantStore,
    *,
    doc_type: DocType | None = None,
    repo: "DocumentRepository | None" = None,
    llm: BaseLLMProvider | None = None,
) -> IngestResult:
    """Index one PDF. ``doc_id`` is a content hash, so re-uploading the same file replaces it.

    Pass ``doc_type`` to override the classifier (e.g. when a person has chosen the type). With a
    ``repo`` the document and its page text are also stored in PostgreSQL.
    """
    name = PurePath(filename).name  # never trust a path from the client
    doc_id = hashlib.sha256(data).hexdigest()[:16]
    doc = await asyncio.to_thread(extract_pages, data)  # OCR is CPU-bound
    if doc_type is not None:
        guess = Classification(doc_type=doc_type, confidence=1.0, matched=[], source="user")
    else:  # rules first; the LLM only if they are unsure (and one is configured)
        guess = await classify_document(doc.full_text, name, llm=llm)
    kind = guess.doc_type
    chunks = chunk_document(doc, kind, doc_id, metadata={"filename": name})
    if repo is not None:  # record the page text first, so the document can be extracted later
        await repo.save_document(
            doc_id, name, kind, doc, len(chunks), guess.source, guess.alternative
        )
    await store.replace_document(doc_id, chunks)
    return IngestResult(
        doc_id=doc_id,
        filename=name,
        doc_type=kind,
        confidence=guess.confidence,
        classified_by=guess.source,
        reason=guess.reason,
        alternative=guess.alternative,
        pages=len(doc.pages),
        ocr_pages=sum(1 for p in doc.pages if p.method == "ocr"),
        chunks=len(chunks),
    )


async def reclassify_document(
    doc_id: str, doc_type: DocType, store: QdrantStore, repo: "DocumentRepository"
) -> int | None:
    """Change a stored document's type, rebuilding its chunks from the saved page text.

    Returns the new chunk count, or None if the document is unknown. Any extraction is dropped
    (it was made for the old type).
    """
    parsed: ParsedDocument | None = await repo.load_parsed(doc_id)
    if parsed is None:
        return None
    chunks = chunk_document(parsed, doc_type, doc_id, metadata={"filename": parsed.source})
    await store.replace_document(doc_id, chunks)
    await repo.set_document_type(doc_id, doc_type, len(chunks))
    return len(chunks)
