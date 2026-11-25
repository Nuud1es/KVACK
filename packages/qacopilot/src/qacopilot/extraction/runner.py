"""Extract every certificate in a stored document, with progress, as a background job.

One LLM call per certificate (20-100 s each with a reasoning model), so a 19-certificate bundle
takes minutes. A certificate that cannot be read is recorded and skipped; it never loses the rest.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from qacopilot.extraction.prompts import PROMPT_VERSION
from qacopilot.extraction.service import (
    SCHEMAS,
    Extracted,
    UnsupportedDocType,
    extract_document,
)
from qacopilot.ingest.bundles import Segment, segment_document, split_certificates
from qacopilot.ingest.models import DocType
from qacopilot.llm import BaseLLMProvider, StructuredOutputError

if (
    TYPE_CHECKING
):  # repositories imports extraction.schemas, so a real import here would be circular
    from qacopilot.repositories import DocumentRepository

logger = logging.getLogger(__name__)


class ExtractionRun(BaseModel):
    total: int
    done: int  # certificates read and stored
    failed: int  # certificates the model could not read


class UnknownDocument(Exception):
    pass


async def plan_segments(repo: "DocumentRepository", doc_id: str) -> tuple[DocType, list[Segment]]:
    """The document's type and its certificates.

    Raises ``UnknownDocument`` or ``UnsupportedDocType``.
    """
    document = await repo.get_document(doc_id)
    parsed = await repo.load_parsed(doc_id)
    if document is None or parsed is None:
        raise UnknownDocument(doc_id)
    doc_type = DocType(document.doc_type)
    if doc_type not in SCHEMAS:
        raise UnsupportedDocType(doc_type.value)
    return doc_type, split_certificates(parsed, doc_type)


async def run_extraction(
    llm: BaseLLMProvider,
    repo: "DocumentRepository",
    doc_id: str,
    *,
    model: str,
    on_progress: Callable[[ExtractionRun], Awaitable[None]] | None = None,
    limit: int | None = None,
    concurrency: int = 1,
    thinking: bool = False,
) -> ExtractionRun:
    """Extract the certificates of ``doc_id`` (the first ``limit`` if given), storing each as it
    finishes."""
    doc_type, segments = await plan_segments(repo, doc_id)
    segments = segments[:limit]
    parsed = await repo.load_parsed(doc_id)
    assert parsed is not None
    await repo.start_job(doc_id, len(segments))
    run = ExtractionRun(total=len(segments), done=0, failed=0)
    errors: list[str] = []
    gate = asyncio.Semaphore(max(1, concurrency))

    async def read(seg: Segment) -> tuple[Segment, Extracted | None, str | None]:
        async with gate:  # at most ``concurrency`` LLM calls in flight
            try:
                return (
                    seg,
                    await extract_document(
                        llm, segment_document(parsed, seg), doc_type, thinking=thinking
                    ),
                    None,
                )
            except StructuredOutputError as exc:
                return seg, None, str(exc)[:120]

    tasks: list[asyncio.Task[tuple[Segment, Extracted | None, str | None]]] = []
    try:
        # Results are saved as they finish, in whatever order the LLM returns them.
        # Tasks are created in page order and the gate is first-come-first-served, so with the
        # default concurrency of 1 the certificates are read strictly in order.
        tasks.extend(asyncio.create_task(read(seg)) for seg in segments)
        for finished in asyncio.as_completed(tasks):
            seg, result, error = await finished
            if result is None:
                run.failed += 1
                errors.append(f"pages {seg.start_page}-{seg.end_page}: {error}")
            else:
                await repo.save_extraction(
                    doc_id,
                    doc_type,
                    result,
                    prompt_version=PROMPT_VERSION,
                    model=model,
                    start_page=seg.start_page,
                    end_page=seg.end_page,
                )
                run.done += 1
            await repo.job_progress(doc_id, done=run.done, failed=run.failed)
            if on_progress:
                await on_progress(run)
        await repo.prune_extractions(doc_id, [s.start_page for s in segments if run.done])
        # A run in which every certificate failed is a failure; a partial one is still "done",
        # with the failures counted so a person can see that some need attention.
        status = "failed" if run.done == 0 else "done"
        await repo.finish_job(doc_id, status, "; ".join(sorted(errors))[:900] or None)
    except (
        BaseException
    ) as exc:  # the LLM was unreachable, the database failed, or we were cancelled
        for task in tasks:
            task.cancel()  # do not leave queued certificates running with nobody to save them
        if not isinstance(exc, Exception):
            await repo.finish_job(doc_id, "failed", "cancelled")
            raise
        logger.exception("extraction job for %s failed", doc_id)
        await repo.finish_job(doc_id, "failed", f"{type(exc).__name__}: {exc}"[:900])
        raise
    return run
