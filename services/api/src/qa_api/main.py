"""KVACK API. Run with: uvicorn qa_api.main:app --reload"""

import asyncio
import contextlib
import logging
import re
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel, Field
from qacopilot import __version__
from qacopilot import auth as accounts
from qacopilot.agent import AgentDeps, AgentService, RunConflict, RunNotFound, RunView
from qacopilot.compliance import CheckFailure, CoACheck, run_coa_check
from qacopilot.config import Settings, get_settings
from qacopilot.extraction import (
    CalibrationCert,
    UnknownDocument,
    UnsupportedDocType,
    certificate_warnings,
    plan_segments,
    run_extraction,
)
from qacopilot.ingest import DocType
from qacopilot.ingest.pdf import sniff_kind
from qacopilot.ingest.pipeline import IngestResult, ingest_pdf, reclassify_document
from qacopilot.llm import BaseLLMProvider, OpenAICompatibleProvider
from qacopilot.mail import make_mailer
from qacopilot.names import NameReview, review_names
from qacopilot.rag import answer_question, answer_with_lookup
from qacopilot.rag.factspec_lookup import scope_for_question, with_note
from qacopilot.rag.specquery import ExportError, SpecQuery, csv_text, export_specs
from qacopilot.repositories import (
    AgentRepository,
    CertificateStatus,
    DocReviewItem,
    DocumentRepository,
    DocumentSummary,
    ReviewEvent,
    ReviewItem,
    UserInfo,
    UserRepository,
    make_engine,
)
from qacopilot.retrieval import (
    FastEmbedEmbedder,
    FastEmbedReranker,
    QdrantStore,
    Reranker,
    Retriever,
)
from qacopilot.tracing import request_id_var, send_feedback_score
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine

from qa_api import metrics
from qa_api.logging_config import configure_logging
from qa_api.sse import stream_events
from qa_api.throttle import Throttle

configure_logging()
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the background metrics refresher around the real startup/shutdown."""
    task: asyncio.Task[None] | None = None
    if get_settings().database_url:
        probe_provider = OpenAICompatibleProvider(
            get_settings()
        )  # one client, reused every refresh

        async def refresh() -> None:
            async def probe_qdrant() -> bool:
                await get_store().count()
                return True

            await metrics.refresh_gauges(
                documents=get_repository(),
                agent_repo=get_agent_repository(),
                probe_qdrant=probe_qdrant,
                probe_llm=probe_provider.health,
            )

        task = asyncio.create_task(metrics.refresh_forever(refresh))
    try:
        async with _lifespan(app):
            yield
    finally:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    try:
        await get_store().ensure_collection()
    except Exception:  # Qdrant may start after the API; /health still reports on the LLM
        logger.exception("could not prepare the Qdrant collection at startup")
    if not settings.database_url:
        logger.warning("QACOPILOT_DATABASE_URL is not set: the agent is disabled")
        yield
        return
    try:  # an extraction still marked running belonged to a process that has since died
        if stale := await get_repository().fail_stale_jobs():
            logger.warning("marked %d interrupted extraction job(s) as failed", stale)
    except Exception:
        logger.exception("could not clean up stale extraction jobs")
    # LangGraph's checkpointer keeps paused runs in Postgres, so an approval can wait for
    # hours and survive a restart. It speaks plain libpq URLs, not SQLAlchemy's +psycopg.
    conninfo = (
        make_url(settings.database_url)
        .set(drivername="postgresql")
        .render_as_string(hide_password=False)
    )
    try:
        async with AsyncPostgresSaver.from_conn_string(conninfo) as saver:
            await saver.setup()
            app.state.agent_service = AgentService(build_agent_deps(settings), saver)
            yield
    except Exception:
        logger.exception("the agent could not start; continuing without it")
        yield


def build_agent_deps(settings: Settings) -> AgentDeps:
    recipients = [r.strip() for r in settings.reminder_recipients.split(",") if r.strip()]
    return AgentDeps(
        llm=OpenAICompatibleProvider(settings),
        retriever=Retriever(get_store(), get_reranker(), min_score=settings.rerank_min_score),
        documents=get_repository(),
        agent_repo=get_agent_repository(),
        mailer=make_mailer(settings),
        recipients=recipients,
    )


app = FastAPI(title="KVACK", version=__version__, lifespan=lifespan)
app.add_middleware(metrics.MetricsMiddleware)


@lru_cache
def get_store() -> QdrantStore:
    settings = get_settings()
    return QdrantStore(FastEmbedEmbedder(settings), settings)


@lru_cache
def get_reranker() -> Reranker:
    return FastEmbedReranker(get_settings())


@lru_cache
def get_engine() -> AsyncEngine:
    return make_engine(get_settings())


@lru_cache
def get_repository() -> DocumentRepository:
    return DocumentRepository(get_engine())


def get_repository_or_none(
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentRepository | None:
    """The database is optional for Ask (document search works without it)."""
    return get_repository() if settings.database_url else None


@lru_cache
def get_agent_repository() -> AgentRepository:
    return AgentRepository(get_engine())


def get_agent_service(request: Request) -> AgentService:
    service: AgentService | None = getattr(request.app.state, "agent_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="The agent is not available")
    return service


def get_retriever(
    store: Annotated[QdrantStore, Depends(get_store)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Retriever:
    return Retriever(store, reranker, min_score=settings.rerank_min_score)


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> BaseLLMProvider:
    return OpenAICompatibleProvider(settings)


def api_users(settings: Settings) -> dict[str, str]:
    """Map each configured key to the person it belongs to (``api_keys="alice:k1,bob:k2"``)."""
    users: dict[str, str] = {}
    for part in settings.api_keys.split(","):
        name, _, key = part.strip().partition(":")
        if name and key:
            users[key] = name
    if settings.api_key:
        users.setdefault(settings.api_key, "api")
    return users


@lru_cache
def get_user_repository() -> UserRepository:
    return UserRepository(get_engine())


def get_user_repository_or_none(
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserRepository | None:
    return get_user_repository() if settings.accounts and settings.database_url else None


async def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    """Return the name of the caller: the owner of a configured API key, or the person a sign-in
    token (from ``/auth/login``, sent in the same header) belongs to. With no keys and accounts
    off (local development only) every request is user ``dev``. Keys are compared in constant
    time with no early exit."""
    users = api_users(settings)
    if not users and not settings.accounts:
        return "dev"
    supplied = (x_api_key or "").encode()
    caller: str | None = None
    for key, name in users.items():
        if secrets.compare_digest(supplied, key.encode()):
            caller = name
    if caller is None and users_repo is not None and supplied:
        caller = await users_repo.user_for_token(x_api_key or "")
    if caller is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return caller


class Health(BaseModel):
    status: str
    version: str
    llm_reachable: bool


async def require_metrics_access(
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """With ``QACOPILOT_METRICS_TOKEN`` set, a scraper must send ``Authorization: Bearer <token>``
    (what Prometheus' ``authorization`` setting does). Without it the normal API key applies."""
    if settings.metrics_token:
        supplied = (authorization or "").removeprefix("Bearer ").encode()
        if not secrets.compare_digest(supplied, settings.metrics_token.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing metrics token")
        return
    await require_api_key(settings, users_repo, x_api_key)


@app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_metrics_access)])
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint (operational numbers only: no document or question text)."""
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.get("/health", response_model=Health)
async def health(llm: Annotated[BaseLLMProvider, Depends(get_llm)]) -> Health:
    reachable = await llm.health()
    return Health(
        status="ok" if reachable else "degraded", version=__version__, llm_reachable=reachable
    )


@app.post("/ingest", response_model=IngestResult, dependencies=[Depends(require_api_key)])
async def ingest(
    store: Annotated[QdrantStore, Depends(get_store)],
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    llm: Annotated[BaseLLMProvider, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File()],
    doc_type: Annotated[DocType | None, Form()] = None,
) -> IngestResult:
    """Upload a PDF or a JPEG/PNG photo: OCR if scanned, classify (or use ``doc_type``), chunk and
    index it."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 50 MB")
    if sniff_kind(data) is None:  # judged by content, not by the file name the client claims
        raise HTTPException(status_code=422, detail="Only PDF, JPEG or PNG files are accepted")
    try:
        result = await ingest_pdf(
            data,
            file.filename or "upload.pdf",
            store,
            doc_type=doc_type,
            repo=repo,
            llm=llm if settings.llm_classify else None,  # asked only if the rules are unsure
        )
        metrics.INGESTED.labels(result.doc_type.value).inc()
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status_code=422, detail="Could not read this PDF") from exc


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_type: DocType | None = None
    limit: int = Field(default=5, ge=1, le=10)
    context: dict[str, Any] | None = Field(
        default=None,
        description="The query behind the previous answer (its export.query), so a follow-up "
        "such as 'give me all from Tesco' refines it",
    )


def _previous_query(body: AskRequest) -> SpecQuery | None:
    """The previous answer's query, if the browser sent a usable one (anything else is ignored)."""
    if not body.context:
        return None
    try:
        return SpecQuery.model_validate(body.context)
    except ValueError:
        return None


@app.post("/ask", dependencies=[Depends(require_api_key)])
async def ask(
    body: AskRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    llm: Annotated[BaseLLMProvider, Depends(get_llm)],
    repo: Annotated[DocumentRepository | None, Depends(get_repository_or_none)],
) -> StreamingResponse:
    """Answer from the indexed documents, streamed as server-sent events.

    Events: ``sources`` (citations), ``token`` (answer text), ``no_answer``, ``done``, ``error``.
    A question about when calibration certificates are due is answered from the certificates
    table (``done.via == "certificates"``), because search cannot tell which certificate for an
    instrument is the current one; everything else is answered from the documents.
    """
    request_id = uuid.uuid4().hex
    # Set here, not inside the generator: stream_events runs each step in its own task, and a
    # value set inside one step would not reach the step that calls the LLM.
    request_id_var.set(request_id)
    scope: str | None = None
    note: str | None = None
    search_question = body.question
    if repo is not None and body.doc_type is None:
        try:  # a question that names one Fact Spec is answered from that spec alone
            scope, note, search_question = await scope_for_question(body.question, repo)
        except Exception:
            logger.exception("could not match the question to a spec; searching all documents")
    events = answer_question(
        search_question, retriever, llm, limit=body.limit, doc_type=body.doc_type, doc_id=scope
    )
    if note is not None:
        events = with_note(events, note)
    if body.doc_type in (None, "calibration_cert"):
        events = answer_with_lookup(body.question, repo, events, llm, _previous_query(body))
    events = metrics.measured_ask(events)

    return StreamingResponse(
        stream_events(events),
        media_type="text/event-stream",
        # Send this id back with POST /feedback to rate the answer.
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )


class SegmentOut(BaseModel):
    """One certificate (or the whole document, for anything that is not a bundle)."""

    start_page: int
    end_page: int | None
    data: dict[str, object]
    warnings: list[str] = []  # implausible dates etc.; a person should check these
    needs_review: bool = False  # an unconfirmed reading from a poor scan or with odd dates
    review_reason: str | None = None
    verified_by: str | None = None  # who last confirmed or corrected it


class ExtractionOut(BaseModel):
    doc_id: str
    doc_type: DocType
    status: Literal["running", "done", "failed"]
    total: int  # certificates found in the document
    done: int
    failed: int  # certificates the model could not read
    error: str | None = None
    prompt_version: str | None = None
    model: str | None = None
    # Pages read by OCR: values from them can contain misread digits, so a person should check
    # them against the original (OCR once turned 4.1E+03 into 41E+03).
    ocr_pages: int = 0
    # Of those, pages read with low confidence (a poor scan): figures there may be misread, and a
    # certificate whose date could not be read may have a wrong one instead.
    poor_ocr_pages: int = 0
    segments: list[SegmentOut] = []
    data: dict[str, object] | None = None  # the first certificate's fields (single-document use)


_extraction_locks: dict[int, asyncio.Semaphore] = {}
_background: set[asyncio.Task[None]] = set()


def _extraction_lock() -> asyncio.Semaphore:
    """One extraction at a time: the LLM serves a single request at once. Created per event loop."""
    loop_id = id(asyncio.get_running_loop())
    return _extraction_locks.setdefault(loop_id, asyncio.Semaphore(1))


async def _extraction_out(repo: DocumentRepository, doc_id: str) -> ExtractionOut | None:
    document = await repo.get_document(doc_id)
    if document is None:
        return None
    rows = await repo.get_extractions(doc_id)
    job = await repo.get_job(doc_id)
    if not rows and job is None:
        return None
    segments = []
    flags = await repo.review_flags(doc_id)
    for row in rows:
        needs, reason, verified_by = flags.get(row.start_page, (False, None, None))
        warnings: list[str] = []
        if row.doc_type == DocType.CALIBRATION_CERT.value:
            warnings = certificate_warnings(CalibrationCert(**row.data))
        segments.append(
            SegmentOut(
                start_page=row.start_page,
                end_page=row.end_page,
                data=row.data,
                warnings=warnings,
                needs_review=needs,
                review_reason=reason,
                verified_by=verified_by,
            )
        )
    if job is not None:
        status, total, done, failed, error = job.status, job.total, job.done, job.failed, job.error
    else:  # extracted before jobs existed
        status, total, done, failed, error = "done", len(rows), len(rows), 0, None
    return ExtractionOut(
        doc_id=doc_id,
        doc_type=DocType(document.doc_type),
        status=status,  # type: ignore[arg-type]
        total=total,
        done=done,
        failed=failed,
        error=error,
        prompt_version=rows[0].prompt_version if rows else None,
        model=rows[0].model if rows else None,
        ocr_pages=document.ocr_pages,
        poor_ocr_pages=await repo.poor_ocr_pages(doc_id),
        segments=segments,
        data=segments[0].data if segments else None,
    )


async def _extract_in_background(
    doc_id: str,
    llm: BaseLLMProvider,
    repo: DocumentRepository,
    model: str,
    concurrency: int,
    thinking: bool,
) -> None:
    async with _extraction_lock():  # waits its turn behind any other extraction
        try:
            await run_extraction(
                llm, repo, doc_id, model=model, concurrency=concurrency, thinking=thinking
            )
        except Exception:  # the job row is already marked failed; do not crash the event loop
            logger.exception("background extraction of %s failed", doc_id)


@app.post(
    "/documents/{doc_id}/extract",
    response_model=ExtractionOut,
    responses={
        202: {"model": ExtractionOut, "description": "A bundle: extraction runs in the background"}
    },
    dependencies=[Depends(require_api_key)],
)
async def extract(
    doc_id: str,
    response: Response,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    llm: Annotated[BaseLLMProvider, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ExtractionOut:
    """Run the LLM extraction for an ingested document and store the results.

    A document with one certificate is extracted now (10-20 s with a reasoning model) and the
    result returned. A bundle of several certificates takes minutes, so it returns **202** at once
    and runs in the background: poll ``GET /documents/{id}/extraction`` for progress. Safe to
    repeat: results are replaced.
    """
    job = await repo.get_job(doc_id)
    if job is not None and job.status == "running":
        raise HTTPException(status_code=409, detail="This document is already being extracted")
    try:
        doc_type, segments = await plan_segments(repo, doc_id)
    except UnknownDocument as exc:
        raise HTTPException(status_code=404, detail="Unknown document") from exc
    except UnsupportedDocType as exc:
        raise HTTPException(
            status_code=422, detail="Extraction is not available for this type of document"
        ) from exc
    if len(segments) > 1:
        await repo.start_job(doc_id, len(segments))  # visible straight away, even if queued
        task = asyncio.create_task(
            _extract_in_background(
                doc_id,
                llm,
                repo,
                settings.llm_model,
                settings.extraction_concurrency,
                settings.extraction_thinking,
            )
        )
        _background.add(task)  # keep a reference so it is not garbage-collected mid-run
        task.add_done_callback(_background.discard)
        response.status_code = 202
    else:
        async with _extraction_lock():
            run = await run_extraction(
                llm, repo, doc_id, model=settings.llm_model, thinking=settings.extraction_thinking
            )
        if run.done == 0:
            logger.warning("extraction failed validation for %s (%s)", doc_id, doc_type.value)
            raise HTTPException(
                status_code=502, detail="The model could not produce valid output for this document"
            )
    out = await _extraction_out(repo, doc_id)
    assert out is not None
    return out


@app.get(
    "/documents/{doc_id}/extraction",
    response_model=ExtractionOut,
    dependencies=[Depends(require_api_key)],
)
async def get_extraction(
    doc_id: str, repo: Annotated[DocumentRepository, Depends(get_repository)]
) -> ExtractionOut:
    """The stored results and, for a bundle, how far a background extraction has got."""
    out = await _extraction_out(repo, doc_id)
    if out is None:
        raise HTTPException(status_code=404, detail="No extraction for this document")
    return out


_WINDOW = re.compile(r"^(\d{1,4})d?$")


@app.get(
    "/certificates",
    response_model=list[CertificateStatus],
    dependencies=[Depends(require_api_key)],
)
async def certificates(
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    expiring_within: str = "30d",
) -> list[CertificateStatus]:
    """Calibration certificates overdue or due within the window (``30`` or ``30d``).

    A plain SQL query over the latest certificate per instrument; no LLM involved.
    """
    m = _WINDOW.match(expiring_within.strip())
    if not m:
        raise HTTPException(status_code=422, detail="expiring_within must look like 30 or 30d")
    return await repo.certificates_due(int(m.group(1)))


class ReviewRequest(BaseModel):
    """Confirm a reading as it is (no ``corrections``), or fix fields and confirm the result."""

    corrections: dict[str, str | None] | None = Field(
        default=None,
        description=(
            "Fields to change: instrument_serial, instrument_description, certificate_number, "
            "calibration_date, next_due (dates as YYYY-MM-DD)"
        ),
    )
    note: str | None = Field(default=None, max_length=500)


@app.get(
    "/certificates/review",
    response_model=list[ReviewItem],
    dependencies=[Depends(require_api_key)],
)
async def certificates_to_review(
    repo: Annotated[DocumentRepository, Depends(get_repository)],
) -> list[ReviewItem]:
    """Readings from poor scans (or with odd dates) that no one has confirmed yet."""
    return await repo.certificates_to_review()


@app.get(
    "/review/documents",
    response_model=list[DocReviewItem],
    dependencies=[Depends(require_api_key)],
)
async def documents_to_review(
    repo: Annotated[DocumentRepository, Depends(get_repository)],
) -> list[DocReviewItem]:
    """Reports and specs that look incomplete or misread and that no one has confirmed yet."""
    return await repo.extractions_to_review()


@app.post("/documents/{doc_id}/extractions/{start_page}/review", response_model=DocReviewItem)
async def review_extraction(
    doc_id: str,
    start_page: int,
    body: ReviewRequest,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    user: Annotated[str, Depends(require_api_key)],
) -> DocReviewItem:
    """A person confirms or corrects a report or spec reading. Who and what changed is recorded."""
    try:
        item = await repo.review_extraction(
            doc_id, start_page, user, corrections=body.corrections, note=body.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Unknown report or spec")
    return item


@app.post("/documents/{doc_id}/certificates/{start_page}/review", response_model=ReviewItem)
async def review_certificate(
    doc_id: str,
    start_page: int,
    body: ReviewRequest,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    user: Annotated[str, Depends(require_api_key)],
) -> ReviewItem:
    """A person confirms or corrects a reading. Who and what changed is recorded."""
    try:
        item = await repo.review_certificate(
            doc_id, start_page, user, corrections=body.corrections, note=body.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Unknown certificate")
    return item


@app.get(
    "/documents/{doc_id}/certificates/{start_page}/reviews",
    response_model=list[ReviewEvent],
    dependencies=[Depends(require_api_key)],
)
async def certificate_review_history(
    doc_id: str, start_page: int, repo: Annotated[DocumentRepository, Depends(get_repository)]
) -> list[ReviewEvent]:
    """Every confirmation or correction of one certificate, oldest first."""
    return await repo.review_history(doc_id, start_page)


@app.get("/compliance/coa-check", response_model=CoACheck, dependencies=[Depends(require_api_key)])
async def coa_check(
    coa: str,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    spec: str | None = None,
) -> CoACheck:
    """Compare an extracted CoA against a supplier spec, in code (no LLM).

    Without ``spec`` the spec is found from the report's product (and, if it has several
    revisions, the one in force on the report date). 404 = nothing matches or something is not
    extracted yet; 409 = several different specs match, and the candidates are listed in the
    message: repeat the call with ``spec`` set to the right one.
    """
    try:
        return await run_coa_check(repo, coa, spec)
    except CheckFailure as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc


# --- agent ------------------------------------------------------------------------------------

User = Annotated[str, Depends(require_api_key)]
Agent = Annotated[AgentService, Depends(get_agent_service)]


class StartRun(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class Decision(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


class RunSummary(BaseModel):
    run_id: str
    created_by: str
    query: str
    status: str
    route: str | None
    message: str
    created_at: str


def _lookup(call: Awaitable[RunView]) -> Awaitable[RunView]:
    async def run() -> RunView:
        try:
            return await call
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail="Unknown run") from exc
        except RunConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return run()


@app.post("/agent/runs", response_model=RunView)
async def start_run(body: StartRun, user: User, agent: Agent) -> RunView:
    """Questions are answered at once. Requests to act (e.g. draft reminders) stop at
    ``awaiting_approval``: nothing is sent until a person approves."""
    view = await agent.start(user, body.query)
    metrics.AGENT_RUNS.labels(view.status).inc()
    return view


@app.get("/agent/runs", response_model=list[RunSummary], dependencies=[Depends(require_api_key)])
async def list_runs(
    repo: Annotated[AgentRepository, Depends(get_agent_repository)],
    status: str | None = None,
    limit: int = 50,
) -> list[RunSummary]:
    """The approval queue: ``?status=awaiting_approval``."""
    runs = await repo.list_runs(status, max(1, min(limit, 200)))
    return [
        RunSummary(
            run_id=r.run_id,
            created_by=r.created_by,
            query=r.query,
            status=r.status,
            route=r.route,
            message=(r.summary or {}).get("message", ""),
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@app.get("/agent/runs/{run_id}", response_model=RunView, dependencies=[Depends(require_api_key)])
async def get_run(run_id: str, agent: Agent) -> RunView:
    return await _lookup(agent.get(run_id))


@app.post("/agent/runs/{run_id}/approve", response_model=RunView)
async def approve_run(run_id: str, user: User, agent: Agent) -> RunView:
    """Approve the drafted actions. The approver is the person who owns the API key."""
    view = await _lookup(agent.decide(run_id, user, "approve"))
    metrics.AGENT_DECISIONS.labels("approve").inc()
    return view


@app.post("/agent/runs/{run_id}/reject", response_model=RunView)
async def reject_run(
    run_id: str, user: User, agent: Agent, body: Decision | None = None
) -> RunView:
    view = await _lookup(agent.decide(run_id, user, "reject", body.reason if body else None))
    metrics.AGENT_DECISIONS.labels("reject").inc()
    return view


@app.post("/agent/runs/{run_id}/retry-delivery", response_model=RunView)
async def retry_delivery(run_id: str, user: User, agent: Agent) -> RunView:
    """Re-attempt emails that failed after approval; delivered ones are not sent again."""
    return await _lookup(agent.retry_delivery(run_id, user))


class AuditEvent(BaseModel):
    ts: str
    actor: str
    event: str
    detail: dict[str, object]


@app.get(
    "/agent/runs/{run_id}/audit",
    response_model=list[AuditEvent],
    dependencies=[Depends(require_api_key)],
)
async def run_audit(
    run_id: str, repo: Annotated[AgentRepository, Depends(get_agent_repository)]
) -> list[AuditEvent]:
    """Who did what on this run, in order (append-only)."""
    return [
        AuditEvent(ts=e.ts.isoformat(), actor=e.actor, event=e.event, detail=e.detail)
        for e in await repo.audit_for_run(run_id)
    ]


class FeedbackIn(BaseModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    kind: Literal["ask", "agent"]
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackOut(BaseModel):
    stored: bool
    langfuse: bool  # whether the score reached Langfuse (tracing may be off or unreachable)


@app.post("/feedback", response_model=FeedbackOut)
async def feedback(
    body: FeedbackIn,
    user: User,
    repo: Annotated[AgentRepository, Depends(get_agent_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FeedbackOut:
    """Thumbs up/down on an answer (``X-Request-Id`` from /ask) or an agent run (its run_id).

    Stored in Postgres, and sent to Langfuse as a score on the request's trace."""
    run = await repo.get_run(body.request_id) if body.kind == "agent" else None
    if body.kind == "agent" and run is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    thumbs_up = body.rating == "up"
    await repo.add_feedback(body.request_id, body.kind, 1 if thumbs_up else -1, user, body.comment)
    sent = await asyncio.to_thread(
        send_feedback_score, settings, body.request_id, thumbs_up, body.comment
    )
    return FeedbackOut(stored=True, langfuse=sent)


# --- front end support --------------------------------------------------------------------------


class Me(BaseModel):
    user: str
    admin: bool = False  # may open the Admin page


class Credentials(BaseModel):
    name: str = Field(max_length=100)
    pin: str = Field(max_length=16)


class SignedIn(BaseModel):
    token: str  # send it as the X-API-Key header, like a key
    user: str
    expires_in_hours: int = accounts.SESSION_HOURS


_register_throttle = Throttle(limit=10, window_s=3600)
_login_throttle = Throttle(limit=30, window_s=600)  # wrong tries per address, any account


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _require_accounts(settings: Settings, users_repo: UserRepository | None) -> UserRepository:
    if not settings.accounts:
        raise HTTPException(status_code=404, detail="Accounts are not turned on")
    if users_repo is None:
        raise HTTPException(status_code=503, detail="Accounts need the database")
    return users_repo


async def _sign_in(users_repo: UserRepository, user_id: int, name: str) -> SignedIn:
    token = accounts.new_token()
    await users_repo.create_session(user_id, token)
    return SignedIn(token=token, user=name)


@app.post("/auth/register", response_model=SignedIn)
async def register(
    body: Credentials,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
) -> SignedIn:
    """Create an account from a name and a 4-digit PIN, and sign it in."""
    repo = _require_accounts(settings, users_repo)
    ip = _client(request)
    if _register_throttle.blocked(ip):
        raise HTTPException(status_code=429, detail="Too many sign-ups from here. Try later.")
    try:
        name = accounts.clean_name(body.name)
        accounts.check_pin(body.pin)
    except accounts.InvalidInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    reserved = (
        {n.casefold() for n in api_users(settings).values()}
        | accounts.RESERVED_NAMES
        | accounts.admin_keys(settings.admins)  # admin accounts are made from the terminal
    )
    if accounts.name_key(name) in reserved:
        raise HTTPException(status_code=409, detail="That name is not available.")
    _register_throttle.record(ip)
    user_id = await repo.register(name, body.pin)
    if user_id is None:
        raise HTTPException(status_code=409, detail="That name is already taken.")
    logger.info("account registered", extra={"user": name})
    return await _sign_in(repo, user_id, name)


@app.post("/auth/login", response_model=SignedIn)
async def login(
    body: Credentials,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
) -> SignedIn:
    """Sign in with a name and PIN. Five wrong PINs lock the account for a few minutes."""
    repo = _require_accounts(settings, users_repo)
    ip = _client(request)
    if _login_throttle.blocked(ip):
        raise HTTPException(status_code=429, detail="Too many wrong tries from here. Try later.")
    result = await repo.authenticate(body.name, body.pin)
    if result.status == "locked":
        raise HTTPException(
            status_code=429,
            detail=f"Too many wrong PINs. This account is locked for {accounts.LOCK_MINUTES} "
            "minutes.",
        )
    if result.status != "ok" or result.user_id is None or result.display_name is None:
        _login_throttle.record(ip)
        raise HTTPException(status_code=401, detail="Name or PIN not recognised.")
    return await _sign_in(repo, result.user_id, result.display_name)


@app.post("/auth/logout", status_code=204)
async def logout(
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> Response:
    if users_repo is not None and x_api_key:
        await users_repo.end_session(x_api_key)
    return Response(status_code=204)


def require_admin(
    settings: Annotated[Settings, Depends(get_settings)],
    user: Annotated[str, Depends(require_api_key)],
) -> str:
    if accounts.name_key(user) not in accounts.admin_keys(settings.admins):
        raise HTTPException(status_code=403, detail="Only an administrator can do that.")
    return user


Admin = Annotated[str, Depends(require_admin)]


class AdminUser(UserInfo):
    admin: bool = False


class NewPin(BaseModel):
    pin: str = Field(max_length=16)


async def _users_for_admin(settings: Settings, repo: UserRepository) -> list[AdminUser]:
    keys = accounts.admin_keys(settings.admins)
    return [
        AdminUser(**u.model_dump(), admin=accounts.name_key(u.name) in keys)
        for u in await repo.list_users()
    ]


def _admin_repo(settings: Settings, repo: UserRepository | None) -> UserRepository:
    return _require_accounts(settings, repo)


@app.get("/admin/users", response_model=list[AdminUser])
async def admin_list_users(
    admin: Admin,
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
) -> list[AdminUser]:
    return await _users_for_admin(settings, _admin_repo(settings, users_repo))


async def _acted(audit: AgentRepository, admin: str, event: str, name: str | None) -> None:
    if name is None:
        raise HTTPException(status_code=404, detail="Unknown account")
    await audit.audit(None, admin, event, {"account": name})


@app.post("/admin/users/{user_id}/pin", status_code=204)
async def admin_set_pin(
    user_id: int,
    body: NewPin,
    admin: Admin,
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
    audit: Annotated[AgentRepository, Depends(get_agent_repository)],
) -> Response:
    """Set a new PIN for someone who forgot theirs. It also lifts a lock and signs them out."""
    repo = _admin_repo(settings, users_repo)
    try:
        accounts.check_pin(body.pin)
    except accounts.InvalidInput as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await _acted(audit, admin, "admin.set_pin", await repo.set_pin(user_id, body.pin))
    return Response(status_code=204)


@app.post("/admin/users/{user_id}/unlock", status_code=204)
async def admin_unlock(
    user_id: int,
    admin: Admin,
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
    audit: Annotated[AgentRepository, Depends(get_agent_repository)],
) -> Response:
    repo = _admin_repo(settings, users_repo)
    await _acted(audit, admin, "admin.unlock", await repo.unlock(user_id))
    return Response(status_code=204)


@app.post("/admin/users/{user_id}/sign-out", status_code=204)
async def admin_sign_out(
    user_id: int,
    admin: Admin,
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
    audit: Annotated[AgentRepository, Depends(get_agent_repository)],
) -> Response:
    repo = _admin_repo(settings, users_repo)
    await _acted(audit, admin, "admin.sign_out", await repo.sign_out_everywhere(user_id))
    return Response(status_code=204)


@app.delete("/admin/users/{user_id}", status_code=204)
async def admin_delete_user(
    user_id: int,
    admin: Admin,
    settings: Annotated[Settings, Depends(get_settings)],
    users_repo: Annotated[UserRepository | None, Depends(get_user_repository_or_none)],
    audit: Annotated[AgentRepository, Depends(get_agent_repository)],
) -> Response:
    """Delete an account. An administrator cannot delete their own, so they cannot lock
    themselves out."""
    repo = _admin_repo(settings, users_repo)
    for u in await repo.list_users():
        if u.id == user_id and accounts.name_key(u.name) == accounts.name_key(admin):
            raise HTTPException(status_code=409, detail="You cannot delete your own account.")
    await _acted(audit, admin, "admin.delete_user", await repo.delete_user(user_id))
    return Response(status_code=204)


NAME_KINDS = ("customer", "supplier", "person")


class NameChoice(BaseModel):
    kind: Literal["customer", "supplier", "person"]
    name: str = Field(min_length=1, max_length=255)
    canonical: str | None = Field(default=None, max_length=255)  # for "same as": the name it is


@app.get("/admin/names", response_model=list[NameReview])
async def admin_names(
    admin: Admin,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
) -> list[NameReview]:
    """Names that need a person: a spelling that may be an alias, or a name that may be new."""
    counts = {kind: await repo.name_counts(kind) for kind in NAME_KINDS}
    aliased = {(k, raw) for k, raw, _ in await repo.list_aliases()}
    targets = {(k, c.lower()) for k, _, c in await repo.list_aliases()}
    return review_names(counts, aliased, await repo.decided_names(), targets)


async def _name_settled(audit: AgentRepository, admin: str, event: str, body: NameChoice) -> None:
    from qacopilot.rag import specquery

    specquery._vocab_cache.clear()  # the next question sees the change at once
    await audit.audit(
        None, admin, event, {"kind": body.kind, "name": body.name, "as": body.canonical}
    )


@app.post("/admin/names/alias", status_code=204)
async def admin_name_alias(
    body: NameChoice,
    admin: Admin,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    audit: Annotated[AgentRepository, Depends(get_agent_repository)],
) -> Response:
    """Treat a spelling as another name from now on ("Sainsbury" is "Sainsbury's")."""
    known = await repo.name_counts(body.kind)
    if not body.canonical or body.canonical not in known:
        raise HTTPException(status_code=422, detail="Pick one of the names already in the specs.")
    if body.canonical.strip().lower() == body.name.strip().lower():
        raise HTTPException(status_code=422, detail="A name cannot be an alias of itself.")
    await repo.set_alias(body.kind, body.name, body.canonical, by=admin)
    await _name_settled(audit, admin, "admin.alias", body)
    return Response(status_code=204)


@app.post("/admin/names/separate", status_code=204)
async def admin_name_separate(
    body: NameChoice,
    admin: Admin,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    audit: Annotated[AgentRepository, Depends(get_agent_repository)],
) -> Response:
    """Confirm a name is its own name: it is not suggested as an alias again."""
    await repo.keep_separate(body.kind, body.name, by=admin)
    await _name_settled(audit, admin, "admin.name_separate", body)
    return Response(status_code=204)


class ExportRequest(BaseModel):
    query: SpecQuery  # the validated query an answer carried in its ``export`` field


@app.post(
    "/export/specs",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "Every matching spec as CSV"}},
)
async def export_spec_rows(
    body: ExportRequest,
    repo: Annotated[DocumentRepository, Depends(get_repository)],
    user: User,
) -> Response:
    """Download every spec a list or count answer selected (the answer shows only the first 25)."""
    try:
        rows = await export_specs(repo, body.query)
    except ExportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("spec export", extra={"user": user, "rows": len(rows)})
    name = f"specs-{date.today():%Y%m%d}.csv"
    return Response(
        content=csv_text(rows).encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/me", response_model=Me)
async def me(user: User, settings: Annotated[Settings, Depends(get_settings)]) -> Me:
    """Who the API key belongs to (also a cheap way for the UI to check a key)."""
    return Me(user=user, admin=accounts.name_key(user) in accounts.admin_keys(settings.admins))


@app.get(
    "/documents", response_model=list[DocumentSummary], dependencies=[Depends(require_api_key)]
)
async def list_documents(
    repo: Annotated[DocumentRepository, Depends(get_repository)],
) -> list[DocumentSummary]:
    return await repo.list_documents()


class PageOut(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    page: int
    method: str  # "text" or "ocr"
    text: str


@app.get(
    "/documents/{doc_id}/pages/{page}",
    response_model=PageOut,
    dependencies=[Depends(require_api_key)],
)
async def get_page(
    doc_id: str, page: int, repo: Annotated[DocumentRepository, Depends(get_repository)]
) -> PageOut:
    """The text of one page, as the model saw it. This is what a citation link opens: original
    PDFs are not stored here (real documents stay on the X: drive)."""
    found = await repo.get_page(doc_id, page)
    if found is None:
        raise HTTPException(status_code=404, detail="Unknown document or page")
    doc, row = found
    return PageOut(
        doc_id=doc.doc_id,
        filename=doc.filename,
        doc_type=doc.doc_type,
        page=row.page,
        method=row.method,
        text=row.text,
    )


def mount_web(application: FastAPI, dist: Path) -> None:
    """Serve the built front end (web/dist) from the same origin, after every API route."""
    if dist.is_dir():
        application.mount("/", StaticFiles(directory=dist, html=True), name="web")


class Reclassify(BaseModel):
    doc_type: DocType


class Reclassified(BaseModel):
    doc_id: str
    doc_type: DocType
    chunks: int


@app.post(
    "/documents/{doc_id}/reclassify",
    response_model=Reclassified,
    dependencies=[Depends(require_api_key)],
)
async def reclassify(
    doc_id: str,
    body: Reclassify,
    store: Annotated[QdrantStore, Depends(get_store)],
    repo: Annotated[DocumentRepository, Depends(get_repository)],
) -> Reclassified:
    """Correct a document's type. It is re-chunked from its saved text (no re-upload) and any
    extraction made for the old type is discarded."""
    chunks = await reclassify_document(doc_id, body.doc_type, store, repo)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Unknown document")
    return Reclassified(doc_id=doc_id, doc_type=body.doc_type, chunks=chunks)


# Must stay the LAST statement: a mount at "/" answers every path, so any route registered
# after it would be unreachable (it returns 405). tests/test_route_order.py guards this.
mount_web(app, Path(__file__).resolve().parents[4] / "web" / "dist")
