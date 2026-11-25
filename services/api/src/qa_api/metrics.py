"""Prometheus metrics for the API: traffic, Ask paths, agent activity and the work queues.

Everything is labelled with a small fixed set of values (a route template, a status, a path), never
with a document id, a question or a user name, so the number of series stays constant.

Counters and histograms are updated where things happen. Gauges that need the database (the
review queue, overdue certificates, the approval queue) and the dependency checks are refreshed by
a background task every ``REFRESH_SECONDS``, not on scrape: a scrape must stay fast even when
Postgres or the LLM is down.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from qacopilot.metrics import counter, histogram
from qacopilot.rag import DoneEvent, NoAnswerEvent
from qacopilot.rag.answer import AskEvent
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 30.0
PROBE_TIMEOUT_SECONDS = 8.0
HTTP_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

HTTP_REQUESTS = Counter(
    "qacopilot_http_requests_total",
    "HTTP requests by method, route template and status",
    ["method", "route", "status"],
)
HTTP_SECONDS = Histogram(
    "qacopilot_http_request_seconds",
    "Time to finish a request (for streamed answers, until the stream ends)",
    ["method", "route"],
    buckets=HTTP_BUCKETS,
)

ASK_REQUESTS = counter(
    "qacopilot_ask_total",
    "Questions asked, by the path that answered (documents, certificates, coa_check) and outcome "
    "(answered, refused, error)",
    ["via", "outcome"],
)
ASK_SECONDS = histogram(
    "qacopilot_ask_seconds", "Time to answer a question, by path", ["via"], HTTP_BUCKETS
)
INGESTED = counter(
    "qacopilot_documents_ingested_total", "Documents ingested, by type", ["doc_type"]
)
AGENT_RUNS = counter(
    "qacopilot_agent_runs_total", "Agent runs started, by resulting status", ["status"]
)
AGENT_DECISIONS = counter(
    "qacopilot_agent_decisions_total", "Approvals and rejections of drafted actions", ["decision"]
)

CERTS_NEEDING_REVIEW = Gauge(
    "qacopilot_certificates_needing_review", "Certificate readings waiting for a person to confirm"
)
CERTS_OVERDUE = Gauge(
    "qacopilot_certificates_overdue", "Calibration certificates past their due date"
)
CERTS_DUE_30D = Gauge(
    "qacopilot_certificates_due_30d", "Calibration certificates overdue or due within 30 days"
)
AWAITING_APPROVAL = Gauge(
    "qacopilot_agent_runs_awaiting_approval", "Agent runs paused for a person to approve"
)
OLDEST_AWAITING_SECONDS = Gauge(
    "qacopilot_agent_oldest_awaiting_approval_seconds",
    "Age of the oldest run still waiting for approval (0 when none)",
)
DEPENDENCY_UP = Gauge(
    "qacopilot_dependency_up", "1 if the API can reach the dependency, else 0", ["service"]
)
LAST_REFRESH = Gauge(
    "qacopilot_metrics_refreshed_timestamp_seconds", "When the queue gauges were last refreshed"
)


def route_label(scope: Scope) -> str:
    """The route *template* (``/documents/{doc_id}/extract``), never the concrete path."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path is None:
        return "unmatched"
    return path or "static"  # the static front-end mount has an empty path


class MetricsMiddleware:
    """Counts and times every HTTP request. Pure ASGI, so a streamed response (an answer arriving
    token by token) is timed until its last byte rather than until its headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500  # what a crash before any response looks like

        async def tracking_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        finally:
            route = route_label(scope)
            method = scope.get("method", "GET")
            HTTP_REQUESTS.labels(method, route, str(status)).inc()
            HTTP_SECONDS.labels(method, route).observe(time.perf_counter() - started)


async def measured_ask(events: AsyncIterator[AskEvent]) -> AsyncIterator[AskEvent]:
    """Pass the answer events through, recording which path answered and how it ended."""
    started = time.perf_counter()
    via, outcome = "documents", "error"
    try:
        async for event in events:
            if isinstance(event, DoneEvent):
                via = event.via
                outcome = "answered" if event.answered else "refused"
            elif isinstance(event, NoAnswerEvent) and outcome == "error":
                outcome = "refused"  # nothing relevant was found; no DoneEvent follows
            yield event
        if outcome == "error":  # the stream ended with neither: treat as refused, not a fault
            outcome = "refused"
    finally:
        ASK_REQUESTS.labels(via, outcome).inc()
        ASK_SECONDS.labels(via).observe(time.perf_counter() - started)


Probe = Callable[[], Awaitable[bool]]


async def refresh_gauges(
    *,
    documents: Any,
    agent_repo: Any,
    probe_qdrant: Probe | None = None,
    probe_llm: Probe | None = None,
    now: datetime | None = None,
) -> None:
    """Update the queue gauges and dependency checks. A failing part never stops the others."""
    now = now or datetime.now(UTC)
    try:
        due = await documents.certificates_due(30)
        CERTS_DUE_30D.set(len(due))
        CERTS_OVERDUE.set(sum(1 for c in due if c.status == "overdue"))
        CERTS_NEEDING_REVIEW.set(len(await documents.certificates_to_review()))
        waiting = await agent_repo.list_runs("awaiting_approval", 200)
        AWAITING_APPROVAL.set(len(waiting))
        OLDEST_AWAITING_SECONDS.set(
            max((now - r.created_at).total_seconds() for r in waiting) if waiting else 0
        )
        DEPENDENCY_UP.labels("postgres").set(1)
        LAST_REFRESH.set(now.timestamp())
    except Exception:
        logger.warning("metrics: could not read the database", exc_info=True)
        DEPENDENCY_UP.labels("postgres").set(0)
    for name, probe in (("qdrant", probe_qdrant), ("llm", probe_llm)):
        if probe is None:
            continue
        try:
            ok = await asyncio.wait_for(probe(), PROBE_TIMEOUT_SECONDS)
        except Exception:
            ok = False
        DEPENDENCY_UP.labels(name).set(1 if ok else 0)


async def refresh_forever(refresh: Callable[[], Awaitable[None]]) -> None:
    """Run ``refresh`` now and every ``REFRESH_SECONDS``; it must never take the API down."""
    while True:
        try:
            await refresh()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("metrics refresh failed")
        await asyncio.sleep(REFRESH_SECONDS)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
