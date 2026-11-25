"""Prometheus metrics: the endpoint, what it counts, and that labels stay low-cardinality."""

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from prometheus_client import REGISTRY
from qacopilot.config import Settings
from qacopilot.rag import DoneEvent, NoAnswerEvent, TokenEvent
from qacopilot.rag.answer import AskEvent

from qa_api import metrics
from qa_api.main import app, get_llm, get_repository, get_settings


def value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class Healthy:
    async def health(self) -> bool:
        return True


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    class NoPages:  # stands in for the database, which CI does not configure for these tests
        async def get_page(self, doc_id: str, page: int) -> None:
            return None

    app.dependency_overrides[get_llm] = lambda: Healthy()
    app.dependency_overrides[get_repository] = lambda: NoPages()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def settings_with() -> Iterator[Any]:
    def use(**kw: Any) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(**kw)

    yield use
    app.dependency_overrides.pop(get_settings, None)


async def test_requests_are_counted_by_route_template_not_by_concrete_path(
    client: httpx.AsyncClient,
) -> None:
    before = value("qacopilot_http_requests_total", method="GET", route="/health", status="200")
    await client.get("/health")
    await client.get("/health")
    assert (
        value("qacopilot_http_requests_total", method="GET", route="/health", status="200")
        == before + 2
    )
    body = (await client.get("/metrics")).text
    assert "qacopilot_http_request_seconds_bucket" in body

    # a path with an id in it is labelled by its template, so series do not multiply per document
    await client.get("/documents/abc123/pages/7")
    await client.get("/documents/zzz999/pages/8")
    body = (await client.get("/metrics")).text
    assert "abc123" not in body and "zzz999" not in body
    assert 'route="/documents/{doc_id}/pages/{page}"' in body


async def test_an_unknown_path_is_one_label_not_a_new_series_each_time(
    client: httpx.AsyncClient,
) -> None:
    for i in range(5):
        await client.get(f"/no/such/path/{i}")
    body = (await client.get("/metrics")).text
    assert "/no/such/path" not in body
    assert 'route="unmatched"' in body


async def test_metrics_needs_the_token_when_one_is_set(
    client: httpx.AsyncClient, settings_with: Any
) -> None:
    settings_with(metrics_token="scrape-secret", api_key="other")
    assert (await client.get("/metrics")).status_code == 401
    assert (
        await client.get("/metrics", headers={"authorization": "Bearer wrong"})
    ).status_code == 401
    # the API key is not a substitute while a metrics token is configured
    assert (await client.get("/metrics", headers={"x-api-key": "other"})).status_code == 401
    ok = await client.get("/metrics", headers={"authorization": "Bearer scrape-secret"})
    assert ok.status_code == 200 and ok.headers["content-type"].startswith("text/plain")


async def test_metrics_falls_back_to_the_api_key_without_a_token(
    client: httpx.AsyncClient, settings_with: Any
) -> None:
    settings_with(api_key="k1")
    assert (await client.get("/metrics")).status_code == 401
    assert (await client.get("/metrics", headers={"x-api-key": "k1"})).status_code == 200
    settings_with()  # local development: nothing configured, nothing required
    assert (await client.get("/metrics")).status_code == 200


async def test_metrics_exposes_no_document_or_question_text(client: httpx.AsyncClient) -> None:
    await client.post("/ask", json={"question": "what is the secret recipe for X"})
    body = (await client.get("/metrics")).text
    assert "secret recipe" not in body


async def _events(*items: AskEvent) -> AsyncIterator[AskEvent]:
    for item in items:
        yield item


async def _drain(events: AsyncIterator[AskEvent]) -> None:
    async for _ in events:
        pass


async def test_ask_outcomes_are_recorded_by_path() -> None:
    def n(via: str, outcome: str) -> float:
        return value("qacopilot_ask_total", via=via, outcome=outcome)

    a, b, c = n("certificates", "answered"), n("documents", "refused"), n("documents", "error")
    await _drain(
        metrics.measured_ask(
            _events(
                TokenEvent(text="x"),
                DoneEvent(answer="x [1]", answered=True, cited=[1], via="certificates"),
            )
        )
    )
    await _drain(metrics.measured_ask(_events(NoAnswerEvent(reason="no_relevant_documents"))))

    async def boom() -> AsyncIterator[AskEvent]:
        yield TokenEvent(text="x")
        raise RuntimeError("llm down")

    with pytest.raises(RuntimeError):
        await _drain(metrics.measured_ask(boom()))
    assert (n("certificates", "answered"), n("documents", "refused"), n("documents", "error")) == (
        a + 1,
        b + 1,
        c + 1,
    )


class FakeDocs:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def certificates_due(self, days: int) -> list[Any]:
        if self.fail:
            raise ConnectionError("postgres down")
        return [SimpleNamespace(status="overdue"), SimpleNamespace(status="due_soon")]

    async def certificates_to_review(self) -> list[Any]:
        return [object(), object(), object()]


class FakeAgents:
    def __init__(self, created: list[datetime]) -> None:
        self.created = created

    async def list_runs(self, status: str | None = None, limit: int = 50) -> list[Any]:
        return [SimpleNamespace(created_at=t) for t in self.created]


async def test_queue_gauges_and_dependency_checks_are_refreshed() -> None:
    now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)

    async def up() -> bool:
        return True

    async def down() -> bool:
        raise ConnectionError("nope")

    await metrics.refresh_gauges(
        documents=FakeDocs(),
        agent_repo=FakeAgents([now - timedelta(hours=5), now - timedelta(minutes=1)]),
        probe_qdrant=up,
        probe_llm=down,
        now=now,
    )
    assert value("qacopilot_certificates_needing_review") == 3
    assert value("qacopilot_certificates_overdue") == 1
    assert value("qacopilot_certificates_due_30d") == 2
    assert value("qacopilot_agent_runs_awaiting_approval") == 2
    assert value("qacopilot_agent_oldest_awaiting_approval_seconds") == 5 * 3600
    assert value("qacopilot_dependency_up", service="postgres") == 1
    assert value("qacopilot_dependency_up", service="qdrant") == 1
    assert value("qacopilot_dependency_up", service="llm") == 0  # one failing probe is just a 0


async def test_a_database_outage_is_reported_and_does_not_stop_the_other_checks() -> None:
    async def up() -> bool:
        return True

    await metrics.refresh_gauges(
        documents=FakeDocs(fail=True), agent_repo=FakeAgents([]), probe_qdrant=up, probe_llm=up
    )
    assert value("qacopilot_dependency_up", service="postgres") == 0
    assert value("qacopilot_dependency_up", service="qdrant") == 1


async def test_a_slow_probe_is_cut_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setattr(metrics, "PROBE_TIMEOUT_SECONDS", 0.05)

    async def hangs() -> bool:
        await asyncio.sleep(5)
        return True

    await metrics.refresh_gauges(documents=FakeDocs(), agent_repo=FakeAgents([]), probe_llm=hangs)
    assert value("qacopilot_dependency_up", service="llm") == 0
