from collections.abc import AsyncIterator
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from qacopilot.agent import AgentDeps, AgentService
from qacopilot.config import Settings
from qacopilot.extraction import CalibrationCert
from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.mail import FileMailer
from qacopilot.repositories import AgentRepository, DocumentRepository
from qacopilot.repositories.models import Feedback
from qacopilot.retrieval import QdrantStore, Retriever
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from stubs import ScriptedLLM, StubEmbedder, StubReranker

from qa_api.main import (
    app,
    get_agent_repository,
    get_agent_service,
    get_repository,
    get_settings,
)

KEYS = Settings(api_keys="alice:key-alice,bob:key-bob")
ALICE = {"x-api-key": "key-alice"}
BOB = {"x-api-key": "key-bob"}
REMIND = "Draft reminders for certificates expiring in the next 30 days"


@pytest.fixture
async def client(
    repos: tuple[DocumentRepository, AgentRepository], tmp_path: Path
) -> AsyncIterator[httpx.AsyncClient]:
    docs, agent_repo = repos
    store = QdrantStore(StubEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
    await store.ensure_collection()
    llm = ScriptedLLM("Cooked duck needs 75C [1].")
    app.state.test_llm = llm
    deps = AgentDeps(
        llm=llm,
        retriever=Retriever(store, StubReranker(), min_score=0.0),
        documents=docs,
        agent_repo=agent_repo,
        mailer=FileMailer(tmp_path / "outbox"),
        recipients=["qa@example.com"],
    )
    service = AgentService(deps, InMemorySaver())
    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[get_agent_repository] = lambda: agent_repo
    app.dependency_overrides[get_repository] = lambda: docs
    app.dependency_overrides[get_settings] = lambda: KEYS
    due = date.today() + timedelta(days=10)
    cert = CalibrationCert(
        instrument_serial="TP-SOON", calibration_date=due - timedelta(days=365), next_due=due,
        result="pass",
    )  # fmt: skip
    parsed = ParsedDocument(source="c.pdf", pages=[PageText(page=1, text="c", method="text")])
    await docs.save_document("c1", "c1.pdf", DocType.CALIBRATION_CERT, parsed, 1)
    await docs.save_extraction("c1", DocType.CALIBRATION_CERT, cert, prompt_version="t", model="t")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        c.outbox = tmp_path / "outbox"  # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()
    await store.aclose()


def outbox(client: httpx.AsyncClient) -> list[Path]:
    path: Path = client.outbox  # type: ignore[attr-defined]
    return list(path.glob("*.eml")) if path.exists() else []


async def test_authentication_names_the_caller(client: httpx.AsyncClient) -> None:
    assert (await client.post("/agent/runs", json={"query": "hi"})).status_code == 401
    bad = await client.post("/agent/runs", json={"query": "hi"}, headers={"x-api-key": "nope"})
    assert bad.status_code == 401
    odd = await client.post(
        "/agent/runs", json={"query": "hi"}, headers=[(b"x-api-key", "café".encode())]
    )
    assert odd.status_code == 401  # not a 500 from comparing non-ASCII text
    ok = await client.post("/agent/runs", json={"query": REMIND}, headers=ALICE)
    assert ok.status_code == 200 and ok.json()["created_by"] == "alice"


async def test_full_flow_start_queue_approve_audit(client: httpx.AsyncClient) -> None:
    started = (await client.post("/agent/runs", json={"query": REMIND}, headers=ALICE)).json()
    run_id = started["run_id"]
    assert started["status"] == "awaiting_approval" and outbox(client) == []

    queue = await client.get("/agent/runs", params={"status": "awaiting_approval"}, headers=BOB)
    assert [r["run_id"] for r in queue.json()] == [run_id]

    done = (await client.post(f"/agent/runs/{run_id}/approve", headers=BOB)).json()
    assert done["status"] == "completed" and done["decided_by"] == "bob"
    assert len(outbox(client)) == 1

    assert (
        await client.get("/agent/runs", params={"status": "awaiting_approval"}, headers=BOB)
    ).json() == []
    audit = (await client.get(f"/agent/runs/{run_id}/audit", headers=BOB)).json()
    assert [(e["actor"], e["event"]) for e in audit] == [
        ("alice", "run_started"),
        ("system", "drafts_created"),
        ("bob", "approved"),
        ("bob", "email_delivered"),
    ]


async def test_second_approval_is_a_conflict_and_sends_nothing_more(
    client: httpx.AsyncClient,
) -> None:
    run_id = (await client.post("/agent/runs", json={"query": REMIND}, headers=ALICE)).json()[
        "run_id"
    ]
    assert (await client.post(f"/agent/runs/{run_id}/approve", headers=BOB)).status_code == 200
    again = await client.post(f"/agent/runs/{run_id}/approve", headers=ALICE)
    assert again.status_code == 409
    assert (await client.post(f"/agent/runs/{run_id}/reject", headers=ALICE)).status_code == 409
    assert len(outbox(client)) == 1


async def test_reject_with_reason(client: httpx.AsyncClient) -> None:
    run_id = (await client.post("/agent/runs", json={"query": REMIND}, headers=ALICE)).json()[
        "run_id"
    ]
    r = await client.post(f"/agent/runs/{run_id}/reject", json={"reason": "not now"}, headers=BOB)
    assert r.json()["status"] == "rejected" and outbox(client) == []
    audit = (await client.get(f"/agent/runs/{run_id}/audit", headers=BOB)).json()
    assert audit[-1]["event"] == "rejected" and audit[-1]["detail"] == {"reason": "not now"}


async def test_questions_and_unknown_runs(client: httpx.AsyncClient) -> None:
    q = await client.post("/agent/runs", json={"query": "what is the shelf life"}, headers=ALICE)
    assert q.json()["route"] == "question" and q.json()["status"] == "completed"
    for path in ("", "/approve", "/reject", "/retry-delivery"):
        method = client.get if path == "" else client.post
        r = await method(f"/agent/runs/missing{path}", headers=ALICE)
        assert r.status_code == 404, path
    assert (await client.post("/agent/runs", json={"query": ""}, headers=ALICE)).status_code == 422


async def test_agent_unavailable_without_a_service() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            r = await c.post("/agent/runs", json={"query": "hi"})
        assert r.status_code == 503
    finally:
        app.dependency_overrides.clear()


async def _feedback_rows(client: httpx.AsyncClient) -> list[Feedback]:
    repo: AgentRepository = app.dependency_overrides[get_agent_repository]()
    async with repo.session() as s:
        return list(await s.scalars(select(Feedback).order_by(Feedback.id)))


async def test_feedback_on_an_agent_run_is_stored_with_the_users_name(
    client: httpx.AsyncClient,
) -> None:
    run_id = (await client.post("/agent/runs", json={"query": REMIND}, headers=ALICE)).json()[
        "run_id"
    ]
    body = {"request_id": run_id, "kind": "agent", "rating": "down", "comment": "wrong people"}
    r = await client.post("/feedback", json=body, headers=BOB)
    assert r.status_code == 200 and r.json() == {"stored": True, "langfuse": False}  # tracing off
    (row,) = await _feedback_rows(client)
    assert (row.request_id, row.kind, row.rating, row.created_by, row.comment) == (
        run_id, "agent", -1, "bob", "wrong people",
    )  # fmt: skip


async def test_feedback_for_an_ask_answer_needs_only_the_request_id(
    client: httpx.AsyncClient,
) -> None:
    body = {"request_id": "0123456789abcdef", "kind": "ask", "rating": "up"}
    assert (await client.post("/feedback", json=body, headers=ALICE)).status_code == 200
    assert [(r.rating, r.created_by) for r in await _feedback_rows(client)] == [(1, "alice")]


@pytest.mark.parametrize(
    "bad",
    [
        {"request_id": "short", "kind": "ask", "rating": "up"},
        {"request_id": "0123456789abcdef", "kind": "ask", "rating": "meh"},
        {"request_id": "0123456789abcdef", "kind": "other", "rating": "up"},
        {"request_id": "../../etc/passwd", "kind": "ask", "rating": "up"},
        {"request_id": "0123456789abcdef", "kind": "ask", "rating": "up", "comment": "x" * 1001},
    ],
)
async def test_feedback_is_validated(client: httpx.AsyncClient, bad: dict[str, str]) -> None:
    assert (await client.post("/feedback", json=bad, headers=ALICE)).status_code == 422


async def test_feedback_needs_a_key_and_a_real_run(client: httpx.AsyncClient) -> None:
    body = {"request_id": "0123456789abcdef", "kind": "agent", "rating": "up"}
    assert (await client.post("/feedback", json=body)).status_code == 401
    assert (await client.post("/feedback", json=body, headers=ALICE)).status_code == 404
    assert await _feedback_rows(client) == []


async def test_the_agents_llm_calls_run_under_the_run_id(client: httpx.AsyncClient) -> None:
    started = await client.post("/agent/runs", json={"query": REMIND}, headers=ALICE)
    run_id = started.json()["run_id"]
    # one LLM call (the email intro), made under the run id so feedback finds its trace
    assert app.state.test_llm.seen_request_ids == [run_id]


async def test_me_documents_and_page_text(client: httpx.AsyncClient) -> None:
    assert (await client.get("/me")).status_code == 401
    assert (await client.get("/me", headers=BOB)).json() == {"user": "bob", "admin": False}

    docs = (await client.get("/documents", headers=ALICE)).json()
    assert [(d["doc_id"], d["doc_type"], d["extracted"]) for d in docs] == [
        ("c1", "calibration_cert", True)
    ]
    page = (await client.get("/documents/c1/pages/1", headers=ALICE)).json()
    assert (page["filename"], page["page"], page["text"], page["method"]) == (
        "c1.pdf", 1, "c", "text",
    )  # fmt: skip
    assert (await client.get("/documents/c1/pages/9", headers=ALICE)).status_code == 404
    assert (await client.get("/documents/nope/pages/1", headers=ALICE)).status_code == 404
    assert (await client.get("/documents", headers=None)).status_code == 401
