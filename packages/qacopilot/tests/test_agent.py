import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from email import message_from_bytes
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEmbedder, FakeReranker
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from qdrant_client import AsyncQdrantClient

from qacopilot.agent import (
    AgentDeps,
    AgentService,
    RuleRouter,
    RunConflict,
    RunNotFound,
    parse_window_days,
    window_cap_note,
)
from qacopilot.agent.drafts import FALLBACK_INTRO
from qacopilot.agent.graph import deliver_all
from qacopilot.config import Settings
from qacopilot.extraction import CalibrationCert
from qacopilot.ingest import DocType, PageText, ParsedDocument
from qacopilot.ingest.pipeline import ingest_pdf
from qacopilot.llm import BaseLLMProvider, LLMResponse
from qacopilot.mail import DeliveryReceipt, EmailMessage, FileMailer, Mailer
from qacopilot.repositories import AgentRepository, DocumentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, libpq_conninfo, temporary_engine
from qacopilot.retrieval import QdrantStore, Retriever

SAMPLES = Path(__file__).resolve().parents[3] / "data" / "samples"
TODAY = date.today()
RECIPIENTS = ["qa@example.com"]


class ScriptedLLM(BaseLLMProvider):
    def __init__(self, intro: str = "Please arrange these calibrations before they lapse.") -> None:
        self.intro = intro
        self.calls = 0

    async def chat(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if isinstance(self.intro, Exception):
            raise self.intro
        return LLMResponse(content=self.intro)

    async def stream(self, messages: Sequence[Any], **kwargs: Any) -> AsyncIterator[str]:
        yield "Cooked duck must reach 75C [1]."

    async def health(self) -> bool:
        return True


class FlakyMailer(Mailer):
    """Fails until ``working`` is set, then delegates."""

    def __init__(self, inner: Mailer) -> None:
        self.inner, self.working = inner, False

    async def deliver(self, message: EmailMessage, *, key: str) -> DeliveryReceipt:
        if not self.working:
            raise ConnectionError("smtp down")
        return await self.inner.deliver(message, key=key)


@dataclass
class Env:
    service: AgentService
    deps: AgentDeps
    docs: DocumentRepository
    outbox: Path
    checkpointer: InMemorySaver
    conninfo: str

    def emails(self) -> list[Path]:
        return sorted(self.outbox.glob("*.eml")) if self.outbox.exists() else []

    def restart(self) -> AgentService:
        """A new service and graph over the same checkpointer and database (a process restart)."""
        return AgentService(self.deps, self.checkpointer)


@pytest.fixture
async def env(tmp_path: Path) -> AsyncIterator[Env]:
    try:
        async with temporary_engine() as engine:
            docs, agent_repo = DocumentRepository(engine), AgentRepository(engine)
            store = QdrantStore(FakeEmbedder(), Settings(), client=AsyncQdrantClient(":memory:"))
            await store.ensure_collection()
            name = "sop_cooking_temperature.pdf"
            await ingest_pdf((SAMPLES / name).read_bytes(), name, store, repo=docs)
            deps = AgentDeps(
                llm=ScriptedLLM(),
                retriever=Retriever(store, FakeReranker(), min_score=0.0),
                documents=docs,
                agent_repo=agent_repo,
                mailer=FileMailer(tmp_path / "outbox"),
                recipients=list(RECIPIENTS),
                router=RuleRouter(),
            )
            saver = InMemorySaver()
            yield Env(
                AgentService(deps, saver), deps, docs, tmp_path / "outbox", saver,
                libpq_conninfo(engine),
            )  # fmt: skip
            await store.aclose()
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


async def add_cert(docs: DocumentRepository, doc_id: str, serial: str, due_in: int) -> None:
    due = TODAY + timedelta(days=due_in)
    cert = CalibrationCert(
        instrument_serial=serial,
        instrument_description="Probe thermometer",
        calibration_date=due - timedelta(days=365),
        next_due=due,
        result="pass",
    )
    parsed = ParsedDocument(source="c.pdf", pages=[PageText(page=1, text="c", method="text")])
    await docs.save_document(doc_id, f"{doc_id}.pdf", DocType.CALIBRATION_CERT, parsed, 1)
    await docs.save_extraction(
        doc_id, DocType.CALIBRATION_CERT, cert, prompt_version="t", model="t"
    )


REMIND = "Find all certificates expiring in the next 30 days and draft reminders"


# --- routing and question answering -------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "route"),
    [
        (REMIND, "action"),
        ("Send an email to the team about expiring probes", "action"),
        ("What temperature must cooked duck reach?", "question"),
        ("Which probe certificates expire next month?", "question"),  # asks, does not act
    ],
)
def test_router(query: str, route: str) -> None:
    assert RuleRouter().route(query) == route


@pytest.mark.parametrize(
    ("query", "days"),
    [
        ("expiring in the next 14 days", 14),
        ("within 2 weeks", 14),
        ("due in 3 months", 90),
        ("this month", 30),
        ("next week", 7),
        ("certs expiring soon", 30),
        ("within 5000 days", 365),  # capped
        ("within 0 days", 1),
        ("within 1000 days", 365),
        ("within 1.5 weeks", 30),  # not understood: falls back to the default, never to 5 days
    ],
)
def test_window_parsing(query: str, days: int) -> None:
    assert parse_window_days(query) == days


def test_window_cap_is_reported_not_silent() -> None:
    note = window_cap_note("expiring in the next 400 days")
    assert "400 days" in note and "365" in note
    assert window_cap_note("next 30 days") == ""
    assert window_cap_note("within 365 days") == ""


async def test_question_is_answered_with_citations_and_sends_nothing(env: Env) -> None:
    view = await env.service.start("alice", "what core temperature must cooked duck reach")
    assert (view.route, view.status, view.answered) == ("question", "completed", True)
    assert view.citations[0]["filename"] == "sop_cooking_temperature.pdf"
    assert view.drafts == [] and env.emails() == []


async def test_unanswerable_question_says_so(env: Env) -> None:
    view = await env.service.start("alice", "where is the fire evacuation assembly point")
    assert view.status == "completed" and view.answered is False and view.citations == []
    assert "not covered" in view.message


# --- the approval gate --------------------------------------------------------------------------


async def start_reminder(env: Env) -> Any:
    await add_cert(env.docs, "overdue", "TP-OVERDUE", -5)
    await add_cert(env.docs, "soon", "TP-SOON", 10)
    await add_cert(env.docs, "later", "TP-LATER", 200)
    return await env.service.start("alice", REMIND)


async def test_reminders_are_drafted_but_nothing_is_sent_before_approval(env: Env) -> None:
    view = await start_reminder(env)
    assert view.status == "awaiting_approval" and view.route == "action"
    (draft,) = view.drafts
    body = draft["message"]["body"]
    assert "TP-OVERDUE" in body and "TP-SOON" in body and "TP-LATER" not in body
    assert "5 days overdue" in body and "in 10 days" in body
    assert draft["message"]["to"] == RECIPIENTS
    assert "(1 overdue)" in draft["message"]["subject"]
    assert env.emails() == []  # the safety property: drafting never delivers


async def test_approval_delivers_once_and_records_who_approved(env: Env) -> None:
    started = await start_reminder(env)
    view = await env.service.decide(started.run_id, "bob", "approve")
    assert (view.status, view.decided_by) == ("completed", "bob")
    (eml,) = env.emails()
    parsed = message_from_bytes(eml.read_bytes())
    assert parsed["To"] == "qa@example.com" and "TP-SOON" in parsed.get_payload()

    events = await env.deps.agent_repo.audit_for_run(started.run_id)
    assert [(e.actor, e.event) for e in events] == [
        ("alice", "run_started"),
        ("system", "drafts_created"),
        ("bob", "approved"),
        ("bob", "email_delivered"),
    ]
    (row,) = await env.deps.agent_repo.outbox_for_run(started.run_id)
    assert row.status == "delivered" and row.approved_by == "bob"


async def test_rejection_sends_nothing_and_is_audited(env: Env) -> None:
    started = await start_reminder(env)
    view = await env.service.decide(started.run_id, "bob", "reject", "wrong list")
    assert view.status == "rejected" and "Nothing was sent" in view.message
    assert env.emails() == []
    events = await env.deps.agent_repo.audit_for_run(started.run_id)
    assert events[-1].event == "rejected" and events[-1].detail == {"reason": "wrong list"}


async def test_a_run_can_only_be_decided_once(env: Env) -> None:
    started = await start_reminder(env)
    await env.service.decide(started.run_id, "bob", "approve")
    with pytest.raises(RunConflict):
        await env.service.decide(started.run_id, "carol", "approve")
    with pytest.raises(RunConflict):
        await env.service.decide(started.run_id, "carol", "reject")
    assert len(env.emails()) == 1


async def test_simultaneous_approvals_send_one_email(env: Env) -> None:
    started = await start_reminder(env)
    results = await asyncio.gather(
        *(env.service.decide(started.run_id, f"user{i}", "approve") for i in range(5)),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert all(isinstance(r, RunConflict) for r in results if isinstance(r, Exception))
    assert len(env.emails()) == 1


async def test_a_retried_delivery_cannot_send_twice(env: Env) -> None:
    started = await start_reminder(env)
    await env.service.decide(started.run_id, "bob", "approve")
    view = await env.service.get(started.run_id)
    again = await deliver_all(env.deps, started.run_id, view.drafts, "bob")  # e.g. a re-driven node
    assert [r["status"] for r in again] == ["delivered"]
    assert len(env.emails()) == 1
    assert len(await env.deps.agent_repo.outbox_for_run(started.run_id)) == 1


async def test_the_pause_survives_a_restart(env: Env) -> None:
    started = await start_reminder(env)
    restarted = env.restart()  # new graph object, same checkpointer + database
    pending = await restarted.get(started.run_id)
    assert pending.status == "awaiting_approval" and len(pending.drafts) == 1
    done = await restarted.decide(started.run_id, "bob", "approve")
    assert done.status == "completed" and len(env.emails()) == 1


async def test_failed_delivery_is_reported_and_can_be_retried_without_duplicates(env: Env) -> None:
    flaky = FlakyMailer(env.deps.mailer)
    env.deps.mailer = flaky
    started = await start_reminder(env)
    failed = await env.service.decide(started.run_id, "bob", "approve")
    assert failed.status == "delivery_failed" and failed.deliveries[0]["status"] == "failed"
    assert env.emails() == []

    flaky.working = True
    fixed = await env.service.retry_delivery(started.run_id, "bob")
    assert fixed.status == "completed" and len(env.emails()) == 1
    with pytest.raises(RunConflict):
        await env.service.retry_delivery(started.run_id, "bob")  # nothing left to retry


# --- cases with nothing to send -----------------------------------------------------------------


async def test_nothing_due_means_no_action_and_no_pause(env: Env) -> None:
    await add_cert(env.docs, "far", "TP-FAR", 300)
    view = await env.service.start("alice", REMIND)
    assert view.status == "no_action" and "No calibration certificates" in view.message


async def test_no_recipients_blocks_instead_of_inventing_one(env: Env) -> None:
    await add_cert(env.docs, "soon", "TP-SOON", 5)
    env.deps.recipients = []
    view = await env.service.start("alice", REMIND)
    assert view.status == "blocked" and "RECIPIENTS" in view.message and env.emails() == []


async def test_unknown_run(env: Env) -> None:
    with pytest.raises(RunNotFound):
        await env.service.get("nope")
    with pytest.raises(RunNotFound):
        await env.service.decide("nope", "bob", "approve")


# --- the LLM writes only the intro --------------------------------------------------------------


@pytest.mark.parametrize(
    "intro",
    [
        "Costs went up 40% so please call 0800 123456 about these.",  # digits: a fact it invented
        "See http://evil.example for details about these calibrations.",
        "Too short",
        RuntimeError("model down"),
    ],
)
async def test_odd_llm_output_falls_back_to_the_fixed_intro(env: Env, intro: Any) -> None:
    env.deps.llm = ScriptedLLM(intro)
    view = await start_reminder(env)
    assert view.drafts[0]["message"]["body"].startswith(FALLBACK_INTRO)


async def test_a_good_intro_is_used_and_the_facts_still_come_from_code(env: Env) -> None:
    view = await start_reminder(env)
    body = view.drafts[0]["message"]["body"]
    assert body.startswith("Please arrange these calibrations before they lapse.")
    assert "TP-SOON" in body


async def test_a_paused_run_survives_a_real_restart_on_the_postgres_checkpointer(env: Env) -> None:
    await add_cert(env.docs, "soon", "TP-SOON", 10)
    async with AsyncPostgresSaver.from_conn_string(env.conninfo) as first:
        await first.setup()
        started = await AgentService(env.deps, first).start("alice", REMIND)
        assert started.status == "awaiting_approval"
    # The first connection and graph are gone: this is what a process restart looks like.
    assert env.emails() == []
    async with AsyncPostgresSaver.from_conn_string(env.conninfo) as second:
        service = AgentService(env.deps, second)
        pending = await service.get(started.run_id)
        assert pending.status == "awaiting_approval" and len(pending.drafts) == 1
        done = await service.decide(started.run_id, "bob", "approve")
        assert (done.status, done.decided_by) == ("completed", "bob")
    assert len(env.emails()) == 1


async def test_reminders_and_answers_flag_readings_nobody_has_confirmed() -> None:
    from qacopilot.agent.drafts import build_reminder, fact_table, unverified_note
    from qacopilot.repositories import CertificateStatus

    def status(serial: str, unverified: bool) -> CertificateStatus:
        return CertificateStatus(
            doc_id=serial,
            instrument_serial=serial,
            instrument_description=None,
            certificate_number=None,
            calibration_date=date(2025, 12, 1),
            next_due=date(2026, 12, 1),
            days_left=12,
            status="due_soon",
            unverified=unverified,
        )

    certs = [status("TP-1", True), status("TP-2", False), status("TP-3", True)]
    table = fact_table(certs)
    assert (
        table.count("UNVERIFIED reading") == 2
        and "TP-2 | - | 2026-12-01 | in 12 days\n" in table + "\n"
    )
    assert "2 of these are unverified readings" in unverified_note(certs)
    assert "1 of these is an unverified reading" in unverified_note(certs[:1])
    assert unverified_note([status("TP-2", False)]) == ""

    draft = await build_reminder(certs, ["qa@example.com"], None, 30, today=date(2026, 11, 19))
    assert "unverified readings" in draft.message.body
    clean = await build_reminder([status("TP-2", False)], ["qa@example.com"], None, 30)
    assert "unverified" not in clean.message.body.lower()
