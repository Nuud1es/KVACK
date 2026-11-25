import asyncio
from collections.abc import AsyncIterator

import pytest

from qacopilot.mail import DeliveryReceipt, EmailMessage
from qacopilot.repositories import AgentRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repositories

MSG = EmailMessage(to=["qa@example.com"], subject="Reminder", body="Body")
KEY = "run1-abcdef12"


@pytest.fixture
async def agent_repo() -> AsyncIterator[AgentRepository]:
    try:
        async with temporary_repositories() as (_, repo):
            yield repo
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


async def test_claim_then_deliver_then_retry_is_a_no_op(agent_repo: AgentRepository) -> None:
    first, _ = await agent_repo.claim_delivery(KEY, "r1", MSG, "alice")
    assert first == "claimed"
    await agent_repo.mark_delivered(
        KEY, DeliveryReceipt(backend="file", outcome="file", reference="x")
    )

    again, row = await agent_repo.claim_delivery(KEY, "r1", MSG, "bob")
    assert again == "already_delivered"
    assert row.approved_by == "alice" and row.reference == "x"  # the original approver is kept


async def test_unfinished_or_failed_delivery_can_be_retried(agent_repo: AgentRepository) -> None:
    await agent_repo.claim_delivery(KEY, "r1", MSG, "alice")
    assert (await agent_repo.claim_delivery(KEY, "r1", MSG, "alice"))[0] == "retry"  # crashed
    await agent_repo.mark_failed(KEY, "boom")
    claim, row = await agent_repo.claim_delivery(KEY, "r1", MSG, "alice")
    assert claim == "retry" and row.error == "boom"


async def test_only_one_of_two_simultaneous_approvals_wins(agent_repo: AgentRepository) -> None:
    await agent_repo.create_run("r1", "alice", "q")
    await agent_repo.update_run("r1", status="awaiting_approval")
    results = await asyncio.gather(
        *(agent_repo.transition("r1", expect="awaiting_approval", to="running") for _ in range(8))
    )
    assert sorted(results) == [False] * 7 + [True]
    run = await agent_repo.get_run("r1")
    assert run is not None and run.status == "running"


async def test_runs_list_filters_by_status(agent_repo: AgentRepository) -> None:
    for run_id, status in (
        ("a", "awaiting_approval"),
        ("b", "completed"),
        ("c", "awaiting_approval"),
    ):
        await agent_repo.create_run(run_id, "alice", f"query {run_id}")
        await agent_repo.update_run(run_id, status=status, route="action", summary={"n": 1})
    waiting = await agent_repo.list_runs("awaiting_approval")
    assert sorted(r.run_id for r in waiting) == ["a", "c"]
    assert len(await agent_repo.list_runs()) == 3


async def test_audit_log_keeps_order_and_detail(agent_repo: AgentRepository) -> None:
    await agent_repo.audit("r1", "alice", "approved", {"emails": 2})
    await agent_repo.audit("r1", "system", "email_delivered", {"key": KEY})
    await agent_repo.audit("other", "bob", "approved")
    events = await agent_repo.audit_for_run("r1")
    assert [(e.actor, e.event) for e in events] == [
        ("alice", "approved"),
        ("system", "email_delivered"),
    ]
    assert events[0].detail == {"emails": 2}


async def test_feedback_is_stored(agent_repo: AgentRepository) -> None:
    await agent_repo.add_feedback("req1", "ask", -1, "alice", "wrong page")
    await agent_repo.add_feedback("req1", "ask", 1, "bob")
