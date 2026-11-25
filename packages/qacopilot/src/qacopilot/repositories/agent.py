"""Database access for agent runs, the outbox (idempotent delivery), audit log and feedback."""

from typing import Any, Literal

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from qacopilot.mail import DeliveryReceipt, EmailMessage
from qacopilot.repositories.models import AgentRun, AuditLog, Feedback, Outbox

Claim = Literal["claimed", "already_delivered", "retry"]


class AgentRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    def session(self) -> AsyncSession:
        return self._sessions()

    # --- runs -------------------------------------------------------------------------------

    async def create_run(self, run_id: str, user: str, query: str) -> None:
        async with self._sessions.begin() as s:
            s.add(AgentRun(run_id=run_id, created_by=user, query=query, status="running"))

    async def update_run(
        self,
        run_id: str,
        *,
        status: str,
        route: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if route is not None:
            values["route"] = route
        if summary is not None:
            values["summary"] = summary
        async with self._sessions.begin() as s:
            await s.execute(update(AgentRun).where(AgentRun.run_id == run_id).values(**values))

    async def transition(self, run_id: str, *, expect: str, to: str) -> bool:
        """Atomically move a run from ``expect`` to ``to``; False if it was not in ``expect``.

        This is what stops two people approving the same run at once: only one UPDATE matches.
        """
        async with self._sessions.begin() as s:
            result = await s.execute(
                update(AgentRun)
                .where(AgentRun.run_id == run_id, AgentRun.status == expect)
                .values(status=to)
            )
            return bool(result.rowcount)  # type: ignore[attr-defined]

    async def get_run(self, run_id: str) -> AgentRun | None:
        async with self._sessions() as s:
            return await s.get(AgentRun, run_id)

    async def list_runs(self, status: str | None = None, limit: int = 50) -> list[AgentRun]:
        stmt = select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit)
        if status:
            stmt = stmt.where(AgentRun.status == status)
        async with self._sessions() as s:
            return list(await s.scalars(stmt))

    # --- outbox -----------------------------------------------------------------------------

    async def claim_delivery(
        self, key: str, run_id: str, message: EmailMessage, approved_by: str
    ) -> tuple[Claim, Outbox]:
        """Reserve ``key`` before sending.

        ``claimed``: first time, go ahead. ``already_delivered``: it was sent earlier, do
        nothing. ``retry``: an earlier attempt did not finish (crash or failure), try again.
        """
        row = {
            "idempotency_key": key,
            "run_id": run_id,
            "message": message.model_dump(),
            "approved_by": approved_by,
            "status": "pending",
        }
        async with self._sessions.begin() as s:
            inserted = await s.execute(
                insert(Outbox)
                .values(row)
                .on_conflict_do_nothing(index_elements=["idempotency_key"])
                .returning(Outbox.idempotency_key)
            )
            existing = await s.get(Outbox, key)
            assert existing is not None
            if inserted.first() is not None:
                return "claimed", existing
            return ("already_delivered" if existing.status == "delivered" else "retry"), existing

    async def mark_delivered(self, key: str, receipt: DeliveryReceipt) -> None:
        async with self._sessions.begin() as s:
            await s.execute(
                update(Outbox)
                .where(Outbox.idempotency_key == key)
                .values(
                    status="delivered",
                    outcome=receipt.outcome,
                    reference=receipt.reference,
                    error=None,
                    delivered_at=func.now(),
                )
            )

    async def mark_failed(self, key: str, error: str) -> None:
        async with self._sessions.begin() as s:
            await s.execute(
                update(Outbox)
                .where(Outbox.idempotency_key == key)
                .values(status="failed", error=error[:1000])
            )

    async def outbox_for_run(self, run_id: str) -> list[Outbox]:
        async with self._sessions() as s:
            rows = await s.scalars(
                select(Outbox).where(Outbox.run_id == run_id).order_by(Outbox.created_at)
            )
            return list(rows)

    # --- audit and feedback -----------------------------------------------------------------

    async def audit(
        self, run_id: str | None, actor: str, event: str, detail: dict[str, Any] | None = None
    ) -> None:
        async with self._sessions.begin() as s:
            s.add(AuditLog(run_id=run_id, actor=actor, event=event, detail=detail or {}))

    async def audit_for_run(self, run_id: str) -> list[AuditLog]:
        async with self._sessions() as s:
            rows = await s.scalars(
                select(AuditLog).where(AuditLog.run_id == run_id).order_by(AuditLog.id)
            )
            return list(rows)

    async def add_feedback(
        self,
        request_id: str,
        kind: str,
        rating: int,
        user: str,
        comment: str | None = None,
    ) -> None:
        async with self._sessions.begin() as s:
            s.add(
                Feedback(
                    request_id=request_id,
                    kind=kind,
                    rating=rating,
                    comment=comment,
                    created_by=user,
                )
            )
