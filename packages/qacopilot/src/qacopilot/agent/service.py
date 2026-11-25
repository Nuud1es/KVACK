"""Run lifecycle around the graph: start, inspect, decide, retry. Used by the API."""

import uuid
from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel

from qacopilot.agent.graph import AgentDeps, build_graph, deliver_all
from qacopilot.repositories.models import AgentRun
from qacopilot.tracing import request_id_var


class RunNotFound(Exception):
    pass


class RunConflict(Exception):
    """The run is not in a state that allows this action (e.g. already decided)."""


class RunView(BaseModel):
    run_id: str
    created_by: str
    query: str
    status: str
    route: str | None = None
    message: str = ""
    answer: str | None = None
    answered: bool | None = None
    citations: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    drafts: list[dict[str, Any]] = []
    deliveries: list[dict[str, Any]] = []
    decided_by: str | None = None


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


class AgentService:
    def __init__(self, deps: AgentDeps, checkpointer: Any) -> None:
        self._deps = deps
        self._graph = build_graph(deps, checkpointer)

    async def _view(self, run: AgentRun) -> RunView:
        snap = await self._graph.aget_state(_config(run.run_id))
        v: dict[str, Any] = snap.values or {}
        return RunView(
            run_id=run.run_id,
            created_by=run.created_by,
            query=run.query,
            status=run.status,
            route=v.get("route") or run.route,
            message=v.get("message", ""),
            answer=v.get("answer"),
            answered=v.get("answered"),
            citations=v.get("citations", []),
            certificates=v.get("certificates", []),
            drafts=v.get("drafts", []),
            deliveries=v.get("deliveries", []),
            decided_by=(v.get("decision") or {}).get("user"),
        )

    async def _settle(self, run_id: str) -> AgentRun:
        """Copy the graph's outcome into the runs table so lists and queues can filter on it."""
        snap = await self._graph.aget_state(_config(run_id))
        v: dict[str, Any] = snap.values or {}
        paused = bool(snap.next)  # stopped at the approval interrupt
        status = "awaiting_approval" if paused else str(v.get("status", "failed"))
        await self._deps.agent_repo.update_run(
            run_id,
            status=status,
            route=v.get("route"),
            summary={"message": v.get("message", ""), "emails": len(v.get("drafts", []))},
        )
        run = await self._deps.agent_repo.get_run(run_id)
        assert run is not None
        return run

    async def start(self, user: str, query: str) -> RunView:
        run_id = uuid.uuid4().hex
        request_id_var.set(run_id)  # the LLM calls in this run share one Langfuse trace
        await self._deps.agent_repo.create_run(run_id, user, query)
        await self._deps.agent_repo.audit(run_id, user, "run_started", {"query": query})
        try:
            await self._graph.ainvoke(
                {"run_id": run_id, "user": user, "query": query}, _config(run_id)
            )
        except Exception:
            await self._deps.agent_repo.update_run(run_id, status="failed")
            await self._deps.agent_repo.audit(run_id, "system", "run_failed", {})
            raise
        return await self._view(await self._settle(run_id))

    async def get(self, run_id: str) -> RunView:
        run = await self._deps.agent_repo.get_run(run_id)
        if run is None:
            raise RunNotFound(run_id)
        return await self._view(run)

    async def decide(
        self,
        run_id: str,
        user: str,
        decision: Literal["approve", "reject"],
        reason: str | None = None,
    ) -> RunView:
        """Approve or reject a paused run. Exactly one caller can win; others get RunConflict."""
        if await self._deps.agent_repo.get_run(run_id) is None:
            raise RunNotFound(run_id)
        if not await self._deps.agent_repo.transition(
            run_id, expect="awaiting_approval", to="running"
        ):
            raise RunConflict("This run is not waiting for approval")
        payload = {"decision": decision, "user": user, "reason": reason}
        try:
            await self._graph.ainvoke(Command(resume=payload), _config(run_id))
        except Exception:
            # The checkpoint still holds the pause, so the decision can be made again.
            await self._deps.agent_repo.update_run(run_id, status="awaiting_approval")
            await self._deps.agent_repo.audit(run_id, "system", "decision_failed", {"by": user})
            raise
        return await self._view(await self._settle(run_id))

    async def retry_delivery(self, run_id: str, user: str) -> RunView:
        """Re-attempt emails that failed after approval. Delivered ones are skipped."""
        run = await self._deps.agent_repo.get_run(run_id)
        if run is None:
            raise RunNotFound(run_id)
        if not await self._deps.agent_repo.transition(
            run_id, expect="delivery_failed", to="running"
        ):
            raise RunConflict("This run has no failed deliveries to retry")
        snap = await self._graph.aget_state(_config(run_id))
        drafts = (snap.values or {}).get("drafts", [])
        approver = ((snap.values or {}).get("decision") or {}).get("user", user)
        await self._deps.agent_repo.audit(run_id, user, "delivery_retried", {})
        results = await deliver_all(self._deps, run_id, drafts, approver)
        failed = any(r["status"] != "delivered" for r in results)
        status = "delivery_failed" if failed else "completed"
        await self._graph.aupdate_state(_config(run_id), {"deliveries": results, "status": status})
        await self._deps.agent_repo.update_run(run_id, status=status)
        return await self.get(run_id)
