"""The agent: a LangGraph graph with a mandatory human approval before any action.

    START -> route --question--> answer ---------------------------------> END
                  \\--action--> plan -> gather -> draft --nothing/blocked--> END
                                                    |
                                              request_approval  (interrupt: pauses for hours)
                                                |           |
                                            approve       reject -> END
                                                |
                                             execute -> END

Rules this file enforces:
* Nothing is sent without a decision from a person (``execute`` is only reachable from approval).
* ``request_approval`` has no side effects before ``interrupt()``: LangGraph re-runs a node from
  its first line when it resumes.
* Delivery is idempotent per (run, draft) through the outbox, so a retried approval cannot send
  the same email twice.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from qacopilot.agent.drafts import ReminderDraft, build_reminder
from qacopilot.agent.router import Router, RuleRouter
from qacopilot.llm import BaseLLMProvider
from qacopilot.mail import Mailer
from qacopilot.rag import NoAnswerEvent, SourcesEvent, TokenEvent, answer_question
from qacopilot.repositories import AgentRepository, DocumentRepository
from qacopilot.retrieval import Retriever
from qacopilot.timewindow import parse_window_days, window_cap_note

Status = Literal[
    "completed",  # a question was answered, or every approved email was delivered
    "awaiting_approval",
    "rejected",
    "no_action",  # nothing was due
    "blocked",  # cannot proceed, e.g. no recipients configured
    "delivery_failed",  # approved, but at least one email could not be delivered
]


class AgentState(TypedDict, total=False):
    run_id: str
    user: str
    query: str
    route: str
    window_days: int
    window_note: str
    answer: str
    answered: bool
    citations: list[dict[str, Any]]
    certificates: list[dict[str, Any]]
    drafts: list[dict[str, Any]]
    decision: dict[str, Any]
    deliveries: list[dict[str, Any]]
    status: Status
    message: str


@dataclass
class AgentDeps:
    llm: BaseLLMProvider
    retriever: Retriever
    documents: DocumentRepository
    agent_repo: AgentRepository
    mailer: Mailer
    recipients: list[str] = field(default_factory=list)
    router: Router = field(default_factory=RuleRouter)


def delivery_key(run_id: str, index: int) -> str:
    return f"{run_id}-{index:02d}"


async def deliver_all(
    deps: AgentDeps, run_id: str, drafts: list[dict[str, Any]], approver: str
) -> list[dict[str, Any]]:
    """Deliver each approved draft once. Safe to call again: delivered drafts are skipped."""
    from qacopilot.mail import EmailMessage

    results: list[dict[str, Any]] = []
    for i, raw in enumerate(drafts):
        key = delivery_key(run_id, i)
        message = EmailMessage(**raw["message"])
        claim, row = await deps.agent_repo.claim_delivery(key, run_id, message, approver)
        if claim == "already_delivered":
            results.append(
                {
                    "key": key,
                    "status": "delivered",
                    "outcome": row.outcome,
                    "reference": row.reference,
                }
            )
            continue
        try:
            receipt = await deps.mailer.deliver(message, key=key)
        except Exception as exc:
            await deps.agent_repo.mark_failed(key, f"{type(exc).__name__}: {exc}")
            await deps.agent_repo.audit(run_id, approver, "email_failed", {"key": key})
            results.append({"key": key, "status": "failed", "error": f"{type(exc).__name__}"})
            continue
        await deps.agent_repo.mark_delivered(key, receipt)
        await deps.agent_repo.audit(
            run_id,
            approver,
            "email_delivered",
            {"key": key, "outcome": receipt.outcome, "to": message.to, "subject": message.subject},
        )
        results.append(
            {
                "key": key,
                "status": "delivered",
                "outcome": receipt.outcome,
                "reference": receipt.reference,
            }
        )
    return results


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any]) -> Any:
    async def route(state: AgentState) -> AgentState:
        return {"route": deps.router.route(state["query"])}

    async def answer(state: AgentState) -> AgentState:
        text, citations, found = [], [], False
        async for event in answer_question(state["query"], deps.retriever, deps.llm):
            if isinstance(event, SourcesEvent):
                citations = [c.model_dump() for c in event.citations]
                found = True
            elif isinstance(event, TokenEvent):
                text.append(event.text)
            elif isinstance(event, NoAnswerEvent):
                found = False
        answered = found and bool("".join(text).strip())
        return {
            "answer": "".join(text).strip(),
            "answered": answered,
            "citations": citations if answered else [],
            "status": "completed",
            "message": "" if answered else "This is not covered by the indexed documents.",
        }

    async def plan(state: AgentState) -> AgentState:
        return {
            "window_days": parse_window_days(state["query"]),
            "window_note": window_cap_note(state["query"]),
        }

    async def gather(state: AgentState) -> AgentState:
        due = await deps.documents.certificates_due(state["window_days"])
        return {"certificates": [c.model_dump(mode="json") for c in due]}

    async def draft(state: AgentState) -> AgentState:
        if not state["certificates"]:
            return {
                "status": "no_action",
                "drafts": [],
                "message": " ".join(
                    part
                    for part in (
                        f"No calibration certificates are due within {state['window_days']} days.",
                        state.get("window_note", ""),
                    )
                    if part
                ),
            }
        if not deps.recipients:
            return {
                "status": "blocked",
                "drafts": [],
                "message": "No reminder recipients are configured (QACOPILOT_REMINDER_RECIPIENTS).",
            }
        from qacopilot.repositories import CertificateStatus

        certs = [CertificateStatus(**c) for c in state["certificates"]]
        reminder: ReminderDraft = await build_reminder(
            certs, deps.recipients, deps.llm, state["window_days"]
        )
        await deps.agent_repo.audit(
            state["run_id"],
            "system",
            "drafts_created",
            {"emails": 1, "certificates": reminder.certificate_count, "to": deps.recipients},
        )
        return {
            "status": "awaiting_approval",
            "drafts": [reminder.model_dump(mode="json")],
            "message": " ".join(
                part
                for part in (
                    f"{reminder.certificate_count} certificates found; 1 email drafted.",
                    state.get("window_note", ""),
                )
                if part
            ),
        }

    def after_draft(state: AgentState) -> str:
        return "request_approval" if state.get("drafts") else END

    async def request_approval(state: AgentState) -> AgentState:
        # Nothing before interrupt() may have side effects: this node re-runs on resume.
        decision = interrupt({"kind": "approve_emails", "drafts": state["drafts"]})
        return {"decision": decision}

    def after_approval(state: AgentState) -> str:
        return "execute" if state["decision"].get("decision") == "approve" else "rejected"

    async def rejected(state: AgentState) -> AgentState:
        who = state["decision"].get("user", "unknown")
        await deps.agent_repo.audit(
            state["run_id"], who, "rejected", {"reason": state["decision"].get("reason")}
        )
        return {"status": "rejected", "message": f"Rejected by {who}. Nothing was sent."}

    async def execute(state: AgentState) -> AgentState:
        who = state["decision"]["user"]
        await deps.agent_repo.audit(
            state["run_id"], who, "approved", {"emails": len(state["drafts"])}
        )
        results = await deliver_all(deps, state["run_id"], state["drafts"], who)
        failed = [r for r in results if r["status"] != "delivered"]
        return {
            "deliveries": results,
            "status": "delivery_failed" if failed else "completed",
            "message": (
                f"{len(failed)} of {len(results)} emails could not be delivered."
                if failed
                else f"{len(results)} email(s) delivered after approval by {who}."
            ),
        }

    g = StateGraph(AgentState)
    for name, fn in [
        ("route", route),
        ("answer", answer),
        ("plan", plan),
        ("gather", gather),
        ("draft", draft),
        ("request_approval", request_approval),
        ("rejected", rejected),
        ("execute", execute),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["route"], {"question": "answer", "action": "plan"})
    g.add_edge("answer", END)
    g.add_edge("plan", "gather")
    g.add_edge("gather", "draft")
    g.add_conditional_edges(
        "draft", after_draft, {"request_approval": "request_approval", END: END}
    )
    g.add_conditional_edges(
        "request_approval", after_approval, {"execute": "execute", "rejected": "rejected"}
    )
    g.add_edge("rejected", END)
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer)
