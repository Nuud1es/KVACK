"""Decide what kind of request this is. Deterministic on purpose: routing picks which code runs
(and whether a human must approve), so it should be predictable and testable. The ``Router``
protocol lets an LLM router replace it later without touching the graph."""

import re
from typing import Literal, Protocol

from qacopilot.timewindow import (  # noqa: F401  (re-exported for callers)
    DEFAULT_WINDOW_DAYS,
    MAX_WINDOW_DAYS,
    parse_window_days,
    requested_window_days,
    window_cap_note,
)

Route = Literal["question", "action"]

# An action is a request to *do* something (draft/send/remind), not to be told something.
_ACTION = re.compile(
    r"\b(draft|write|compose|prepare|send|email|e-mail|remind(er|ers)?|notify|chase|alert)\b",
    re.IGNORECASE,
)


class Router(Protocol):
    def route(self, query: str) -> Route: ...


class RuleRouter:
    def route(self, query: str) -> Route:
        return "action" if _ACTION.search(query) else "question"
