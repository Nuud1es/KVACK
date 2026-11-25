"""Prometheus metrics for the library. Optional: without ``prometheus-client`` every call here is
a no-op, so the library stays usable (and its tests pass) without it installed.

Labels are kept to a handful of fixed values on purpose (an operation, an outcome, a path): a
label that takes a document id or a user's question would create a new series per request.
"""

import asyncio
import time
from types import TracebackType
from typing import Any

try:
    from prometheus_client import Counter, Histogram

    ENABLED = True
except ImportError:  # pragma: no cover - exercised only where the package is absent
    ENABLED = False


class _Noop:
    """Stands in for a metric when prometheus-client is not installed."""

    def labels(self, *_: Any, **__: Any) -> "_Noop":
        return self

    def inc(self, *_: Any, **__: Any) -> None: ...

    def observe(self, *_: Any, **__: Any) -> None: ...

    def set(self, *_: Any, **__: Any) -> None: ...


def counter(name: str, doc: str, labels: list[str]) -> Any:
    return Counter(name, doc, labels) if ENABLED else _Noop()


def histogram(name: str, doc: str, labels: list[str], buckets: tuple[float, ...]) -> Any:
    return Histogram(name, doc, labels, buckets=buckets) if ENABLED else _Noop()


# A reasoning model can take minutes, a refused connection fails in seconds: wide buckets.
LLM_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)

LLM_REQUESTS = counter(
    "qacopilot_llm_requests_total",
    "Calls to the LLM server by operation and outcome (ok, error, unreachable, cancelled)",
    ["operation", "outcome"],
)
LLM_SECONDS = histogram(
    "qacopilot_llm_request_seconds",
    "Time spent in LLM calls, by operation",
    ["operation"],
    LLM_BUCKETS,
)
LLM_TOKENS = counter(
    "qacopilot_llm_tokens_total",
    "Tokens the LLM reported using (kind: prompt or completion)",
    ["kind"],
)


def outcome_of(exc_type: type[BaseException] | None, exc: BaseException | None) -> str:
    """ok / cancelled / unreachable / error for the exception (if any) that ended a call."""
    if exc_type is None:
        return "ok"
    if issubclass(exc_type, (GeneratorExit, asyncio.CancelledError)):
        return "cancelled"  # the caller went away: not a fault of the LLM
    try:
        import openai

        if isinstance(exc, openai.APIConnectionError):  # includes timeouts
            return "unreachable"
    except ImportError:  # pragma: no cover
        pass
    return "error"


class llm_call:  # noqa: N801 - used like a context manager: ``with llm_call("chat"):``
    """Times one LLM call and counts it by outcome."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self._start = 0.0

    def __enter__(self) -> "llm_call":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        LLM_SECONDS.labels(self.operation).observe(time.perf_counter() - self._start)
        LLM_REQUESTS.labels(self.operation, outcome_of(exc_type, exc)).inc()
