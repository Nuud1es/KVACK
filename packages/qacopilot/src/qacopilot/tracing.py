"""Ties LLM calls and user feedback to one Langfuse trace per request.

A request id (an ``/ask`` request, or an agent run id) is set in a context variable; the LLM
provider then tags its calls with the trace id derived from it, and a later thumbs-up/down is
recorded as a score on that same trace. Everything here is a no-op unless both Langfuse keys are
configured.
"""

from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from qacopilot.config import Settings

request_id_var: ContextVar[str | None] = ContextVar("qacopilot_request_id", default=None)


def tracing_enabled(settings: Settings) -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def trace_id_for(request_id: str) -> str:
    """The Langfuse trace id for a request id (deterministic, so feedback can find the trace)."""
    from langfuse import Langfuse

    return str(Langfuse.create_trace_id(seed=request_id))


def current_trace_id(settings: Settings) -> str | None:
    request_id = request_id_var.get()
    if request_id and tracing_enabled(settings):
        return trace_id_for(request_id)
    return None


@lru_cache
def _client(public_key: str, secret_key: str, host: str) -> Any:
    from langfuse import Langfuse

    return Langfuse(public_key=public_key, secret_key=secret_key, host=host or None)


def send_feedback_score(
    settings: Settings, request_id: str, thumbs_up: bool, comment: str | None
) -> bool:
    """Record the rating on the request's trace. Returns False (never raises) if it cannot."""
    if not tracing_enabled(settings):
        return False
    try:
        client = _client(
            settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host
        )
        client.create_score(
            trace_id=trace_id_for(request_id),
            name="user-feedback",
            value=1 if thumbs_up else 0,
            data_type="BOOLEAN",
            comment=comment,
        )
        client.flush()
        return True
    except Exception:
        return False
