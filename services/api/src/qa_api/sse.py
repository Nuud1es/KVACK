"""Server-sent events helpers."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel

logger = logging.getLogger(__name__)

LLM_UNREACHABLE = (
    "The language model server could not be reached. Check that it is running and that this "
    "machine can connect to it, then try again."
)


def failure_detail(exc: BaseException) -> str:
    """What to tell the person: a specific message when the LLM server is unreachable."""
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency of the library
        return "The request failed. Please try again."
    if isinstance(exc, openai.APIConnectionError):  # includes timeouts
        return LLM_UNREACHABLE
    return "The request failed. Please try again."


PING = ": ping\n\n"


def encode(event: str, data: BaseModel | dict[str, object]) -> str:
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_events(
    events: AsyncIterator[BaseModel], *, keepalive_s: float = 10.0
) -> AsyncIterator[str]:
    """Encode events as SSE, sending a comment ping whenever the source is quiet.

    The model can reason for a long time before its first token; without pings, proxies and
    browsers may drop an idle connection. Errors become a final ``error`` event.
    """
    iterator = events.__aiter__()
    pending: asyncio.Future[BaseModel] = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
            if not done:
                yield PING
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                return
            yield encode(str(getattr(item, "event", "message")), item)
            pending = asyncio.ensure_future(iterator.__anext__())
    except Exception as exc:
        logger.exception("stream failed")
        yield encode("error", {"event": "error", "detail": failure_detail(exc)})
    finally:
        pending.cancel()
        aclose = getattr(iterator, "aclose", None)
        if aclose:
            await aclose()
