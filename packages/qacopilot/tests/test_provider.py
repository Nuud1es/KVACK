import json

import httpx
import pytest
from pydantic import BaseModel

from qacopilot.config import Settings
from qacopilot.llm import OpenAICompatibleProvider, StructuredOutputError


class Cert(BaseModel):
    serial: str
    result: str


def _completion(
    content: str = "", tool_calls: list | None = None, finish_reason: str = "stop"
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _provider(replies: list[dict], seen: list[dict]) -> OpenAICompatibleProvider:
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider(Settings(llm_base_url="http://llm.test/v1"), http_client=client)


async def test_thinking_off_by_default_and_usage_parsed() -> None:
    seen: list[dict] = []
    p = _provider([_completion("hi")], seen)
    resp = await p.chat([{"role": "user", "content": "hello"}])
    assert resp.content == "hi"
    assert resp.usage.completion_tokens == 7
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["enable_thinking"] is False  # this server only reads the top-level flag


async def test_thinking_can_be_enabled_per_request() -> None:
    seen: list[dict] = []
    p = _provider([_completion("hi")], seen)
    await p.chat([{"role": "user", "content": "hello"}], thinking=True)
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen[0]["enable_thinking"] is True


async def test_tool_calls_are_parsed() -> None:
    seen: list[dict] = []
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Norwich"}'},
    }
    p = _provider([_completion(tool_calls=[call])], seen)
    resp = await p.chat([{"role": "user", "content": "weather?"}], tools=[{"type": "function"}])
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "Norwich"}


async def test_extract_retries_after_invalid_json() -> None:
    seen: list[dict] = []
    p = _provider(
        [_completion('{"serial": "A1"}'), _completion('{"serial": "A1", "result": "pass"}')],
        seen,
    )
    cert = await p.extract([{"role": "user", "content": "read cert"}], Cert)
    assert cert == Cert(serial="A1", result="pass")
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert len(seen) == 2  # one retry with the validation error fed back


async def test_extract_gives_up_after_retries() -> None:
    seen: list[dict] = []
    p = _provider([_completion("{}")] * 3, seen)
    with pytest.raises(StructuredOutputError):
        await p.extract([{"role": "user", "content": "x"}], Cert, retries=2)


async def test_extract_accepts_fenced_json_and_sends_schema_in_prompt() -> None:
    seen: list[dict] = []
    fenced = 'Sure!\n```json\n{"serial": "A1", "result": "pass"}\n```'
    p = _provider([_completion(fenced)], seen)
    cert = await p.extract([{"role": "user", "content": "read cert"}], Cert)
    assert cert == Cert(serial="A1", result="pass")
    assert seen[0]["messages"][0]["role"] == "system"
    assert "serial" in seen[0]["messages"][0]["content"]


def test_plain_openai_client_when_tracing_not_configured() -> None:
    import openai

    from qacopilot.llm.openai_compat import _client_class

    assert _client_class(Settings()) is openai.AsyncOpenAI


async def test_stream_yields_content_deltas_and_skips_reasoning() -> None:
    def sse(delta: dict) -> str:
        chunk = {
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }  # fmt: skip
        return f"data: {json.dumps(chunk)}\n\n"

    body = (
        sse({"role": "assistant", "reasoning_content": "thinking..."})
        + sse({"content": "Hel"})
        + sse({"content": "lo"})
        + "data: [DONE]\n\n"
    )
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    p = OpenAICompatibleProvider(Settings(llm_base_url="http://llm.test/v1"), http_client=client)
    parts = [t async for t in p.stream([{"role": "user", "content": "hi"}])]
    assert parts == ["Hel", "lo"]
    assert seen[0]["stream"] is True
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["enable_thinking"] is False  # this server only reads the top-level flag


async def test_extract_gives_a_cut_off_reply_more_tokens_instead_of_repeating_it() -> None:
    seen: list[dict] = []
    p = _provider(
        [
            _completion("", finish_reason="length"),  # all 100 tokens spent reasoning
            _completion('{"serial": "TP-1', finish_reason="length"),  # truncated mid-object
            _completion('{"serial": "TP-1", "result": "pass"}'),
        ],
        seen,
    )
    cert = await p.extract([{"role": "user", "content": "read"}], Cert, max_tokens=100)
    assert cert.serial == "TP-1"
    assert [r["max_tokens"] for r in seen] == [100, 200, 400]
    # no misleading "Invalid JSON" turn was added for the truncated attempts
    assert all(m["role"] != "assistant" for r in seen for m in r["messages"])


async def test_extract_budget_is_capped() -> None:
    seen: list[dict] = []
    p = _provider([_completion("", finish_reason="length")] * 3, seen)
    with pytest.raises(StructuredOutputError, match="token limit"):
        await p.extract([{"role": "user", "content": "x"}], Cert, max_tokens=10_000)
    assert [r["max_tokens"] for r in seen] == [10_000, 16_384, 16_384]


# --- metrics ----------------------------------------------------------------------------------


def _sample(name: str, **labels: str) -> float:
    pytest.importorskip("prometheus_client")  # the library works without it; these tests need it
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


def _failing_provider(handler) -> OpenAICompatibleProvider:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(llm_base_url="http://llm.test/v1")
    return OpenAICompatibleProvider(settings, http_client=client)


async def test_llm_calls_are_counted_timed_and_token_usage_is_recorded() -> None:
    ok = lambda: _sample("qacopilot_llm_requests_total", operation="chat", outcome="ok")  # noqa: E731
    tokens = lambda: _sample("qacopilot_llm_tokens_total", kind="completion")  # noqa: E731
    count = lambda: _sample("qacopilot_llm_request_seconds_count", operation="chat")  # noqa: E731
    o, t, c = ok(), tokens(), count()
    await _provider([_completion("hi")], []).chat([{"role": "user", "content": "hello"}])
    assert (ok(), tokens(), count()) == (o + 1, t + 7, c + 1)


async def test_an_unreachable_server_and_a_server_error_are_told_apart() -> None:
    def n(outcome: str) -> float:
        return _sample("qacopilot_llm_requests_total", operation="chat", outcome=outcome)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    unreachable, error = n("unreachable"), n("error")
    with pytest.raises(Exception):  # noqa: B017 - the SDK's own error types
        await _failing_provider(refuse).chat([{"role": "user", "content": "x"}])
    with pytest.raises(Exception):  # noqa: B017
        await _failing_provider(broken).chat([{"role": "user", "content": "x"}])
    assert (n("unreachable"), n("error")) == (unreachable + 1, error + 1)
