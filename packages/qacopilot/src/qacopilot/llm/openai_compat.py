"""Provider for any OpenAI-compatible server (llama.cpp ``llama-server``, Ollama, vLLM...)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

from openai import AsyncOpenAI

from qacopilot.config import Settings, get_settings
from qacopilot.llm.base import BaseLLMProvider, LLMResponse, Message, ToolCall, Usage
from qacopilot.metrics import LLM_TOKENS, llm_call
from qacopilot.tracing import current_trace_id

logger = logging.getLogger(__name__)


def _client_class(settings: Settings) -> type[AsyncOpenAI]:
    """Return Langfuse's traced drop-in AsyncOpenAI when tracing is configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return AsyncOpenAI
    try:
        # The Langfuse SDK reads its config from the environment
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        if settings.langfuse_host:
            os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        from langfuse.openai import AsyncOpenAI as TracedAsyncOpenAI  # type: ignore[attr-defined]

        return TracedAsyncOpenAI
    except ImportError:
        logger.warning("Langfuse keys set but 'langfuse' is not installed; tracing disabled")
        return AsyncOpenAI


def _thinking_body(thinking: bool) -> dict[str, Any]:
    """Switch the model's reasoning on or off, in both places servers look for it.

    The community Nemotron GGUF behind our proxy reads ``enable_thinking`` at the TOP LEVEL of
    the request and ignores it nested under ``chat_template_kwargs`` (measured: 52 completion
    tokens against 538 for the same prompt). Servers built on llama.cpp's chat templates read the
    nested form, so both are sent.
    """
    return {"enable_thinking": thinking, "chat_template_kwargs": {"enable_thinking": thinking}}


def _timeout(settings: Settings) -> Any:
    """Long read timeout (reasoning can take minutes) but a short connect timeout."""
    import openai

    return openai.Timeout(settings.llm_timeout_s, connect=settings.llm_connect_timeout_s)


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        http_client: Any = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = _client_class(self._settings)(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            timeout=_timeout(self._settings),
            max_retries=1,  # the default of 2 triples the wait when the server is down
            http_client=http_client,
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if thinking is None:
            thinking = self._settings.llm_thinking_default

        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
            # llama.cpp reads this to toggle the model's reasoning mode
            "extra_body": _thinking_body(thinking),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = list(tools)
        if json_schema is not None:
            # llama.cpp compiles this into a grammar, so output is guaranteed to parse
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }

        if self._settings.langfuse_public_key:
            kwargs["name"] = "qacopilot.chat"  # trace name, consumed by the Langfuse wrapper
            if trace_id := current_trace_id(self._settings):
                kwargs["trace_id"] = trace_id  # join this request's trace (see tracing.py)
        with llm_call("chat"):
            completion = await self._client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message

        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (msg.tool_calls or [])
            if tc.type == "function"
        ]
        usage = completion.usage
        if usage:
            LLM_TOKENS.labels("prompt").inc(usage.prompt_tokens or 0)
            LLM_TOKENS.labels("completion").inc(usage.completion_tokens or 0)
        extra = getattr(msg, "model_extra", None) or {}
        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            reasoning=extra.get("reasoning_content"),
            finish_reason=completion.choices[0].finish_reason,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            ),
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        if thinking is None:
            thinking = self._settings.llm_thinking_default
        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": True,
            "extra_body": _thinking_body(thinking),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._settings.langfuse_public_key:
            kwargs["name"] = "qacopilot.stream"
            if trace_id := current_trace_id(self._settings):
                kwargs["trace_id"] = trace_id
        with llm_call("stream"):
            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                # reasoning arrives as delta.reasoning_content and is deliberately skipped
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def health(self) -> bool:
        try:
            models = await self._client.models.list()
        except Exception:
            return False
        return len(models.data) > 0
