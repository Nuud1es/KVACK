"""Provider-agnostic LLM interface. Business logic depends on this, never on a vendor SDK."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

Message = dict[str, Any]

# extract() doubles its token budget after a reply is cut off, up to this many tokens.
MAX_EXTRACT_TOKENS = 16384


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = []
    reasoning: str | None = None
    finish_reason: str | None = None  # "length" = cut off by max_tokens
    usage: Usage = Usage()


class StructuredOutputError(Exception):
    """The model's output could not be validated against the requested schema."""


class BaseLLMProvider(ABC):
    @abstractmethod
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
        """Send a chat request. ``thinking=None`` means use the configured default."""

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield the reply text as it is generated (reasoning tokens are not included).

        The default waits for the whole reply; providers that can stream override it.
        """
        resp = await self.chat(
            messages, thinking=thinking, max_tokens=max_tokens, temperature=temperature
        )
        yield resp.content

    @abstractmethod
    async def health(self) -> bool:
        """True if the backing server is reachable and serving a model."""

    async def extract(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        retries: int = 2,
        max_tokens: int | None = 8192,
        thinking: bool = False,
    ) -> T:
        """Structured extraction validated with Pydantic.

        Works even when the server ignores ``response_format``: the schema is also put in
        the prompt and JSON is pulled leniently out of the reply (code fences, stray text).
        A reply cut off by ``max_tokens`` is retried with a doubled budget (up to
        ``MAX_EXTRACT_TOKENS``). Other retries feed the validation error back to the model,
        then raise :class:`StructuredOutputError`. ``thinking=True`` lets a reasoning model
        think first: slower, so only for judgement (grading, classifying). ``max_tokens`` is
        generous so a server that cannot switch reasoning off does not spend the whole budget
        thinking.
        """
        instruction = (
            "Reply with a single JSON object and nothing else (no prose, no code fences) "
            f"that conforms to this JSON Schema: {json.dumps(schema.model_json_schema())}"
        )
        convo: list[Message] = [{"role": "system", "content": instruction}, *messages]
        last_error: Exception | None = None
        budget = max_tokens
        for _ in range(retries + 1):
            resp = await self.chat(
                convo,
                json_schema=schema.model_json_schema(),
                thinking=thinking,
                max_tokens=budget,
            )
            if resp.finish_reason == "length" and budget is not None:
                # A reasoning model can spend the whole budget thinking and return nothing (or
                # half a JSON object). Repeating the request unchanged would do the same, so
                # allow more tokens instead of sending the model a misleading "invalid JSON".
                budget = min(budget * 2, MAX_EXTRACT_TOKENS)
                last_error = StructuredOutputError("reply was cut off by the token limit")
                continue
            try:
                return schema.model_validate_json(_extract_json(resp.content))
            except ValidationError as exc:
                last_error = exc
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                convo = [
                    *convo,
                    {"role": "assistant", "content": resp.content},
                    {
                        "role": "user",
                        "content": f"Invalid ({problems}). Return the corrected JSON object only.",
                    },
                ]
        raise StructuredOutputError(str(last_error))


def _extract_json(text: str) -> str:
    """Return the JSON object inside ``text``, tolerating code fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    first, last = text.find("{"), text.rfind("}")
    return text[first : last + 1] if 0 <= first < last else text
