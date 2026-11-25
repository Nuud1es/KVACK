"""Probe how the LLM server handles thinking flags and JSON-schema enforcement.

    python evals/probe.py

Sends 5 tiny requests to QACOPILOT_LLM_BASE_URL (from .env). Never prints the API key.
"""

import asyncio

import httpx
from qacopilot.config import Settings

s = Settings()
HEADERS = {"Authorization": f"Bearer {s.llm_api_key}"}
PROMPT = [{"role": "user", "content": "Reply with the single word: ready"}]


async def post(c: httpx.AsyncClient, body: dict) -> dict:
    r = await c.post(s.llm_base_url + "/chat/completions", headers=HEADERS, json=body)
    return r.json()


def show(label: str, data: dict) -> None:
    msg = data["choices"][0]["message"]
    keys = ("content", "reasoning_content", "reasoning")
    print(f"{label:28} ->", {k: str(msg[k])[:70] for k in keys if msg.get(k)})


async def main() -> None:
    base = {"model": s.llm_model, "temperature": 0}
    async with httpx.AsyncClient(timeout=120) as c:
        models = (await c.get(s.llm_base_url + "/models", headers=HEADERS)).json()
        print("models:", [m.get("id") for m in models.get("data", [])])

        show("no flag", await post(c, {**base, "messages": PROMPT}))
        show(
            "enable_thinking=false, nested in chat_template_kwargs",
            await post(
                c,
                {
                    **base,
                    "messages": PROMPT,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ),
        )
        show(
            "enable_thinking=false, top level",
            await post(c, {**base, "messages": PROMPT, "enable_thinking": False}),
        )
        show(
            "/no_think system prompt",
            await post(
                c,
                {
                    **base,
                    "messages": [{"role": "system", "content": "/no_think"}, *PROMPT],
                },
            ),
        )

        schema = {
            "type": "object",
            "properties": {"d": {"type": "string", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}},
            "required": ["d"],
            "additionalProperties": False,
        }
        body = {
            **base,
            "messages": [
                {
                    "role": "user",
                    "content": "Calibrated on 14 March 2026. Give the date.",
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "o", "schema": schema, "strict": True},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        show("schema with date pattern", await post(c, body))
        print('(a result like {"d": "2026-03-14"} means the schema IS enforced)')


asyncio.run(main())
