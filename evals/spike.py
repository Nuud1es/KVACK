"""Phase 1 feasibility spike: does the LLM server support what KVACK needs?

    python evals/spike.py [--base-url http://<linux-ip>:8080/v1] [--model nemotron]

Checks: reachability, tool calling, JSON-schema extraction, thinking-off output.
Add real cert text to evals/spike_docs/*.txt to measure extraction accuracy too.
"""

import argparse
import asyncio
import time
from datetime import date

from pydantic import BaseModel, Field
from qacopilot.config import Settings
from qacopilot.llm import OpenAICompatibleProvider


class CalibrationCert(BaseModel):
    instrument_serial: str
    cert_date: date = Field(description="Calibration date as ISO 8601 YYYY-MM-DD, e.g. 2026-03-14")
    next_due: date = Field(description="Next due date as ISO 8601 YYYY-MM-DD, e.g. 2027-03-14")
    result: str


SAMPLE = """CERTIFICATE OF CALIBRATION
Instrument: Digital probe thermometer   Serial No: TP-20931
Calibrated on 14 March 2026. Recommended recalibration date: 14 March 2027.
All points within tolerance. Overall result: PASS."""

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


async def timed(label: str, coro):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    try:
        result = await coro
        print(f"[ OK ] {label} ({time.perf_counter() - start:.1f}s)")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    overrides = {
        k: v
        for k, v in {
            "llm_base_url": args.base_url,
            "llm_model": args.model,
            "llm_api_key": args.api_key,
        }.items()
        if v
    }
    llm = OpenAICompatibleProvider(Settings(**overrides))

    print("Reachable:", await llm.health())

    resp = await timed(
        "tool calling",
        llm.chat(
            [{"role": "user", "content": "What is the weather in Norwich?"}],
            tools=[WEATHER_TOOL],
        ),
    )
    if resp:
        ok = bool(resp.tool_calls) and resp.tool_calls[0].arguments.get("city")
        print(
            "       tool_calls:",
            resp.tool_calls or f"NONE (content={resp.content[:80]!r})",
        )
        print("       ->", "PASS" if ok else "FAIL: tool call not returned in tool_calls")

    cert = await timed(
        "schema extraction",
        llm.extract(
            [
                {
                    "role": "system",
                    "content": "Extract the calibration certificate fields.",
                },
                {"role": "user", "content": SAMPLE},
            ],
            CalibrationCert,
        ),
    )
    if cert:
        expected = ("TP-20931", date(2026, 3, 14), date(2027, 3, 14))
        got = (cert.instrument_serial, cert.cert_date, cert.next_due)
        print("       got:", cert)
        print("       ->", "PASS" if got == expected else f"MISMATCH, expected {expected}")

    resp = await timed(
        "thinking off",
        llm.chat(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            thinking=False,
        ),
    )
    if resp:
        print(
            "       reasoning present:",
            bool(resp.reasoning),
            "| content:",
            repr(resp.content[:80]),
        )


if __name__ == "__main__":
    asyncio.run(main())
