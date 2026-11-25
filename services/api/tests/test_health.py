import httpx
from qacopilot.llm import BaseLLMProvider

from qa_api.main import app, get_llm


class FakeLLM(BaseLLMProvider):
    def __init__(self, ok: bool) -> None:
        self._ok = ok

    async def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def health(self) -> bool:
        return self._ok


async def _get(ok: bool) -> httpx.Response:
    app.dependency_overrides[get_llm] = lambda: FakeLLM(ok)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.get("/health")
    finally:
        app.dependency_overrides.clear()


async def test_health_ok() -> None:
    r = await _get(True)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_health_degraded_when_llm_down() -> None:
    r = await _get(False)
    assert r.json() == {"status": "degraded", "version": "0.1.0", "llm_reachable": False}
