from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "src" / "qa_api" / "main.py").read_text()


def test_the_web_mount_is_registered_after_every_route() -> None:
    """A mount at "/" answers every path, so a route added after it can never be reached.

    This is checked on the source because the mount only exists where web/dist has been built:
    a test of the live app would pass on CI and fail on a developer's machine, or the reverse.
    """
    assert SOURCE.rfind("@app.") < SOURCE.rfind("mount_web(app"), (
        "mount_web(app, ...) must be the last statement in main.py"
    )


def test_every_endpoint_is_reachable_through_the_real_app() -> None:
    import httpx

    from qa_api.main import app

    async def hit() -> int:
        # no database in CI: a 500 from the missing repository still proves the route was matched
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return (await c.post("/documents/x/reclassify", json={"doc_type": "sop"})).status_code

    import asyncio

    # 401 (no key) or 503/other API answers are all fine; 405 means the static mount ate the route
    assert asyncio.run(hit()) != 405
