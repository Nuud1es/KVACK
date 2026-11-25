from collections.abc import AsyncIterator

import httpx
import pytest
from qacopilot.config import Settings
from qacopilot.repositories import AgentRepository, UserRepository
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_engine

import qa_api.main as main
from qa_api.main import (
    app,
    get_agent_repository,
    get_repository,
    get_settings,
    get_user_repository_or_none,
)
from qa_api.throttle import Throttle


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    try:
        async with temporary_engine() as engine:
            users = UserRepository(engine)
            # a database_url is needed only so that accounts count as available
            app.dependency_overrides[get_settings] = lambda: Settings(
                accounts=True,
                database_url="x",
                api_keys="alice:key-alice",
                admins="Boss Person",
            )
            app.dependency_overrides[get_user_repository_or_none] = lambda: users
            app.dependency_overrides[get_agent_repository] = lambda: AgentRepository(engine)
            app.state.users = users
            main._register_throttle = Throttle(limit=10, window_s=3600)
            main._login_throttle = Throttle(limit=30, window_s=600)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                yield c
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))
    finally:
        app.dependency_overrides.clear()


def register(client: httpx.AsyncClient, name: str = "Joe Rudling", pin: str = "1234"):  # type: ignore[no-untyped-def]
    return client.post("/auth/register", json={"name": name, "pin": pin})


async def test_register_signs_in_and_the_token_works_like_a_key(client) -> None:  # type: ignore[no-untyped-def]
    r = await register(client)
    assert r.status_code == 200 and r.json()["user"] == "Joe Rudling"
    token = r.json()["token"]
    me = await client.get("/me", headers={"x-api-key": token})
    assert me.json() == {"user": "Joe Rudling", "admin": False}


async def test_requests_need_a_sign_in_once_accounts_are_on(client) -> None:  # type: ignore[no-untyped-def]
    assert (await client.get("/me")).status_code == 401
    assert (await client.get("/me", headers={"x-api-key": "nonsense"})).status_code == 401
    assert (await client.get("/documents")).status_code == 401
    # a configured API key still works beside accounts
    assert (await client.get("/me", headers={"x-api-key": "key-alice"})).json() == {
        "user": "alice",
        "admin": False,
    }


async def test_bad_names_and_pins_are_refused(client) -> None:  # type: ignore[no-untyped-def]
    for name, pin in [
        ("Joe", "123"),
        ("Joe", "12345"),
        ("Joe", "abcd"),
        ("J", "1234"),
        ("<b>", "1234"),
    ]:
        assert (await register(client, name, pin)).status_code == 422
    for reserved in ["admin", "API", "alice"]:  # alice owns a configured API key
        assert (await register(client, reserved)).status_code == 409


async def test_a_name_can_only_be_registered_once(client) -> None:  # type: ignore[no-untyped-def]
    assert (await register(client, "Joe Rudling", "1234")).status_code == 200
    again = await register(client, "joe rudling", "9999")
    assert again.status_code == 409


async def test_login_wrong_pin_then_lockout(client) -> None:  # type: ignore[no-untyped-def]
    await register(client)
    good = {"name": "joe rudling", "pin": "1234"}
    assert (await client.post("/auth/login", json=good)).status_code == 200
    bad = {"name": "Joe Rudling", "pin": "0000"}
    for _ in range(4):
        assert (await client.post("/auth/login", json=bad)).status_code == 401
    assert (await client.post("/auth/login", json=bad)).status_code == 429
    assert (
        await client.post("/auth/login", json=good)
    ).status_code == 429  # locked for the right PIN too


async def test_unknown_name_and_wrong_pin_look_the_same(client) -> None:  # type: ignore[no-untyped-def]
    await register(client)
    a = await client.post("/auth/login", json={"name": "Nobody", "pin": "1234"})
    b = await client.post("/auth/login", json={"name": "Joe Rudling", "pin": "0000"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


async def test_one_address_cannot_try_endlessly(client) -> None:  # type: ignore[no-untyped-def]
    main._login_throttle = Throttle(limit=3, window_s=600)
    for i in range(3):
        await client.post("/auth/login", json={"name": f"Nobody{i}", "pin": "1234"})
    blocked = await client.post("/auth/login", json={"name": "Anyone", "pin": "1234"})
    assert blocked.status_code == 429


async def test_signups_are_limited_per_address(client) -> None:  # type: ignore[no-untyped-def]
    main._register_throttle = Throttle(limit=2, window_s=3600)
    assert (await register(client, "Person One")).status_code == 200
    assert (await register(client, "Person Two")).status_code == 200
    assert (await register(client, "Person Three")).status_code == 429


async def test_logout_ends_the_session(client) -> None:  # type: ignore[no-untyped-def]
    token = (await register(client)).json()["token"]
    headers = {"x-api-key": token}
    assert (await client.post("/auth/logout", headers=headers)).status_code == 204
    assert (await client.get("/me", headers=headers)).status_code == 401


async def test_accounts_off_means_the_endpoints_do_not_exist() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(accounts=False)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            assert (
                await c.post("/auth/register", json={"name": "Joe", "pin": "1234"})
            ).status_code == 404
    finally:
        app.dependency_overrides.clear()


async def _admin(client: httpx.AsyncClient) -> dict[str, str]:
    """An administrator's session: their account is made outside the web form."""
    await app.state.users.register("Boss Person", "7777")
    token = (await client.post("/auth/login", json={"name": "Boss Person", "pin": "7777"})).json()[
        "token"
    ]
    return {"x-api-key": token}


async def test_admin_names_cannot_be_claimed_through_the_web_form(client) -> None:  # type: ignore[no-untyped-def]
    assert (await register(client, "boss person")).status_code == 409


async def test_only_an_administrator_may_manage_accounts(client) -> None:  # type: ignore[no-untyped-def]
    joe = {"x-api-key": (await register(client)).json()["token"]}
    assert (await client.get("/admin/users", headers=joe)).status_code == 403
    assert (await client.get("/admin/users")).status_code == 401
    assert (await client.get("/me", headers=joe)).json()["admin"] is False
    boss = await _admin(client)
    assert (await client.get("/me", headers=boss)).json()["admin"] is True
    users = (await client.get("/admin/users", headers=boss)).json()
    assert [u["name"] for u in users] == ["Boss Person", "Joe Rudling"]
    assert "pin_hash" not in users[0] and users[0]["admin"] is True


async def test_an_admin_can_reset_a_pin_and_it_signs_the_person_out(client) -> None:  # type: ignore[no-untyped-def]
    joe = (await register(client)).json()
    boss = await _admin(client)
    joe_id = next(
        u["id"]
        for u in (await client.get("/admin/users", headers=boss)).json()
        if u["name"] == "Joe Rudling"
    )
    assert (
        await client.post(f"/admin/users/{joe_id}/pin", json={"pin": "12"}, headers=boss)
    ).status_code == 422
    done = await client.post(f"/admin/users/{joe_id}/pin", json={"pin": "5555"}, headers=boss)
    assert done.status_code == 204
    assert (await client.get("/me", headers={"x-api-key": joe["token"]})).status_code == 401
    old = await client.post("/auth/login", json={"name": "Joe Rudling", "pin": "1234"})
    assert old.status_code == 401
    new = await client.post("/auth/login", json={"name": "Joe Rudling", "pin": "5555"})
    assert new.status_code == 200


async def test_an_admin_can_unlock_a_locked_account(client) -> None:  # type: ignore[no-untyped-def]
    await register(client)
    for _ in range(5):
        await client.post("/auth/login", json={"name": "Joe Rudling", "pin": "0000"})
    good = {"name": "Joe Rudling", "pin": "1234"}
    assert (await client.post("/auth/login", json=good)).status_code == 429
    boss = await _admin(client)
    users = (await client.get("/admin/users", headers=boss)).json()
    joe = next(u for u in users if u["name"] == "Joe Rudling")
    assert joe["locked"] is True
    assert (await client.post(f"/admin/users/{joe['id']}/unlock", headers=boss)).status_code == 204
    assert (await client.post("/auth/login", json=good)).status_code == 200


async def test_delete_removes_the_account_but_not_your_own(client) -> None:  # type: ignore[no-untyped-def]
    await register(client)
    boss = await _admin(client)
    users = (await client.get("/admin/users", headers=boss)).json()
    ids = {u["name"]: u["id"] for u in users}
    own = await client.delete(f"/admin/users/{ids['Boss Person']}", headers=boss)
    assert own.status_code == 409
    assert (
        await client.delete(f"/admin/users/{ids['Joe Rudling']}", headers=boss)
    ).status_code == 204
    gone = await client.post("/auth/login", json={"name": "Joe Rudling", "pin": "1234"})
    assert gone.status_code == 401
    assert (await client.delete("/admin/users/99999", headers=boss)).status_code == 404


async def test_admin_actions_are_written_to_the_audit_log(client) -> None:  # type: ignore[no-untyped-def]
    await register(client)
    boss = await _admin(client)
    joe = next(
        u
        for u in (await client.get("/admin/users", headers=boss)).json()
        if u["name"] == "Joe Rudling"
    )
    await client.post(f"/admin/users/{joe['id']}/sign-out", headers=boss)
    await client.post(f"/admin/users/{joe['id']}/unlock", headers=boss)
    from qacopilot.repositories.models import AuditLog
    from sqlalchemy import select

    async with app.state.users._sessions() as s:
        rows = (await s.scalars(select(AuditLog).order_by(AuditLog.id))).all()
    assert [(r.actor, r.event, r.detail) for r in rows] == [
        ("Boss Person", "admin.sign_out", {"account": "Joe Rudling"}),
        ("Boss Person", "admin.unlock", {"account": "Joe Rudling"}),
    ]


class _Names:
    """The few repository methods the name review uses, kept in memory."""

    def __init__(self) -> None:
        self.aliases: dict[tuple[str, str], str] = {}
        self.separate: set[tuple[str, str]] = set()

    async def name_counts(self, kind: str) -> dict[str, int]:
        return {"customer": {"Sainsbury's": 30, "Sainsbury": 12, "Tesco": 40}}.get(kind, {})

    async def list_aliases(self) -> list[tuple[str, str, str]]:
        return [(k, raw, c) for (k, raw), c in self.aliases.items()]

    async def decided_names(self) -> set[tuple[str, str]]:
        return self.separate

    async def set_alias(
        self, kind: str, raw: str, canonical: str, *, by: str | None = None
    ) -> None:
        self.aliases[(kind, raw.lower())] = canonical

    async def keep_separate(self, kind: str, name: str, *, by: str | None = None) -> None:
        self.separate.add((kind, name.lower()))


async def test_an_admin_reviews_names_and_makes_an_alias_or_keeps_one_separate(client) -> None:  # type: ignore[no-untyped-def]
    names = _Names()
    app.dependency_overrides[get_repository] = lambda: names
    joe = {"x-api-key": (await register(client)).json()["token"]}
    assert (await client.get("/admin/names", headers=joe)).status_code == 403
    boss = await _admin(client)
    (item,) = (await client.get("/admin/names", headers=boss)).json()
    assert (item["name"], item["recommendation"], item["suggested"]) == (
        "Sainsbury",
        "alias",
        "Sainsbury's",
    )
    bad = {"kind": "customer", "name": "Sainsbury", "canonical": "Nobody Ltd"}
    assert (await client.post("/admin/names/alias", json=bad, headers=boss)).status_code == 422
    good = {"kind": "customer", "name": "Sainsbury", "canonical": "Sainsbury's"}
    assert (await client.post("/admin/names/alias", json=good, headers=boss)).status_code == 204
    assert names.aliases == {("customer", "sainsbury"): "Sainsbury's"}
    assert (await client.get("/admin/names", headers=boss)).json() == []

    names.aliases.clear()
    apart = {"kind": "customer", "name": "Sainsbury"}
    assert (await client.post("/admin/names/separate", json=apart, headers=boss)).status_code == 204
    assert (await client.get("/admin/names", headers=boss)).json() == []
