from datetime import UTC, datetime, timedelta

import pytest

from qacopilot import auth
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_user_repository


def test_names_are_tidied_and_checked() -> None:
    assert auth.clean_name("  Joe   Rudling ") == "Joe Rudling"
    assert auth.clean_name("Anne-Marie O'Neil") == "Anne-Marie O'Neil"
    for bad in ["J", "x" * 41, "1234", "Joe; DROP", "<b>Joe</b>", ""]:
        with pytest.raises(auth.InvalidInput):
            auth.clean_name(bad)
    assert auth.name_key("JOE  rudling") == auth.name_key("joe rudling")


def test_pin_must_be_four_digits() -> None:
    assert auth.check_pin("0042") == "0042"
    for bad in ["123", "12345", "12a4", "", " 123", "١٢٣٤"]:  # the last is Arabic-Indic digits
        with pytest.raises(auth.InvalidInput):
            auth.check_pin(bad)


def test_pin_hash_is_salted_and_verifies() -> None:
    a, b = auth.hash_pin("1234"), auth.hash_pin("1234")
    assert a != b and "1234" not in a
    assert auth.verify_pin("1234", a)
    assert not auth.verify_pin("1235", a)
    assert not auth.verify_pin("1234", None)  # unknown account is never a match


def test_tokens_are_stored_only_as_a_digest() -> None:
    token = auth.new_token()
    assert len(token) >= 40 and token != auth.new_token()
    assert auth.token_digest(token) != token and len(auth.token_digest(token)) == 64


@pytest.fixture
async def users():  # type: ignore[no-untyped-def]
    try:
        async with temporary_user_repository() as repo:
            yield repo
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


async def test_register_then_login(users) -> None:  # type: ignore[no-untyped-def]
    assert await users.register("Alice Smith", "1234") is not None
    assert await users.register("alice  smith", "9999") is None  # same person, any case
    ok = await users.authenticate("ALICE SMITH", "1234")
    assert ok.status == "ok" and ok.display_name == "Alice Smith"
    assert (await users.authenticate("Alice Smith", "0000")).status == "bad"
    assert (await users.authenticate("Nobody", "1234")).status == "bad"


async def test_five_wrong_pins_lock_the_account_even_for_the_right_pin(users) -> None:  # type: ignore[no-untyped-def]
    await users.register("Alice", "1234")
    now = datetime.now(UTC)
    for _ in range(auth.MAX_FAILED - 1):
        assert (await users.authenticate("Alice", "0000", now=now)).status == "bad"
    assert (await users.authenticate("Alice", "0000", now=now)).status == "locked"
    assert (await users.authenticate("Alice", "1234", now=now)).status == "locked"
    later = now + timedelta(minutes=auth.LOCK_MINUTES, seconds=1)
    assert (await users.authenticate("Alice", "1234", now=later)).status == "ok"


async def test_a_good_login_resets_the_count(users) -> None:  # type: ignore[no-untyped-def]
    await users.register("Alice", "1234")
    for _ in range(auth.MAX_FAILED - 1):
        await users.authenticate("Alice", "0000")
    assert (await users.authenticate("Alice", "1234")).status == "ok"
    for _ in range(auth.MAX_FAILED - 1):
        assert (await users.authenticate("Alice", "0000")).status == "bad"


async def test_sessions_expire_and_end(users) -> None:  # type: ignore[no-untyped-def]
    uid = await users.register("Alice", "1234")
    assert uid is not None
    now = datetime.now(UTC)
    await users.create_session(uid, "tok", now=now)
    assert await users.user_for_token("tok", now=now) == "Alice"
    assert await users.user_for_token("other", now=now) is None
    assert (
        await users.user_for_token("tok", now=now + timedelta(hours=auth.SESSION_HOURS + 1)) is None
    )
    await users.end_session("tok")
    assert await users.user_for_token("tok", now=now) is None
