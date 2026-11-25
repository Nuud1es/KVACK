"""Accounts and sign-in sessions."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from qacopilot.auth import (
    LOCK_MINUTES,
    MAX_FAILED,
    SESSION_HOURS,
    hash_pin,
    name_key,
    token_digest,
    verify_pin,
)
from qacopilot.repositories.models import User, UserSession


class UserInfo(BaseModel):
    """An account as the admin page shows it (never the PIN hash)."""

    id: int
    name: str
    created_at: datetime
    failed_attempts: int
    locked_until: datetime | None
    locked: bool
    sessions: int  # sign-ins that have not expired
    last_sign_in: datetime | None


@dataclass(frozen=True)
class LoginResult:
    status: Literal["ok", "bad", "locked"]
    user_id: int | None = None
    display_name: str | None = None
    locked_until: datetime | None = None  # set when status is "locked"


class UserRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def register(self, name: str, pin: str) -> int | None:
        """Create the account and return its id; None if the name is already taken."""
        user = User(name_key=name_key(name), display_name=name, pin_hash=hash_pin(pin))
        try:
            async with self._sessions.begin() as s:
                s.add(user)
                await s.flush()
                return user.id
        except IntegrityError:
            return None

    async def authenticate(
        self, name: str, pin: str, *, now: datetime | None = None
    ) -> LoginResult:
        """Check a PIN. Wrong tries are counted on the account and lock it after MAX_FAILED, so
        guessing 10,000 PINs is not possible; while locked even the right PIN is refused."""
        now = now or datetime.now(UTC)
        async with self._sessions.begin() as s:
            user = await s.scalar(
                select(User).where(User.name_key == name_key(name)).with_for_update()
            )
            if user is None:
                verify_pin(pin, None)  # same time as a real check, so a name cannot be probed
                return LoginResult("bad")
            if user.locked_until is not None and user.locked_until > now:
                return LoginResult("locked", locked_until=user.locked_until)
            if verify_pin(pin, user.pin_hash):
                user.failed_attempts = 0
                user.locked_until = None
                return LoginResult("ok", user.id, user.display_name)
            user.failed_attempts += 1
            if user.failed_attempts >= MAX_FAILED:
                user.failed_attempts = 0
                user.locked_until = now + timedelta(minutes=LOCK_MINUTES)
                return LoginResult("locked", locked_until=user.locked_until)
            return LoginResult("bad")

    async def create_session(
        self, user_id: int, token: str, *, now: datetime | None = None
    ) -> None:
        now = now or datetime.now(UTC)
        async with self._sessions.begin() as s:
            await s.execute(delete(UserSession).where(UserSession.expires_at < now))
            s.add(
                UserSession(
                    token_digest=token_digest(token),
                    user_id=user_id,
                    expires_at=now + timedelta(hours=SESSION_HOURS),
                )
            )

    async def user_for_token(self, token: str, *, now: datetime | None = None) -> str | None:
        now = now or datetime.now(UTC)
        async with self._sessions() as s:
            return await s.scalar(
                select(User.display_name)
                .join(UserSession, UserSession.user_id == User.id)
                .where(UserSession.token_digest == token_digest(token))
                .where(UserSession.expires_at > now)
            )

    async def end_session(self, token: str) -> None:
        async with self._sessions.begin() as s:
            await s.execute(
                delete(UserSession).where(UserSession.token_digest == token_digest(token))
            )

    # --- administration ---------------------------------------------------------------------

    async def list_users(self, *, now: datetime | None = None) -> list[UserInfo]:
        now = now or datetime.now(UTC)
        live = (
            select(
                UserSession.user_id,
                func.count().label("n"),
                func.max(UserSession.created_at).label("last"),
            )
            .where(UserSession.expires_at > now)
            .group_by(UserSession.user_id)
            .subquery()
        )
        stmt = (
            select(User, live.c.n, live.c.last)
            .outerjoin(live, live.c.user_id == User.id)
            .order_by(User.name_key)
        )
        async with self._sessions() as s:
            return [
                UserInfo(
                    id=u.id,
                    name=u.display_name,
                    created_at=u.created_at,
                    failed_attempts=u.failed_attempts,
                    locked_until=u.locked_until,
                    locked=u.locked_until is not None and u.locked_until > now,
                    sessions=n or 0,
                    last_sign_in=last,
                )
                for u, n, last in (await s.execute(stmt)).all()
            ]

    async def set_pin(self, user_id: int, pin: str) -> str | None:
        """Give an account a new PIN, lift any lock and sign it out everywhere. Returns the
        name, or None if there is no such account."""
        async with self._sessions.begin() as s:
            user = await s.get(User, user_id)
            if user is None:
                return None
            user.pin_hash = hash_pin(pin)
            user.failed_attempts = 0
            user.locked_until = None
            await s.execute(delete(UserSession).where(UserSession.user_id == user_id))
            return user.display_name

    async def unlock(self, user_id: int) -> str | None:
        async with self._sessions.begin() as s:
            user = await s.get(User, user_id)
            if user is None:
                return None
            user.failed_attempts = 0
            user.locked_until = None
            return user.display_name

    async def sign_out_everywhere(self, user_id: int) -> str | None:
        async with self._sessions.begin() as s:
            user = await s.get(User, user_id)
            if user is None:
                return None
            await s.execute(delete(UserSession).where(UserSession.user_id == user_id))
            return user.display_name

    async def delete_user(self, user_id: int) -> str | None:
        """Remove an account (its sessions go with it). Returns the name, or None."""
        async with self._sessions.begin() as s:
            user = await s.get(User, user_id)
            if user is None:
                return None
            name = user.display_name
            await s.delete(user)
            return name

    async def find(self, name: str) -> int | None:
        async with self._sessions() as s:
            return await s.scalar(select(User.id).where(User.name_key == name_key(name)))
