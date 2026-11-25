"""Helpers for tests: repositories on a throwaway schema of a real PostgreSQL."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from qacopilot.config import Settings
from qacopilot.repositories.agent import AgentRepository
from qacopilot.repositories.documents import DocumentRepository
from qacopilot.repositories.models import Base
from qacopilot.repositories.users import UserRepository


class DatabaseUnavailable(Exception):
    """No database is configured or it cannot be reached; tests should skip."""


@asynccontextmanager
async def temporary_engine() -> AsyncIterator[AsyncEngine]:
    """Create a uniquely named schema, yield a repository bound to it, drop it afterwards.

    Uses ``QACOPILOT_TEST_DATABASE_URL`` if set, else the configured database. Only the
    throwaway schema is ever created or dropped, so real tables are never touched.
    """
    url = os.environ.get("QACOPILOT_TEST_DATABASE_URL") or Settings().database_url
    if not url:
        raise DatabaseUnavailable("no database configured")
    schema = f"test_{uuid4().hex[:10]}"
    admin = create_async_engine(url)
    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    except Exception as exc:
        await admin.dispose()
        raise DatabaseUnavailable("PostgreSQL is not reachable") from exc
    # The schema is part of the URL, so the same URL also works for LangGraph's checkpointer.
    pinned = make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    engine = create_async_engine(pinned)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


def libpq_conninfo(engine: AsyncEngine) -> str:
    """The engine's URL as plain ``postgresql://`` (what LangGraph's checkpointer expects)."""
    url = make_url(engine.url.render_as_string(hide_password=False))
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


@asynccontextmanager
async def temporary_repository() -> AsyncIterator[DocumentRepository]:
    async with temporary_engine() as engine:
        yield DocumentRepository(engine)


@asynccontextmanager
async def temporary_repositories() -> AsyncIterator[tuple[DocumentRepository, AgentRepository]]:
    async with temporary_engine() as engine:
        yield DocumentRepository(engine), AgentRepository(engine)


@asynccontextmanager
async def temporary_user_repository() -> AsyncIterator[UserRepository]:
    async with temporary_engine() as engine:
        yield UserRepository(engine)
