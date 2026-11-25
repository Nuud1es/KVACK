"""Alembic environment. Run: alembic -c packages/qacopilot/alembic.ini upgrade head"""

from alembic import context
from sqlalchemy import create_engine

from qacopilot.config import Settings
from qacopilot.repositories.models import Base

target_metadata = Base.metadata


def run() -> None:
    url = Settings().database_url
    if not url:
        raise SystemExit("QACOPILOT_DATABASE_URL is not set")
    engine = create_engine(url)  # psycopg works for both sync and async use
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run()
