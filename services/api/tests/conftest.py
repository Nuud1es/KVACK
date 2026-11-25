import os

import pytest
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repositories

# Settings() reads the developer's .env (real Langfuse keys); keep tests from using them.
os.environ["QACOPILOT_LANGFUSE_PUBLIC_KEY"] = ""
os.environ["QACOPILOT_LANGFUSE_SECRET_KEY"] = ""
os.environ["QACOPILOT_ACCOUNTS"] = "false"  # a developer .env may turn accounts on
os.environ["QACOPILOT_ADMINS"] = ""
os.environ["QACOPILOT_METRICS_TOKEN"] = ""  # likewise: tests choose their own


@pytest.fixture
async def repos():  # type: ignore[no-untyped-def]
    """(DocumentRepository, AgentRepository) on one throwaway schema."""
    try:
        async with temporary_repositories() as pair:
            yield pair
    except DatabaseUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture
def repo(repos):  # type: ignore[no-untyped-def]
    return repos[0]
