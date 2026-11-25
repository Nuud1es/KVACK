import subprocess
import sys

import pytest

MODULES = [
    "qacopilot.extraction",
    "qacopilot.extraction.runner",
    "qacopilot.repositories",
    "qacopilot.repositories.testing",
    "qacopilot.agent",
    "qacopilot.ingest",
    "qacopilot.ingest.pipeline",
    "qacopilot.evaluation.recorded",
]


@pytest.mark.parametrize("module", MODULES)
def test_each_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """Circular imports depend on which module is imported first, and a test run imports in one
    fixed order, so each module gets its own process."""
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", f"import {module}"],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-400:]
