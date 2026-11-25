from typing import Any

import pytest

from qacopilot import tracing
from qacopilot.config import Settings

pytest.importorskip("langfuse")

KEYS = Settings(langfuse_public_key="pk-lf-x", langfuse_secret_key="sk-lf-x")


def test_trace_id_is_deterministic_and_per_request() -> None:
    a = tracing.trace_id_for("req-00000001")
    assert a == tracing.trace_id_for("req-00000001")
    assert a != tracing.trace_id_for("req-00000002")
    assert len(a) == 32 and int(a, 16) >= 0  # the format Langfuse expects


def test_current_trace_id_needs_both_a_request_and_configured_keys() -> None:
    token = tracing.request_id_var.set("req-00000001")
    try:
        assert tracing.current_trace_id(Settings()) is None  # tracing off
        assert tracing.current_trace_id(KEYS) == tracing.trace_id_for("req-00000001")
    finally:
        tracing.request_id_var.reset(token)
    assert tracing.current_trace_id(KEYS) is None  # no request in progress


class FakeClient:
    def __init__(self) -> None:
        self.scores: list[dict[str, Any]] = []
        self.flushed = False

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def test_feedback_becomes_a_boolean_score_on_the_requests_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    monkeypatch.setattr(tracing, "_client", lambda *a: fake)
    assert tracing.send_feedback_score(KEYS, "req-00000001", False, "wrong page") is True
    (score,) = fake.scores
    assert score["trace_id"] == tracing.trace_id_for("req-00000001")
    assert (score["name"], score["value"], score["data_type"]) == ("user-feedback", 0, "BOOLEAN")
    assert score["comment"] == "wrong page" and fake.flushed


def test_feedback_never_raises_when_langfuse_is_down_or_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tracing.send_feedback_score(Settings(), "req-00000001", True, None) is False

    def boom(*a: Any) -> None:
        raise ConnectionError("langfuse down")

    monkeypatch.setattr(tracing, "_client", boom)
    assert tracing.send_feedback_score(KEYS, "req-00000001", True, None) is False
