from pathlib import Path
from typing import Any

import pytest

from qacopilot import tracing
from qacopilot.config import Settings
from qacopilot.evaluation import report

pytest.importorskip("langfuse")
REAL = Path(__file__).resolve().parents[3] / "evals"


def test_collect_scores_reads_every_headline_metric() -> None:
    scores = report.collect_scores(REAL)
    assert set(scores) >= {
        "retrieval.doc_hit@1", "retrieval.mrr", "answers.answer_correct", "answers.false_refusal",
        "answers.correct_refusal", "extraction.accuracy", "judge.faithfulness",
    }  # fmt: skip
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in scores.values())


class FakeSpan:
    def __enter__(self) -> "FakeSpan":
        return self

    def __exit__(self, *a: Any) -> None:
        return None


class FakeClient:
    def __init__(self) -> None:
        self.scores: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.flushed = False

    def start_as_current_observation(self, **kw: Any) -> FakeSpan:
        self.spans.append(kw)
        return FakeSpan()

    def create_score(self, **kw: Any) -> None:
        self.scores.append(kw)

    def flush(self) -> None:
        self.flushed = True


KEYS = Settings(langfuse_public_key="pk", langfuse_secret_key="sk")


def test_scores_go_to_one_trace_per_release_with_stable_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClient()
    monkeypatch.setattr(report, "_client", lambda *a: fake)
    ok = report.log_release_scores(KEYS, "v1", {"a.x": 0.5, "a.y": 1.0}, {"git_sha": "abc"})
    assert ok and fake.flushed
    trace = tracing.trace_id_for("eval-v1")
    assert fake.spans[0]["trace_context"] == {"trace_id": trace}
    assert fake.spans[0]["metadata"] == {"release": "v1", "git_sha": "abc"}
    assert [(s["trace_id"], s["name"], s["value"]) for s in fake.scores] == [
        (trace, "a.x", 0.5), (trace, "a.y", 1.0),
    ]  # fmt: skip
    ids = {s["score_id"] for s in fake.scores}
    assert len(ids) == 2  # one per metric, and the same on a re-run so it updates, not duplicates
    fake.scores.clear()
    report.log_release_scores(KEYS, "v1", {"a.x": 0.6}, {})
    assert fake.scores[0]["score_id"] in ids


def test_logging_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    assert report.log_release_scores(Settings(), "v1", {"a": 1.0}, {}) is False  # tracing off

    def boom(*a: Any) -> None:
        raise ConnectionError("down")

    monkeypatch.setattr(report, "_client", boom)
    assert report.log_release_scores(KEYS, "v1", {"a": 1.0}, {}) is False
