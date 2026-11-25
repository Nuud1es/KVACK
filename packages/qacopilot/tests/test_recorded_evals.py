"""The CI gate over recorded eval results, and proof that it fails when it should."""

import json
import shutil
from pathlib import Path

import pytest

from qacopilot.evaluation import recorded

REAL = Path(__file__).resolve().parents[3] / "evals"


@pytest.fixture
def evals(tmp_path: Path) -> Path:
    """A throwaway copy, so a test can tamper with it."""
    target = tmp_path / "evals"
    shutil.copytree(REAL, target, ignore=shutil.ignore_patterns("__pycache__", "*.py", "*.md"))
    return target


def edit(path: Path, change) -> None:  # type: ignore[no-untyped-def]
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_the_committed_results_pass_the_gate() -> None:
    assert recorded.check_all(REAL) == []


def test_a_lowered_floor_check_catches_a_worse_recorded_score(evals: Path) -> None:
    edit(evals / "thresholds.json", lambda t: t["answers"]["min"].update(answer_correct=1.01))
    problems = recorded.check_all(evals)
    assert any("answers answer_correct" in p and "below the floor" in p for p in problems)


def test_editing_a_recorded_answer_is_caught_by_rescoring(evals: Path) -> None:
    def worsen(d: dict) -> None:
        item = next(i for i in d["items"] if i["id"] == "q01")
        item["answer"] = "I think about 60C."  # the summary still claims everything was correct

    edit(evals / "results" / "answers_v3.json", worsen)
    problems = recorded.check_all(evals)
    assert any("re-scoring the recorded answers gives different numbers" in p for p in problems)


def test_a_table_question_answered_by_search_is_a_routing_regression(evals: Path) -> None:
    def reroute(d: dict) -> None:
        item = next(i for i in d["items"] if i["id"] == "q22")
        item["via"] = "documents"  # right text, wrong path: the stale-certificate bug can return

    edit(evals / "results" / "answers_v3.json", reroute)
    assert any("re-scoring the recorded answers" in p for p in recorded.check_all(evals))


def test_a_changed_answer_golden_set_means_the_eval_must_be_re_run(evals: Path) -> None:
    edit(evals / "golden" / "v3.json", lambda g: g.pop())
    assert any("is not in the golden set" in p for p in recorded.check_all(evals))


def test_a_made_up_extraction_accuracy_is_caught(evals: Path) -> None:
    edit(evals / "results" / "extraction_v2.json", lambda d: d.update(accuracy=1.0))
    assert any("re-scoring gives" in p for p in recorded.check_all(evals))


def test_a_changed_golden_question_set_means_the_eval_must_be_re_run(evals: Path) -> None:
    edit(evals / "golden" / "v2.json", lambda g: g.pop())
    problems = recorded.check_all(evals)
    assert any("differ from the golden set" in p for p in problems)


def test_a_wrong_compliance_verdict_is_caught(evals: Path) -> None:
    edit(evals / "golden" / "compliance_v1.json", lambda c: c[2].update(overall="pass"))
    assert any("compliance" in p and "overall" in p for p in recorded.check_all(evals))


def test_a_prompt_edit_without_a_version_bump_fails(
    evals: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recorded, "ANSWER_SYSTEM", recorded.ANSWER_SYSTEM + "\n- Be chatty.")
    problems = recorded.check_prompts(evals)
    assert len(problems) == 1 and "text changed but the version did not" in problems[0]


def test_a_new_prompt_version_must_be_recorded(
    evals: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recorded, "ANSWER_VERSION", "answer_v99")  # a version never recorded
    assert any("answer_v99: new prompt version" in p for p in recorded.check_prompts(evals))


def test_results_recorded_with_an_older_prompt_are_stale(
    evals: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded_with = recorded.EXTRACT_VERSION
    monkeypatch.setattr(recorded, "EXTRACT_VERSION", "extract_v99")
    assert any(f"recorded with prompt {recorded_with}" in p for p in recorded.check_all(evals))


def test_changing_an_extraction_schema_is_noticed(
    evals: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qacopilot.extraction.schemas import CalibrationCert

    monkeypatch.setattr(CalibrationCert.model_fields["result"], "description", "something else")
    CalibrationCert.model_rebuild(force=True)
    try:
        assert any(recorded.EXTRACT_VERSION in p for p in recorded.check_prompts(evals))
    finally:
        monkeypatch.undo()
        CalibrationCert.model_rebuild(force=True)
