"""Collect the headline eval numbers and record them in Langfuse, one trace per release."""

import json
from pathlib import Path
from typing import Any

from qacopilot.config import Settings
from qacopilot.tracing import _client, trace_id_for, tracing_enabled


def collect_scores(evals: Path, files: dict[str, str] | None = None) -> dict[str, float]:
    """Flat ``{"retrieval.doc_hit@1": 0.93, ...}``.

    By default the files are the recorded results named in thresholds.json. ``files`` maps
    ``retrieval`` / ``answers`` / ``extraction`` / ``judge`` to other result files in
    ``evals/results`` (e.g. those of a fresh CI run).
    """
    t = json.loads((evals / "thresholds.json").read_text(encoding="utf-8"))
    files = files or {}

    def summary(spec: dict[str, Any], key: str) -> dict[str, Any]:
        name = files.get(key, spec["file"])
        return dict(json.loads((evals / "results" / name).read_text(encoding="utf-8")))

    scores: dict[str, float] = {}
    r = summary(t["retrieval"], "retrieval")["summary"]
    scores.update({f"retrieval.{k}": r[k] for k in ("doc_hit@1", "doc_hit@3", "doc_hit@5", "mrr")})
    a = summary(t["answers"], "answers")["summary"]
    scores.update({f"answers.{k}": a[k] for k in (*t["answers"]["min"], "false_refusal")})
    scores["extraction.accuracy"] = summary(t["extraction"], "extraction")["accuracy"]
    if "judge" in t:
        j = summary(t["judge"], "judge")["summary"]
        scores.update({"judge.faithfulness": j["faithfulness"], "judge.relevance": j["relevance"]})
    for name in t.get("classification", {}).get("sets", {}):
        path = evals / "results" / f"classification_{name}.json"
        if path.exists():
            acc = json.loads(path.read_text(encoding="utf-8"))["accuracy"]
            scores.update(
                {f"classification.{name}.{k}": acc[k] for k in ("rules", "llm", "hybrid")}
            )
    return {k: float(v) for k, v in scores.items()}


def log_release_scores(
    settings: Settings, release: str, scores: dict[str, float], metadata: dict[str, str]
) -> bool:
    """One Langfuse trace ``eval-<release>`` carrying each metric as a numeric score.

    Re-logging the same release updates it rather than duplicating it. Returns False, without
    raising, when tracing is off or Langfuse cannot be reached.
    """
    if not tracing_enabled(settings):
        return False
    try:
        client = _client(
            settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host
        )
        trace_id = trace_id_for(f"eval-{release}")
        with client.start_as_current_observation(
            name="eval-run",
            as_type="span",
            trace_context={"trace_id": trace_id},
            metadata={"release": release, **metadata},
        ):
            pass
        for name, value in scores.items():
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type="NUMERIC",
                score_id=trace_id_for(f"{release}:{name}"),  # same id = update, not duplicate
                comment=f"release {release}",
            )
        client.flush()
        return True
    except Exception:
        return False
