"""Offline checks over the *recorded* eval results in ``evals/results``. No LLM, no models.

This is what GitHub-hosted CI runs. It cannot say how a new prompt or model performs (that needs
the live evals on the self-hosted runner), but it does catch: scoring or golden-set changes that
silently move the numbers, recorded results that fall below the committed floors, results that
are stale because a prompt changed, and prompt text edited without a version bump.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from qacopilot.compliance import Verdict, check_coa_against_spec, overall
from qacopilot.evaluation.answers import AnswerRecord, score_answer, summarize_answers
from qacopilot.evaluation.extraction import score_extraction
from qacopilot.evaluation.judge import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM
from qacopilot.evaluation.retrieval import ItemResult, load_golden, summarize
from qacopilot.extraction import PROMPT_VERSION as EXTRACT_VERSION
from qacopilot.extraction.prompts import EXTRACT_SYSTEM
from qacopilot.extraction.schemas import CoA, SupplierSpec
from qacopilot.extraction.service import SCHEMAS
from qacopilot.ingest.llm_classifier import CLASSIFY_PROMPT_VERSION, CLASSIFY_SYSTEM
from qacopilot.ingest.models import DocType
from qacopilot.rag.prompts import ANSWER_SYSTEM
from qacopilot.rag.prompts import PROMPT_VERSION as ANSWER_VERSION


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_fingerprints() -> dict[str, str]:
    """Hash of each prompt's full text (and the extraction schemas the model is shown)."""
    schemas = json.dumps(
        {t.value: m.model_json_schema() for t, m in SCHEMAS.items()}, sort_keys=True
    )
    texts = {
        ANSWER_VERSION: ANSWER_SYSTEM,
        EXTRACT_VERSION: EXTRACT_SYSTEM + schemas,
        JUDGE_PROMPT_VERSION: JUDGE_SYSTEM,
        CLASSIFY_PROMPT_VERSION: CLASSIFY_SYSTEM,
    }
    return {
        version: hashlib.sha256(text.encode()).hexdigest()[:16] for version, text in texts.items()
    }


def check_prompts(evals: Path) -> list[str]:
    known: dict[str, str] = _load(evals / "prompt_hashes.json")
    problems = []
    for version, digest in prompt_fingerprints().items():
        if version not in known:
            problems.append(f"{version}: new prompt version, record it (check_recorded --write)")
        elif known[version] != digest:
            problems.append(
                f"{version}: the prompt or schema text changed but the version did not. Bump the "
                "version, re-run the live evals, then record the new hashes and results"
            )
    return problems


def check_retrieval(evals: Path, spec: dict[str, Any], golden: Path) -> list[str]:
    data = _load(evals / "results" / spec["file"])
    items = [ItemResult(**i) for i in data["items"]]
    ids = {g.id for g in load_golden(golden)}
    problems = []
    if {i.id for i in items} != ids:
        problems.append("retrieval: recorded questions differ from the golden set; re-run the eval")
    summary = summarize(items)
    for key, floor in spec["min"].items():
        if summary[key] < floor:
            problems.append(f"retrieval {key}: {summary[key]} is below the floor {floor}")
    if summary != data["summary"]:
        problems.append("retrieval: recorded summary does not match its own items")
    return problems


def check_answers(evals: Path, spec: dict[str, Any], golden: Path) -> list[str]:
    data = _load(evals / "results" / spec["file"])
    golden_items = {g.id: g for g in load_golden(golden)}
    problems = []
    if data["prompt_version"] != ANSWER_VERSION:
        problems.append(
            f"answers: recorded with prompt {data['prompt_version']} but the code is on "
            f"{ANSWER_VERSION}; re-run the live evals"
        )
    scores = []
    for raw in data["items"]:
        item = golden_items.get(raw["id"])
        if item is None:
            problems.append(f"answers: {raw['id']} is not in the golden set")
            continue
        scores.append(score_answer(item, AnswerRecord(**raw)))
    if len(scores) != len(golden_items):
        problems.append("answers: recorded questions differ from the golden set; re-run the eval")
    summary = summarize_answers(scores)
    if summary != data["summary"]:
        problems.append("answers: re-scoring the recorded answers gives different numbers")
    for key, floor in spec["min"].items():
        if summary[key] < floor:
            problems.append(f"answers {key}: {summary[key]} is below the floor {floor}")
    for key, ceiling in spec.get("max", {}).items():
        if summary[key] > ceiling:
            problems.append(f"answers {key}: {summary[key]} is above the ceiling {ceiling}")
    return problems


def check_extraction(evals: Path, spec: dict[str, Any], golden: Path) -> list[str]:
    data = _load(evals / "results" / spec["file"])
    expected = {g["file"]: g for g in _load(golden)}
    problems = []
    if data["prompt_version"] != EXTRACT_VERSION:
        problems.append(
            f"extraction: recorded with prompt {data['prompt_version']} but the code is on "
            f"{EXTRACT_VERSION}; re-run the live evals"
        )
    matched = total = 0
    for rec in data["items"]:
        want = expected.get(rec["file"])
        if want is None:
            problems.append(f"extraction: {rec['file']} is not in the golden set")
            continue
        if "extracted" not in rec:
            total += rec["total"]
            continue
        model = SCHEMAS[DocType(want["doc_type"])](**rec["extracted"])
        score = score_extraction(want["fields"], model)
        matched, total = matched + score.matched, total + score.total
    accuracy = round(matched / total, 3) if total else 0.0
    if accuracy != data["accuracy"]:
        problems.append(f"extraction: re-scoring gives {accuracy}, recorded {data['accuracy']}")
    if accuracy < spec["min_accuracy"]:
        problems.append(f"extraction accuracy {accuracy} is below the floor {spec['min_accuracy']}")
    return problems


def check_compliance(evals: Path, spec: dict[str, Any]) -> list[str]:
    """The CoA-vs-spec verdicts, computed by code from the recorded extractions."""
    extracted = {
        i["file"]: i["extracted"]
        for i in _load(evals / "results" / spec["extraction_file"])["items"]
        if "extracted" in i
    }
    problems = []
    for case in _load(evals / "golden" / "compliance_v1.json"):
        coa, sp = extracted.get(case["coa"]), extracted.get(case["spec"])
        if coa is None or sp is None:
            problems.append(
                f"compliance: no recorded extraction for {case['coa']} / {case['spec']}"
            )
            continue
        rows = check_coa_against_spec(CoA(**coa), SupplierSpec(**sp))
        got = {r.test: r.verdict.value for r in rows}
        if got != case["verdicts"]:
            problems.append(f"compliance {case['coa']}: expected {case['verdicts']}, got {got}")
        if overall(rows) is not Verdict(case["overall"]):
            problems.append(f"compliance {case['coa']}: overall {overall(rows).value}")
    return problems


def check_judge(evals: Path, spec: dict[str, Any]) -> list[str]:
    summary = _load(evals / "results" / spec["file"])["summary"]
    return [
        f"judge {key}: {summary[key]} is below the floor {floor}"
        for key, floor in spec["min"].items()
        if summary[key] < floor
    ]


def check_classification(evals: Path, spec: dict[str, Any]) -> list[str]:
    """Re-score the recorded classifier comparisons and hold them to the committed floors."""
    problems: list[str] = []
    for name, floors in spec["sets"].items():
        data = _load(evals / "results" / f"classification_{name}.json")
        if data["prompt_version"] != CLASSIFY_PROMPT_VERSION:
            problems.append(
                f"classification {name}: recorded with prompt {data['prompt_version']} but the "
                f"code is on {CLASSIFY_PROMPT_VERSION}; re-run the live evals"
            )
        items = data["items"]
        for who in ("rules", "llm", "hybrid"):
            acc = round(sum(i[who] == i["truth"] for i in items) / len(items), 3)
            if acc != data["accuracy"][who]:
                problems.append(f"classification {name} {who}: re-scoring gives {acc}")
            if who in floors and acc < floors[who]:
                problems.append(
                    f"classification {name} {who}: {acc} is below the floor {floors[who]}"
                )
    return problems


def check_all(evals: Path) -> list[str]:
    t = _load(evals / "thresholds.json")
    problems = check_prompts(evals)
    problems += check_retrieval(evals, t["retrieval"], evals / "golden" / "v2.json")
    problems += check_answers(
        evals, t["answers"], evals / "golden" / t["answers"].get("golden", "v2.json")
    )
    problems += check_extraction(evals, t["extraction"], evals / "golden" / "extraction_v2.json")
    problems += check_compliance(evals, t["compliance"])
    if "judge" in t:
        problems += check_judge(evals, t["judge"])
    if "classification" in t:
        problems += check_classification(evals, t["classification"])
    return problems
