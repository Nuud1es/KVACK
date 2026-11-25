"""Compare document classifiers: keyword rules, the LLM alone, and the hybrid (rules, then LLM
only when the rules are unsure). One LLM call per document serves both the LLM and hybrid scores.

  python evals/run_classification.py --set synthetic   # data/samples + golden/classification_v1
  python evals/run_classification.py --set real        # trainingData/ + trainingData/labels.json

The real set is never committed. Its results use anonymous ids (r01, r02, ...); a local detail
file with the real names stays under trainingData/.cache (gitignored). Text extracted from the
real documents is cached there too, because OCR takes minutes.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

from common import GOLDEN, RESULTS, ROOT, SAMPLES
from qacopilot.config import Settings
from qacopilot.ingest import classify_text, classify_with_llm, extract_pages
from qacopilot.ingest.llm_classifier import (
    CLASSIFY_PROMPT_VERSION,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    needs_second_opinion,
)
from qacopilot.ingest.models import DocType
from qacopilot.llm import OpenAICompatibleProvider, StructuredOutputError

REAL = ROOT / "trainingData" / "Training Data"
CACHE = ROOT / "trainingData" / ".cache"


def load_text(path: Path, use_cache: bool) -> str:
    key = CACHE / f"{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}.txt"
    if use_cache and key.exists():
        return key.read_text(encoding="utf-8")
    text = extract_pages(path).full_text
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        key.write_text(text, encoding="utf-8")
    return text


async def run(which: str) -> dict[str, object]:
    llm = OpenAICompatibleProvider(Settings())
    if which == "synthetic":
        labels = json.loads((GOLDEN / "classification_v1.json").read_text())
        base, cache = SAMPLES, False
    else:
        labels = json.loads((REAL.parent / "labels.json").read_text())
        base, cache = REAL, True
    # exact-name lookup: names can contain [brackets], which glob would read as a pattern
    by_name = {p.name: p for p in base.rglob("*") if p.is_file()} if which == "real" else {}
    rows = []
    for n, item in enumerate(labels, start=1):
        path = by_name[item["file"]] if which == "real" else base / item["file"]
        text = load_text(path, cache)
        rules = classify_text(text, path.name)
        t = time.time()
        try:
            by_llm: str | None = (await classify_with_llm(llm, text, path.name)).doc_type.value
        except StructuredOutputError:
            by_llm = None
        sure = not needs_second_opinion(rules)  # the production policy, including the margin
        # the production policy (see classify_document): the LLM fills a gap when the rules find
        # nothing, and only flags a disagreement when they named a type unconfidently
        if sure or by_llm is None:
            hybrid = rules.doc_type.value
        elif rules.doc_type is DocType.OTHER:
            hybrid = by_llm
        else:
            hybrid = rules.doc_type.value
        row = {
            "id": f"r{n:02d}" if which == "real" else item["file"],
            "truth": item["doc_type"],
            "rules": rules.doc_type.value,
            "rules_confidence": rules.confidence,
            "rules_margin": rules.margin,  # lead over the runner-up type; small = a near tie
            "rules_runner_up": rules.runner_up.value if rules.runner_up else None,
            "asked_llm": not sure,
            "llm": by_llm,
            "hybrid": hybrid,
            "flagged": bool(
                not sure
                and by_llm is not None
                and rules.doc_type is not DocType.OTHER
                and by_llm != rules.doc_type.value
            ),  # fmt: skip
            "llm_seconds": round(time.time() - t, 1),
        }
        rows.append(row)
        mark = "".join("y" if row[k] == row["truth"] else "N" for k in ("rules", "llm", "hybrid"))
        print(
            f"{row['id'][:36]:36} rules/llm/hybrid {mark}  {row['llm_seconds']:5.1f}s", flush=True
        )
        if which == "real":
            detail = CACHE / "classification_detail.json"
            names = json.loads(detail.read_text()) if detail.exists() else {}
            names[row["id"]] = path.name
            detail.write_text(json.dumps(names, indent=1))
    n = len(rows)
    acc = {
        k: round(sum(r[k] == r["truth"] for r in rows) / n, 3) for k in ("rules", "llm", "hybrid")
    }
    return {
        "set": which,
        "prompt_version": CLASSIFY_PROMPT_VERSION,
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
        "min_margin": DEFAULT_MIN_MARGIN,
        "n": n,
        "accuracy": acc,
        "flagged_for_review": sum(1 for r in rows if r["flagged"]),
        "llm_asked_in_hybrid": sum(1 for r in rows if r["asked_llm"]),
        # Right answers that the rules got by a narrow margin: the ones most likely to be wrong on
        # documents the rules were not tuned on. A high count is a reason to collect more documents.
        "fragile_but_correct": sum(
            1
            for r in rows
            if r["rules"] == r["truth"]
            and r["rules"] != "other"
            and r["rules_margin"] is not None
            and r["rules_margin"] < DEFAULT_MIN_MARGIN
        ),
        "items": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["synthetic", "real"], required=True)
    args = ap.parse_args()
    out = asyncio.run(run(args.set))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"classification_{args.set}.json").write_text(json.dumps(out, indent=2))
    a = out["accuracy"]
    assert isinstance(a, dict)
    print(
        f"\n{out['set']}: n={out['n']}  rules {a['rules']} | llm {a['llm']} | hybrid {a['hybrid']}"
        f"  (LLM consulted for {out['llm_asked_in_hybrid']} of {out['n']}; "
        f"{out['flagged_for_review']} flagged for review)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
