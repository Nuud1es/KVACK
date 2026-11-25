"""Offline gate over the recorded results (what GitHub-hosted CI runs): no LLM, no models.

  python evals/check_recorded.py            # exit 1 on any problem
  python evals/check_recorded.py --write    # accept the current recorded results as the new floors

--write copies the numbers in evals/results into evals/thresholds.json and the prompt fingerprints
into evals/prompt_hashes.json. Run it deliberately after a live eval you are happy with, and
review the diff: it is what stops scores being lowered by accident.
"""

import argparse
import json
import sys

from common import RESULTS, ROOT
from qacopilot.evaluation.recorded import check_all, prompt_fingerprints

EVALS = ROOT / "evals"

FILES = {
    "retrieval": "rerank_v2.json",
    "answers": "answers_v3.json",
    "extraction": "extraction_v2.json",
    "judge": "judge_v3.json",
}


def write() -> None:
    def load(name: str) -> dict:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    r, a, e = (
        load(FILES["retrieval"])["summary"],
        load(FILES["answers"])["summary"],
        load(FILES["extraction"]),
    )
    thresholds: dict[str, object] = {
        "retrieval": {
            "file": FILES["retrieval"],
            "min": {k: r[k] for k in ("doc_hit@1", "doc_hit@3", "doc_hit@5", "page_hit@1", "mrr")},
        },
        "answers": {
            "file": FILES["answers"],
            "golden": "v3.json",
            "min": {
                k: a[k]
                for k in (
                    "answer_correct",
                    "citation_presence",
                    "citation_correct",
                    "correct_refusal",
                )
            },
            "max": {"false_refusal": a["false_refusal"]},
        },
        "extraction": {"file": FILES["extraction"], "min_accuracy": e["accuracy"]},
        "compliance": {"extraction_file": FILES["extraction"]},
    }
    sets = {}
    for name in ("synthetic", "real"):
        path = RESULTS / f"classification_{name}.json"
        if path.exists():
            acc = json.loads(path.read_text(encoding="utf-8"))["accuracy"]
            sets[name] = {"rules": acc["rules"], "hybrid": acc["hybrid"]}
    if sets:
        thresholds["classification"] = {"sets": sets}
    if (RESULTS / FILES["judge"]).exists():
        j = load(FILES["judge"])["summary"]
        thresholds["judge"] = {
            "file": FILES["judge"],
            "min": {"faithfulness": j["faithfulness"], "relevance": j["relevance"]},
        }
    (EVALS / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n", encoding="utf-8"
    )
    (EVALS / "prompt_hashes.json").write_text(
        json.dumps(prompt_fingerprints(), indent=2) + "\n", encoding="utf-8"
    )
    print("wrote evals/thresholds.json and evals/prompt_hashes.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    if ap.parse_args().write:
        write()
        return 0
    problems = check_all(EVALS)
    for p in problems:
        print("FAIL", p)
    print(f"{len(problems)} problem(s)" if problems else "recorded evals OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
