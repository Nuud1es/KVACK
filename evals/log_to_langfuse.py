"""Record the current eval results in Langfuse as the scores for a release.

  python evals/log_to_langfuse.py --release v0.5.0
  python evals/log_to_langfuse.py            # release = git describe, else the short commit

Reads the numbers named in evals/thresholds.json, so run it after the live evals are recorded.
"""

import argparse
import subprocess
import sys

from common import ROOT
from qacopilot.config import Settings
from qacopilot.evaluation.report import collect_scores, log_release_scores
from qacopilot.extraction import PROMPT_VERSION as EXTRACT_VERSION
from qacopilot.rag.prompts import PROMPT_VERSION as ANSWER_VERSION


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return out.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=None)
    ap.add_argument("--ci", action="store_true", help="log the fresh results of a --name ci run")
    args = ap.parse_args()
    sha = git("rev-parse", "--short", "HEAD")
    release = args.release or git("describe", "--tags", "--always") or sha or "dev"
    files = (
        {
            "retrieval": "ci.json",
            "answers": "answers_ci.json",
            "extraction": "extraction_ci.json",
            "judge": "judge_ci.json",
        }  # fmt: skip
        if args.ci
        else None
    )
    scores = collect_scores(ROOT / "evals", files)
    settings = Settings()
    ok = log_release_scores(
        settings,
        release,
        scores,
        {"git_sha": sha, "answer_prompt": ANSWER_VERSION, "extract_prompt": EXTRACT_VERSION},
    )
    for name, value in scores.items():
        print(f"{name:32} {value}")
    print(
        f"\nrelease {release}: "
        + ("logged to Langfuse" if ok else "NOT logged (tracing off or unreachable)")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
