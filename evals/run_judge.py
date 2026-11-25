"""LLM-judged faithfulness and relevance over answers recorded by run_answers.py.

  python evals/run_judge.py --answers v2 --name v2          # about 15-30 s per answer
  python evals/run_judge.py --answers v2 --name v2 --check evals/results/judge_v2.json

Only answered questions are judged (refusals have nothing to ground). The judge never sees the
golden answer. See qacopilot.evaluation.judge for the caveat about self-judging.
"""

import argparse
import asyncio
import json
import sys

from common import RESULTS
from qacopilot.config import Settings
from qacopilot.evaluation.answers import AnswerRecord
from qacopilot.evaluation.judge import (
    JUDGE_PROMPT_VERSION,
    JudgedAnswer,
    judge_answer,
    summarize_judgements,
)
from qacopilot.llm import OpenAICompatibleProvider


async def run(answers_name: str) -> dict[str, object]:
    settings = Settings()
    llm = OpenAICompatibleProvider(settings)
    recorded = json.loads((RESULTS / f"answers_{answers_name}.json").read_text())
    judged: list[JudgedAnswer] = []
    for raw in recorded["items"]:
        rec = AnswerRecord(**raw)
        if not (rec.answered and rec.sources):
            continue
        j = await judge_answer(llm, rec)
        judged.append(j)
        flag = "ok  " if j.faithful and j.relevant else "FLAG"
        print(
            f"{flag} {j.id} faithful={j.faithful} relevant={j.relevant} {j.unsupported or ''}",
            flush=True,
        )
    if judged and sum(1 for j in judged if j.error) > len(judged) // 2:
        raise SystemExit("Most judgements failed. Is the LLM reachable? Run from qa-copilot/.")
    return {
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_model": settings.llm_model,
        "answers": answers_name,
        "summary": summarize_judgements(judged),
        "items": [j.model_dump() for j in judged],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="v2", help="name of the recorded answers_<name>.json")
    ap.add_argument("--name", default="latest")
    ap.add_argument("--check", help="baseline judge JSON; exit 1 if a metric dropped")
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()
    out = asyncio.run(run(args.answers))
    (RESULTS / f"judge_{args.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    s = out["summary"]
    assert isinstance(s, dict)
    print(
        f"\nfaithfulness {s['faithfulness']} | relevance {s['relevance']} | "
        f"judge errors {s['n_judge_errors']}"
    )
    if args.check:
        base = json.loads(open(args.check, encoding="utf-8").read())["summary"]
        bad = [
            f"{k}: {base[k]} -> {s[k]}"
            for k in ("faithfulness", "relevance")
            if s[k] < base[k] - args.tolerance
        ]
        if bad:
            print("\nREGRESSION:\n  " + "\n  ".join(bad))
            return 1
        print("\nno regression against", args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
