"""Extraction eval: run the real LLM over the sample documents and score field accuracy.

  python evals/run_extraction.py --name latest        # needs the LLM (about 20-30 s per document)
  python evals/run_extraction.py --name v2 --check evals/results/extraction_v2.json

The raw extracted objects are stored with the scores, so scoring can be re-run offline (see
check_recorded.py) without the LLM.
"""

import argparse
import asyncio
import json
import sys
import time

from common import GOLDEN, RESULTS, SAMPLES
from qacopilot.config import Settings
from qacopilot.evaluation.extraction import score_extraction
from qacopilot.extraction import PROMPT_VERSION, extract_document
from qacopilot.ingest import DocType, extract_pages
from qacopilot.llm import OpenAICompatibleProvider, StructuredOutputError


def field_count(fields: dict[str, object]) -> int:
    return sum(len(v) if isinstance(v, dict) else 1 for v in fields.values())


async def run(name: str, golden: str) -> dict[str, object]:
    settings = Settings()
    llm = OpenAICompatibleProvider(settings)
    items = json.loads((GOLDEN / f"{golden}.json").read_text())
    out: list[dict[str, object]] = []
    matched = total = 0
    for item in items:
        doc = extract_pages(SAMPLES / item["file"])
        started = time.time()
        try:
            result = await extract_document(
                llm, doc, DocType(item["doc_type"]), thinking=settings.extraction_thinking
            )
        except StructuredOutputError as exc:
            n = field_count(item["fields"])
            total += n
            print(f"FAIL  0/{n:<2} {item['file']}: {str(exc)[:100]}", flush=True)
            out.append({"file": item["file"], "error": str(exc)[:300], "matched": 0, "total": n})
            continue
        score = score_extraction(item["fields"], result)
        matched, total = matched + score.matched, total + score.total
        print(
            f"{score.matched:>2}/{score.total:<2} {time.time() - started:5.1f}s  {item['file']}",
            flush=True,
        )
        for miss in score.misses:
            print(f"        miss {miss}")
        out.append(
            {
                "file": item["file"],
                "extracted": result.model_dump(mode="json"),
                "matched": score.matched,
                "total": score.total,
                "misses": score.misses,
            }
        )
    errored = sum(1 for o in out if "error" in o)
    if errored > len(out) // 2:
        raise SystemExit(
            f"{errored} of {len(out)} documents failed to extract. "
            "Is the LLM reachable? Run from the qa-copilot/ folder so .env is read."
        )
    return {
        "name": name,
        "golden": golden,
        "prompt_version": PROMPT_VERSION,
        "model": settings.llm_model,
        "accuracy": round(matched / total, 3),
        "items": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="latest")
    ap.add_argument("--golden", default="extraction_v2")
    ap.add_argument("--check", help="baseline extraction JSON; exit 1 if accuracy dropped")
    ap.add_argument("--tolerance", type=float, default=0.0)
    args = ap.parse_args()
    out = asyncio.run(run(args.name, args.golden))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"extraction_{args.name}.json").write_text(json.dumps(out, indent=2))
    print(f"\nfield accuracy: {out['accuracy']} (prompt {out['prompt_version']})")
    if args.check:
        base = json.loads(open(args.check).read())["accuracy"]
        assert isinstance(out["accuracy"], float)
        if out["accuracy"] < base - args.tolerance:
            print(f"\nREGRESSION: accuracy {base} -> {out['accuracy']}")
            return 1
        print("\nno regression against", args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
