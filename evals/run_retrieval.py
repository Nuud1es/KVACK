"""Retrieval eval: index data/samples with real local embeddings, ask the golden questions.

  python evals/run_retrieval.py                       # print scores, write results/<name>.json
  python evals/run_retrieval.py --name baseline       # record a baseline
  python evals/run_retrieval.py --check evals/results/baseline.json   # exit 1 on a regression

Uses its own Qdrant collection (qa_eval), rebuilt each run. Needs Qdrant and the local models,
not the LLM.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from common import GOLDEN, RESULTS, build_index
from qacopilot.evaluation import load_golden, regressions, score_item, summarize
from qacopilot.retrieval import (
    FastEmbedReranker,
    Retriever,
)

TOP_K = 10


async def run(name: str, rerank: bool, golden: Path) -> dict[str, object]:
    store, settings = await build_index()
    retriever = Retriever(store, FastEmbedReranker(settings) if rerank else None)
    items = load_golden(golden)
    results = []
    for item in items:
        hits = await retriever.retrieve(item.question, limit=TOP_K)
        results.append(score_item(item, hits))
    await store.aclose()
    return {
        "name": name,
        "summary": summarize(results),
        "items": [r.model_dump() for r in results],
    }


def show(out: dict[str, object]) -> None:
    summary = out["summary"]
    assert isinstance(summary, dict)
    keys = ("doc_hit@1", "doc_hit@3", "doc_hit@5", "page_hit@1", "page_hit@3", "mrr")
    print(f"{'':16}{'n':>3}  " + "  ".join(f"{k:>10}" for k in keys))
    print(f"{'ALL':16}{summary['n_in_scope']:>3}  " + "  ".join(f"{summary[k]:>10}" for k in keys))
    for cat, m in summary["by_category"].items():
        print(f"{cat:16}{m['n']:>3}  " + "  ".join(f"{m[k]:>10}" for k in keys))
    print(f"\nout-of-scope top scores: {summary['out_of_scope']['top_scores']}")
    print(f"lowest in-scope top score: {summary['out_of_scope']['in_scope_top_scores_min']}")
    misses = [i for i in out["items"] if i["doc_rank"] is None or i["doc_rank"] > 1]  # type: ignore[attr-defined]
    for i in misses:
        if i["category"] != "out_of_scope":
            print(f"  miss/low: {i['id']} doc_rank={i['doc_rank']} top={i['top_doc']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="latest")
    ap.add_argument("--golden", default="v2", help="golden set name in evals/golden/ (default v2)")
    ap.add_argument("--rerank", action="store_true", help="rerank with the cross-encoder")
    ap.add_argument("--check", help="baseline JSON; exit 1 if any tracked metric is lower")
    ap.add_argument("--tolerance", type=float, default=0.0)
    args = ap.parse_args()

    out = asyncio.run(run(args.name, args.rerank, GOLDEN / f"{args.golden}.json"))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    show(out)
    if args.check:
        base = json.loads(Path(args.check).read_text(encoding="utf-8"))["summary"]
        bad = regressions(base, out["summary"], args.tolerance)  # type: ignore[arg-type]
        if bad:
            print("\nREGRESSION:\n  " + "\n  ".join(bad))
            return 1
        print("\nno regression against", args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
