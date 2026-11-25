"""Answer eval: ask every golden question end to end with the real LLM and score the answers.

  python evals/run_answers.py --name latest                   # needs Qdrant, local models, the LLM
  python evals/run_answers.py --name v2 --check evals/results/answers_v2.json   # exit 1 on a drop

Scoring is deterministic (required/forbidden phrases, correct refusal, citations). Faithfulness
and relevance, which need a judge model, are scored separately by run_judge.py from the answers
recorded here.
"""

import argparse
import asyncio
import json
import sys
import time

from common import GOLDEN, RESULTS, build_index, seed_repository
from qacopilot.evaluation import (
    AnswerRecord,
    CitationRecord,
    answer_regressions,
    load_golden,
    score_answer,
    summarize_answers,
)
from qacopilot.llm import OpenAICompatibleProvider
from qacopilot.rag import (
    DoneEvent,
    NoAnswerEvent,
    SourcesEvent,
    TokenEvent,
    answer_question,
    answer_with_lookup,
)
from qacopilot.rag.prompts import PROMPT_VERSION
from qacopilot.repositories.testing import DatabaseUnavailable, temporary_repository
from qacopilot.retrieval import FastEmbedReranker, Hit, Retriever


async def run(name: str, golden: str) -> dict[str, object]:
    store, settings = await build_index()
    retriever = Retriever(store, FastEmbedReranker(settings), min_score=settings.rerank_min_score)
    llm = OpenAICompatibleProvider(settings)
    items = load_golden(GOLDEN / f"{golden}.json")
    # Questions are asked through answer_with_lookup, as /ask does: due dates come from the
    # certificates table and "does batch X meet spec" from the check, so they need a database
    # holding the golden extractions (a throwaway schema; real tables are never touched).
    try:
        async with temporary_repository() as repo:
            print(f"seeded {await seed_repository(repo)} documents", flush=True)
            return await _ask_all(name, golden, items, store, retriever, llm, settings, repo)
    except DatabaseUnavailable as exc:
        raise SystemExit(f"PostgreSQL is needed for the lookup questions: {exc}") from exc


async def _ask_all(name, golden, items, store, retriever, llm, settings, repo):  # type: ignore[no-untyped-def]
    records, scores = [], []
    for item in items:
        rec = AnswerRecord(id=item.id, question=item.question)
        hits: list[Hit] = []
        started = time.time()
        try:
            text: list[str] = []
            events = answer_question(item.question, retriever, llm, capture=hits)
            async for e in answer_with_lookup(item.question, repo, events):
                if isinstance(e, SourcesEvent):
                    rec.citations = [
                        CitationRecord(n=c.n, doc_id=c.doc_id, filename=c.filename, page=c.page)
                        for c in e.citations
                    ]
                elif isinstance(e, TokenEvent):
                    text.append(e.text)
                elif isinstance(e, NoAnswerEvent):
                    rec.refusal = e.reason
                elif isinstance(e, DoneEvent):
                    rec.answered, rec.cited, rec.via = e.answered, e.cited, e.via
            rec.answer = "".join(text).strip()
            rec.sources = {i: h.chunk.text for i, h in enumerate(hits, start=1)}
        except Exception as exc:
            rec.error = f"{type(exc).__name__}: {exc}"[:300]
        rec.seconds = round(time.time() - started, 1)
        score = score_answer(item, rec)
        records.append(rec)
        scores.append(score)
        mark = "ok  " if score.correct else "FAIL"
        print(
            f"{mark} {item.id} {item.category:14} {rec.seconds:5.1f}s {'; '.join(score.reasons)}",
            flush=True,
        )
    await store.aclose()
    failed = sum(1 for r in records if r.error)
    if failed > len(records) // 2:
        raise SystemExit(
            f"{failed} of {len(records)} questions errored (first: {records[0].error}). "
            "Is the LLM reachable? Run from the qa-copilot/ folder so .env is read."
        )
    return {
        "name": name,
        "golden": golden,
        "prompt_version": PROMPT_VERSION,
        "model": settings.llm_model,
        "summary": summarize_answers(scores),
        "items": [
            {**r.model_dump(), "score": s.model_dump()}
            for r, s in zip(records, scores, strict=True)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="latest")
    ap.add_argument("--golden", default="v3")
    ap.add_argument("--check", help="baseline answers JSON; exit 1 if a metric dropped")
    ap.add_argument("--tolerance", type=float, default=0.0)
    args = ap.parse_args()
    out = asyncio.run(run(args.name, args.golden))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"answers_{args.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    s = out["summary"]
    assert isinstance(s, dict)
    print(
        f"\nanswer correct {s['answer_correct']} | false refusal {s['false_refusal']} | "
        f"citation presence {s['citation_presence']} | "
        f"citation correct {s['citation_correct']} | correct refusal {s['correct_refusal']}"
    )
    for cat, m in s["by_category"].items():
        print(f"  {cat:14} n={m['n']} correct={m['answer_correct']}")
    if args.check:
        base = json.loads(open(args.check, encoding="utf-8").read())["summary"]
        bad = answer_regressions(base, s, args.tolerance)
        if bad:
            print("\nREGRESSION:\n  " + "\n  ".join(bad))
            return 1
        print("\nno regression against", args.check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
