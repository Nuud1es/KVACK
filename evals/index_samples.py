"""Smoke test: ingest data/samples, index into live Qdrant with real local embeddings, query.

Run: python evals/index_samples.py ["your question"]
Uses a separate collection (qa_samples_smoke) so it never touches real data.
"""

import asyncio
import sys
from pathlib import Path

from qacopilot.config import Settings
from qacopilot.ingest import chunk_document, classify_text, extract_pages
from qacopilot.retrieval import FastEmbedEmbedder, QdrantStore

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
QUESTIONS = [
    "What temperature must cooked duck reach before release?",
    "What is the serial number of the probe thermometer?",
    "How long is the shelf life of the orange sauce?",
    "When does the BRCGS certificate expire?",
]


async def main(questions: list[str]) -> None:
    settings = Settings(qdrant_collection="qa_samples_smoke")
    store = QdrantStore(FastEmbedEmbedder(settings), settings)
    await store.ensure_collection()
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        doc = extract_pages(pdf)
        kind = classify_text(doc.full_text, pdf.name)
        chunks = chunk_document(doc, kind.doc_type, pdf.name)
        await store.replace_document(pdf.name, chunks)
        print(f"{pdf.name:42} {kind.doc_type.value:17} {len(chunks)} chunk(s)")
    print(f"\n{await store.count()} chunks indexed\n")
    for q in questions:
        print(f"Q: {q}")
        for h in await store.search(q, limit=3):
            c = h.chunk
            print(f"   {h.score:.3f}  {c.doc_id} p{c.page} [{c.doc_type.value}] {c.section or '-'}")
    await store.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or QUESTIONS))
