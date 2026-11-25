"""Shared by the eval runners: index the synthetic corpus with the real local models."""

import json
import re
from pathlib import Path

from qacopilot.config import Settings
from qacopilot.ingest import chunk_document, classify_text, extract_pages
from qacopilot.repositories import DocumentRepository  # noqa: F401  (type of seed_repository)
from qacopilot.retrieval import FastEmbedEmbedder, QdrantStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
GOLDEN = ROOT / "evals" / "golden"
RESULTS = ROOT / "evals" / "results"


async def build_index(collection: str = "qa_eval") -> tuple[QdrantStore, Settings]:
    """A fresh collection holding every PDF in data/samples (doc_id = file name)."""
    settings = Settings(qdrant_collection=collection)
    store = QdrantStore(FastEmbedEmbedder(settings), settings)
    await store.recreate_collection()
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        doc = extract_pages(pdf)
        kind = classify_text(doc.full_text, pdf.name)
        await store.upsert(chunk_document(doc, kind.doc_type, pdf.name))
    return store, settings


async def seed_repository(repo: "DocumentRepository") -> int:
    """Load the golden extraction expectations into a repository, as if every document had been
    extracted perfectly. Lookups (certificates table, CoA check) are then scored on their own
    logic; how well the LLM extracts is measured separately by run_extraction.py.

    doc_id = file name, the same convention as the search index, so citations compare equal.
    Returns the number of documents seeded.
    """
    from qacopilot.extraction import CalibrationCert, CoA, MicroLimit, SupplierSpec, TestResult
    from qacopilot.ingest.models import DocType

    items = json.loads((GOLDEN / "extraction_v2.json").read_text(encoding="utf-8"))
    for item in items:
        kind = DocType(item["doc_type"])
        fields = item["fields"]
        if kind is DocType.CALIBRATION_CERT:
            model = CalibrationCert(**fields)
        elif kind is DocType.COA:
            model = CoA(
                product=fields.get("product"),
                batch=fields["batch"],
                results=[TestResult(test=t, result=r) for t, r in fields["results"].items()],
            )
        else:
            model = SupplierSpec(
                supplier=fields.get("supplier"),
                product=fields["product"],
                product_code=fields.get("product_code"),
                shelf_life=fields.get("shelf_life"),
                storage=fields.get("storage"),
                allergens=fields.get("allergens", []),
                micro_limits=[
                    MicroLimit(test=t, limit=lim)
                    for t, lim in fields.get("micro_limits", {}).items()
                ],
            )
        doc = extract_pages(SAMPLES / item["file"])
        if isinstance(model, CalibrationCert) and not model.instrument_description:
            # the golden lists only some fields; a real extraction reads the printed line
            printed = re.search(r"Instrument:\s*(.+)", doc.full_text)
            if printed:
                model = model.model_copy(
                    update={"instrument_description": printed.group(1).strip()}
                )
        await repo.save_document(item["file"], item["file"], kind, doc, 0)
        await repo.save_extraction(
            item["file"], kind, model, prompt_version="golden", model="golden"
        )
    return len(items)
