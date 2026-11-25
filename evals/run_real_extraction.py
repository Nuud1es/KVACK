"""Extraction on the real calibration files, reporting AGGREGATES ONLY (no document text).

  python evals/run_real_extraction.py --limit 6

Runs the production pipeline (split into certificates, extract each with the LLM, store in a
throwaway PostgreSQL schema) on trainingData/Training Data/1 Calibration certificates. There is
no ground truth for these files, so quality is judged by what can be checked without one: how many
certificates were read, how many serials are distinct, how many have both dates, and how many
dates are plausible (about a year apart, in a sensible range). Files appear as f01, f02, ...
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import date

from common import RESULTS, ROOT
from qacopilot.config import Settings
from qacopilot.extraction import CalibrationCert, certificate_warnings, run_extraction
from qacopilot.ingest import DocType, PageText, ParsedDocument, extract_pages
from qacopilot.llm import OpenAICompatibleProvider
from qacopilot.repositories.testing import temporary_repository

FOLDER = ROOT / "trainingData" / "Training Data" / "1 Calibration certificates"
CACHE = ROOT / "trainingData" / ".cache"


def load_pages(path):  # type: ignore[no-untyped-def]
    key = CACHE / f"pages_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}.json"
    if key.exists():
        pages = json.loads(key.read_text())
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        pages = [
            {"page": p.page, "text": p.text, "method": p.method} for p in extract_pages(path).pages
        ]
        key.write_text(json.dumps(pages))
    return ParsedDocument(source=path.name, pages=[PageText(**p) for p in pages])


async def run(limit: int, only: set[int]) -> dict[str, object]:
    settings = Settings()
    llm = OpenAICompatibleProvider(settings)
    out = []
    async with temporary_repository() as repo:
        for n, path in enumerate(sorted(p for p in FOLDER.iterdir() if p.is_file()), start=1):
            if only and n not in only:
                continue
            doc = load_pages(path)
            doc_id = f"f{n:02d}"
            await repo.save_document(doc_id, f"{doc_id}.pdf", DocType.CALIBRATION_CERT, doc, 1)
            started = time.time()
            run_ = await run_extraction(llm, repo, doc_id, model=settings.llm_model, limit=limit)
            rows = await repo.get_extractions(doc_id)
            certs = [CalibrationCert(**r.data) for r in rows]
            serials = [c.instrument_serial for c in certs]
            warned = sum(1 for c in certs if certificate_warnings(c))
            plausible = sum(
                1 for c in certs if c.next_due and not certificate_warnings(c)
                and date(2015, 1, 1) <= c.calibration_date <= date(2027, 12, 31)
            )  # fmt: skip
            rec = {
            "file": doc_id, "pages": len(doc.pages), "read": run_.done, "failed": run_.failed,
            "distinct_serials": len(set(serials)),
            "with_next_due": sum(1 for c in certs if c.next_due),
            "with_warnings": warned, "plausible_dates": plausible,
            "seconds_per_certificate": round((time.time() - started) / max(run_.total, 1), 1),
            }  # fmt: skip
            out.append(rec)
            print(rec, flush=True)
    total = sum(r["read"] + r["failed"] for r in out)  # type: ignore[operator]
    return {"limit_per_file": limit, "files": out, "certificates_attempted": total,
            "certificates_read": sum(r["read"] for r in out)}  # type: ignore[misc]  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--only", default="", help="comma-separated file numbers (1-5); default all")
    args = ap.parse_args()
    only = {int(x) for x in args.only.split(",") if x}
    result = asyncio.run(run(args.limit, only))
    (
        RESULTS
        / ("extraction_real_calibration.json" if not only else "extraction_real_partial.json")
    ).write_text(json.dumps(result, indent=2))
    print(
        f"\nread {result['certificates_read']} of {result['certificates_attempted']} certificates"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
