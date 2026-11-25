"""Database access for documents, extractions and certificates."""

import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import distinct_on, insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from qacopilot.config import Settings
from qacopilot.extraction.checks import (
    certificate_warnings,
    coa_problems,
    spec_problems,
    ungrounded_fields,
)
from qacopilot.extraction.schemas import CalibrationCert, CoA, SupplierSpec
from qacopilot.extraction.service import Extracted
from qacopilot.ingest.factspec import parse_fact_spec
from qacopilot.ingest.models import POOR_OCR_CONFIDENCE, DocType, PageText, ParsedDocument
from qacopilot.repositories.models import (
    CalibrationCertRow,
    CertificateReview,
    Document,
    DocumentPage,
    Extraction,
    ExtractionJob,
    FactSpecRow,
    NameAlias,
    NameDecision,
    SpecPackaging,
    SpecVersion,
)


def make_engine(settings: Settings | None = None) -> AsyncEngine:
    url = (settings or Settings()).database_url
    if not url:
        raise RuntimeError("QACOPILOT_DATABASE_URL is not set")
    return create_async_engine(url, pool_pre_ping=True)


class CertificateStatus(BaseModel):
    """A calibration certificate's due state, relative to ``today`` (decided in code)."""

    doc_id: str
    start_page: int = 1  # which certificate in a bundle PDF
    instrument_serial: str
    instrument_description: str | None
    certificate_number: str | None
    calibration_date: date
    next_due: date
    days_left: int  # negative = overdue
    status: Literal["overdue", "due_soon"]
    # Read from a poor scan (or with odd dates) and not yet confirmed by a person.
    unverified: bool = False
    review_reason: str | None = None


class OlderCertificate(BaseModel):
    """An earlier certificate for the same instrument, superseded by a later calibration."""

    certificate_number: str | None
    calibration_date: date
    next_due: date | None


class InstrumentCertificate(BaseModel):
    """An instrument's *latest* certificate (by calibration date), with where it came from."""

    doc_id: str
    filename: str
    start_page: int = 1
    instrument_serial: str
    instrument_description: str | None
    certificate_number: str | None
    calibration_date: date
    next_due: date | None
    days_left: int | None  # None when the certificate states no due date; negative = overdue
    older: list[OlderCertificate] = []
    unverified: bool = False  # see CertificateStatus
    review_reason: str | None = None


class ReviewItem(BaseModel):
    """A certificate reading that a person should confirm or correct."""

    doc_id: str
    filename: str
    start_page: int
    end_page: int | None
    instrument_serial: str
    instrument_description: str | None
    certificate_number: str | None
    calibration_date: date
    next_due: date | None
    needs_review: bool
    reason: str | None
    min_ocr_confidence: float | None  # lowest page confidence in the certificate, if OCR'd
    verified_by: str | None = None
    verified_at: datetime | None = None


class DocReviewItem(BaseModel):
    """A report or spec reading that looks incomplete or misread and that nobody has confirmed."""

    doc_id: str
    filename: str
    doc_type: str
    start_page: int
    end_page: int | None
    reason: str | None
    fields: dict[str, str | None]  # the scalar fields a person may correct (results are not)
    min_ocr_confidence: float | None
    verified_by: str | None = None


class PackagingUse(BaseModel):
    """A packaging code and how many current specs use it."""

    code: str
    specs: int
    description: str | None = None  # the most common description given for this code
    supplier: str | None = None
    section: str | None = None  # primary / secondary / tertiary, when asked per product group


class SpecUsing(BaseModel):
    """A current Fact Spec that uses a packaging code (or the spec a packaging list belongs to)."""

    doc_id: str
    filename: str
    nav_code: str | None
    product: str | None
    customer: str | None
    version: int | None
    section: str | None = None
    qty: str | None = None
    page: int = 1


class SpecPackagingRow(BaseModel):
    code: str
    section: str
    qty: str
    description: str
    supplier: str
    dimension: str
    weight: str
    page: int


class SpecVocab(BaseModel):
    """What a question can name, read from the current specs."""

    customers: list[str]
    suppliers: list[str]
    product_words: list[str]
    codes: list[str]
    people: list[str] = []  # who appears in the version histories
    # kind -> {spelling as typed or printed, normalised -> the one name used for it}
    aliases: dict[str, dict[str, str]] = {}


class SpecVersionLine(BaseModel):
    version: int | None
    person: str
    pages: str
    amends: str
    changed_on: date | None


class SpecDetails(BaseModel):
    """What a Fact Spec says about the product (see ``parse_fact_spec``)."""

    doc_id: str
    filename: str
    nav_code: str | None
    product: str | None
    weight: str | None
    weight_type: str | None
    storage: str | None
    temp_min_c: float | None
    temp_max_c: float | None
    max_life_text: str | None
    max_life_days: int | None
    life_basis: str | None
    life_notes: str | None
    min_life_text: str | None
    min_life_days: int | None
    pallet: dict[str, Any] | None
    pallet_type: str | None
    metal_detection: dict[str, Any] | None
    metal_detection_text: str | None
    ferrous_mm: float | None
    non_ferrous_mm: float | None
    stainless_mm: float | None
    pack_tare_g: float | None
    case_tare_g: float | None
    product_weight: str | None
    process_steps: list[str] | None
    raw_materials: list[dict[str, Any]] | None
    history: list[SpecVersionLine] = []


class SpecIndexEntry(BaseModel):
    """What is needed to recognise which current Fact Spec a question is about."""

    doc_id: str
    filename: str
    nav_codes: list[str]
    product: str | None
    customer: str | None


class SpecHeader(BaseModel):
    doc_id: str
    filename: str
    nav_code: str | None
    plu: str | None
    version: int | None
    version_date: date | None
    brand: str | None
    site: str | None
    season: str | None
    product: str | None
    spec_type: str | None
    customer: str | None
    certification: str | None
    allergens: str | None
    status: str | None


class ReviewEvent(BaseModel):
    reviewed_by: str
    reviewed_at: datetime
    action: Literal["confirm", "correct"]
    before: dict[str, Any]
    after: dict[str, Any]
    note: str | None = None


# Scalar fields a person may correct on a report or spec. The list of results or limits is not
# editable here: a missing row means running the extraction again or reading the original.
CORRECTABLE_DOC: dict[str, tuple[str, ...]] = {
    DocType.COA.value: ("product", "batch", "report_date", "lab"),
    DocType.SUPPLIER_SPEC.value: ("supplier", "product", "product_code", "shelf_life", "storage"),
}


# Fields a person may correct (the same ones the typed table and the extraction hold).
CORRECTABLE = (
    "instrument_serial",
    "instrument_description",
    "certificate_number",
    "calibration_date",
    "next_due",
)


class ExtractionRecord(BaseModel):
    """A stored extraction with the document it came from (CoAs and specs, for pairing)."""

    doc_id: str
    filename: str
    data: dict[str, Any]
    text: str = ""  # leading page text, for details the schema does not hold (revision, dates)
    ingested_at: datetime
    ocr_pages: int = 0


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    classified_by: str  # rules | llm | user
    alternative_type: str | None = None  # the LLM suggested this instead: a person should check
    pages: int
    ocr_pages: int
    poor_ocr_pages: int = 0  # OCR'd pages read with low confidence: figures may be misread
    chunks: int
    ingested_at: datetime
    extracted: bool  # whether structured fields have been extracted yet
    extraction_status: str | None = None  # running | done | failed, if a job has run
    extraction_done: int = 0
    extraction_total: int = 0


class DocumentRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    def session(self) -> AsyncSession:
        return self._sessions()

    async def save_document(
        self,
        doc_id: str,
        filename: str,
        doc_type: DocType,
        doc: ParsedDocument,
        chunks: int,
        classified_by: str = "rules",
        alternative: DocType | None = None,
    ) -> None:
        """Insert or replace a document and its page text. Re-ingesting the same file is safe."""
        row = {
            "doc_id": doc_id,
            "filename": filename,
            "doc_type": doc_type.value,
            "classified_by": classified_by,
            "alternative_type": alternative.value if alternative else None,
            "pages": len(doc.pages),
            "ocr_pages": sum(1 for p in doc.pages if p.method == "ocr"),
            "chunks": chunks,
        }
        async with self._sessions.begin() as s:
            stmt = insert(Document).values(row)
            await s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["doc_id"], set_={k: v for k, v in row.items() if k != "doc_id"}
                )
            )
            await s.execute(delete(DocumentPage).where(DocumentPage.doc_id == doc_id))
            s.add_all(
                DocumentPage(
                    doc_id=doc_id,
                    page=p.page,
                    text=p.text,
                    method=p.method,
                    ocr_confidence=p.confidence,
                )
                for p in doc.pages
            )
            await self._write_fact_spec(
                s, doc_id, [p.text for p in doc.pages] if doc_type is DocType.FACT_SPEC else None
            )

    async def get_document(self, doc_id: str) -> Document | None:
        async with self._sessions() as s:
            return await s.get(Document, doc_id)

    async def load_parsed(self, doc_id: str) -> ParsedDocument | None:
        async with self._sessions() as s:
            doc = await s.get(Document, doc_id)
            if doc is None:
                return None
            rows = await s.scalars(
                select(DocumentPage)
                .where(DocumentPage.doc_id == doc_id)
                .order_by(DocumentPage.page)
            )
            pages = [
                PageText(
                    page=r.page,
                    text=r.text,
                    method=r.method,  # type: ignore[arg-type]
                    confidence=r.ocr_confidence,
                )
                for r in rows
            ]
            return ParsedDocument(source=doc.filename, pages=pages)

    async def save_extraction(
        self,
        doc_id: str,
        doc_type: DocType,
        result: Extracted,
        *,
        prompt_version: str,
        model: str,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> None:
        """Store one extraction (one certificate of a bundle, or the whole document).

        A calibration certificate also gets its typed row. Saving again for the same
        ``(doc_id, start_page)`` replaces it.
        """
        data = result.model_dump(mode="json")
        values = {
            "doc_id": doc_id,
            "start_page": start_page,
            "end_page": end_page,
            "doc_type": doc_type.value,
            "data": data,
            "prompt_version": prompt_version,
            "model": model,
        }
        async with self._sessions.begin() as s:
            previous_ext = await s.get(Extraction, (doc_id, start_page))
            if isinstance(result, CoA | SupplierSpec):
                flag, why = await self._document_review_state(
                    s, doc_id, start_page, end_page, result
                )
                values.update(self._carry_confirmation(previous_ext, data, flag, why))
            stmt = insert(Extraction).values(values)
            await s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["doc_id", "start_page"],
                    set_={
                        **{k: v for k, v in values.items() if k not in ("doc_id", "start_page")},
                        "extracted_at": func.now(),
                    },
                )
            )
            previous = await s.get(CalibrationCertRow, (doc_id, start_page))
            await s.execute(
                delete(CalibrationCertRow).where(
                    CalibrationCertRow.doc_id == doc_id, CalibrationCertRow.start_page == start_page
                )
            )
            if isinstance(result, CalibrationCert):
                needs_review, reason = await self._review_state(
                    s, doc_id, start_page, end_page, result
                )
                verified_by = verified_at = None
                if previous is not None and previous.verified_at is not None:
                    same = (
                        previous.instrument_serial,
                        previous.calibration_date,
                        previous.next_due,
                    ) == (
                        result.instrument_serial.strip(),
                        result.calibration_date,
                        result.next_due,
                    )
                    if same:  # a person already confirmed exactly this reading: keep that
                        needs_review, reason = False, None
                        verified_by, verified_at = previous.verified_by, previous.verified_at
                    else:  # never silently drop a person's confirmation: ask again, and say why
                        needs_review = True
                        reason = "; ".join(
                            r
                            for r in (
                                reason,
                                f"re-read, and it differs from what {previous.verified_by} "
                                "confirmed",
                            )
                            if r
                        )
                s.add(
                    CalibrationCertRow(
                        doc_id=doc_id,
                        start_page=start_page,
                        end_page=end_page,
                        instrument_serial=result.instrument_serial.strip(),
                        instrument_description=result.instrument_description,
                        certificate_number=result.certificate_number,
                        calibrated_by=result.calibrated_by,
                        calibration_date=result.calibration_date,
                        next_due=result.next_due,
                        result=result.result,
                        needs_review=needs_review,
                        review_reason=reason,
                        verified_by=verified_by,
                        verified_at=verified_at,
                    )
                )

    @staticmethod
    async def _pages(
        s: AsyncSession, doc_id: str, start_page: int, end_page: int | None
    ) -> list[tuple[str, float | None]]:
        """(text, OCR confidence) of each page of a certificate."""
        stmt = (
            select(DocumentPage.text, DocumentPage.ocr_confidence)
            .where(DocumentPage.doc_id == doc_id, DocumentPage.page >= start_page)
            .order_by(DocumentPage.page)
        )
        if end_page is not None:
            stmt = stmt.where(DocumentPage.page <= end_page)
        return [(t, c) for t, c in (await s.execute(stmt)).all()]

    @staticmethod
    async def _page_confidences(
        s: AsyncSession, doc_id: str, start_page: int, end_page: int | None
    ) -> list[float | None]:
        return [c for _, c in await DocumentRepository._pages(s, doc_id, start_page, end_page)]

    @staticmethod
    def _carry_confirmation(
        previous: "Extraction | None", data: dict[str, Any], needs: bool, why: str | None
    ) -> dict[str, Any]:
        """Review columns for a new reading of a report or spec. A person's confirmation stays only
        if the new reading is identical; otherwise it is asked again and says why."""
        out: dict[str, Any] = {
            "needs_review": needs,
            "review_reason": why,
            "verified_by": None,
            "verified_at": None,
        }
        if previous is not None and previous.verified_at is not None:
            if previous.data == data:
                out.update(
                    needs_review=False,
                    review_reason=None,
                    verified_by=previous.verified_by,
                    verified_at=previous.verified_at,
                )
            else:
                differs = f"re-read, and it differs from what {previous.verified_by} confirmed"
                out.update(
                    needs_review=True, review_reason="; ".join(r for r in (why, differs) if r)
                )
        return out

    async def _document_review_state(
        self,
        s: AsyncSession,
        doc_id: str,
        start_page: int,
        end_page: int | None,
        result: CoA | SupplierSpec,
    ) -> tuple[bool, str | None]:
        """Whether a report or spec reading needs a person, and why."""
        pages = await self._pages(s, doc_id, start_page, end_page)
        text = "\n".join(t for t, _ in pages)
        reasons: list[str] = []
        poor = sum(1 for _, c in pages if c is not None and c < POOR_OCR_CONFIDENCE)
        if poor:
            reasons.append(f"{poor} page(s) are a poor scan, so the values may be misread")
        reasons.extend(
            coa_problems(result, text) if isinstance(result, CoA) else spec_problems(result, text)
        )
        return bool(reasons), "; ".join(reasons) or None

    async def _review_state(
        self,
        s: AsyncSession,
        doc_id: str,
        start_page: int,
        end_page: int | None,
        cert: CalibrationCert,
    ) -> tuple[bool, str | None]:
        """Whether a reading needs a person, and why: poor scan pages or implausible dates."""
        reasons: list[str] = []
        pages = await self._pages(s, doc_id, start_page, end_page)
        poor = sum(1 for _, c in pages if c is not None and c < POOR_OCR_CONFIDENCE)
        if poor:
            reasons.append(f"{poor} page(s) are a poor scan, so the values may be misread")
        reasons.extend(certificate_warnings(cert))
        reasons.extend(ungrounded_fields(cert, "\n".join(t for t, _ in pages)))
        return bool(reasons), "; ".join(reasons) or None

    async def prune_extractions(self, doc_id: str, keep_start_pages: list[int]) -> None:
        """Drop rows for certificates that a re-run no longer finds (e.g. a changed split)."""
        async with self._sessions.begin() as s:
            await s.execute(
                delete(Extraction).where(
                    Extraction.doc_id == doc_id, Extraction.start_page.not_in(keep_start_pages)
                )
            )
            await s.execute(
                delete(CalibrationCertRow).where(
                    CalibrationCertRow.doc_id == doc_id,
                    CalibrationCertRow.start_page.not_in(keep_start_pages),
                )
            )

    async def clear_extractions(self, doc_id: str) -> None:
        """Forget every extraction of a document (before re-running it)."""
        async with self._sessions.begin() as s:
            await s.execute(delete(Extraction).where(Extraction.doc_id == doc_id))
            await s.execute(delete(CalibrationCertRow).where(CalibrationCertRow.doc_id == doc_id))

    async def get_extraction(self, doc_id: str) -> Extraction | None:
        """The first extraction of a document (the only one, unless it is a bundle)."""
        async with self._sessions() as s:
            return await s.scalar(
                select(Extraction)
                .where(Extraction.doc_id == doc_id)
                .order_by(Extraction.start_page)
                .limit(1)
            )

    async def get_extractions(self, doc_id: str) -> list[Extraction]:
        async with self._sessions() as s:
            rows = await s.scalars(
                select(Extraction)
                .where(Extraction.doc_id == doc_id)
                .order_by(Extraction.start_page)
            )
            return list(rows)

    # --- background extraction jobs -------------------------------------------------------

    async def start_job(self, doc_id: str, total: int) -> None:
        values = {"doc_id": doc_id, "status": "running", "total": total, "done": 0, "failed": 0}
        async with self._sessions.begin() as s:
            stmt = insert(ExtractionJob).values(values)
            await s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["doc_id"],
                    set_={**values, "error": None, "started_at": func.now(), "finished_at": None},
                )
            )

    async def job_progress(self, doc_id: str, *, done: int, failed: int) -> None:
        async with self._sessions.begin() as s:
            await s.execute(
                update(ExtractionJob)
                .where(ExtractionJob.doc_id == doc_id)
                .values(done=done, failed=failed)
            )

    async def finish_job(self, doc_id: str, status: str, error: str | None = None) -> None:
        async with self._sessions.begin() as s:
            await s.execute(
                update(ExtractionJob)
                .where(ExtractionJob.doc_id == doc_id)
                .values(status=status, error=error, finished_at=func.now())
            )

    async def get_job(self, doc_id: str) -> ExtractionJob | None:
        async with self._sessions() as s:
            return await s.get(ExtractionJob, doc_id)

    async def fail_stale_jobs(self) -> int:
        """At startup: a job still 'running' belongs to a process that no longer exists."""
        async with self._sessions.begin() as s:
            result = await s.execute(
                update(ExtractionJob)
                .where(ExtractionJob.status == "running")
                .values(status="failed", error="interrupted by a restart", finished_at=func.now())
            )
            return int(result.rowcount)  # type: ignore[attr-defined]

    async def certificates_due(
        self, within_days: int, *, today: date | None = None
    ) -> list[CertificateStatus]:
        """Instruments overdue or due within ``within_days``, soonest first.

        Only each instrument's *latest* certificate counts: if a probe was recalibrated, the old
        certificate's due date must not raise an alert.
        """
        today = today or date.today()
        latest = (
            select(CalibrationCertRow)
            .ext(distinct_on(CalibrationCertRow.instrument_serial))
            .order_by(
                CalibrationCertRow.instrument_serial,
                CalibrationCertRow.calibration_date.desc(),
                CalibrationCertRow.doc_id,
                CalibrationCertRow.start_page,
            )
            .subquery()
        )
        stmt = (
            select(latest)
            .where(
                latest.c.next_due.is_not(None),
                latest.c.next_due <= today + timedelta(days=within_days),
            )
            .order_by(latest.c.next_due, latest.c.instrument_serial)
        )
        out: list[CertificateStatus] = []
        async with self._sessions() as s:
            for r in (await s.execute(stmt)).mappings():
                days_left = (r["next_due"] - today).days
                out.append(
                    CertificateStatus(
                        doc_id=r["doc_id"],
                        start_page=r["start_page"],
                        instrument_serial=r["instrument_serial"],
                        instrument_description=r["instrument_description"],
                        certificate_number=r["certificate_number"],
                        calibration_date=r["calibration_date"],
                        next_due=r["next_due"],
                        days_left=days_left,
                        status="overdue" if days_left < 0 else "due_soon",
                        unverified=bool(r["needs_review"]),
                        review_reason=r["review_reason"],
                    )
                )
        return out

    def _item(
        self, row: CalibrationCertRow, filename: str, confidences: list[float | None]
    ) -> ReviewItem:
        known = [c for c in confidences if c is not None]
        return ReviewItem(
            doc_id=row.doc_id,
            filename=filename,
            start_page=row.start_page,
            end_page=row.end_page,
            instrument_serial=row.instrument_serial,
            instrument_description=row.instrument_description,
            certificate_number=row.certificate_number,
            calibration_date=row.calibration_date,
            next_due=row.next_due,
            needs_review=row.needs_review,
            reason=row.review_reason,
            min_ocr_confidence=round(min(known), 1) if known else None,
            verified_by=row.verified_by,
            verified_at=row.verified_at,
        )

    async def certificates_to_review(self) -> list[ReviewItem]:
        """Readings waiting for a person, oldest document first."""
        stmt = (
            select(CalibrationCertRow, Document.filename)
            .join(Document, Document.doc_id == CalibrationCertRow.doc_id)
            .where(CalibrationCertRow.needs_review.is_(True))
            .order_by(
                Document.ingested_at, CalibrationCertRow.doc_id, CalibrationCertRow.start_page
            )
        )
        async with self._sessions() as s:
            rows = (await s.execute(stmt)).all()
            return [
                self._item(
                    row,
                    filename,
                    await self._page_confidences(s, row.doc_id, row.start_page, row.end_page),
                )
                for row, filename in rows
            ]

    async def review_certificate(
        self,
        doc_id: str,
        start_page: int,
        user: str,
        *,
        corrections: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> ReviewItem | None:
        """A person confirms a reading as it is, or corrects some fields and confirms the result.

        The corrected values replace the stored extraction too, so nothing downstream sees the
        misread ones. The values before and after are kept in ``certificate_reviews``. Raises
        ``ValueError`` if the corrected values are not a valid certificate; None if there is no
        such certificate.
        """
        unknown = set(corrections or {}) - set(CORRECTABLE)
        if unknown:
            raise ValueError(f"These fields cannot be corrected: {', '.join(sorted(unknown))}")
        async with self._sessions.begin() as s:
            row = await s.get(CalibrationCertRow, (doc_id, start_page))
            ext = await s.get(Extraction, (doc_id, start_page))
            doc = await s.get(Document, doc_id)
            if row is None or ext is None or doc is None:
                return None
            fields = {k: (ext.data.get(k)) for k in CORRECTABLE}
            merged = {**ext.data, **(corrections or {})}
            try:
                cert = CalibrationCert.model_validate(merged)
            except Exception as exc:  # pydantic's message names the bad field
                raise ValueError(f"Not a valid certificate: {exc}") from exc
            after_fields = {k: cert.model_dump(mode="json").get(k) for k in CORRECTABLE}
            changed = after_fields != fields
            if changed:
                ext.data = cert.model_dump(mode="json")  # reassigned, so JSONB change is seen
                row.instrument_serial = cert.instrument_serial.strip()
                row.instrument_description = cert.instrument_description
                row.certificate_number = cert.certificate_number
                row.calibration_date = cert.calibration_date
                row.next_due = cert.next_due
            row.needs_review = False
            row.review_reason = None
            row.verified_by = user
            row.verified_at = datetime.now(UTC)
            s.add(
                CertificateReview(
                    doc_id=doc_id,
                    start_page=start_page,
                    reviewed_by=user,
                    action="correct" if changed else "confirm",
                    before=fields,
                    after=after_fields,
                    note=note,
                )
            )
            confidences = await self._page_confidences(s, doc_id, start_page, row.end_page)
            return self._item(row, doc.filename, confidences)

    @staticmethod
    async def _write_fact_spec(s: AsyncSession, doc_id: str, pages: list[str] | None) -> None:
        """Replace a document's Fact Spec tables: filled for a Fact Spec, emptied otherwise."""
        await s.execute(delete(SpecPackaging).where(SpecPackaging.doc_id == doc_id))
        await s.execute(delete(SpecVersion).where(SpecVersion.doc_id == doc_id))
        await s.execute(delete(FactSpecRow).where(FactSpecRow.doc_id == doc_id))
        if pages is None:
            return
        spec = parse_fact_spec(pages)
        s.add(
            FactSpecRow(
                doc_id=doc_id,
                nav_code=spec.nav_code,
                plu=spec.plu,
                version=spec.version,
                version_date=spec.version_date,
                brand=spec.brand,
                site=spec.site,
                season=spec.season,
                product=spec.product,
                spec_type=spec.spec_type,
                customer=spec.customer,
                certification=spec.certification,
                allergens=spec.allergens,
                status=spec.status,
                weight=spec.weight,
                weight_type=spec.weight_type,
                storage=spec.storage,
                temp_min_c=spec.temp_min_c,
                temp_max_c=spec.temp_max_c,
                max_life_text=spec.max_life_text,
                max_life_days=spec.max_life_days,
                life_basis=spec.life_basis,
                life_notes=spec.life_notes,
                min_life_text=spec.min_life_text,
                min_life_days=spec.min_life_days,
                boxes_per_layer=spec.boxes_per_layer,
                layers_per_pallet=spec.layers_per_pallet,
                cases_per_pallet=spec.cases_per_pallet,
                pallet_type=spec.pallet_type,
                pallet=spec.pallet or None,
                metal_detection=spec.metal_detection or None,
                metal_detection_text=spec.metal_detection_text,
                ferrous_mm=spec.ferrous_mm,
                non_ferrous_mm=spec.non_ferrous_mm,
                stainless_mm=spec.stainless_mm,
                pack_tare_g=spec.pack_tare_g,
                case_tare_g=spec.case_tare_g,
                product_weight=spec.product_weight,
                process_steps=spec.process_steps or None,
                raw_materials=spec.raw_materials or None,
            )
        )
        s.add_all(
            SpecVersion(
                doc_id=doc_id,
                version=v.version,
                person=v.person[:128],
                pages=v.pages[:64],
                amends=v.amends,
                changed_on=v.changed_on,
            )
            for v in spec.history
        )
        s.add_all(
            SpecPackaging(
                doc_id=doc_id,
                section=r.section,
                code=r.code,
                qty=r.qty,
                description=r.description,
                supplier=r.supplier,
                dimension=r.dimension,
                weight=r.weight,
                cells=r.cells,
                page=r.page,
            )
            for r in spec.packaging
        )

    # --- Fact Spec packaging ----------------------------------------------------------------
    # "Current" = the latest version of each product (by NAV code): an old version's packaging
    # must not count as in use.

    _CURRENT = """
        current AS (
            SELECT DISTINCT ON (coalesce(f.nav_code, f.doc_id)) f.doc_id
            FROM fact_specs f
            ORDER BY coalesce(f.nav_code, f.doc_id), f.version DESC NULLS LAST,
                     f.version_date DESC NULLS LAST, f.doc_id
        )"""

    async def fact_spec_count(self) -> int:
        async with self._sessions() as s:
            return int(
                await s.scalar(text(f"WITH {self._CURRENT} SELECT count(*) FROM current")) or 0
            )

    async def packaging_usage(
        self, *, limit: int = 5, section: str | None = None
    ) -> list[PackagingUse]:
        """Packaging codes by the number of current specs that use them, most used first."""
        stmt = text(
            f"""
            WITH {self._CURRENT},
            usage AS (
                SELECT p.code, count(DISTINCT p.doc_id) AS specs
                FROM spec_packaging p JOIN current c ON c.doc_id = p.doc_id
                WHERE (CAST(:section AS text) IS NULL OR p.section = :section)
                GROUP BY p.code
            )
            SELECT u.code, u.specs,
              (SELECT p.description FROM spec_packaging p
                WHERE p.code = u.code AND p.description <> ''
                GROUP BY p.description ORDER BY count(*) DESC, p.description LIMIT 1),
              (SELECT p.supplier FROM spec_packaging p
                WHERE p.code = u.code AND p.supplier <> ''
                GROUP BY p.supplier ORDER BY count(*) DESC, p.supplier LIMIT 1)
            FROM usage u ORDER BY u.specs DESC, u.code LIMIT :limit"""
        )
        async with self._sessions() as s:
            rows = (await s.execute(stmt, {"limit": limit, "section": section})).all()
        return [PackagingUse(code=r[0], specs=r[1], description=r[2], supplier=r[3]) for r in rows]

    async def packaging_code(self, code: str) -> PackagingUse | None:
        """One code: its usage and its most common description and supplier."""
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT p.code, count(DISTINCT p.doc_id),
              (SELECT x.description FROM spec_packaging x
                WHERE x.code = p.code AND x.description <> ''
                GROUP BY x.description ORDER BY count(*) DESC, x.description LIMIT 1),
              (SELECT x.supplier FROM spec_packaging x
                WHERE x.code = p.code AND x.supplier <> ''
                GROUP BY x.supplier ORDER BY count(*) DESC, x.supplier LIMIT 1)
            FROM spec_packaging p
            WHERE p.code = :code AND p.doc_id IN (SELECT doc_id FROM current)
            GROUP BY p.code"""
        )
        async with self._sessions() as s:
            row = (await s.execute(stmt, {"code": code})).first()
        return (
            None
            if row is None
            else PackagingUse(code=row[0], specs=row[1], description=row[2], supplier=row[3])
        )

    async def specs_using_code(self, code: str, *, limit: int = 200) -> list[SpecUsing]:
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT DISTINCT ON (f.doc_id) f.doc_id, d.filename, f.nav_code, f.product, f.customer,
                   f.version, p.section, p.qty, p.page
            FROM spec_packaging p
              JOIN current c ON c.doc_id = p.doc_id
              JOIN fact_specs f ON f.doc_id = p.doc_id
              JOIN documents d ON d.doc_id = p.doc_id
            WHERE p.code = :code
            ORDER BY f.doc_id, p.id LIMIT :limit"""
        )
        async with self._sessions() as s:
            rows = (await s.execute(stmt, {"code": code, "limit": limit})).all()
        out = [
            SpecUsing(
                doc_id=r[0],
                filename=r[1],
                nav_code=r[2],
                product=r[3],
                customer=r[4],
                version=r[5],
                section=r[6],
                qty=r[7],
                page=r[8],
            )
            for r in rows
        ]
        return sorted(out, key=lambda x: (x.nav_code or "", x.filename))

    async def find_fact_specs(
        self, *, nav_code: str | None = None, name: str | None = None
    ) -> list[SpecUsing]:
        """Current Fact Specs by NAV code or by words of the product name."""
        words = [w for w in re.findall(r"[A-Za-z0-9]+", name or "") if len(w) > 2][:6]
        conds = ["TRUE"]
        params: dict[str, Any] = {}
        if nav_code:
            conds.append("f.nav_code = :nav")
            params["nav"] = nav_code
        for i, w in enumerate(words):
            conds.append(f"(f.product ILIKE :w{i} OR d.filename ILIKE :w{i})")
            params[f"w{i}"] = f"%{w}%"
        if not nav_code and not words:
            return []
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT f.doc_id, d.filename, f.nav_code, f.product, f.customer, f.version
            FROM fact_specs f JOIN current c ON c.doc_id = f.doc_id
              JOIN documents d ON d.doc_id = f.doc_id
            WHERE {" AND ".join(conds)} ORDER BY f.nav_code, d.filename LIMIT 25"""
        )
        async with self._sessions() as s:
            rows = (await s.execute(stmt, params)).all()
        return [
            SpecUsing(
                doc_id=r[0], filename=r[1], nav_code=r[2], product=r[3], customer=r[4], version=r[5]
            )
            for r in rows
        ]

    async def packaging_code_details(self, code: str) -> dict[str, str]:
        """The most common description, supplier, dimension and weight given for a code."""
        out: dict[str, str] = {}
        for field in ("description", "supplier", "dimension", "weight"):
            rows = await self.query_rows(
                f"SELECT {field} FROM spec_packaging WHERE code = :code AND {field} <> '' "
                f"GROUP BY {field} ORDER BY count(*) DESC, {field} LIMIT 1",
                {"code": code},
            )
            out[field] = rows[0][0] if rows else ""
        return out

    # --- name aliases -----------------------------------------------------------------------

    async def set_alias(
        self, kind: str, raw: str, canonical: str, *, by: str | None = None, note: str | None = None
    ) -> None:
        """Treat the spelling ``raw`` as the name ``canonical`` from now on."""
        key = raw.strip().lower()
        async with self._sessions.begin() as s:
            stmt = insert(NameAlias).values(
                kind=kind, raw_key=key, canonical=canonical.strip(), created_by=by, note=note
            )
            await s.execute(
                stmt.on_conflict_do_update(
                    index_elements=["kind", "raw_key"],
                    set_={"canonical": stmt.excluded.canonical, "note": stmt.excluded.note},
                )
            )

    async def resolve_name(self, kind: str, name: str) -> str:
        """The name a spelling stands for ("Sainsburys" -> "Sainsbury's"), or the name itself.

        Case, apostrophes, spaces and a trailing s do not matter when looking a spelling up.
        """
        key = re.sub(r"[^a-z0-9]+", "", name.lower().replace("\u2019", "").replace("'", ""))
        rows = await self.query_rows(
            "SELECT canonical FROM name_aliases WHERE kind = :kind AND "
            "regexp_replace(raw_key, '[^a-z0-9]+', '', 'g') IN (:a, :b) LIMIT 1",
            {"kind": kind, "a": key, "b": key.rstrip("s")},
        )
        return str(rows[0][0]) if rows else name

    async def remove_alias(self, kind: str, raw: str) -> bool:
        async with self._sessions.begin() as s:
            result = await s.execute(
                delete(NameAlias).where(
                    NameAlias.kind == kind, NameAlias.raw_key == raw.strip().lower()
                )
            )
            return bool(result.rowcount)  # type: ignore[attr-defined]

    async def list_aliases(self) -> list[tuple[str, str, str]]:
        """(kind, spelling, canonical name) of every alias."""
        rows = await self.query_rows(
            "SELECT kind, raw_key, canonical FROM name_aliases ORDER BY kind, canonical, raw_key"
        )
        return [(r[0], r[1], r[2]) for r in rows]

    async def keep_separate(self, kind: str, name: str, *, by: str | None = None) -> None:
        """Record that ``name`` is its own name: it is not suggested as an alias again."""
        async with self._sessions.begin() as s:
            stmt = insert(NameDecision).values(
                kind=kind, name_key=name.strip().lower(), decision="separate", decided_by=by
            )
            await s.execute(stmt.on_conflict_do_nothing(index_elements=["kind", "name_key"]))

    async def decided_names(self) -> set[tuple[str, str]]:
        """(kind, lower-cased name) of every name a person confirmed is separate."""
        rows = await self.query_rows("SELECT kind, name_key FROM name_decisions")
        return {(r[0], r[1]) for r in rows}

    async def name_counts(self, kind: str) -> dict[str, int]:
        """Every spelling of a name as printed in the current specs, and how many specs use it."""
        column, table, join = {
            "customer": ("f.customer", "fact_specs f", "f.doc_id"),
            "supplier": ("p.supplier", "spec_packaging p", "p.doc_id"),
            "person": ("v.person", "spec_versions v", "v.doc_id"),
        }[kind]
        rows = await self.query_rows(
            f"WITH {self._CURRENT} SELECT {column}, count(DISTINCT {join}) FROM {table} "
            f"JOIN current c ON c.doc_id = {join} WHERE {column} IS NOT NULL AND {column} <> '' "
            f"GROUP BY {column}"
        )
        return {r[0]: r[1] for r in rows}

    async def spec_index(self) -> list[SpecIndexEntry]:
        """Every current Fact Spec, for matching a question to one of them."""
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT f.doc_id, d.filename, f.nav_code, f.product, f.customer
            FROM fact_specs f JOIN current c ON c.doc_id = f.doc_id
              JOIN documents d ON d.doc_id = f.doc_id"""
        )
        async with self._sessions() as s:
            rows = (await s.execute(stmt)).all()
        return [
            SpecIndexEntry(
                doc_id=r[0],
                filename=r[1],
                nav_codes=re.findall(r"\d{7}", r[2] or ""),
                product=r[3],
                customer=r[4],
            )
            for r in rows
        ]

    async def spec_details(self, doc_id: str) -> SpecDetails | None:
        """The fields read from a spec's text, with its version history (oldest first)."""
        async with self._sessions() as s:
            row = await s.get(FactSpecRow, doc_id)
            doc = await s.get(Document, doc_id)
            if row is None or doc is None:
                return None
            history = list(
                await s.scalars(
                    select(SpecVersion)
                    .where(SpecVersion.doc_id == doc_id)
                    .order_by(SpecVersion.changed_on.nulls_first(), SpecVersion.id)
                )
            )
        fields = {k: getattr(row, k) for k in SpecDetails.model_fields if hasattr(row, k)}
        return SpecDetails(
            **{**fields, "doc_id": doc_id, "filename": doc.filename},
            history=[
                SpecVersionLine(
                    version=h.version,
                    person=h.person,
                    pages=h.pages,
                    amends=h.amends,
                    changed_on=h.changed_on,
                )
                for h in history
            ],
        )

    async def spec_header(self, doc_id: str) -> SpecHeader | None:
        async with self._sessions() as s:
            row = await s.get(FactSpecRow, doc_id)
            doc = await s.get(Document, doc_id)
        if row is None or doc is None:
            return None
        return SpecHeader(
            doc_id=doc_id,
            filename=doc.filename,
            nav_code=row.nav_code,
            plu=row.plu,
            version=row.version,
            version_date=row.version_date,
            brand=row.brand,
            site=row.site,
            season=row.season,
            product=row.product,
            spec_type=row.spec_type,
            customer=row.customer,
            certification=row.certification,
            allergens=row.allergens,
            status=row.status,
        )

    async def count_specs(
        self,
        *,
        spec_type: str | None = None,
        customer: str | None = None,
        has_allergens: bool = False,
        supplier: str | None = None,
        words: list[str] | None = None,
    ) -> int:
        """How many current specs match every given condition (``words``: a product group)."""
        conds: list[str] = ["TRUE"]
        params: dict[str, Any] = {}
        if customer:
            customer = await self.resolve_name("customer", customer)
        if words:
            group, group_params, _ = await self._group_condition(words)
            conds += group
            params.update(group_params)
        if spec_type:
            conds.append("f.spec_type ILIKE :stype")
            params["stype"] = f"{spec_type}%"
        if customer:
            conds.append(f"lower({self.canon('customer', 'f.customer')}) = lower(:customer)")
            params["customer"] = customer
        if has_allergens:
            conds.append(
                "f.allergens IS NOT NULL AND lower(f.allergens) NOT IN ('none', 'nil', 'n/a', '')"
            )
        if supplier:
            supplier = await self.resolve_name("supplier", supplier)
            conds.append(
                "EXISTS (SELECT 1 FROM spec_packaging p WHERE p.doc_id = f.doc_id "
                f"AND lower({self.canon('supplier', 'p.supplier')}) = lower(:supplier))"
            )
            params["supplier"] = supplier
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT count(*) FROM fact_specs f JOIN current c ON c.doc_id = f.doc_id
              JOIN documents d ON d.doc_id = f.doc_id
            WHERE {" AND ".join(conds)}"""
        )
        async with self._sessions() as s:
            return int(await s.scalar(stmt, params) or 0)

    async def customer_counts(self, limit: int = 1000) -> list[tuple[str, int]]:
        """Customers by number of current specs: aliases applied, case ignored."""
        cc = self.canon("customer", "f.customer")
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT min({cc}) AS name, count(*) AS n FROM fact_specs f
            JOIN current c ON c.doc_id = f.doc_id WHERE f.customer IS NOT NULL
            GROUP BY lower({cc}) ORDER BY n DESC, name LIMIT :limit"""
        )
        async with self._sessions() as s:
            return [(r[0], r[1]) for r in (await s.execute(stmt, {"limit": limit})).all()]

    async def supplier_counts(self, limit: int = 1000) -> list[tuple[str, int, int]]:
        """(packaging supplier, current specs using it, distinct codes), most specs first."""
        sc = self.canon("supplier", "p.supplier")
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT min({sc}) AS name, count(DISTINCT p.doc_id) AS specs, count(DISTINCT p.code)
            FROM spec_packaging p JOIN current c ON c.doc_id = p.doc_id
            WHERE p.supplier <> ''
            GROUP BY lower({sc}) ORDER BY specs DESC, name LIMIT :limit"""
        )
        async with self._sessions() as s:
            return [(r[0], r[1], r[2]) for r in (await s.execute(stmt, {"limit": limit})).all()]

    async def packaging_totals(self) -> tuple[int, int]:
        """(distinct packaging codes, distinct packaging suppliers) across current specs."""
        sc = self.canon("supplier", "p.supplier")
        stmt = text(
            f"""
            WITH {self._CURRENT}
            SELECT count(DISTINCT p.code),
                   count(DISTINCT lower({sc})) FILTER (WHERE p.supplier <> '')
            FROM spec_packaging p JOIN current c ON c.doc_id = p.doc_id"""
        )
        async with self._sessions() as s:
            row = (await s.execute(stmt)).one()
        return int(row[0]), int(row[1])

    @staticmethod
    def _phrase_regex(words: list[str]) -> str:
        """A Postgres regex for the words as one phrase: adjacent and in order, a plural on the
        last word allowed ("duck legs" finds "Duck Leg"), separated by spaces, hyphens or
        slashes. The words are letters and digits only (see ``product_words``)."""
        last = words[-1]
        if len(last) > 3 and last.lower().endswith("s") and not last.lower().endswith("ss"):
            last = last[:-1]
        body = r"[\s/-]+".join([*words[:-1], last])
        return rf"\m{body}s?\M"

    async def _group_condition(self, words: list[str]) -> tuple[list[str], dict[str, Any], str]:
        """SQL conditions (on ``f`` fact_specs and ``d`` documents) for a product group, and how
        they match: ``customer``, ``phrase`` or ``words`` (``none`` if nothing matches at all).

        - A phrase that IS a customer's name means that customer's specs ("Tesco").
        - Otherwise the words must appear together, in order, as a phrase ("whole duck" is not
          "whole aromatic ... duck").
        - Only when no spec has the phrase, every word must appear somewhere in the product name
          or file name; the caller says so.
        Customers are never matched by substring: "whole" once matched "Wholesale".
        """
        phrase = " ".join(words)
        if not phrase:
            return [], {}, "none"
        phrase = await self.resolve_name("customer", phrase)  # a spelling of a customer
        base = (
            f"WITH {self._CURRENT} SELECT 1 FROM fact_specs f "
            "JOIN current c ON c.doc_id = f.doc_id JOIN documents d ON d.doc_id = f.doc_id WHERE "
        )
        async with self._sessions() as s:
            if await s.scalar(
                text(
                    base + f"lower({self.canon('customer', 'f.customer')}) = lower(:phrase) LIMIT 1"
                ),
                {"phrase": phrase},
            ):
                return (
                    [f"lower({self.canon('customer', 'f.customer')}) = lower(:phrase)"],
                    {"phrase": phrase},
                    "customer",
                )
        return await self._product_condition(words)

    async def _product_condition(self, words: list[str]) -> tuple[list[str], dict[str, Any], str]:
        """Conditions for a product name: the words as one phrase, else (only if no spec has the
        phrase) all the words anywhere. Never matches a customer. Mode: phrase, words or none."""
        if not words:
            return [], {}, "none"
        base = (
            f"WITH {self._CURRENT} SELECT 1 FROM fact_specs f "
            "JOIN current c ON c.doc_id = f.doc_id JOIN documents d ON d.doc_id = f.doc_id WHERE "
        )
        regex = self._phrase_regex(words)
        async with self._sessions() as s:
            if await s.scalar(
                text(base + "(f.product ~* :rx OR d.filename ~* :rx) LIMIT 1"), {"rx": regex}
            ):
                return ["(f.product ~* :rx OR d.filename ~* :rx)"], {"rx": regex}, "phrase"
        conds = [f"(f.product ILIKE :w{i} OR d.filename ILIKE :w{i})" for i in range(len(words))]
        params = {f"w{i}": f"%{w}%" for i, w in enumerate(words)}
        async with self._sessions() as s:
            found = await s.scalar(text(base + " AND ".join(conds) + " LIMIT 1"), params)
        return conds, params, "words" if found else "none"

    CURRENT_SQL = _CURRENT

    @staticmethod
    def canon(kind: str, column: str) -> str:
        """SQL for a name with its alias applied: the canonical name if one is set, else as is."""
        return (
            f"coalesce((SELECT a.canonical FROM name_aliases a "
            f"WHERE a.kind = '{kind}' AND a.raw_key = lower({column})), {column})"
        )

    async def query_rows(self, sql: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Run a read-only query built by the spec query layer (only validated pieces go in)."""
        async with self._sessions() as s:
            return list((await s.execute(text(sql), params or {})).all())

    def vocab_key(self) -> int:
        """Identifies the database behind this repository (for caching its vocabulary)."""
        return id(self._sessions.kw["bind"])

    async def spec_vocab(self) -> "SpecVocab":
        """Customers, packaging suppliers, product words and packaging codes in use now: what a
        question can name. Names have their aliases applied, so one customer is one name."""
        cc, sc, pc = (
            self.canon("customer", "f.customer"),
            self.canon("supplier", "p.supplier"),
            self.canon("person", "v.person"),
        )
        customers = [
            r[0]
            for r in await self.query_rows(
                f"WITH {self._CURRENT} SELECT DISTINCT {cc} FROM fact_specs f "
                "JOIN current c ON c.doc_id = f.doc_id WHERE f.customer IS NOT NULL"
            )
        ]
        suppliers = [
            r[0]
            for r in await self.query_rows(
                f"WITH {self._CURRENT} SELECT DISTINCT {sc} FROM spec_packaging p "
                "JOIN current c ON c.doc_id = p.doc_id WHERE p.supplier <> ''"
            )
        ]
        products = [
            r[0]
            for r in await self.query_rows(
                f"WITH {self._CURRENT} SELECT f.product FROM fact_specs f "
                "JOIN current c ON c.doc_id = f.doc_id WHERE f.product IS NOT NULL"
            )
        ]
        codes = [
            r[0]
            for r in await self.query_rows(
                f"WITH {self._CURRENT} SELECT DISTINCT p.code FROM spec_packaging p "
                "JOIN current c ON c.doc_id = p.doc_id"
            )
        ]
        people = [
            r[0]
            for r in await self.query_rows(
                f"WITH {self._CURRENT} SELECT DISTINCT {pc} FROM spec_versions v "
                "JOIN current c ON c.doc_id = v.doc_id"
            )
        ]
        aliases: dict[str, dict[str, str]] = {}
        for kind, raw_key, canonical in await self.query_rows(
            "SELECT kind, raw_key, canonical FROM name_aliases"
        ):
            aliases.setdefault(kind, {})[raw_key] = canonical
        return SpecVocab(
            aliases=aliases,
            people=sorted(set(people)),
            customers=sorted(set(customers)),
            suppliers=sorted(set(suppliers)),
            product_words=sorted(
                {w for p in products for w in re.findall(r"[a-z0-9]+", p.lower())}
            ),
            codes=sorted(set(codes)),
        )

    async def group_mode(self, words: list[str]) -> str:
        """How ``words`` match specs: ``customer``, ``phrase``, ``words`` or ``none``."""
        return (await self._group_condition(words))[2]

    async def packaging_group_totals(self, words: list[str]) -> tuple[int, int, int]:
        """(matching current specs, distinct packaging codes, distinct suppliers) for a group."""
        conds, params, _ = await self._group_condition(words)
        if not conds:
            return 0, 0, 0
        stmt = text(
            f"""
            WITH {self._CURRENT},
            matched AS (
                SELECT f.doc_id FROM fact_specs f JOIN current c ON c.doc_id = f.doc_id
                  JOIN documents d ON d.doc_id = f.doc_id WHERE {" AND ".join(conds)}
            )
            SELECT (SELECT count(*) FROM matched),
                   count(DISTINCT p.code),
                   count(DISTINCT lower({self.canon("supplier", "p.supplier")}))
                     FILTER (WHERE p.supplier <> '')
            FROM spec_packaging p JOIN matched m ON m.doc_id = p.doc_id"""
        )
        async with self._sessions() as s:
            row = (await s.execute(stmt, params)).one()
        if not row[0]:  # no packaging rows joined: still report how many specs match
            plain = text(
                f"""
                WITH {self._CURRENT}
                SELECT count(*) FROM fact_specs f JOIN current c ON c.doc_id = f.doc_id
                  JOIN documents d ON d.doc_id = f.doc_id WHERE {" AND ".join(conds)}"""
            )
            async with self._sessions() as s:
                return int(await s.scalar(plain, params) or 0), 0, 0
        return int(row[0]), int(row[1]), int(row[2])

    async def packaging_for_products(
        self, words: list[str], *, limit: int = 15
    ) -> tuple[list[SpecUsing], list[PackagingUse]]:
        """Current specs of the customer named by ``words``, or whose product name or file name
        contains every word, and the packaging codes they use, most used first (``specs`` counts
        only the matching specs)."""
        conds, params, _ = await self._group_condition(words)
        if not conds:
            return [], []
        matched = f"""
            WITH {self._CURRENT},
            matched AS (
                SELECT f.doc_id FROM fact_specs f JOIN current c ON c.doc_id = f.doc_id
                  JOIN documents d ON d.doc_id = f.doc_id WHERE {" AND ".join(conds)}
            )"""
        specs_stmt = text(
            matched
            + """
            SELECT f.doc_id, d.filename, f.nav_code, f.product, f.customer, f.version
            FROM matched m JOIN fact_specs f ON f.doc_id = m.doc_id
              JOIN documents d ON d.doc_id = f.doc_id ORDER BY f.nav_code, d.filename"""
        )
        usage_stmt = text(
            matched
            + """,
            usage AS (
                SELECT p.code, count(DISTINCT p.doc_id) AS specs
                FROM spec_packaging p JOIN matched m ON m.doc_id = p.doc_id GROUP BY p.code
            )
            SELECT u.code, u.specs,
              (SELECT x.description FROM spec_packaging x
                WHERE x.code = u.code AND x.description <> ''
                GROUP BY x.description ORDER BY count(*) DESC, x.description LIMIT 1),
              (SELECT x.supplier FROM spec_packaging x
                WHERE x.code = u.code AND x.supplier <> ''
                GROUP BY x.supplier ORDER BY count(*) DESC, x.supplier LIMIT 1),
              (SELECT x.section FROM spec_packaging x JOIN matched m2 ON m2.doc_id = x.doc_id
                WHERE x.code = u.code GROUP BY x.section ORDER BY count(*) DESC LIMIT 1)
            FROM usage u ORDER BY u.specs DESC, u.code LIMIT :limit"""
        )
        async with self._sessions() as s:
            spec_rows = (await s.execute(specs_stmt, params)).all()
            usage_rows = (await s.execute(usage_stmt, {**params, "limit": limit})).all()
        specs = [
            SpecUsing(
                doc_id=r[0], filename=r[1], nav_code=r[2], product=r[3], customer=r[4], version=r[5]
            )
            for r in spec_rows
        ]
        usage = [
            PackagingUse(
                code=r[0],
                specs=r[1],
                description=r[2],
                supplier=r[3],
                section=(r[4] or "").lower() or None,
            )
            for r in usage_rows
        ]
        return specs, usage

    async def packaging_of_spec(self, doc_id: str) -> list[SpecPackagingRow]:
        """A spec's packaging rows. A row whose description or supplier the PDF text did not give
        borrows the usual one for that code (the most common among all specs)."""
        stmt = (
            select(SpecPackaging).where(SpecPackaging.doc_id == doc_id).order_by(SpecPackaging.id)
        )
        async with self._sessions() as s:
            rows = list(await s.scalars(stmt))
            usual: dict[str, tuple[str, str]] = {}
            blank = {r.code for r in rows if not r.description or not r.supplier}
            if blank:
                found = await s.execute(
                    text(
                        """
                        SELECT code,
                          (SELECT x.description FROM spec_packaging x
                            WHERE x.code = c.code AND x.description <> ''
                            GROUP BY x.description ORDER BY count(*) DESC, x.description LIMIT 1),
                          (SELECT x.supplier FROM spec_packaging x
                            WHERE x.code = c.code AND x.supplier <> ''
                            GROUP BY x.supplier ORDER BY count(*) DESC, x.supplier LIMIT 1)
                        FROM (SELECT unnest(CAST(:codes AS text[])) AS code) c"""
                    ),
                    {"codes": sorted(blank)},
                )
                usual = {r[0]: (r[1] or "", r[2] or "") for r in found.all()}
        return [
            SpecPackagingRow(
                code=r.code,
                section=r.section,
                qty=r.qty,
                description=r.description or usual.get(r.code, ("", ""))[0],
                supplier=r.supplier or usual.get(r.code, ("", ""))[1],
                dimension=r.dimension,
                weight=r.weight,
                page=r.page,
            )
            for r in rows
        ]

    async def rebuild_fact_specs(self) -> int:
        """Read every Fact Spec's tables again from its stored text (after the parser improved)."""
        async with self._sessions.begin() as s:
            ids = list(
                await s.scalars(select(Document.doc_id).where(Document.doc_type == "fact_spec"))
            )
            for doc_id in ids:
                pages = list(
                    await s.scalars(
                        select(DocumentPage.text)
                        .where(DocumentPage.doc_id == doc_id)
                        .order_by(DocumentPage.page)
                    )
                )
                await self._write_fact_spec(s, doc_id, pages)
        return len(ids)

    async def extractions_to_review(self) -> list[DocReviewItem]:
        """Reports and specs waiting for a person, oldest document first."""
        stmt = (
            select(Extraction, Document.filename)
            .join(Document, Document.doc_id == Extraction.doc_id)
            .where(Extraction.needs_review.is_(True))
            .order_by(Document.ingested_at, Extraction.doc_id, Extraction.start_page)
        )
        async with self._sessions() as s:
            rows = (await s.execute(stmt)).all()
            return [
                self._doc_item(
                    row,
                    filename,
                    await self._page_confidences(s, row.doc_id, row.start_page, row.end_page),
                )
                for row, filename in rows
            ]

    @staticmethod
    def _doc_item(row: Extraction, filename: str, confidences: list[float | None]) -> DocReviewItem:
        known = [c for c in confidences if c is not None]
        keys = CORRECTABLE_DOC.get(row.doc_type, ())
        return DocReviewItem(
            doc_id=row.doc_id,
            filename=filename,
            doc_type=row.doc_type,
            start_page=row.start_page,
            end_page=row.end_page,
            reason=row.review_reason,
            fields={k: (None if row.data.get(k) is None else str(row.data[k])) for k in keys},
            min_ocr_confidence=round(min(known), 1) if known else None,
            verified_by=row.verified_by,
        )

    async def review_extraction(
        self,
        doc_id: str,
        start_page: int,
        user: str,
        *,
        corrections: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> DocReviewItem | None:
        """A person confirms a report or spec reading, or corrects scalar fields and confirms it.

        Recorded in ``certificate_reviews`` (the table is the audit trail for every document
        type). Raises ``ValueError`` for a field that cannot be corrected or an invalid value;
        None if there is no such extraction or it is a calibration certificate.
        """
        async with self._sessions.begin() as s:
            ext = await s.get(Extraction, (doc_id, start_page))
            doc = await s.get(Document, doc_id)
            if ext is None or doc is None or ext.doc_type not in CORRECTABLE_DOC:
                return None
            keys = CORRECTABLE_DOC[ext.doc_type]
            unknown = set(corrections or {}) - set(keys)
            if unknown:
                raise ValueError(f"These fields cannot be corrected: {', '.join(sorted(unknown))}")
            schema = CoA if ext.doc_type == DocType.COA.value else SupplierSpec
            try:
                merged = schema.model_validate({**ext.data, **(corrections or {})})
            except Exception as exc:  # pydantic's message names the bad field
                raise ValueError(f"Not valid: {exc}") from exc
            before = {k: ext.data.get(k) for k in keys}
            new_data = merged.model_dump(mode="json")
            after = {k: new_data.get(k) for k in keys}
            changed = after != before
            if changed:
                ext.data = new_data  # reassigned, so the JSONB change is seen
            ext.needs_review = False
            ext.review_reason = None
            ext.verified_by = user
            ext.verified_at = datetime.now(UTC)
            s.add(
                CertificateReview(
                    doc_id=doc_id,
                    start_page=start_page,
                    reviewed_by=user,
                    action="correct" if changed else "confirm",
                    before=before,
                    after=after,
                    note=note,
                )
            )
            confidences = await self._page_confidences(s, doc_id, start_page, ext.end_page)
            return self._doc_item(ext, doc.filename, confidences)

    async def review_flags(self, doc_id: str) -> dict[int, tuple[bool, str | None, str | None]]:
        """``start_page -> (needs_review, reason, verified_by)`` for a document's certificates."""
        stmt = select(
            CalibrationCertRow.start_page,
            CalibrationCertRow.needs_review,
            CalibrationCertRow.review_reason,
            CalibrationCertRow.verified_by,
        ).where(CalibrationCertRow.doc_id == doc_id)
        others = select(
            Extraction.start_page,
            Extraction.needs_review,
            Extraction.review_reason,
            Extraction.verified_by,
        ).where(Extraction.doc_id == doc_id, Extraction.doc_type != DocType.CALIBRATION_CERT.value)
        async with self._sessions() as s:
            rows = [*(await s.execute(stmt)).all(), *(await s.execute(others)).all()]
            return {page: (needs, reason, by) for page, needs, reason, by in rows}

    async def review_history(self, doc_id: str, start_page: int) -> list[ReviewEvent]:
        stmt = (
            select(CertificateReview)
            .where(CertificateReview.doc_id == doc_id, CertificateReview.start_page == start_page)
            .order_by(CertificateReview.id)
        )
        async with self._sessions() as s:
            return [
                ReviewEvent(
                    reviewed_by=r.reviewed_by,
                    reviewed_at=r.reviewed_at,
                    action=r.action,  # type: ignore[arg-type]
                    before=r.before,
                    after=r.after,
                    note=r.note,
                )
                for r in (await s.scalars(stmt))
            ]

    async def latest_certificates(
        self, *, today: date | None = None
    ) -> list[InstrumentCertificate]:
        """Every instrument's latest certificate, soonest due first (no due date last).

        This is the authority for "when is X due": earlier certificates for the same serial
        are attached as ``older`` rather than competing with the latest one.
        """
        today = today or date.today()
        stmt = (
            select(CalibrationCertRow, Document.filename)
            .join(Document, Document.doc_id == CalibrationCertRow.doc_id)
            .order_by(
                CalibrationCertRow.instrument_serial,
                CalibrationCertRow.calibration_date.desc(),
                CalibrationCertRow.doc_id,
                CalibrationCertRow.start_page,
            )
        )
        out: dict[str, InstrumentCertificate] = {}
        async with self._sessions() as s:
            for row, filename in (await s.execute(stmt)).all():
                current = out.get(row.instrument_serial)
                if current is not None:
                    older = OlderCertificate(
                        certificate_number=row.certificate_number,
                        calibration_date=row.calibration_date,
                        next_due=row.next_due,
                    )
                    # the same certificate filed twice (e.g. a clean PDF and a scan) is one
                    same_as_latest = (
                        older.calibration_date == current.calibration_date
                        and older.next_due == current.next_due
                    )
                    if older not in current.older and not same_as_latest:
                        current.older.append(older)
                    continue
                out[row.instrument_serial] = InstrumentCertificate(
                    doc_id=row.doc_id,
                    filename=filename,
                    start_page=row.start_page,
                    instrument_serial=row.instrument_serial,
                    instrument_description=row.instrument_description,
                    certificate_number=row.certificate_number,
                    calibration_date=row.calibration_date,
                    next_due=row.next_due,
                    days_left=(row.next_due - today).days if row.next_due else None,
                    unverified=row.needs_review,
                    review_reason=row.review_reason,
                )
        return sorted(
            out.values(),
            key=lambda c: (c.next_due is None, c.next_due or date.max, c.instrument_serial),
        )

    async def extraction_records(
        self, doc_type: DocType, *, with_text_pages: int = 0
    ) -> list[ExtractionRecord]:
        """Every stored extraction of one document type, newest document first."""
        stmt = (
            select(Extraction, Document)
            .join(Document, Document.doc_id == Extraction.doc_id)
            .where(Extraction.doc_type == doc_type.value)
            .order_by(Document.ingested_at.desc(), Document.doc_id)
        )
        out: list[ExtractionRecord] = []
        async with self._sessions() as s:
            for ext, doc in (await s.execute(stmt)).all():
                text = ""
                if with_text_pages:
                    pages = (
                        await s.execute(
                            select(DocumentPage.text)
                            .where(
                                DocumentPage.doc_id == doc.doc_id,
                                DocumentPage.page <= with_text_pages,
                            )
                            .order_by(DocumentPage.page)
                        )
                    ).scalars()
                    text = "\n".join(pages)
                out.append(
                    ExtractionRecord(
                        doc_id=doc.doc_id,
                        filename=doc.filename,
                        data=ext.data,
                        text=text,
                        ingested_at=doc.ingested_at,
                        ocr_pages=doc.ocr_pages,
                    )
                )
        return out

    async def unextracted_calibration_documents(self) -> int:
        """Calibration documents with no certificate rows yet (so absent from due-date answers)."""
        stmt = (
            select(func.count())
            .select_from(Document)
            .where(
                Document.doc_type == "calibration_cert",
                ~select(CalibrationCertRow.doc_id)
                .where(CalibrationCertRow.doc_id == Document.doc_id)
                .exists(),
            )
        )
        async with self._sessions() as s:
            return int((await s.execute(stmt)).scalar_one())

    async def list_documents(self, limit: int = 200) -> list[DocumentSummary]:
        """Newest first, with whether each has been extracted and how a running job is going."""
        has_rows = select(Extraction.doc_id).where(Extraction.doc_id == Document.doc_id).exists()
        poor = (
            select(func.count())
            .where(
                DocumentPage.doc_id == Document.doc_id,
                DocumentPage.ocr_confidence < POOR_OCR_CONFIDENCE,
            )
            .scalar_subquery()
        )
        stmt = (
            select(Document, has_rows, ExtractionJob, poor)
            .outerjoin(ExtractionJob, ExtractionJob.doc_id == Document.doc_id)
            .order_by(Document.ingested_at.desc())
            .limit(limit)
        )
        async with self._sessions() as s:
            return [
                DocumentSummary(
                    doc_id=d.doc_id,
                    filename=d.filename,
                    doc_type=d.doc_type,
                    classified_by=d.classified_by,
                    alternative_type=d.alternative_type,
                    pages=d.pages,
                    ocr_pages=d.ocr_pages,
                    poor_ocr_pages=int(poor_pages or 0),
                    chunks=d.chunks,
                    ingested_at=d.ingested_at,
                    extracted=bool(extracted),
                    extraction_status=job.status if job else None,
                    extraction_done=job.done if job else 0,
                    extraction_total=job.total if job else 0,
                )
                for d, extracted, job, poor_pages in (await s.execute(stmt)).all()
            ]

    async def poor_ocr_pages(self, doc_id: str) -> int:
        """How many OCR'd pages of a document were read with low confidence."""
        async with self._sessions() as s:
            return int(
                await s.scalar(
                    select(func.count()).where(
                        DocumentPage.doc_id == doc_id,
                        DocumentPage.ocr_confidence < POOR_OCR_CONFIDENCE,
                    )
                )
                or 0
            )

    async def get_page(self, doc_id: str, page: int) -> tuple[Document, DocumentPage] | None:
        async with self._sessions() as s:
            doc = await s.get(Document, doc_id)
            row = await s.get(DocumentPage, (doc_id, page))
            return (doc, row) if doc is not None and row is not None else None

    async def set_document_type(
        self, doc_id: str, doc_type: DocType, chunks: int, classified_by: str = "user"
    ) -> None:
        """A person corrected the type. The old extraction was for another type, so it goes."""
        async with self._sessions.begin() as s:
            await s.execute(
                update(Document)
                .where(Document.doc_id == doc_id)
                .values(
                    doc_type=doc_type.value,
                    classified_by=classified_by,
                    alternative_type=None,
                    chunks=chunks,
                )
            )
            pages = (
                list(
                    await s.scalars(
                        select(DocumentPage.text)
                        .where(DocumentPage.doc_id == doc_id)
                        .order_by(DocumentPage.page)
                    )
                )
                if doc_type is DocType.FACT_SPEC
                else None
            )
            await self._write_fact_spec(s, doc_id, pages)
            await s.execute(delete(Extraction).where(Extraction.doc_id == doc_id))
            await s.execute(delete(CalibrationCertRow).where(CalibrationCertRow.doc_id == doc_id))
