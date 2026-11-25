"""PostgreSQL tables (SQLAlchemy 2.0). Schema changes go through Alembic migrations."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # content hash
    filename: Mapped[str] = mapped_column(String(255))
    doc_type: Mapped[str] = mapped_column(String(32), index=True)
    classified_by: Mapped[str] = mapped_column(String(8), server_default="rules")  # rules|llm|user
    alternative_type: Mapped[str | None] = mapped_column(String(32))  # LLM disagreed: check this
    pages: Mapped[int] = mapped_column(Integer)
    ocr_pages: Mapped[int] = mapped_column(Integer)
    chunks: Mapped[int] = mapped_column(Integer)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentPage(Base):
    """Full page text, so a document can be re-extracted without uploading it again."""

    __tablename__ = "document_pages"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    page: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(8))  # "text" or "ocr"
    ocr_confidence: Mapped[float | None] = mapped_column(Float)  # OCR only: mean word confidence


class Extraction(Base):
    """The LLM's structured output for a document, as JSON (any document type)."""

    __tablename__ = "extractions"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    # A bundle PDF holds several certificates: one row per certificate, keyed by its first page.
    start_page: Mapped[int] = mapped_column(Integer, primary_key=True, server_default="1")
    end_page: Mapped[int | None] = mapped_column(Integer)
    doc_type: Mapped[str] = mapped_column(String(32))
    data: Mapped[dict[str, Any]] = mapped_column(JSONB)
    prompt_version: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(255))
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # For reports and specs (a calibration certificate keeps these on its typed row): a reading
    # that looks incomplete or misread and that no person has confirmed yet.
    needs_review: Mapped[bool] = mapped_column(Boolean, server_default="false", default=False)
    review_reason: Mapped[str | None] = mapped_column(Text)
    verified_by: Mapped[str | None] = mapped_column(String(128))
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CalibrationCertRow(Base):
    """Typed copy of a calibration certificate's fields, so expiry questions are plain SQL."""

    __tablename__ = "calibration_certs"
    __table_args__ = (
        Index("ix_calibration_certs_serial_date", "instrument_serial", "calibration_date"),
    )

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    start_page: Mapped[int] = mapped_column(Integer, primary_key=True, server_default="1")
    end_page: Mapped[int | None] = mapped_column(Integer)
    instrument_serial: Mapped[str] = mapped_column(String(128))
    instrument_description: Mapped[str | None] = mapped_column(String(255))
    certificate_number: Mapped[str | None] = mapped_column(String(128))
    calibrated_by: Mapped[str | None] = mapped_column(String(255))
    calibration_date: Mapped[date] = mapped_column()
    next_due: Mapped[date | None] = mapped_column(index=True)
    result: Mapped[str] = mapped_column(String(16))
    # A reading from a poor scan (or with implausible dates) is unverified until a person confirms
    # or corrects it. Due-date answers and reminders say so while it is.
    needs_review: Mapped[bool] = mapped_column(Boolean, server_default="false", index=True)
    review_reason: Mapped[str | None] = mapped_column(Text)
    verified_by: Mapped[str | None] = mapped_column(String(128))
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CertificateReview(Base):
    """Who confirmed or corrected a certificate's values, with the values before and after."""

    __tablename__ = "certificate_reviews"
    __table_args__ = (Index("ix_certificate_reviews_cert", "doc_id", "start_page"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(64))
    start_page: Mapped[int] = mapped_column(Integer)
    reviewed_by: Mapped[str] = mapped_column(String(128))
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    action: Mapped[str] = mapped_column(String(16))  # confirm | correct
    before: Mapped[dict[str, Any]] = mapped_column(JSONB)
    after: Mapped[dict[str, Any]] = mapped_column(JSONB)
    note: Mapped[str | None] = mapped_column(Text)


class AgentRun(Base):
    """One agent run. The graph state itself lives in the LangGraph checkpointer."""

    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(64))
    query: Mapped[str] = mapped_column(Text)
    route: Mapped[str | None] = mapped_column(String(32))
    # running | awaiting_approval | completed | rejected | failed
    status: Mapped[str] = mapped_column(String(24), index=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Outbox(Base):
    """Every email the agent sends, keyed so an approval that is retried cannot send twice."""

    __tablename__ = "outbox"

    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[dict[str, Any]] = mapped_column(JSONB)
    approved_by: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))  # pending | delivered | failed
    outcome: Mapped[str | None] = mapped_column(String(16))  # file | draft | sent
    reference: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    """Append-only (a database trigger rejects UPDATE and DELETE; see migration 0002)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(64))
    event: Mapped[str] = mapped_column(String(48), index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # ask | agent
    rating: Mapped[int] = mapped_column(Integer)  # 1 = thumbs up, -1 = thumbs down
    comment: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(64))


class ExtractionJob(Base):
    """Progress of a background extraction (a bundle can take minutes, one LLM call each)."""

    __tablename__ = "extraction_jobs"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[str] = mapped_column(String(12))  # running | done | failed
    total: Mapped[int] = mapped_column(Integer)
    done: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)  # certificates that could not be read
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class User(Base):
    """A person who registered with a name and a 4-digit PIN (stored only as a salted hash)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name_key: Mapped[str] = mapped_column(String(64), unique=True)  # case-folded, for matching
    display_name: Mapped[str] = mapped_column(String(64))
    pin_hash: Mapped[str] = mapped_column(String(200))
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserSession(Base):
    """A sign-in. ``token_digest`` is a hash of the token the browser holds."""

    __tablename__ = "user_sessions"

    token_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FactSpecRow(Base):
    """The header of a Fact Spec, read in code from its page 1 (one row per document)."""

    __tablename__ = "fact_specs"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    nav_code: Mapped[str | None] = mapped_column(String(32), index=True)
    plu: Mapped[str | None] = mapped_column(String(32))
    version: Mapped[int | None] = mapped_column(Integer)
    version_date: Mapped[date | None] = mapped_column(Date)
    brand: Mapped[str | None] = mapped_column(String(128))
    site: Mapped[str | None] = mapped_column(String(128))
    season: Mapped[str | None] = mapped_column(String(64))
    product: Mapped[str | None] = mapped_column(Text)
    spec_type: Mapped[str | None] = mapped_column(String(64))
    customer: Mapped[str | None] = mapped_column(String(255))
    certification: Mapped[str | None] = mapped_column(Text)
    allergens: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(32))
    # what the spec says about the product (read in code from its text)
    weight: Mapped[str | None] = mapped_column(String(64))
    weight_type: Mapped[str | None] = mapped_column(String(32))
    storage: Mapped[str | None] = mapped_column(Text)
    temp_min_c: Mapped[float | None] = mapped_column(Float)
    temp_max_c: Mapped[float | None] = mapped_column(Float)
    max_life_text: Mapped[str | None] = mapped_column(Text)
    max_life_days: Mapped[int | None] = mapped_column(Integer)
    life_basis: Mapped[str | None] = mapped_column(String(32))
    life_notes: Mapped[str | None] = mapped_column(Text)
    min_life_text: Mapped[str | None] = mapped_column(Text)
    min_life_days: Mapped[int | None] = mapped_column(Integer)
    boxes_per_layer: Mapped[int | None] = mapped_column(Integer)
    layers_per_pallet: Mapped[int | None] = mapped_column(Integer)
    cases_per_pallet: Mapped[int | None] = mapped_column(Integer)
    pallet_type: Mapped[str | None] = mapped_column(String(64))
    pallet: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metal_detection: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metal_detection_text: Mapped[str | None] = mapped_column(Text)
    ferrous_mm: Mapped[float | None] = mapped_column(Float)
    non_ferrous_mm: Mapped[float | None] = mapped_column(Float)
    stainless_mm: Mapped[float | None] = mapped_column(Float)
    pack_tare_g: Mapped[float | None] = mapped_column(Float)
    case_tare_g: Mapped[float | None] = mapped_column(Float)
    product_weight: Mapped[str | None] = mapped_column(String(64))
    process_steps: Mapped[list[str] | None] = mapped_column(JSONB)
    raw_materials: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)


class SpecPackaging(Base):
    """One packaging row of a Fact Spec. ``cells`` keeps every cell as printed."""

    __tablename__ = "spec_packaging"
    __table_args__ = (Index("ix_spec_packaging_code", "code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    section: Mapped[str] = mapped_column(String(12))  # PRIMARY | SECONDARY | TERTIARY
    code: Mapped[str] = mapped_column(String(16))
    qty: Mapped[str] = mapped_column(String(32), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    supplier: Mapped[str] = mapped_column(String(255), default="")
    dimension: Mapped[str] = mapped_column(String(128), default="")
    weight: Mapped[str] = mapped_column(String(64), default="")
    cells: Mapped[list[str]] = mapped_column(JSONB)
    page: Mapped[int] = mapped_column(Integer, default=1, server_default="1")


class SpecVersion(Base):
    """One line of a Fact Spec's version history: who changed what, and when."""

    __tablename__ = "spec_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int | None] = mapped_column(Integer)  # None for a "Creation" line
    person: Mapped[str] = mapped_column(String(128), index=True)
    pages: Mapped[str] = mapped_column(String(64), default="")
    amends: Mapped[str] = mapped_column(Text, default="")
    changed_on: Mapped[date | None] = mapped_column(Date)


class NameAlias(Base):
    """Which spellings are one name: "Sainsbury" and "Sainsbury's" are one customer.

    ``raw_key`` is the spelling as it appears in the specs, lower-cased; ``canonical`` is the name
    every question and answer uses for it. ``kind`` is customer, supplier or person.
    """

    __tablename__ = "name_aliases"

    kind: Mapped[str] = mapped_column(String(12), primary_key=True)
    raw_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    canonical: Mapped[str] = mapped_column(String(255))
    note: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NameDecision(Base):
    """A person's decision that a name is its own name, so it is not suggested as an alias."""

    __tablename__ = "name_decisions"

    kind: Mapped[str] = mapped_column(String(12), primary_key=True)
    name_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    decision: Mapped[str] = mapped_column(String(16))
    decided_by: Mapped[str | None] = mapped_column(String(128))
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
