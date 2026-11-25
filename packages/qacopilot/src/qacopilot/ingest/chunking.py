"""Per-document-type chunking (see PLAN.md section 6).

* Calibration cert / CoA / audit cert: not chunked for Q&A, one record per page.
* SOP / HACCP / supplier spec / other: heading-aware chunks, parent section path prepended,
  tables kept intact, page number kept for citations.
* Customer code of practice: one chunk per numbered clause, clause id in the metadata.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from qacopilot.ingest.models import DocType, ParsedDocument

DEFAULT_MAX_CHARS = 1200
# An unnumbered heading-looking line ("PLU", "NAV CODE", "CAT3") only counts as a heading if at
# least this much text follows it before the next heading; otherwise it is a label or table cell
# and stays in the text. Without this a 6-page table became 1300 tiny chunks.
MIN_SECTION_CHARS = 40
# A page of many very short lines (box labels, step numbers, codes) is a diagram or form, not
# prose: heading detection means nothing there, so it is packed by size like plain text.
DIAGRAM_MIN_LINES = 40
DIAGRAM_MAX_MEDIAN_CHARS = 15
# This many numbered 'headings' in a row, each with almost no text, are a list or flow chart.
FLOW_RUN = 3

_PAGE_ONLY = {DocType.CALIBRATION_CERT, DocType.COA, DocType.AUDIT_CERT}

# The title after the number must start with a capital: "12 months ambient" is body text.
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].*)$")
_CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+\S")
_HEADER_FIELD = re.compile(
    r"^\s*(supplier|product|product name|customer|site)\s*:\s*(.+?)\s*$", re.IGNORECASE
)
_TABLE_LIKE = re.compile(r"\|| {2,}\S|\t")
_NS = uuid.UUID("6f1c2f3e-9d0a-4c57-8a55-0c1d2e3f4a5b")


class Chunk(BaseModel):
    id: str
    doc_id: str
    doc_type: DocType
    index: int
    text: str  # what is embedded and shown to the LLM (includes the section prefix)
    page: int  # 1-based page the chunk starts on
    section: str | None = None
    clause: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


@dataclass
class _Unit:
    kind: Literal["heading", "para", "table"]
    text: str
    page: int
    depth: int = 1  # headings only
    numbered: bool = False  # "2.1 Release": structure we always trust


def _heading_depth(line: str) -> int | None:
    """Return the heading depth for a line, or None if it is not a heading."""
    s = line.strip()
    if not s or len(s) > 80:
        return None
    m = _NUMBERED.match(s)
    if m and not s.endswith((".", ",", ";")) and len(s.split()) <= 10 and s[len(m.group(1)) :]:
        return m.group(1).count(".") + 1
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return 1
    if s.endswith(":") and len(s) <= 40:
        return 2
    return None


def _is_diagram_page(text: str) -> bool:
    """Many lines, almost all of them a few words long: a flow chart, diagram or form."""
    lengths = sorted(len(ln.strip()) for ln in text.splitlines() if ln.strip())
    return (
        len(lengths) >= DIAGRAM_MIN_LINES and lengths[len(lengths) // 2] <= DIAGRAM_MAX_MEDIAN_CHARS
    )


def _units(doc: ParsedDocument) -> list[_Unit]:
    units: list[_Unit] = []
    para: list[str] = []
    table: list[str] = []

    def flush(page_no: int) -> None:
        if table:
            kind: Literal["para", "table"] = "table" if len(table) >= 2 else "para"
            units.append(_Unit(kind, "\n".join(table), page_no))
            table.clear()
        if para:
            units.append(_Unit("para", "\n".join(para), page_no))
            para.clear()

    for page in doc.pages:
        if _is_diagram_page(page.text):
            flush(page.page)
            units.append(
                _Unit(
                    "para",
                    "\n".join(ln.strip() for ln in page.text.splitlines() if ln.strip()),
                    page.page,
                )
            )
            continue
        for raw in page.text.splitlines():
            line = raw.rstrip()
            if not line.strip():
                flush(page.page)
                continue
            table_like = bool(_TABLE_LIKE.search(line))
            depth = None if table_like else _heading_depth(line)
            if depth is not None:
                flush(page.page)
                units.append(
                    _Unit(
                        "heading",
                        line.strip().rstrip(":"),
                        page.page,
                        depth,
                        numbered=bool(_NUMBERED.match(line.strip())),
                    )
                )
            elif table_like:
                if para:
                    units.append(_Unit("para", "\n".join(para), page.page))
                    para.clear()
                table.append(line)
            else:
                if table:
                    flush(page.page)
                para.append(line)
        flush(page.page)
    return _demote_label_headings(units)


def _text_after(units: list[_Unit], i: int) -> int:
    """Characters of body text between heading ``i`` and the next heading."""
    total = 0
    for nxt in units[i + 1 :]:
        if nxt.kind == "heading":
            break
        total += len(nxt.text)
    return total


def _demote_label_headings(units: list[_Unit]) -> list[_Unit]:
    """Turn 'headings' that are really labels or list items back into plain text.

    * An unnumbered heading-looking line with almost no text after it is a label or table cell.
    * Numbered headings are structure and normally kept, but a *run* of three or more of them
      with almost no text between (the steps of a process flow chart: "2. Unload", "3. Inspect",
      "4. Store") is one list, not that many sections. A single short numbered section is kept.
    """
    tiny = [
        u.kind == "heading" and _text_after(units, i) < MIN_SECTION_CHARS
        for i, u in enumerate(units)
    ]
    demote = {i for i, u in enumerate(units) if tiny[i] and not u.numbered}
    i = 0
    while i < len(units):
        if tiny[i] and units[i].numbered:
            j = i
            while j + 1 < len(units) and tiny[j + 1] and units[j + 1].numbered:
                j += 1
            if j - i + 1 >= FLOW_RUN:
                demote.update(range(i, j + 1))
            i = j + 1
        else:
            i += 1
    return [_Unit("para", u.text, u.page) if i in demote else u for i, u in enumerate(units)]


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph at line, then sentence, boundaries."""
    if len(text) <= max_chars:
        return [text]
    pieces = [s for line in text.splitlines() for s in re.split(r"(?<=[.!?])\s+", line) if s]
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > max_chars:
            out.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        out.append(cur)
    return out


def _title(doc: ParsedDocument) -> str:
    for page in doc.pages:
        for line in page.text.splitlines():
            if line.strip():
                return line.strip()[:120]
    return doc.source


def _without_title(doc: ParsedDocument) -> ParsedDocument:
    """Copy of ``doc`` with its first non-empty line removed (it becomes the chunk prefix)."""
    pages = [p.model_copy() for p in doc.pages]
    for page in pages:
        lines = page.text.splitlines()
        for i, line in enumerate(lines):
            if line.strip():
                page.text = "\n".join(lines[i + 1 :])
                return doc.model_copy(update={"pages": pages})
    return doc


def header_fields(doc: ParsedDocument) -> dict[str, str]:
    """Pick up 'Supplier: ...' / 'Product: ...' style lines from the first page."""
    found: dict[str, str] = {}
    if not doc.pages:
        return found
    for line in doc.pages[0].text.splitlines():
        # A line can hold several fields ("Product: X   Batch: Y"); take the first only.
        m = _HEADER_FIELD.match(re.split(r" {3,}", line.strip())[0])
        if m:
            found.setdefault(m.group(1).lower().replace(" ", "_"), m.group(2))
    return found


def _make(
    doc_id: str,
    doc_type: DocType,
    index: int,
    body: str,
    page: int,
    section: str | None,
    prefix: str,
    metadata: dict[str, str],
    clause: str | None = None,
) -> Chunk:
    text = f"{prefix}\n\n{body}" if prefix else body
    return Chunk(
        id=str(uuid.uuid5(_NS, f"{doc_id}:{index}")),
        doc_id=doc_id,
        doc_type=doc_type,
        index=index,
        text=text,
        page=page,
        section=section,
        clause=clause,
        metadata=metadata,
    )


def _prefix(title: str, stack: list[str]) -> str:
    parts = [title] if not stack or stack[0] != title else []
    return " > ".join(parts + stack)


def _chunk_sections(
    doc: ParsedDocument,
    doc_id: str,
    doc_type: DocType,
    metadata: dict[str, str],
    max_chars: int,
) -> list[Chunk]:
    title = _title(doc)
    doc = _without_title(doc)
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    body_len = 0
    start_page = 1

    def flush() -> None:
        nonlocal body, body_len
        if not body:
            return
        names = [name for _, name in stack]
        chunks.append(
            _make(
                doc_id,
                doc_type,
                len(chunks),
                "\n\n".join(body),
                start_page,
                " > ".join(names) or None,
                _prefix(title, names),
                metadata,
            )
        )
        body, body_len = [], 0

    for unit in _units(doc):
        if unit.kind == "heading":
            flush()
            while stack and stack[-1][0] >= unit.depth:
                stack.pop()
            stack.append((unit.depth, unit.text))
            start_page = unit.page
            continue
        pieces = [unit.text] if unit.kind == "table" else _split_long(unit.text, max_chars)
        for piece in pieces:
            if body and body_len + len(piece) > max_chars:
                flush()
            if not body:
                start_page = unit.page
            body.append(piece)
            body_len += len(piece) + 2
    flush()
    return chunks


def _chunk_clauses(
    doc: ParsedDocument,
    doc_id: str,
    doc_type: DocType,
    metadata: dict[str, str],
    max_chars: int,
) -> list[Chunk]:
    title = _title(doc)
    doc = _without_title(doc)
    segments: list[tuple[str | None, int, list[str]]] = [(None, 1, [])]
    for page in doc.pages:
        for line in page.text.splitlines():
            m = _CLAUSE.match(line)
            if m:
                segments.append((m.group(1), page.page, [line.strip()]))
            elif line.strip():
                if not segments[-1][2]:
                    segments[-1] = (segments[-1][0], page.page, [])
                segments[-1][2].append(line.strip())
    chunks: list[Chunk] = []
    for clause, page_no, lines in segments:
        if not lines:
            continue
        for piece in _split_long("\n".join(lines), max_chars):
            chunks.append(
                _make(
                    doc_id,
                    doc_type,
                    len(chunks),
                    piece,
                    page_no,
                    f"Clause {clause}" if clause else None,
                    f"{title} > Clause {clause}" if clause else title,
                    metadata,
                    clause,
                )
            )
    return chunks


def chunk_document(
    doc: ParsedDocument,
    doc_type: DocType,
    doc_id: str,
    *,
    metadata: dict[str, str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """Chunk a parsed document using the strategy for its type."""
    meta = {**header_fields(doc), **(metadata or {})}
    if doc_type in _PAGE_ONLY:
        return [
            _make(doc_id, doc_type, i, p.text, p.page, None, "", meta)
            for i, p in enumerate(p for p in doc.pages if p.text.strip())
        ]
    if doc_type is DocType.CUSTOMER_COP:
        return _chunk_clauses(doc, doc_id, doc_type, meta, max_chars)
    return _chunk_sections(doc, doc_id, doc_type, meta, max_chars)
