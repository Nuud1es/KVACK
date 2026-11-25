"""Read a Gressingham Fact Spec into data, in code (no LLM): the header and the packaging table.

A Fact Spec's page text comes out one table cell per line, so the packaging table reads as
``code / qty / description / supplier / dimension / weight`` lines. Cells can be missing and a
description can wrap, so a row keeps its raw ``cells`` beside the guessed columns: the packaging
code and quantity are reliable, the other columns are a best reading.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

PACKAGING_CODE = re.compile(r"7[09]\d{5}")
_ANY_CODE = re.compile(r"\d{7}|W[A-Z]{3}\d{4}")
_SECTION = re.compile(r"(PRIMARY|SECONDARY|TERTIARY)\s+PACKAGING", re.I)
_END_OF_PACKAGING = re.compile(
    r"(RAW MATERIAL|PROCESS STEPS|PRODUCT DESCRIPTION|LABEL|PALLET|FINISHED PRODUCT"
    r"|Pack Tare|OCM|Case Tare|Product weight)\b",
    re.I,
)
_NOISE = re.compile(
    r"Page \d+ of \d+|GRESSINGHAM FOODS|QTY|PACKAGING|Supplier|Dimension|Weight", re.I
)
_UNIT = r"(?:g|kg|mg)"
# a weight: 3.4g, 12.8g ±0.5g, 20 - 25g, a lone tolerance ±0.5g, or a bare number such as 2.6
_NUM = r"[\d.,]+"
_WEIGHT = re.compile(
    rf"(?:{_NUM}(?:\s?-\s?{_NUM})?\s?{_UNIT}?(?:\s*±\s*{_NUM}\s?{_UNIT}?)?|±\s*{_NUM}\s?{_UNIT}?)",
    re.I,
)
# a size: 530x850mm, 9mm, 28µ, 1/3 footprint, 20mm
_DIMENSION = re.compile(
    r"\d\s?x\s?\d|\b\d+(?:[.,]\d+)?\s?(?:mm|cm|m|µ|mu)\b|\d\s?µ|footprint", re.I
)
_PLACEHOLDER = re.compile(r"-|N/?A", re.I)

# Header labels as printed on page 1, and the field each fills.
_LABELS = {
    "NAV CODE": "nav_code",
    "NAVCODE": "nav_code",
    "VER5SION": "version",  # a typo that is in some specs
    "EXCHEQUER": "nav_code",  # the label on older specs, before the move to NAV
    "PLU": "plu",
    "VERSION": "version",
    "BRAND": "brand",
    "SITE": "site",
    "SEASON": "season",
    "PRODUCT": "product",
    "NLT": "nlt",
    "TYPE": "spec_type",
    "CUSTOMER": "customer",
    "CERTIFICATION": "certification",
    "ALLERGENS": "allergens",
    "STATUS": "status",
}


# How many lines after its label a value may take (the status is followed by unrelated text).
_VALUE_LINES = {"version": 2, "product": 2, "allergens": 2, "customer": 2}


@dataclass
class PackagingRow:
    section: str  # PRIMARY | SECONDARY | TERTIARY
    code: str
    qty: str
    description: str
    supplier: str
    dimension: str
    weight: str
    cells: list[str] = field(default_factory=list)  # every cell as printed, in order
    page: int = 1  # the page the row is printed on


@dataclass
class VersionRow:
    """One line of a spec's version history: who changed what, and when."""

    version: int | None
    person: str
    pages: str
    amends: str
    changed_on: date | None


@dataclass
class FactSpec:
    nav_code: str | None = None
    plu: str | None = None
    version: int | None = None
    version_date: date | None = None
    brand: str | None = None
    site: str | None = None
    season: str | None = None
    product: str | None = None
    spec_type: str | None = None
    customer: str | None = None
    certification: str | None = None
    allergens: str | None = None
    status: str | None = None
    packaging: list[PackagingRow] = field(default_factory=list)
    # --- the rest of the template: what the spec says about the product -------------------
    weight: str | None = None  # "170g", "15kg"
    weight_type: str | None = None  # "Fixed Weight", "Catchweight"
    storage: str | None = None  # "Keep refrigerated 0°C to 3°C"
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    max_life_text: str | None = None  # "Kill +15 days"
    max_life_days: int | None = None
    life_basis: str | None = None  # kill, pack, production
    life_notes: str | None = None  # other shelf-life lines when no single figure is given
    min_life_text: str | None = None  # "9 days" (into the customer's depot)
    min_life_days: int | None = None
    pallet: dict[str, str] = field(default_factory=dict)  # label -> value as printed
    boxes_per_layer: int | None = None
    layers_per_pallet: int | None = None
    cases_per_pallet: int | None = None
    pallet_type: str | None = None
    metal_detection: dict[str, list[float]] = field(default_factory=dict)  # inline / xray, mm
    metal_detection_text: str | None = None  # as printed, e.g. "2x 6mm | N/A | N/A"
    ferrous_mm: float | None = None
    non_ferrous_mm: float | None = None
    stainless_mm: float | None = None
    pack_tare_g: float | None = None
    case_tare_g: float | None = None
    product_weight: str | None = None
    process_steps: list[str] = field(default_factory=list)
    raw_materials: list[dict[str, Any]] = field(default_factory=list)
    history: list[VersionRow] = field(default_factory=list)


def is_fact_spec(text: str) -> bool:
    """Page 1 of a Fact Spec carries these labels; nothing else does."""
    upper = text[:1500].upper()
    return bool(re.search(r"NAV ?CODE|EXCHEQUER", upper)) and all(
        re.search(pattern, upper) for pattern in ("PLU", r"VER\d?SION", "SITE")
    )


def _lines(text: str) -> list[str]:
    """Non-empty lines, with a code that the PDF wrapped ("79041" / "68") put back together."""
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        joined = line + nxt
        if (
            line.isdigit()
            and nxt.isdigit()
            and len(joined) == 7
            and PACKAGING_CODE.fullmatch(joined)
        ):
            out.append(joined)
            i += 2
        elif (line.lower(), nxt.lower()) in {
            ("pack", "tare"),
            ("case", "tare"),
            ("product", "weight"),
        }:
            out.append(f"{line} {nxt}")  # a label the PDF wrapped: it ends the packaging table
            i += 2
        else:
            out.append(line)
            i += 1
    return out


def _header(page1: str) -> dict[str, str]:
    """Label line followed by its value line(s), up to the next label."""
    out: dict[str, str] = {}
    current: str | None = None
    taken = 0
    for line in _lines(page1)[:60]:
        label = _LABELS.get(line.upper())
        if label is not None:
            current, taken = label, 0
            out.setdefault(label, "")
            continue
        if "ODUCT DESCRIPTION" in line.upper():  # a clipped "RODUCT DESCRIPTION" happens too
            break
        if line.upper().startswith("PAGE "):
            continue
        if current is not None and taken < _VALUE_LINES.get(current, 1):
            out[current] = f"{out[current]} {line}".strip()
            taken += 1
    return out


def _row(section: str, code: str, cells: list[str], page: int) -> PackagingRow:
    """Sort a row's cells by what they look like, not by position.

    Real rows carry extras (a film thickness "28µ", a machine tag "ULMA LINE", a weight with no
    unit) that shift a fixed column order. So: the first cell after the quantity is the
    description; weights and sizes are recognised by their form; of the remaining text cells the
    last is the supplier and any earlier ones continue the description.
    """
    qty = cells[0] if cells and cells[0].isdigit() else ""
    rest = cells[1:] if qty else list(cells)
    description_head = rest[0] if rest else ""
    weights: list[str] = []
    dimensions: list[str] = []
    text: list[str] = []
    for cell in rest[1:]:
        if _PLACEHOLDER.fullmatch(cell):
            continue
        if _WEIGHT.fullmatch(cell):
            weights.append(cell)
        elif _DIMENSION.search(cell) and len(cell) <= 30:
            dimensions.append(cell)
        else:
            text.append(cell)
    with_unit = [w for w in weights if re.search(r"[a-z]", w, re.I)]
    weight = (with_unit or weights or [""])[0]
    supplier = text[-1] if text else ""
    description = " ".join(([description_head] if description_head else []) + text[:-1])
    return PackagingRow(
        section=section,
        code=code,
        qty=qty,
        description=description,
        supplier=supplier,
        dimension=" ".join(dimensions),
        weight=weight,
        cells=[code, *cells],
        page=page,
    )


_MM = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.I)
_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "year": 365}
_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")


def _all_lines(pages: list[str]) -> list[str]:
    return [ln for page in pages for ln in _lines(page)]


def _after(lines: list[str], pattern: str, *, skip: int = 1) -> str | None:
    """The line ``skip`` places after the first line that matches ``pattern`` (a label)."""
    rx = re.compile(pattern, re.I)
    for i, line in enumerate(lines):
        if rx.fullmatch(line) and i + skip < len(lines):
            return lines[i + skip]
    return None


def _temperatures(text: str) -> tuple[float | None, float | None]:
    """The temperature range in a storage line: "0°C to 3°C", "0°-3°C", "-18°C", "-18oC",
    "frozen - 18°C", "frozen-18°C"."""
    degree = r"[°º]|o(?=\s*C\b)"
    dash = r"[-\u2013\u2212]"
    # a hyphen between two numbers is a range, not a minus sign: "0°-3°C" -> "0° to 3°C"
    spaced = re.sub(
        rf"(?<=\d)(\s*(?:{degree})\s*C?)?\s*{dash}\s*(?=\d)",
        lambda m: f"{m.group(1) or ''} to ",
        text,
    )
    # any other hyphen before a number is a minus sign, however it is spaced: "frozen - 18"
    spaced = re.sub(rf"\s*{dash}\s*(?=\d)", " -", spaced)
    numbers = [
        float(n)
        for n in re.findall(rf"(?<![\w.])-?\d+(?:\.\d+)?(?=\s*(?:{degree})|\s*to\b|\s*C\b)", spaced)
    ]
    if not numbers:
        return None, None
    lo, hi = min(numbers), max(numbers)
    if lo > 0 and re.search(r"\b(frozen|freezer|deep[- ]?freeze)\b", text, re.I):
        lo, hi = -hi, -lo  # a frozen product is never stored above zero: the minus was lost
    return lo, hi


def _duration_days(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(days?|weeks?|months?|years?)", text, re.I)
    if not m:
        return None
    unit = m.group(2).lower()
    return int(m.group(1)) * _UNIT_DAYS[unit if unit in _UNIT_DAYS else unit.rstrip("s")]


def _grams(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g)\b", text, re.I)
    if not m:
        return None
    return float(m.group(1)) * (1000 if m.group(2).lower() == "kg" else 1)


def _shelf_life(lines: list[str], spec: FactSpec) -> None:
    for i, line in enumerate(lines):
        if re.match(r"shelf[- ]?life information", line, re.I):
            block = lines[i + 1 : i + 8]
            notes: list[str] = []
            for text in block:
                if re.fullmatch(r"PLU|Customer code|Pack Barcode|Case Barcode", text, re.I):
                    break
                if not re.match(r"(max|min)\.?\s*life", text, re.I):
                    notes.append(text)
                    continue
                if m := re.match(r"max\.?\s*life:?\s*(.+)", text, re.I):
                    spec.max_life_text = m.group(1).strip()
                    spec.max_life_days = _duration_days(spec.max_life_text)
                    basis = re.search(
                        r"\b(kill|pack|production|manufacture)\b", spec.max_life_text, re.I
                    )
                    spec.life_basis = basis.group(1).lower() if basis else None
                elif m := re.match(r"min\.?\s*life[^:]*:?\s*(.+)", text, re.I):
                    spec.min_life_text = m.group(1).strip()
                    spec.min_life_days = _duration_days(spec.min_life_text)
            if notes and spec.max_life_days is None:
                spec.life_notes = " ".join(notes)
            return


_PALLET_LABELS = re.compile(r"(packs|pieces|boxes|layers|cases|type)\b", re.I)


def _pallet(lines: list[str], spec: FactSpec) -> None:
    """The pallet table prints its labels first (up to "Type of Pallet"), then the values in the
    same order. The first label varies: "Packs per box", "Bags per box", "Pieces per bag"."""
    for i, line in enumerate(lines):
        if not (
            re.fullmatch(r"PALLET", line, re.I)
            and i + 1 < len(lines)
            and lines[i + 1].upper().startswith("CONFIGURATIO")
        ):
            continue
        j = i + 2
        if j < len(lines) and lines[j].upper() == "N":  # "CONFIGURATIO" / "N"
            j += 1
        end = next(
            (
                k
                for k in range(j, min(j + 8, len(lines)))
                if re.match(r"type of pallet", lines[k], re.I)
            ),
            None,
        )
        if end is None:
            return
        labels: list[str] = []
        for text in lines[j : end + 1]:
            if labels and labels[-1].lower().endswith(("per", "of")):  # a wrapped label
                labels[-1] += " " + text
            else:
                labels.append(text)
        values = lines[end + 1 : end + 1 + len(labels)]
        if len(values) < len(labels):
            return
        spec.pallet = dict(zip(labels, values, strict=True))
        for label, value in spec.pallet.items():
            low = label.lower()
            number = int(value) if value.isdigit() else None
            if low.startswith("boxes per"):
                spec.boxes_per_layer = number
            elif low.startswith("layers per"):
                spec.layers_per_pallet = number
            elif low.startswith("cases per"):
                spec.cases_per_pallet = number
            elif low.startswith("type"):
                spec.pallet_type = value
        return


def _metal_detection(lines: list[str], spec: FactSpec) -> None:
    for i, line in enumerate(lines):
        if re.fullmatch(r"metal stick sizes", line, re.I):
            window = lines[i : i + 30]
            at = next(
                (k for k, t in enumerate(window) if re.match(r"in line metal", t, re.I)), None
            )
            if at is not None:
                spec.metal_detection_text = " | ".join(window[at + 1 : at + 4])
            readings = []
            for k, text in enumerate(window):
                if m := re.fullmatch(r"(\d+(?:\.\d+)?)\s*mm", text, re.I):
                    readings.append((k, float(m.group(1))))
            xray_at = next(
                (k for k, t in enumerate(window) if re.fullmatch(r"x[- ]?ray", t, re.I)), None
            )
            inline = [v for k, v in readings if xray_at is None or k < xray_at][:3]
            xray = [v for k, v in readings if xray_at is not None and k > xray_at][:3]
            if len(inline) == 3:
                spec.metal_detection["inline"] = inline
                spec.ferrous_mm, spec.non_ferrous_mm, spec.stainless_mm = inline
            if len(xray) == 3:
                spec.metal_detection["xray"] = xray
            return


def _tare(lines: list[str], spec: FactSpec) -> None:
    for i, line in enumerate(lines):
        low = line.lower()
        if i + 1 >= len(lines):
            break
        if low == "pack tare" and spec.pack_tare_g is None:
            spec.pack_tare_g = _grams(lines[i + 1])
        elif low.startswith("ocm") and "tare" in low and spec.case_tare_g is None:
            spec.case_tare_g = _grams(lines[i + 1])
        elif low == "product weight" and spec.product_weight is None:
            spec.product_weight = lines[i + 1]


def _between(lines: list[str], start: str, stop: str) -> list[str]:
    rx_start, rx_stop = re.compile(start, re.I), re.compile(stop, re.I)
    for i, line in enumerate(lines):
        if rx_start.match(line):
            out: list[str] = []
            for text in lines[i + 1 :]:
                if rx_stop.match(text):
                    break
                out.append(text)
            return out
    return []


def _process_steps(lines: list[str]) -> list[str]:
    block = _between(lines, r"PROCESS STEPS", r"METAL\b|DETECTION\b")
    return [t for t in block if t not in ("•", "-") and not _NOISE.fullmatch(t)]


def _raw_materials(lines: list[str]) -> list[dict[str, Any]]:
    """The ingredient table: a quantity, a description, a supplier and a weight per row."""
    block = [
        t
        for t in _between(lines, r"RAW MATERIAL", r"PROCESS STEPS")
        if not re.fullmatch(r"Code|QTY|Ingredients|Supplier|Weight", t, re.I)
    ]
    rows: list[list[str]] = []
    for text in block:
        if re.fullmatch(r"\d{1,3}|-", text) and (not rows or len(rows[-1]) > 1):
            rows.append([text])
        elif rows:
            rows[-1].append(text)
    out: list[dict[str, Any]] = []
    for cells in rows:
        qty, rest = cells[0], cells[1:]
        weights = [c for c in rest if _WEIGHT.fullmatch(c) or re.search(r"\d\s*(g|kg)\b", c, re.I)]
        names = [c for c in rest if c not in weights and not _PLACEHOLDER.fullmatch(c)]
        out.append(
            {
                "qty": qty,
                "description": " ".join(names[:-1])
                if len(names) > 1
                else (names[0] if names else ""),
                "supplier": names[-1] if len(names) > 1 else "",
                "weight": " ".join(weights),
                "cells": cells,
            }
        )
    return out


_HISTORY_DATE = re.compile(r"(\d{1,2})[./](\d{1,2})[./](\d{4}|\d{2})")


def _history_date(text: str) -> date | None:
    m = _HISTORY_DATE.fullmatch(text)
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    year = year + 2000 if year < 100 else year
    try:
        return date(year, month, day)
    except ValueError:
        return None


_PERSON = re.compile(r"[A-Za-z][A-Za-z.'’-]*(?: [A-Za-z][A-Za-z.'’-]*){0,3}")
_FIRST_CELL = re.compile(r"\d{1,3}|creation|created|original|originated|new", re.I)
_PAGE = re.compile(r"all|creation|\d+(?:\s*[-&,/]\s*\d+)*", re.I)
_HISTORY_HEADER = re.compile(r"Page changed|Amends made|Date|Page|changed", re.I)


def _history_row(cells: list[str], when: date | None) -> VersionRow | None:
    """The row that ends here: the last place in ``cells`` that looks like version, person,
    page. Anything before it (a page header that interrupted the table) is ignored. A name can
    be split over two lines ("Peter" / "Bringlesnap")."""
    for k in range(len(cells) - 3, -1, -1):
        if not _FIRST_CELL.fullmatch(cells[k]):
            continue
        name, after = cells[k + 1], cells[k + 2]
        if _PERSON.fullmatch(name) and _PAGE.fullmatch(after):
            person, page, rest = name, after, cells[k + 3 :]
        elif (
            k + 3 < len(cells)
            and _PERSON.fullmatch(name)
            and _PERSON.fullmatch(after)
            and len(name.split()) + len(after.split()) <= 4
            and _PAGE.fullmatch(cells[k + 3])
        ):
            person, page, rest = f"{name} {after}", cells[k + 3], cells[k + 4 :]
        else:
            continue
        version = int(cells[k]) if cells[k].isdigit() else None
        return VersionRow(version, person, page, " ".join(rest), when)
    return None


def _history(lines: list[str]) -> list[VersionRow]:
    """The version table: version (or "Creation"), person responsible, page, amends, date.

    The table can run across pages, with the page header in the middle of it, so it is read to the
    end of the document: every date closes a group, and a group is a row only if it contains
    version, person, page in that order. Text that is not a row is skipped, never read as one.
    """
    start = next((i for i, t in enumerate(lines) if re.fullmatch(r"responsible", t, re.I)), None)
    if start is None:
        start = next(
            (i for i, t in enumerate(lines) if re.fullmatch(r"person responsible", t, re.I)), None
        )
    if start is None:
        return []
    rows: list[VersionRow] = []
    cells: list[str] = []
    for text in lines[start + 1 :]:
        if _HISTORY_HEADER.fullmatch(text):
            continue
        if _HISTORY_DATE.fullmatch(text):
            if row := _history_row(cells, _history_date(text)):
                rows.append(row)
            cells = []
        else:
            cells.append(text)
            if len(cells) > 80:  # a long stretch of ordinary text: keep only the recent lines
                cells = cells[-40:]
    return rows


def _read_template(pages: list[str], spec: FactSpec) -> None:
    lines = _all_lines(pages)
    spec.weight = _after(lines, r"WEIGHT:")
    kind = _after(lines, r"WEIGHT:", skip=2)
    spec.weight_type = kind if kind and re.search(r"weight", kind, re.I) else None
    if (storage := _after(lines, r"TEMP:")) is not None:
        spec.storage = storage
        spec.temp_min_c, spec.temp_max_c = _temperatures(storage)
    _shelf_life(lines, spec)
    _pallet(lines, spec)
    _metal_detection(lines, spec)
    _tare(lines, spec)
    spec.process_steps = _process_steps(lines)
    spec.raw_materials = _raw_materials(lines)
    spec.history = _history(lines)


def parse_fact_spec(pages: list[str]) -> FactSpec:
    """The header from page 1 and every packaging row (primary, secondary, tertiary)."""
    spec = FactSpec()
    if not pages:
        return spec
    head = _header(pages[0])
    spec.nav_code = head.get("nav_code") or None
    for key in ("plu", "brand", "site", "season", "product", "spec_type", "customer", "status"):
        setattr(spec, key, head.get(key) or None)
    spec.certification = head.get("certification") or None
    spec.allergens = head.get("allergens") or None
    if m := re.match(r"(\d+)\s+(\d\d/\d\d/\d{4})", head.get("version", "")):
        spec.version = int(m.group(1))
        try:
            spec.version_date = datetime.strptime(m.group(2), "%d/%m/%Y").date()
        except ValueError:
            spec.version_date = None
    elif m := re.match(r"(\d+)", head.get("version", "")):
        spec.version = int(m.group(1))

    section: str | None = None
    code: str | None = None
    code_page = 1
    cells: list[str] = []

    def close() -> None:
        nonlocal code, cells
        if code is not None and section is not None:
            spec.packaging.append(_row(section, code, cells, code_page))
        code, cells = None, []

    for number, line in ((n, ln) for n, page in enumerate(pages, start=1) for ln in _lines(page)):
        if m := _SECTION.fullmatch(line):
            close()
            section = m.group(1).upper()
        elif section is not None and _END_OF_PACKAGING.match(line):
            close()
            section = None
        elif section is None:
            continue
        elif PACKAGING_CODE.fullmatch(line):
            close()
            code, code_page = line, number
        elif _ANY_CODE.fullmatch(line):  # an ingredient or WIP code ends the packaging row
            close()
        elif code is not None and not _NOISE.fullmatch(line):
            cells.append(line)
    close()
    _read_template(pages, spec)
    return spec
