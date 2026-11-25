"""Rules-based document classifier (first pass; an LLM fallback comes later).

Each type has weighted regex rules matched against the filename and the first pages' text.
The winner is the type with the highest score; below ``MIN_SCORE`` the answer is ``OTHER``.
"""

import re
from collections.abc import Mapping

from qacopilot.ingest.models import Classification, DocType

MIN_SCORE = (
    3.0  # a strong phrase (3) or at least three weaker signals; one stray word is not enough
)
# Only the start of a document is read. Its type is evident up front, and a 100-page manual that
# mentions calibration, TVC and HACCP somewhere would otherwise score on every type.
CLASSIFY_CHARS = 4000
TITLE_BONUS = 2.0
STRONG = 3
TITLE_LINE_CHARS = 70  # longer lines are body text, not a title
_NUMBERED_LINE = re.compile(r"^\d+(\.\d+)*\.?\s")  # '1.1 Suppliers shall...' is a clause

# (pattern, weight[, minimum number of matches]). Matched case-insensitively against the filename
# and the start of the text.
_RULES: Mapping[DocType, list[tuple[str, float] | tuple[str, float, int]]] = {
    DocType.CALIBRATION_CERT: [
        (
            r"certificate of calibration|calibration certificate"
            r"|performance validation certificate",
            3,
        ),
        (r"\bcalibrat", 1),
        (r"traceab\w+ to (national|nist|ukas)|ukas", 1),
        (r"serial (no|number)|instrument", 1),
        (r"next (test |calibration )?due|recalibration|due date|valid to", 1),
        (r"metal detector|probe|thermometer|balance|scale", 1),
    ],
    DocType.COA: [
        (
            r"certificate of analysis|\bcoa\b|analysis report|test report|test certificate"
            r"|certificate of testing|analytical report|report of analysis",
            3,
        ),
        (r"date analysis started|date reported|sample (ref|reference|description|id|received)", 1),
        (r"\bspecification\b.{0,40}\bresult|\bresult\b.{0,40}\blimit", 1),
        (r"total viable count|\btvc\b|enterobacteriaceae|salmonella|listeria", 1),
        (r"batch|lot (no|number)", 1),
        (r"cfu/g|cfu/ml|\bmg/kg\b|\bppm\b", 1),
    ],
    DocType.FACT_SPEC: [
        (r"fact spec(ification)?\b", 3),
        (r"\b(nav code|exchequer)\b", 3),
        (r"\bplu\b", 1),
        (r"raw material|process steps|primary packaging|secondary packaging", 1),
        (r"\bsite\b.{0,20}\bredgrave\b|gressingham foods", 1),
    ],
    DocType.SUPPLIER_SPEC: [
        (
            r"product specification|technical specification|supplier (technical )?specification"
            r"|specification data sheet|(technical|product) data sheet"
            r"|raw material specification",
            3,
        ),
        (r"\bsupersedes\b|effective\s*(date)?\s*:", 1),
        (r"^\s*product\s*:", 1),
        (r"ingredients?|allergens?", 1),
        (r"shelf life|storage", 1),
        (r"microbiolog\w+ (limits?|criteria)", 1),
        (r"nutrition(al)?", 1),
        (r"packaging|pack size|net weight", 1),
    ],
    DocType.AUDIT_CERT: [
        (r"brcgs|brc global standards|global standard for food safety", 3),
        (r"certificate (no|number)|certificate of (approval|registration)", 1),
        (r"\bgrade\b|\bissue\b.{0,20}\bexpiry|expiry date|valid until", 1),
        (r"audit|certification body|accredit", 1),
    ],
    DocType.HACCP: [
        (r"\bhaccp\b|hazard analysis", 3),
        (r"critical control point|\bccp\b", 2),
        (r"critical limit|corrective action|monitoring", 1),
    ],
    DocType.SOP: [
        (r"standard operating procedure|\bsop\b", 3),
        (r"\bscope\b|\bpurpose\b|\bresponsibilit", 1),
        (r"procedure|document (no|number|control)|revision", 1),
    ],
    DocType.CUSTOMER_COP: [
        (
            r"code of practice|customer requirements|technical standards?"
            r"|supplier (technical )?(manual|handbook|standards?|requirements|delivery)"
            r"|supply requirements|product standards",
            3,
        ),
        (r"^\s*\d+\.\d+(\.\d+)?\s", 1, 3),  # numbered clauses: three or more, not one price
        (r"\bsupplier(s)? (must|shall)\b|\bshall\b", 1),
    ],
}

_COMPILED = {
    t: [
        (
            re.compile(rule[0], re.IGNORECASE | re.MULTILINE),
            float(rule[1]),
            rule[2] if len(rule) > 2 else 1,
        )
        for rule in rules
    ]
    for t, rules in _RULES.items()
}


def _title_text(text: str, filename: str) -> str:
    """The filename plus the first few short lines. Covers are a logo, a tagline, then the title;
    long lines and numbered clauses are body text (a clause that mentions BRCGS is not a BRCGS
    certificate's title)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
    titles = (ln for ln in lines if len(ln) <= TITLE_LINE_CHARS and not _NUMBERED_LINE.match(ln))
    return "\n".join([filename, *titles])


def classify_text(text: str, filename: str = "") -> Classification:
    title = _title_text(text, filename)
    haystack = f"{filename}\n{text[:CLASSIFY_CHARS]}"
    # (key, score, type, matched rules, sum of the type's rule weights)
    scored: list[tuple[tuple[bool, float], DocType, list[str], float]] = []
    for doc_type, rules in _COMPILED.items():
        score, titled = 0.0, False
        matched: list[str] = []
        for pattern, weight, min_count in rules:
            if len(pattern.findall(haystack)) >= min_count:
                score += weight
                if weight >= STRONG and pattern.search(title):
                    score += TITLE_BONUS
                    titled = True
                matched.append(pattern.pattern)
        # A strong phrase in the title outranks any amount of body text: a technical manual that
        # lists "product specification" and "allergens" in its contents is still a manual.
        scored.append(
            ((titled and score >= MIN_SCORE, score), doc_type, matched, sum(w for _, w, _ in rules))
        )
    scored.sort(key=lambda row: row[0], reverse=True)
    (best_key, best_type, best_matched, best_total), runner = scored[0], scored[1]
    if best_key[1] < MIN_SCORE:
        return Classification(doc_type=DocType.OTHER, confidence=0.0, matched=[])
    return Classification(
        doc_type=best_type,
        confidence=round(min(best_key[1] / best_total, 1.0), 2),
        matched=best_matched,
        margin=round(best_key[1] - runner[0][1], 2),
        runner_up=runner[1] if runner[0][1] > 0 else None,
    )


def has_title_for(text: str, doc_type: DocType, filename: str = "") -> bool:
    """True if a strong phrase for ``doc_type`` is in the document's title lines."""
    title = _title_text(text, filename)
    return any(
        weight >= STRONG and pattern.search(title)
        for pattern, weight, _ in _COMPILED.get(doc_type, [])
    )
