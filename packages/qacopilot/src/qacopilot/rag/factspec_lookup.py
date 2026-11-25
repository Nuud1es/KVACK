"""Questions about Fact Specs answered from the header table: one spec's fields and counts.

Two things search cannot do reliably across hundreds of near-identical specs:
- tell which spec a question means (so a question that names a product or a NAV code is first
  matched to exactly one spec, and search is then limited to that spec: ``resolve_spec``);
- count or list across all of them ("how many specs are frozen", "which customers have most").
Anything not clearly one of these returns None and goes to document search.
"""

import re
from collections.abc import AsyncIterator
from typing import Any

from qacopilot.rag.answer import (
    AskEvent,
    Citation,
    DoneEvent,
    ExportRef,
    SourcesEvent,
    TokenEvent,
)
from qacopilot.rag.specquery import SpecFilter, SpecQuery, _mentioned, get_vocab
from qacopilot.repositories.documents import SpecIndexEntry

NAV = re.compile(r"\b(4\d{6})\b")
MIN_PRODUCT_WORDS = 3
MIN_COVERAGE = 0.85  # how much of a product's name the question must contain
SNIPPET_CHARS = 240
BASIS = "current Fact Specs (the latest version of each product)"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t}


def match_specs(question: str, index: list[SpecIndexEntry]) -> list[SpecIndexEntry]:
    """Every current spec the question fits equally well (usually one, or none).

    A NAV code is exact. A product name must be (almost) wholly in the question and the longest
    name wins; several specs with the same product name (one per customer or code) are all
    returned, because the question does not say which of them is meant.
    """
    codes = set(NAV.findall(question))
    if codes:
        return [s for s in index if codes & set(s.nav_codes)]
    q = _tokens(question)
    scored: list[tuple[float, int, SpecIndexEntry]] = []
    for s in index:
        words = _tokens(s.product or "")
        if len(words) < MIN_PRODUCT_WORDS:
            continue
        coverage = len(words & q) / len(words)
        if coverage >= MIN_COVERAGE:
            scored.append((coverage, len(words), s))
    if not scored:
        return []
    best = max((t[0], t[1]) for t in scored)
    return [t[2] for t in scored if (t[0], t[1]) == best]


def resolve_spec(question: str, index: list[SpecIndexEntry]) -> SpecIndexEntry | None:
    """The one current spec a question is about, or None if it names none or is ambiguous."""
    found = match_specs(question, index)
    return found[0] if len(found) == 1 else None


# (field, question pattern, label); checked in this order
_ATTRIBUTES: list[tuple[str, re.Pattern[str], str]] = [
    ("customer", re.compile(r"\bcustomers?\b|who is (?:it|this|that) for", re.I), "Customer"),
    ("version", re.compile(r"\bversion\b|\brevision\b", re.I), "Version"),
    (
        "status",
        re.compile(r"\bstatus\b|\bis\b.*\b(?:active|approved|live|obsolete)\b", re.I),
        "Status",
    ),
    ("plu", re.compile(r"\bplu\b", re.I), "PLU"),
    ("season", re.compile(r"\bseason\b", re.I), "Season"),
    ("brand", re.compile(r"\bbrand\b", re.I), "Brand"),
    ("certification", re.compile(r"\bcertif\w*", re.I), "Certification"),
    ("allergens", re.compile(r"\ballergens?\b", re.I), "Allergens"),
    ("site", re.compile(r"\bsite\b", re.I), "Site"),
    (
        "spec_type",
        re.compile(r"fresh or frozen|frozen or fresh|\bis\b.*\b(?:fresh|frozen)\b|\btype\b", re.I),
        "Type",
    ),
    ("product", re.compile(r"product name|\bcalled\b|name of|\bdescription\b", re.I), "Product"),
]


def _value(header: Any, field: str) -> str | None:
    if field == "version":
        if header.version is None:
            return None
        when = f" (dated {header.version_date:%d/%m/%Y})" if header.version_date else ""
        return f"{header.version}{when}"
    value = getattr(header, field)
    return None if value in (None, "") else str(value)


async def _citation(repo: Any, header: Any) -> Citation:
    found = await repo.get_page(header.doc_id, 1)
    snippet = found[1].text.strip()[:SNIPPET_CHARS] if found is not None else header.filename
    return Citation(
        n=1, doc_id=header.doc_id, filename=header.filename, page=1, score=1.0, snippet=snippet
    )


def _done(
    text: str, citations: list[Citation] | None = None, export: ExportRef | None = None
) -> list[AskEvent]:
    events: list[AskEvent] = []
    if citations:
        events.append(SourcesEvent(citations=citations))
    events.append(TokenEvent(text=text))
    cited = [c.n for c in citations or []]
    events.append(
        DoneEvent(answer=text, answered=True, cited=cited, via="fact_spec", export=export)
    )
    return events


_SUMMARY = re.compile(
    r"\btell me about\b|\bsummar\w+|\boverview\b|\bdetails? (?:of|for|on)\b"
    r"|\binformation (?:on|about)\b|\bwhat is (?:spec )?\d{7}\??$",
    re.I,
)


async def _summary_answer(question: str, repo: Any) -> list[AskEvent] | None:
    """A short card for one spec: its header fields and how much packaging it lists."""
    if not _SUMMARY.search(question):
        return None
    spec = resolve_spec(question, await repo.spec_index())
    if spec is None:
        return None
    header = await repo.spec_header(spec.doc_id)
    if header is None:
        return None
    packaging = await repo.packaging_of_spec(spec.doc_id)
    facts = [
        ("Customer", header.customer),
        ("Type", header.spec_type),
        ("Status", header.status),
        ("Version", _value(header, "version")),
        ("PLU", header.plu),
        ("Season", header.season),
        ("Brand", header.brand),
        ("Allergens", header.allergens),
        ("Certification", header.certification),
    ]
    lines = [f"{header.nav_code or ''} {header.product or header.filename} [1]".strip()]
    lines += [f"{label}: {value}" for label, value in facts if value]
    lines.append(f"Packaging: {len(packaging)} coded items listed (ask what packaging it uses).")
    return _done("\n".join(lines), [await _citation(repo, header)])


# --- what the spec says: storage, shelf life, pallet, metal detection, weight, steps, history ---

_DETAILS: list[tuple[str, re.Pattern[str]]] = [
    (
        "history",
        re.compile(
            r"version history|\bhistory\b|"
            r"who (?:last |has |have )?(?:updated|changed|amended|edited|"
            r"wrote|written|created|originated|approved|signed|authori[sz]ed|reviewed|owns)|"
            r"(?:updated|changed|amended|edited|written|created|approved|signed off|reviewed) by|"
            r"\bauthor\b|last (?:updated|changed|amended|edited)",
            re.I,
        ),
    ),
    (
        "life",
        re.compile(
            r"max(?:imum)?\.? ?life|shelf[- ]?life|min(?:imum)?\.? ?life|life into depot|"
            r"how long (?:does|will|is|can).*(?:last|keep|life)|\buse[- ]by\b",
            re.I,
        ),
    ),
    (
        "storage",
        re.compile(
            r"temperature|\btemp\b|(?:stored?|kept|keep)\b.*\b(?:at|below|between)\b|"
            r"storage|how cold",
            re.I,
        ),
    ),
    ("pallet", re.compile(r"pallet|cases per|layers per|boxes per|layers|palletis", re.I)),
    ("metal", re.compile(r"metal|ferrous|stainless|x[- ]?ray|detector|detection", re.I)),
    ("raw", re.compile(r"ingredients?|raw materials?|made (?:of|from)|what.*contain", re.I)),
    (
        "steps",
        re.compile(r"process steps?|how is .* (?:made|processed|prepared)|\bprocess\b", re.I),
    ),
    ("weight", re.compile(r"\bweight\b|\btare\b|how heavy|\bweighs?\b", re.I)),
]


def _num(value: float | None, unit: str = "") -> str:
    if value is None:
        return "not given"
    text = f"{value:g}"
    return f"{text}{unit}"


def _detail_text(kind: str, d: Any, who: str, question: str = "") -> str | None:
    """The answer to a question about one spec's content, from its stored fields."""
    if kind == "life":
        if d.max_life_text is None and d.min_life_text is None and d.life_notes is None:
            return f"The shelf life is not given in the Fact Spec for {who} [1]."
        parts: list[str] = []
        if d.max_life_text:
            show = d.max_life_days is not None and "day" not in d.max_life_text.lower()
            days = f" ({d.max_life_days} days)" if show else ""
            parts.append(f"Max life: {d.max_life_text}{days}")
        if d.min_life_text:
            show = d.min_life_days is not None and "day" not in d.min_life_text.lower()
            days = f" ({d.min_life_days} days)" if show else ""
            parts.append(f"minimum life into the depot: {d.min_life_text}{days}")
        if d.life_notes and not d.max_life_text:
            parts.append(d.life_notes)
        return f"Shelf life of {who}: " + "; ".join(parts) + " [1]"
    if kind == "storage":
        if not d.storage:
            return f"The storage temperature is not given in the Fact Spec for {who} [1]."
        rng = ""
        return f"Storage of {who}: {d.storage}{rng} [1]"
    if kind == "pallet":
        if not d.pallet:
            return f"No pallet configuration is given in the Fact Spec for {who} [1]."
        listed = "; ".join(f"{k}: {v}" for k, v in d.pallet.items())
        return f"Pallet configuration of {who}: {listed} [1]"
    if kind == "metal":
        md = d.metal_detection or {}
        parts = []
        if "inline" in md:
            f, nf, ss = md["inline"]
            parts.append(
                f"in-line stick sizes: ferrous {_num(f, ' mm')}, non-ferrous {_num(nf, ' mm')}, "
                f"stainless steel {_num(ss, ' mm')}"
            )
        if "xray" in md:
            f, nf, ss = md["xray"]
            parts.append(
                f"X-ray: ferrous {_num(f, ' mm')}, non-ferrous {_num(nf, ' mm')}, "
                f"stainless steel {_num(ss, ' mm')}"
            )
        if not parts and d.metal_detection_text:
            parts.append(f"as printed: {d.metal_detection_text}")
        if not parts:
            return f"No metal detection settings are given in the Fact Spec for {who} [1]."
        return f"Metal detection for {who}: " + "; ".join(parts) + " [1]"
    if kind == "weight":
        facts: list[str] = []
        if d.weight:
            facts.append(f"weight {d.weight}" + (f" ({d.weight_type})" if d.weight_type else ""))
        if d.pack_tare_g is not None:
            facts.append(f"pack tare {_num(d.pack_tare_g, ' g')}")
        if d.case_tare_g is not None:
            facts.append(f"case tare {_num(d.case_tare_g, ' g')}")
        if d.product_weight:
            facts.append(f"product weight {d.product_weight}")
        if not facts:
            return f"No weight is given in the Fact Spec for {who} [1]."
        return f"Weight of {who}: " + "; ".join(facts) + " [1]"
    if kind == "raw":
        rows = d.raw_materials or []
        if not rows:
            return f"No ingredients are listed in the Fact Spec for {who} [1]."
        lines = [
            f"{i}. {r.get('description') or '-'}"
            + (f" ({r['supplier']})" if r.get("supplier") else "")
            + (f", {r['weight']}" if r.get("weight") else "")
            for i, r in enumerate(rows, start=1)
        ]
        return f"Ingredients (raw materials) of {who} [1]:\n" + "\n".join(lines)
    if kind == "steps":
        steps = d.process_steps or []
        if not steps:
            return f"No process steps are listed in the Fact Spec for {who} [1]."
        lines = [f"{i}. {t}" for i, t in enumerate(steps[:15], start=1)]
        more = f"\n...and {len(steps) - 15} more steps." if len(steps) > 15 else ""
        return f"Process steps for {who} [1]:\n" + "\n".join(lines) + more
    if kind == "history":
        rows = d.history
        if not rows:
            return f"No version history is given in the Fact Spec for {who} [1]."
        first, last = rows[0], rows[-1]

        def when(r: Any) -> str:
            return f" on {r.changed_on:%d/%m/%Y}" if r.changed_on else ""

        approval = re.search(r"approv|sign|authori", question, re.I)
        note = (
            "A Fact Spec does not record an approver; its version history names the person "
            "responsible for each change.\n"
            if approval
            else ""
        )
        head = note + (
            f"{who}: created by {first.person}{when(first)}; last changed by {last.person}"
            f"{when(last)} (version {last.version if last.version is not None else '?'}"
            + (f": {last.amends}" if last.amends else "")
            + f") [1]. {len(rows)} entries in the version history:"
        )
        shown = rows[-6:]
        lines = [
            f"- version {r.version if r.version is not None else 'creation'}, {r.person}"
            f"{when(r)}: {r.amends or '-'}"
            for r in shown
        ]
        older = (
            f"\n...and {len(rows) - len(shown)} earlier entries." if len(rows) > len(shown) else ""
        )
        return head + "\n" + "\n".join(lines) + older
    return None


async def _detail_answer(question: str, repo: Any) -> list[AskEvent] | None:
    kind = next((k for k, pat in _DETAILS if pat.search(question)), None)
    if kind is None:
        return None
    spec = resolve_spec(question, await repo.spec_index())
    if spec is None:
        return None
    details = await repo.spec_details(spec.doc_id)
    header = await repo.spec_header(spec.doc_id)
    if details is None or header is None:
        return None
    who = f"{header.nav_code or ''} {header.product or header.filename}".strip()
    text = _detail_text(kind, details, who, question)
    if text is None:
        return None
    return _done(text, [await _citation(repo, header)])


async def _attribute_answer(question: str, repo: Any) -> list[AskEvent] | None:
    if (card := await _summary_answer(question, repo)) is not None:
        return card
    if (detail := await _detail_answer(question, repo)) is not None:
        return detail
    field = next(((f, label) for f, pat, label in _ATTRIBUTES if pat.search(question)), None)
    if field is None:
        return None
    codes = NAV.findall(question)
    index = await repo.spec_index()
    spec = resolve_spec(question, index)
    if spec is None:
        if codes:
            return _done(f"No current Fact Spec has the NAV code {codes[0]}.")
        return None
    header = await repo.spec_header(spec.doc_id)
    if header is None:
        return None
    attr, label = field
    value = _value(header, attr)
    who = f"{header.nav_code or ''} {header.product or header.filename}".strip()
    if attr == "version" and value is not None:
        text = f"Version {value} of {who} [1]"
    else:
        text = (
            f"{label} of {who}: {value} [1]"
            if value is not None
            else f"The {label.lower()} is not given in the Fact Spec for {who} [1]."
        )
    return _done(text, [await _citation(repo, header)])


_COUNT = re.compile(r"\bhow (?:many|much)\b|\bnumber of\b|\bcount\b", re.I)
_SPECS = re.compile(r"\bspecs?\b|\bspecifications?\b|\bproducts?\b", re.I)
_CUSTOMERS = re.compile(
    r"\bwhich customers?\b|\bwho are (?:our|the) customers\b|\btop customers\b"
    r"|\bbiggest customers\b",
    re.I,
)


# Where the product group sits in a count question: "how many whole duck specs",
# "number of turkey products", "how many specs are there for whole duck".
_GROUP_PHRASES = [
    re.compile(
        r"\bhow (?:many|much)\s+(.+?)\s+(?:fact\s+)?(?:specs?|specifications?|products?)\b", re.I
    ),
    re.compile(
        r"\b(?:number of|count(?: of)?|total)\s+(?:the\s+)?(.+?)\s+"
        r"(?:fact\s+)?(?:specs?|specifications?|products?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:specs?|specifications?|products?)\b.*?\b(?:for|of|on|in|about|with)\s+"
        r"(?:the |our |all )?(.+)$",
        re.I,
    ),
]
# Words that carry no product: structure of the question and the conditions handled elsewhere.
_GROUP_STOP = {
    "how",
    "many",
    "much",
    "the",
    "our",
    "all",
    "any",
    "are",
    "there",
    "is",
    "do",
    "does",
    "we",
    "have",
    "has",
    "own",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "about",
    "number",
    "count",
    "total",
    "current",
    "fact",
    "spec",
    "specs",
    "specification",
    "specifications",
    "product",
    "products",
    "different",
    "distinct",
    "unique",
    "packaging",
    "code",
    "codes",
    "supplier",
    "suppliers",
    "use",
    "used",
    "using",
    "frozen",
    "fresh",
    "allergen",
    "allergens",
    "contain",
    "contains",
    "declare",
    "declaring",
    "made",
    "make",
    "list",
    "show",
    "what",
    "which",
    "please",
    "customer",
    "customers",
    "currently",
    "active",
    "per",
}


def count_group_words(question: str) -> list[str]:
    """Product words a count question is narrowed to ("whole duck"), or [] if it names none.

    A word that is not understood must narrow the count or be refused, never dropped: "how many
    whole duck specs" once answered with every spec in the table.
    """
    for pattern in _GROUP_PHRASES:
        if m := pattern.search(question):
            words = [
                w
                for w in re.findall(r"[A-Za-z0-9]+", m.group(1))
                if len(w) > 2 and w.lower() not in _GROUP_STOP
            ]
            if words:
                return words[:6]
    return []


def _name_in(question: str, names: list[str]) -> str | None:
    """The longest known name that appears in the question as whole words."""
    found = [
        n
        for n in names
        if n and re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", question, re.I)
    ]
    return max(found, key=len) if found else None


async def _aggregate_answer(question: str, repo: Any) -> list[AskEvent] | None:
    if _CUSTOMERS.search(question):
        counts = await repo.customer_counts(limit=8)
        if not counts:
            return None
        lines = [f"Customers with the most specs, across the {BASIS}:"]
        lines += [f"{i}. {name}: {n} specs" for i, (name, n) in enumerate(counts, start=1)]
        return _done("\n".join(lines))
    if not (_COUNT.search(question) and _SPECS.search(question)):
        return None
    if re.search(r"packag\w*", question, re.I):
        return None  # a packaging count: the packaging table answers it
    conditions: dict[str, Any] = {}
    parts: list[str] = []
    if re.search(r"\bfrozen\b", question, re.I):
        conditions["spec_type"] = "frozen"
        parts.append("frozen")
    elif re.search(r"\bfresh\b", question, re.I):
        conditions["spec_type"] = "fresh"
        parts.append("fresh")
    if re.search(r"\ballergens?\b", question, re.I) and not re.search(
        r"\b(?:no|without|free of|none)\b", question, re.I
    ):
        conditions["has_allergens"] = True
        parts.append("declaring allergens")
    vocab = await get_vocab(repo)
    found = _mentioned(question, vocab, "customer", vocab.customers)
    if found:  # a spelling people type ("Sainsbury") is found as the customer ("Sainsbury's")
        conditions["customer"] = found[0]
        parts.append(f"for {found[0]}")
    elif words := count_group_words(question):
        conditions["words"] = words
        parts.append(f'match "{" ".join(words)}"')
    count = await repo.count_specs(**conditions)
    if "words" in conditions and count == 0:
        group = " ".join(conditions["words"])
        if await repo.count_specs(words=conditions["words"]) == 0:  # the name matches nothing
            return _done(f'No current Fact Spec has a product name matching "{group}".')
    if not parts:
        return _done(f"There are {count} {BASIS}.")
    total = await repo.fact_spec_count()
    note = ""
    if "words" in conditions and await repo.group_mode(conditions["words"]) == "words":
        note = (
            f' No spec name contains "{" ".join(conditions["words"])}" as a phrase, so this uses '
            "the specs whose names contain all of those words, in any order."
        )
    verbs = {
        "frozen": "are frozen",
        "fresh": "are fresh",
        "declaring allergens": "declare allergens",
    }
    what = " and ".join(verbs.get(p, p if p.startswith("match") else f"are {p}") for p in parts)
    # the query behind the count, so the next question can refine it and every row can be downloaded
    query = SpecQuery(
        intent="count",
        filter=SpecFilter(
            product=" ".join(conditions["words"]) if "words" in conditions else None,
            customer=conditions.get("customer"),
            spec_type=conditions.get("spec_type"),
            allergens="yes" if conditions.get("has_allergens") else None,
        ),
    )
    export = ExportRef(query=query.model_dump(mode="json"), count=count) if count else None
    return _done(f"{count} of the {total} {BASIS} {what}.{note}", None, export)


async def answer_from_factspec(question: str, repo: Any) -> list[AskEvent] | None:
    """Events for a header or count question the table can answer, else None."""
    if await repo.fact_spec_count() == 0:
        return None
    return await _aggregate_answer(question, repo) or await _attribute_answer(question, repo)


async def scope_for_question(question: str, repo: Any) -> tuple[str | None, str | None, str]:
    """(doc_id to search, a note for the answer, the question to search with).

    One spec: search only that spec, with its NAV code swapped for its product name (a question
    made mostly of a code scores below the relevance cut-off and finds nothing). Several specs
    with the same product name: search stays open and the note lists their NAV codes.
    """
    found = match_specs(question, await repo.spec_index())
    if len(found) == 1:
        spec = found[0]
        rewritten = question
        if spec.product:
            rewritten = NAV.sub(spec.product, question, count=1)
        return spec.doc_id, None, rewritten
    if len(found) > 1:
        listed = "; ".join(
            f"{' '.join(s.nav_codes) or s.filename} ({s.customer or 'no customer'})"
            for s in found[:6]
        )
        more = f" and {len(found) - 6} more" if len(found) > 6 else ""
        return (
            None,
            (
                f"Note: {len(found)} current specs have this product name ({listed}{more}). The "
                "answer comes from one of them; give the NAV code to get a specific one."
            ),
            question,
        )
    return None, None, question


async def with_note(events: AsyncIterator[AskEvent], note: str) -> AsyncIterator[AskEvent]:
    """Add ``note`` to an answer that was given (not to a refusal), before its ``done`` event."""
    async for event in events:
        if isinstance(event, DoneEvent) and event.answered:
            text = f"\n\n{note}"
            yield TokenEvent(text=text)
            event = event.model_copy(update={"answer": event.answer + text})
        yield event
