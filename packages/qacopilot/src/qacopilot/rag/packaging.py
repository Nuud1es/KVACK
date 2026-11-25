"""Packaging questions answered from the Fact Spec packaging table (no LLM, no search).

Counting across every spec ("what is the most used packaging code") cannot be answered from the
five passages search returns, so the packaging rows are read into a table at upload and answered
with SQL. Only clear shapes are taken: a ranking, a packaging code, or a NAV code with the word
"packaging". Anything else returns None and goes to document search.
"""

import difflib
import re
from typing import Any

from qacopilot.rag.answer import (
    AskEvent,
    Citation,
    DoneEvent,
    ExportRef,
    SourcesEvent,
    TokenEvent,
)

PACKAGING_WORD = re.compile(r"\b(packag\w*|packing|film|tray|pad|label)\b", re.I)
CODE = re.compile(r"\b(7[09]\d{5})\b")
NAV = re.compile(r"\b(4\d{6})\b")
RANKING = re.compile(
    r"\b(most|top|commonest|common|popular|often|frequent\w*|widely|biggest|highest)\b", re.I
)
TOP_N = re.compile(r"\btop\s+(\d{1,2})\b", re.I)
MAX_LISTED = 10
SNIPPET_CHARS = 240


# "packaging codes used for whole duck", "what packaging does whole duck use",
# "which packaging is on duck breast": the product group is what follows.
_PHRASE_PATTERNS = [
    re.compile(
        r"\bpackag\w*\s+(?:does|do|is used (?:for|on|in)|is on|is in)\s+"
        r"(?:the |our )?(.+?)\s+use\b",
        re.I,
    ),
    re.compile(
        r"\b(?:codes?|packag\w*|packing|materials?|items?)\b.*?"
        r"\b(?:used\s+)?(?:for|on|in|by|with)\s+(?:the |our |all )?(.+)$",
        re.I,
    ),
    re.compile(r"\b(?:codes?|packag\w*)\s+(?:of|for)\s+(?:the |our |all )?(.+)$", re.I),
]
_PHRASE_TRAILING = re.compile(
    r"\s*(?:products?|specs?|specifications?|ones|items|lines|please|\?|\.)+\s*$", re.I
)
_STOP = {"the", "our", "all", "and", "for", "with", "use", "used", "are", "what", "which", "any"}


def product_words(question: str) -> list[str]:
    """The product-group words a packaging question is about ("whole duck"), or []."""
    for pattern in _PHRASE_PATTERNS:
        if m := pattern.search(question):
            phrase = _PHRASE_TRAILING.sub("", m.group(1).strip())
            words = [
                w
                for w in re.findall(r"[A-Za-z0-9]+", phrase)
                if len(w) > 2 and w.lower() not in _STOP
            ]
            if words:
                return words[:6]
    return []


def words_note(group: str) -> str:
    """Said when no spec name has the phrase and the answer uses specs with all the words."""
    return (
        f' No spec name contains "{group}" as a phrase, so this uses the specs whose names '
        "contain all of those words, in any order."
    )


_ATTRIBUTE = re.compile(r"\b(weigh|weight|dimensions?|size)\b", re.I)


def fix_typos(question: str) -> str:
    """Put a misspelt "packaging" ("packging", "pakaging", "packagin") right.

    People type fast, and a question that is not recognised as a packaging question goes to
    search, which cannot count or list codes and answered something unrelated.
    """

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word) < 6 or word.lower() in {"packaging", "package", "packages", "packing"}:
            return word
        ratio = difflib.SequenceMatcher(None, word.lower(), "packaging").ratio()
        return "packaging" if ratio >= 0.8 else word

    return re.sub(r"[A-Za-z]+", fix, question)


def is_packaging_question(question: str) -> bool:
    return bool(CODE.search(question) or PACKAGING_WORD.search(question))


def _row_text(use: Any) -> str:
    parts = [use.description or "description not read", use.supplier or ""]
    return ", ".join(p for p in parts if p)


async def _citation(n: int, spec: Any, repo: Any, code: str | None = None) -> Citation:
    rows = await repo.packaging_of_spec(spec.doc_id)
    mine = [r for r in rows if code is None or r.code == code]
    snippet = "; ".join(f"{r.code} x{r.qty or '?'} {r.description}".strip() for r in mine[:4])[
        :SNIPPET_CHARS
    ]
    page = mine[0].page if mine else 1
    return Citation(
        n=n,
        doc_id=spec.doc_id,
        filename=spec.filename,
        page=page,
        score=1.0,
        snippet=snippet or spec.filename,
    )


def _finish(
    text: str, citations: list[Citation], export: ExportRef | None = None
) -> list[AskEvent]:
    return [
        *([SourcesEvent(citations=citations)] if citations else []),
        TokenEvent(text=text),
        DoneEvent(
            answer=text,
            answered=True,
            cited=[c.n for c in citations],
            via="packaging",
            export=export,
        ),
    ]


_HOW_MANY_TOTAL = re.compile(
    r"how (?:many|much) (?:different |distinct |unique )?(?:packaging )?(codes|suppliers)\b", re.I
)
_HOW_MANY_SUPPLIER = re.compile(
    r"how (?:many|much) (?:current )?(?:specs?|products?)\b.*\bpackaging\b", re.I
)
_TOP_SUPPLIER = re.compile(
    r"which (?:packaging )?suppliers?\b.*\b(?:most|top|common|often)\b"
    r"|(?:most|top|biggest)\b.*\bpackaging suppliers?",
    re.I,
)


def _name_in(question: str, names: list[str]) -> str | None:
    found = [
        n
        for n in names
        if n and re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", question, re.I)
    ]
    return max(found, key=len) if found else None


async def _aggregate(question: str, repo: Any, total: int, basis: str) -> list[AskEvent] | None:
    """Counts over the whole packaging table: codes, suppliers, a supplier's share."""
    if m := _HOW_MANY_TOTAL.search(question):
        if words := product_words(question):  # "how many packaging codes for whole duck"
            group = " ".join(words)
            specs, codes, suppliers = await repo.packaging_group_totals(words)
            if specs == 0:
                return _finish(
                    f'No current Fact Spec has a product name or customer matching "{group}".', []
                )
            n, kind = (
                (codes, "codes") if m.group(1).lower() == "codes" else (suppliers, "suppliers")
            )
            note = words_note(group) if await repo.group_mode(words) == "words" else ""
            return _finish(
                f"{n} different packaging {kind} are used by the {specs} current specs "
                f'matching "{group}".{note}',
                [],
            )
        codes, suppliers = await repo.packaging_totals()
        text = (
            f"{codes} different packaging codes appear {basis}."
            if m.group(1).lower() == "codes"
            else f"{suppliers} different packaging suppliers appear {basis}."
        )
        return _finish(text, [])
    suppliers_all = await repo.supplier_counts()
    if _HOW_MANY_SUPPLIER.search(question):
        from qacopilot.rag.specquery import _mentioned, get_vocab

        vocab = await get_vocab(repo)  # a spelling of a supplier finds the supplier
        found = _mentioned(question, vocab, "supplier", [s[0] for s in suppliers_all])
        name = found[0] if found else _name_in(question, [s[0] for s in suppliers_all])
        if name is not None and any(s[0] == name for s in suppliers_all):
            _, specs, codes = next(s for s in suppliers_all if s[0] == name)
            return _finish(
                f"{specs} of {total} current specs use packaging from {name} ({codes} codes).", []
            )
    if _TOP_SUPPLIER.search(question) and suppliers_all:
        lines = [f"Packaging suppliers used by the most specs {basis}:"]
        for i, (name, specs, codes) in enumerate(suppliers_all[:5], start=1):
            lines.append(f"{i}. {name}: {specs} specs, {codes} codes")
        return _finish("\n".join(lines), [])
    return None


async def _for_product_group(
    question: str, words: list[str], repo: Any, basis: str
) -> list[AskEvent] | None:
    """The codes used by the specs of one product group, e.g. every "whole duck" spec."""
    specs, usage = await repo.packaging_for_products(words, limit=15)
    name = " ".join(words)
    if not specs:
        return _finish(
            f'No current Fact Spec has a product name or customer matching "{name}".', []
        )
    if not usage:
        return _finish(
            f'{len(specs)} current specs match "{name}", but none lists coded packaging.', []
        )
    shown = specs[:5]
    citations = [await _citation(i, s, repo) for i, s in enumerate(shown, start=1)]
    lines = [
        f'Packaging codes used by the {len(specs)} current specs matching "{name}" '
        f"(most used first):"
    ]
    for i, u in enumerate(usage, start=1):
        kind = f" ({u.section})" if u.section else ""
        lines.append(
            f"{i}. {u.code}: {_row_text(u)}{kind}, used by {u.specs} of {len(specs)} specs"
        )
    examples = "; ".join(
        f"[{i}] {s.nav_code or '-'} {s.product or s.filename}" for i, s in enumerate(shown, start=1)
    )
    lines.append(
        f"Example specs: {examples}" + (f" and {len(specs) - 5} more." if len(specs) > 5 else ".")
    )
    if await repo.group_mode(words) == "words":
        lines.append(words_note(name).strip())
    lines.append("Ask for one spec by its NAV code to see its full packaging list.")
    return _finish("\n".join(lines), citations)


async def answer_from_packaging(question: str, repo: Any) -> list[AskEvent] | None:
    """Events for a packaging question the table can answer, else None (use search)."""
    question = fix_typos(question)
    if not is_packaging_question(question):
        return None
    total = await repo.fact_spec_count()
    if total == 0:
        return None
    basis = f"across the {total} current Fact Specs (the latest version of each product)"

    if m := CODE.search(question):
        code = m.group(1)
        use = await repo.packaging_code(code)
        if use is None:
            text = f"Packaging code {code} does not appear in any current Fact Spec."
            return [
                TokenEvent(text=text),
                DoneEvent(answer=text, answered=True, cited=[], via="packaging"),
            ]
        asked = _ATTRIBUTE.findall(question)
        if asked:  # "what is the weight of 7903818": answer that, not the usage list
            details = await repo.packaging_code_details(code)
            facts = {
                "weight": details["weight"],
                "dimension": details["dimension"],
                "supplier": details["supplier"],
                "description": details["description"],
            }
            wanted = {
                "weigh": "weight",
                "weight": "weight",
                "dimensions": "dimension",
                "dimension": "dimension",
                "size": "dimension",
                "supplier": "supplier",
            }
            keys = [wanted[a.lower()] for a in asked if a.lower() in wanted] or [
                "weight",
                "dimension",
            ]
            said = ", ".join(f"{k}: {facts[k] or 'not given'}" for k in dict.fromkeys(keys))
            text = (
                f"{code} ({details['description'] or 'description not read'}): {said} "
                f"(the most common value across {use.specs} specs)."
            )
            return _finish(text, [])
        specs = await repo.specs_using_code(code)
        shown = specs[:MAX_LISTED]
        citations = [await _citation(i, s, repo, code) for i, s in enumerate(shown, start=1)]
        lines = [
            f"{code}: {_row_text(use)}. Used by {use.specs} of {total} current specs.",
            *[
                f"[{i}] {s.nav_code or '-'} {s.product or s.filename}"
                + (f" ({s.customer})" if s.customer else "")
                + (f", {s.section.lower()}, qty {s.qty}" if s.section and s.qty else "")
                for i, s in enumerate(shown, start=1)
            ],
        ]
        if len(specs) > len(shown):
            lines.append(f"...and {len(specs) - len(shown)} more.")
        from qacopilot.rag.specquery import PackCond, SpecFilter, SpecQuery

        used = SpecQuery(intent="list", filter=SpecFilter(packaging=[PackCond(code=code)]))
        export = ExportRef(query=used.model_dump(mode="json"), count=use.specs)
        return _finish("\n".join(lines), citations, export if use.specs else None)

    if m := NAV.search(question):
        specs = await repo.find_fact_specs(nav_code=m.group(1))
        if len(specs) == 1:
            spec = specs[0]
            rows = await repo.packaging_of_spec(spec.doc_id)
            lines = [
                f"Packaging for {spec.nav_code} {spec.product or ''} "
                f"(version {spec.version or '?'}) [1]:",
                *[
                    f"{r.section.title()}: {r.code}, qty {r.qty or '?'}, "
                    f"{r.description or 'description not read'}"
                    + (f", {r.supplier}" if r.supplier else "")
                    for r in rows
                ],
            ]
            if not rows:
                lines.append("No coded packaging rows were found in this spec.")
            return _finish("\n".join(lines), [await _citation(1, spec, repo)])
        return None

    if answered := await _aggregate(question, repo, total, basis):
        return answered

    if words := product_words(question):
        found = await _for_product_group(question, words, repo, basis)
        if found is not None:
            return found

    if RANKING.search(question) and re.search(r"\bcodes?\b|\bpackag\w*|\bused\b", question, re.I):
        limit = min(int(n.group(1)), MAX_LISTED) if (n := TOP_N.search(question)) else 5
        section = (
            "PRIMARY"
            if re.search(r"\bprimary\b", question, re.I)
            else "SECONDARY"
            if re.search(r"\bsecondary\b", question, re.I)
            else None
        )
        usage = await repo.packaging_usage(limit=limit, section=section)
        if not usage:
            return None
        ranked: list[Citation] = []
        lines = [f"Most used {section.lower() + ' ' if section else ''}packaging codes {basis}:"]
        for i, u in enumerate(usage, start=1):
            first = (await repo.specs_using_code(u.code, limit=1))[:1]
            if first:
                ranked.append(await _citation(len(ranked) + 1, first[0], repo, u.code))
            ref = f" [{len(ranked)}]" if first else ""
            lines.append(f"{i}. {u.code}: {_row_text(u)}, used by {u.specs} specs{ref}")
        lines.append("Each citation is one example spec that uses the code.")
        return _finish("\n".join(lines), ranked)
    return None
