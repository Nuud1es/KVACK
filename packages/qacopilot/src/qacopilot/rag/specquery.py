"""Questions that combine conditions, negate, compare, rank or list across the Fact Specs.

One handler per question shape kept dropping the parts it did not know: "how many Tesco duck
breast specs" counted every Tesco spec, "specs with no allergens" counted all of them. Here a
question is read once into a structured :class:`SpecQuery` (the model translates, code validates),
one SQL query is built from it, and the answer says how the question was understood, so a
misreading shows at once.

The language model only turns words into the structure. Every name in it is checked against what
exists in the data (customers, suppliers, product words, codes); the numbers come from SQL.
Anything that cannot be checked is reported, never guessed.
"""

import csv
import difflib
import io
import logging
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from qacopilot.llm import BaseLLMProvider
from qacopilot.rag.answer import (
    AskEvent,
    Citation,
    DoneEvent,
    ExportRef,
    SourcesEvent,
    TokenEvent,
)
from qacopilot.repositories.documents import DocumentRepository, SpecVocab

logger = logging.getLogger(__name__)

MAX_LISTED = 25
SNIPPET_CHARS = 240
VOCAB_TTL_S = 60.0

# --- the structure a question is read into -----------------------------------------------------


class PackCond(BaseModel):
    """A condition on a spec's packaging rows: the spec must (or must not) have such a row."""

    code: str | None = Field(default=None, description="A packaging code such as 7903818")
    supplier: str | None = Field(default=None, description="A packaging supplier, e.g. Reflex")
    kind: str | None = Field(
        default=None,
        description="A word in the packaging description: bag, film, label, box, tray, pad, "
        "strapping, tag, 'gas flush bag'",
    )
    section: Literal["primary", "secondary"] | None = None
    negate: bool = Field(
        default=False, description="True when the spec must NOT have such a packaging row"
    )


NumField = Literal[
    "max_life_days",
    "min_life_days",
    "cases_per_pallet",
    "layers_per_pallet",
    "boxes_per_layer",
    "ferrous_mm",
    "non_ferrous_mm",
    "stainless_mm",
    "pack_tare_g",
    "case_tare_g",
    "temp_min_c",
    "temp_max_c",
]
_NUM_FIELDS = set(NumField.__args__)  # type: ignore[attr-defined]
_OPS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "="}


class NumCond(BaseModel):
    """A number the spec states, compared with a value: "max life over 14 days"."""

    field: NumField
    op: Literal["gt", "gte", "lt", "lte", "eq"]
    value: float


class SpecFilter(BaseModel):
    product: str | None = Field(
        default=None, description="Words from the product name, as a phrase: 'whole duck'"
    )
    customer: str | None = None
    spec_type: Literal["frozen", "fresh"] | None = None
    allergens: Literal["yes", "no"] | None = Field(
        default=None, description="'yes': declares allergens; 'no': declares none"
    )
    version: int | None = Field(default=None, description="The spec is on exactly this version")
    year: int | None = Field(default=None, description="Version dated in this calendar year")
    packaging: list[PackCond] = Field(default_factory=list)
    numeric: list[NumCond] = Field(default_factory=list)
    storage_c: float | None = Field(
        default=None, description="Specs whose storage temperature range includes this value"
    )
    person: str | None = Field(
        default=None, description="A person named in the spec's version history"
    )
    person_role: Literal["any", "last", "created"] = Field(
        default="any", description="any: made any change; last: made the latest; created: wrote it"
    )


class SpecQuery(BaseModel):
    intent: Literal[
        "count",
        "list",
        "distinct",
        "rank",
        "compare",
        "common",
        "packaging",
        "oldest",
        "newest",
        "most_revised",
        "content",
        "unsupported",
    ]
    reason: str | None = Field(
        default=None, description="For unsupported: why the tables cannot answer it"
    )
    filter: SpecFilter = Field(default_factory=SpecFilter)
    compare_with: str | None = Field(
        default=None,
        description="For compare: the second product phrase (or customer) to set against the "
        "first one in filter",
    )
    group_by: (
        Literal["customer", "supplier", "code", "spec_type", "version", "year", "person"] | None
    ) = Field(default=None, description="For rank: what to count by")
    order_by: NumField | None = Field(
        default=None, description="For list: sort by this number ('the longest max life')"
    )
    descending: bool = Field(default=True, description="With order_by: highest first")
    distinct_of: Literal["customer", "supplier", "code"] | None = Field(
        default=None, description="For distinct: what to count the different values of"
    )
    section: Literal["primary", "secondary"] | None = Field(
        default=None, description="For packaging or rank by code/supplier: only this section"
    )
    kind: str | None = Field(
        default=None,
        description="For packaging or rank by code/supplier: only rows whose description has "
        "this word (bag, film, label, box, tray, pad, strapping)",
    )
    limit: int | None = Field(default=None, description="How many to show: 'top 3' means 3")


PROMPT = """You turn a question about Gressingham Foods' product specifications ("Fact Specs")
into a JSON query. Each spec has a customer, a product name, a type (frozen or fresh),
allergens, a version and date, and a list of packaging rows (code, description, supplier,
primary or secondary). Reply with the JSON only.

Intents:
- count: how many specs match.        - list: which/what specs or products match.
- distinct: how many DIFFERENT customers/suppliers/codes (distinct_of), among the matching specs.
- rank: which customer/supplier/code/type/version/year has the most (group_by), among the
  matching specs. "Which customers buy whole duck" = rank by customer with product 'whole duck'.
- compare: two groups side by side ("turkey or duck"): product = the first, compare_with = the
  second. For two customers use customer and compare_with.
- packaging: which packaging codes the matching specs use (use section or kind to narrow rows).
- common: which packaging codes BOTH of two groups use ("common to whole duck and duck breast"):
  product = the first, compare_with = the second.
- oldest / newest: the specs with the oldest / newest version date.
- most_revised: the specs on the highest version numbers ("revised the most").
- content: the question is about what a spec SAYS (storage temperature, process steps,
  ingredients, weights, shelf life, metal detection, who approved it) and not about counting,
  listing or ranking specs.
- unsupported: it cannot be answered from customers, product names, type, allergens, version
  dates or packaging rows (give the reason), e.g. "which packaging codes are not used by any
  spec" (only codes that are used are recorded).

Each spec also states: storage temperature (temp_min_c, temp_max_c), max life and the minimum
life into the customer's depot in days (max_life_days, min_life_days), the pallet layout
(cases_per_pallet, layers_per_pallet, boxes_per_layer), metal detection stick sizes in mm
(ferrous_mm, non_ferrous_mm, stainless_mm), pack_tare_g and case_tare_g, and a version history
naming the person responsible for each change.

Rules:
- A number comparison goes in filter.numeric ("max life over 14 days" = max_life_days gt 14).
  'kept at -18' / 'stored at 3 degrees' is storage_c. A month is 30 days.
- 'updated/changed by X' is person (role any); 'last updated by X' role last; 'written/created
  by X' role created. 'who has updated the most specs' is rank with group_by person.
- 'the longest max life', 'the heaviest case' is a list with order_by and descending, limit 1 or
  as asked. 'the shortest' sets descending false. It is NEVER a number condition: 'shortest max
  life' is order_by max_life_days ascending, not max_life_days lte 0.
- 'more than'/'over' is gt, 'fewer than'/'less than'/'under' is lt, 'at least' is gte,
  'at most'/'or less' is lte.
- Put EVERY condition in the question into the query. Never leave one out.
- product holds only product-name words ('whole duck', 'duck breast', 'turkey'); a customer
  name goes in customer, never in product. 'frozen'/'fresh' go in spec_type, not in product.
- 'don't use', 'no', 'without' a packaging item: a packaging condition with negate true.
  'specs with no allergens' means allergens 'no'.
- 'both X and Y' packaging: two packaging conditions.
- 'primary'/'secondary' packaging: section.
- Use the customer, supplier and product words below exactly as written when they match.
- 'top 3' / 'the 5 most' sets limit.

Examples:
Q: how many Tesco duck breast specs are there
{"intent":"count","filter":{"product":"duck breast","customer":"Tesco"}}
Q: how many specs do not use a gas flush bag
{"intent":"count","filter":{"packaging":[{"kind":"gas flush bag","negate":true}]}}
Q: which whole duck specs use 7903818
{"intent":"list","filter":{"product":"whole duck","packaging":[{"code":"7903818"}]}}
Q: which specs use both 7903818 and 7901619
{"intent":"list","filter":{"packaging":[{"code":"7903818"},{"code":"7901619"}]}}
Q: which has more specs, turkey or duck
{"intent":"compare","filter":{"product":"turkey"},"compare_with":"duck"}
Q: what primary packaging does Ocado use
{"intent":"packaging","filter":{"customer":"Ocado"},"section":"primary"}
Q: which customer buys the most whole duck
{"intent":"rank","filter":{"product":"whole duck"},"group_by":"customer","limit":1}
Q: how many different customers do we have
{"intent":"distinct","distinct_of":"customer"}
Q: what film do we use for duck legs
{"intent":"packaging","filter":{"product":"duck legs"},"kind":"film"}
Q: which specs are on version 1
{"intent":"list","filter":{"version":1}}
Q: what is the oldest spec
{"intent":"oldest"}
Q: which specs were updated in 2025
{"intent":"list","filter":{"year":2025}}
Q: which specs have been revised the most
{"intent":"most_revised"}
Q: what packaging is common to turkey and duck breast
{"intent":"common","filter":{"product":"turkey"},"compare_with":"duck breast"}
Q: how many specs use packaging from Reflex and Krehalon UK
{"intent":"count","filter":{"packaging":[{"supplier":"Reflex"},{"supplier":"Krehalon UK"}]}}
Q: which specs have a max life over 14 days
{"intent":"list","filter":{"numeric":[{"field":"max_life_days","op":"gt","value":14}]}}
Q: how many specs have more than 100 cases per pallet
{"intent":"count","filter":{"numeric":[{"field":"cases_per_pallet","op":"gt","value":100}]}}
Q: which specs are kept at -18
{"intent":"list","filter":{"storage_c":-18}}
Q: which specs did James Walton last update
{"intent":"list","filter":{"person":"James Walton","person_role":"last"}}
Q: who has updated the most specs
{"intent":"rank","group_by":"person"}
Q: which spec has the longest max life
{"intent":"list","order_by":"max_life_days","descending":true,"limit":1}
Q: which spec has the shortest max life
{"intent":"list","order_by":"max_life_days","descending":false,"limit":1}
Q: how many specs have fewer than 50 cases per pallet
{"intent":"count","filter":{"numeric":[{"field":"cases_per_pallet","op":"lt","value":50}]}}
Q: what is the storage temperature for frozen turkey
{"intent":"content"}
Q: which packaging codes are not used by any spec
{"intent":"unsupported","reason":"only packaging codes that are used are recorded"}
"""

# --- vocabulary: what a question can name ------------------------------------------------------

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "our",
    "all",
    "any",
    "are",
    "there",
    "have",
    "has",
    "how",
    "many",
    "much",
    "what",
    "which",
    "who",
    "list",
    "show",
    "spec",
    "specs",
    "specification",
    "product",
    "products",
    "packaging",
    "code",
    "codes",
    "use",
    "used",
    "uses",
    "using",
    "from",
    "that",
    "this",
    "about",
    "does",
    "not",
    "per",
    "also",
    "than",
    "most",
    "more",
    "less",
    "top",
    "different",
    "distinct",
    "unique",
    "customer",
    "customers",
    "supplier",
    "suppliers",
    "version",
    "frozen",
    "fresh",
    "allergen",
    "allergens",
    "primary",
    "secondary",
    "current",
    "latest",
    "oldest",
    "newest",
    "buy",
    "buys",
    "make",
    "made",
    "between",
    "both",
    "either",
    "when",
    "were",
    "was",
    "been",
    "being",
    "each",
    "every",
    "number",
    "count",
    "total",
    "common",
    "popular",
    "often",
    "bags",
    "bag",
    "film",
    "films",
    "label",
    "labels",
    "box",
    "boxes",
    "tray",
    "trays",
    "pad",
    "pads",
    "strapping",
    "tag",
    "tags",
    "updated",
    "revised",
    "year",
    "versions",
    "like",
    "tell",
    "give",
    "can",
    "you",
    "get",
    "name",
    "names",
    "its",
    "their",
    "them",
    "where",
}
KIND_WORDS = {
    "bag": ["bag", "pouch", "bdf"],
    "bags": ["bag", "pouch", "bdf"],
    "film": ["film", "lid", "web"],
    "films": ["film", "lid", "web"],
    "label": ["lab", "lbl", "label"],
    "labels": ["lab", "lbl", "label"],
    "box": ["box", "crate", "carton"],
    "boxes": ["box", "crate", "carton"],
    "crate": ["crate", "box"],
    "crates": ["crate", "box"],
    "tray": ["tray"],
    "trays": ["tray"],
    "pad": ["pad"],
    "pads": ["pad"],
    "strapping": ["strapping", "strap", "tape"],
    "strap": ["strapping", "strap", "tape"],
    "tag": ["tag"],
    "tags": ["tag"],
}


def _norm_name(text: str) -> str:
    """Compare names without case, apostrophes or a trailing s: Sainsbury's = Sainsburys."""
    base = re.sub(r"[^a-z0-9]+", "", text.lower().replace("’", "").replace("'", ""))
    return base[:-1] if base.endswith("s") and len(base) > 3 else base


def _windows(question: str) -> set[str]:
    """Runs of one to four words of the question with case, apostrophes and a plural s ignored."""
    tokens = re.findall(r"[A-Za-z0-9]+", question.replace("\u2019", "").replace("'", ""))
    return {_norm_name("".join(tokens[i : i + n])) for n in range(1, 5) for i in range(len(tokens))}


def _names_in(question: str, names: list[str]) -> list[str]:
    """Names mentioned as whole words, longest first, without one inside another: "Smart" and
    "Smart Packaging" in "from Smart Packaging" are one mention. Apostrophes, case and a plural s
    do not matter ("Sainsburys" finds "Sainsbury's")."""
    windows = _windows(question)
    found = sorted(
        {
            n
            for n in names
            if n
            and (
                re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", question, re.I)
                or (len(_norm_name(n)) >= 4 and _norm_name(n) in windows)
            )
        },
        key=len,
        reverse=True,
    )
    kept: list[str] = []
    for n in found:
        if not any(n.lower() in k.lower() for k in kept):
            kept.append(n)
    return kept


def _name_in(question: str, names: list[str]) -> str | None:
    found = _names_in(question, names)
    return found[0] if found else None


def _canon(kind: str, column: str) -> str:
    """SQL for a name with its alias applied (see ``DocumentRepository.canon``)."""
    return DocumentRepository.canon(kind, column)


def _with_aliases(vocab: SpecVocab, kind: str, names: list[str]) -> list[str]:
    """The names plus every spelling that is an alias of one, so either is found in a question."""
    return [*names, *vocab.aliases.get(kind, {})]


def canonical(vocab: SpecVocab, kind: str, name: str | None) -> str | None:
    """The one name a spelling stands for ("Sainsbury" -> "Sainsbury's"), or the name itself."""
    if not name:
        return name
    key = _norm_name(name)
    for raw, target in vocab.aliases.get(kind, {}).items():
        if _norm_name(raw) == key:
            return target
    return name


def _mentioned(question: str, vocab: SpecVocab, kind: str, names: list[str]) -> list[str]:
    """Names of this kind in the question as their canonical names, no name twice."""
    out: list[str] = []
    for found in _names_in(question, _with_aliases(vocab, kind, names)):
        target = canonical(vocab, kind, found)
        if target and not any(_norm_name(target) == _norm_name(x) for x in out):
            out.append(target)
    return out


def match_name(wanted: str, names: list[str]) -> tuple[str | None, list[str]]:
    """(canonical name, close suggestions): case, apostrophes and plurals do not matter."""
    index = {_norm_name(n): n for n in names}
    if (key := _norm_name(wanted)) in index:
        return index[key], []
    close = difflib.get_close_matches(key, list(index), n=3, cutoff=0.82)
    if close and difflib.SequenceMatcher(None, key, close[0]).ratio() >= 0.9:
        return index[close[0]], []
    return None, [index[c] for c in close]


def fix_product_words(phrase: str, vocab_words: set[str]) -> tuple[list[str], list[str]]:
    """The phrase as words, a misspelt one put right: ("duk breast" -> duck breast)."""
    words, notes = [], []
    for w in re.findall(r"[A-Za-z0-9]+", phrase):
        low = w.lower()
        if len(low) <= 2 or low in vocab_words or low.rstrip("s") in vocab_words:
            words.append(w)
        elif len(low) >= 4 and (c := difflib.get_close_matches(low, vocab_words, n=1, cutoff=0.8)):
            words.append(c[0])
            notes.append(f"{w} read as {c[0]}")
        else:
            words.append(w)
    return words, notes


# --- deciding whether a question needs this layer ---------------------------------------------

_NAV = re.compile(r"\b4\d{6}\b")
_CODE = re.compile(r"\b7[09]\d{5}\b")
_NEGATION = re.compile(
    r"\b(no|not|without|don'?t|doesn'?t|do not|does not|except|excluding)\b", re.I
)
_COMPARE = re.compile(
    r"\b(both|versus|vs\.?|compare|compared|more than|fewer|less than)\b|,\s*\w+ or \w+", re.I
)
_LISTING = re.compile(
    r"\b(list|show|which|who|what|name)\b.*\b(specs?|products?|customers?|suppliers?|versions?)\b"
    r"|\bwhich\b.*\b(most|least|fewest|biggest|largest|top)\b",
    re.I,
)
_FIELD_CUES = re.compile(
    r"\b(?:max(?:imum)?\.? ?life|shelf[- ]?life|min(?:imum)?\.? ?life|cases per pallet|"
    r"layers per pallet|"
    r"boxes per layer|ferrous|stainless|pack tare|case tare|(?:kept|stored?) at|"
    r"(?:updated?|changed?|amended?|written|created?|edited?) by|"
    r"who (?:has |have )?(?:updated?|changed?|wrote|created?)|"
    r"last (?:updated?|changed?)|longest|shortest|heaviest|lightest)\b",
    re.I,
)
_DISTINCT = re.compile(r"\bhow many\b.*\b(different|distinct|unique)\b", re.I)
_DATES = re.compile(
    r"\b(oldest|newest|revised|updated|on version \d+|version \d+|in 20\d\d|since 20\d\d)\b", re.I
)
_RANK = re.compile(r"\b(most|least|fewest|biggest|largest|top \d+|commonest)\b", re.I)
_AGGREGATE = re.compile(
    r"\bhow (many|much)\b|\bnumber of\b|\bcount\b|\blist\b|\bwhich (specs?|products?|customers?)\b"
    r"|\bwhat (specs?|products?) (do|are|have)\b",
    re.I,
)
_DATA_NOUN = re.compile(r"\b(specs?|specifications?|products?|customers?|suppliers?)\b", re.I)


def signals(question: str, vocab: SpecVocab) -> dict[str, Any]:
    """What the question mentions that a spec can be filtered or grouped by."""
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", question)}
    product_set = set(vocab.product_words)
    customers_found = _mentioned(question, vocab, "customer", vocab.customers)
    customer = customers_found[0] if customers_found else None
    suppliers = _mentioned(question, vocab, "supplier", vocab.suppliers)
    people = _mentioned(question, vocab, "person", vocab.people)
    if customer:  # "Tesco" is also a supplier name: here it is the customer
        suppliers = [x for x in suppliers if _norm_name(x) != _norm_name(customer)]
    named = {w.lower() for n in [customer, *suppliers] if n for w in re.findall(r"[A-Za-z0-9]+", n)}
    product_hits = [
        w
        for w in words
        if len(w) > 2
        and w not in _STOP
        and w not in named
        and (w in product_set or w.rstrip("s") in product_set)
    ]
    typos = [
        w
        for w in words
        if len(w) >= 3
        and w not in _STOP
        and w not in named
        and w not in product_set
        and w.rstrip("s") not in product_set
        and difflib.get_close_matches(w, product_set, n=1, cutoff=0.8)
    ]
    return {
        "customer": customer,
        "supplier": suppliers[0] if suppliers else None,
        "suppliers": suppliers,
        "people": people,
        "codes": _CODE.findall(question),
        "products": product_hits,
        "typos": typos,
        "type": bool(re.search(r"\b(frozen|fresh)\b", question, re.I)),
        "allergens": bool(re.search(r"\ballergens?\b", question, re.I)),
        "kind": bool(words & set(KIND_WORDS)),
        "section": bool(re.search(r"\b(primary|secondary)\b", question, re.I)),
        "negation": bool(_NEGATION.search(question)),
        "compare": bool(
            _COMPARE.search(question)
            or re.search(
                r"\b\w+ or \w+\b.*\b(more|most|common)\b|\b(more|most|common)\b.*\b\w+ or \w+\b",
                question,
                re.I,
            )
        ),
        "listing": bool(_LISTING.search(question) or _RANK.search(question)),
        "distinct": bool(_DISTINCT.search(question)),
        "dates": bool(_DATES.search(question)),
        "noun": bool(_DATA_NOUN.search(question)),
        "packaging": bool(re.search(r"\b(packag\w*|packing|codes?)\b", question, re.I)),
    }


def needs_engine(question: str, vocab: SpecVocab) -> bool:
    """True for a question the single-shape handlers would answer only in part.

    A question with one condition (one customer, one product group, one code, one supplier) is
    left to the handlers that already answer it exactly; this layer takes what combines,
    negates, compares, ranks, lists or has a word it must correct.
    """
    if _NAV.search(question):  # one named spec: the header and packaging handlers read it
        return False
    s = signals(question, vocab)
    others = sum(
        [
            bool(s["customer"]),
            bool(s["suppliers"]),
            bool(s["products"]),
            s["type"],
            s["allergens"],
            len(s["codes"]) > 0,
            s["kind"],
        ]
    )
    if len(s["codes"]) >= 2 or len(s["suppliers"]) >= 2 or s["dates"] or s["typos"]:
        return True
    if s["distinct"] and re.search(r"\bcustomers?\b", question, re.I):
        return True
    if re.search(r"\bcommon (to|between|in)\b|\bin common\b", question, re.I):
        return True
    if s["people"]:  # "which specs did James Walton change": a person in the version histories
        return True
    if _FIELD_CUES.search(question):
        return True  # a number the spec states, or who changed it: the model decides if it is
        # a question about content (then search reads it) or about the specs as a whole
    if s["negation"] and (others >= 1 or s["noun"]):
        return True
    if s["compare"] and (len(s["products"]) >= 2 or others >= 1):
        return True
    if s["section"] and others >= 1:
        return True
    if others >= 2:  # a second condition would be dropped by a single-shape handler
        return True
    only_code = others == 1 and len(s["codes"]) == 1
    return bool(others == 1 and s["listing"] and not s["packaging"] and not only_code)


def looks_like_aggregate(question: str, vocab: SpecVocab) -> bool:
    """A count or list over the specs that nothing could answer: search must not guess it."""
    if _NAV.search(question):
        return False
    return bool(_AGGREGATE.search(question)) and signals(question, vocab)["noun"]


GUARD_MESSAGE = (
    "I can't answer that reliably: it needs counting or listing across the specs, and I could "
    "not read the question into a precise query (search only sees a few passages and would "
    'guess). Try naming a customer, a product such as "whole duck", frozen or fresh, a '
    'packaging code or supplier, for example: "how many Tesco duck breast specs are frozen", '
    '"which customers buy whole duck", "which specs use 7903818 and 7901619".'
)

# --- reading a question ------------------------------------------------------------------------

_vocab_cache: dict[int, tuple[float, SpecVocab]] = {}


async def get_vocab(repo: Any) -> SpecVocab:
    """The vocabulary, kept for a minute per database (a request makes its own repository, so
    the key is the engine behind it)."""
    key = repo.vocab_key()
    cached = _vocab_cache.get(key)
    if cached and time.monotonic() - cached[0] < VOCAB_TTL_S:
        return cached[1]
    vocab: SpecVocab = await repo.spec_vocab()
    if len(_vocab_cache) > 8:
        _vocab_cache.clear()
    _vocab_cache[key] = (time.monotonic(), vocab)
    return vocab


async def read_question(llm: BaseLLMProvider, question: str, vocab: SpecVocab) -> SpecQuery | None:
    """Ask the model for the query; None if it cannot be read (then nothing is guessed)."""
    context = (
        f"Customers: {'; '.join(vocab.customers)}\n"
        f"Packaging suppliers: {'; '.join(vocab.suppliers[:80])}\n"
    )
    try:
        return await llm.extract(
            [
                {"role": "system", "content": PROMPT + "\n" + context},
                {"role": "user", "content": f"Q: {question}"},
            ],
            SpecQuery,
            retries=1,
            max_tokens=1024,
        )
    except Exception:
        logger.exception("could not read the question into a spec query")
        return None


class Understood(BaseModel):
    """The query after every name was checked against the data, and what was changed or wrong."""

    query: SpecQuery
    notes: list[str] = []
    problems: list[str] = []
    product_words: list[str] = []
    compare_words: list[str] = []
    kinds: list[str] = []  # description keywords for the packaging-row restriction


def _kind_keywords(kind: str) -> list[str]:
    """ILIKE patterns for a packaging type. A known word has synonyms (any of them); a phrase
    such as "gas flush bag" must appear with its words in order, not as any one of them."""
    low = kind.lower().strip()
    if low in KIND_WORDS:
        return KIND_WORDS[low]
    words = [w for w in re.findall(r"[a-z0-9]+", low) if w not in {"a", "the", "of"}]
    if not words:
        return [low]
    if len(words) == 1 and words[0] in KIND_WORDS:
        return KIND_WORDS[words[0]]
    return ["%".join(words)]


def _exact_in_question(value: str, question: str) -> bool:
    """The name is in the question as whole words (not just as part of a longer word)."""
    tokens = re.findall(r"[A-Za-z0-9]+", question.lower().replace("\u2019", "").replace("'", ""))
    windows = {
        _norm_name("".join(tokens[i : i + n])) for n in range(1, 5) for i in range(len(tokens))
    }
    return _norm_name(value) in windows


def _in_question(value: str, question: str) -> bool:
    """Whether a name the model put in the query is really in the question (it can copy a
    name from an example). Case, apostrophes, plurals and spacing do not matter."""
    squashed = re.sub(r"[^a-z0-9]+", "", question.lower().replace("\u2019", "").replace("'", ""))
    word = _norm_name(value)
    if bool(word) and (word in squashed or word.rstrip("s") in squashed):
        return True
    # a word the model corrected from a typo in the question ("turky" -> "turkey") still counts
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    return len(word) >= 3 and bool(difflib.get_close_matches(word, tokens, n=1, cutoff=0.78))


_FIELD_PHRASES: list[tuple[str, str]] = [
    (r"max(?:imum)?\.? ?life|shelf[- ]?life", "max_life_days"),
    (r"min(?:imum)?\.? ?life|life into depot", "min_life_days"),
    (r"cases per pallet|cases a pallet", "cases_per_pallet"),
    (r"layers per pallet", "layers_per_pallet"),
    (r"boxes per layer", "boxes_per_layer"),
    (r"case tare", "case_tare_g"),
    (r"pack tare", "pack_tare_g"),
    (r"non[- ]?ferrous", "non_ferrous_mm"),
    (r"stainless", "stainless_mm"),
    (r"ferrous", "ferrous_mm"),
]
_HIGH = re.compile(r"\b(longest|highest|heaviest|biggest|largest|greatest|most)\b", re.I)
_LOW = re.compile(r"\b(shortest|lowest|lightest|smallest|fewest|least)\b", re.I)


def _numbers_in(question: str) -> list[float]:
    return [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", question)]


def _ground_numbers(q: SpecQuery, question: str, notes: list[str]) -> None:
    """A number in the query must be in the question: the model once turned "shortest" into
    "at most 0". A month, a week or a kilogram may have been converted to days or grams."""
    nums = _numbers_in(question)

    def present(value: float) -> bool:
        return any(
            abs(value - n * k) < 1e-6 or (k == 1 and abs(value - n) < 1e-6)
            for n in nums
            for k in (1, 7, 30, 365, 1000)
        )

    f = q.filter
    kept = []
    for num in f.numeric:
        if present(num.value):
            kept.append(num)
        else:
            notes.append(
                f"ignored {num.field.replace('_', ' ')} {num.value:g} (not in the question)"
            )
    f.numeric = kept
    if f.storage_c is not None and not any(abs(f.storage_c - n) < 1e-6 for n in nums):
        notes.append(f"ignored storage {f.storage_c:g}°C (not in the question)")
        f.storage_c = None


def _order_by_field(q: SpecQuery, question: str, notes: list[str]) -> None:
    """ "The shortest max life" names a field and a direction: that is ordering, not a filter."""
    if q.order_by or q.intent not in ("list", "count"):
        return
    field = next((f for rx, f in _FIELD_PHRASES if re.search(rx, question, re.I)), None)
    if field is None or not (_HIGH.search(question) or _LOW.search(question)):
        return
    if re.search(r"\b(more|fewer|less|over|under|above|below|at least|at most)\b", question, re.I):
        return  # a comparison with a number, not a superlative
    q.intent = "list"
    q.order_by = field  # type: ignore[assignment]
    q.descending = not _LOW.search(question)
    if q.limit is None:
        single = re.search(r"\b(which|what) (?:spec|product)\b(?!s)", question, re.I)
        q.limit = 1 if single else 5
    notes.append(f"ordered by {field.replace('_', ' ')} (a superlative in the question)")


def _ground_flags(q: SpecQuery, question: str, notes: list[str]) -> None:
    """A condition the question does not mention is the model's invention: frozen/fresh,
    allergens, a version or a year ("how many turkey specs for Wholesale" once came back as
    "customer Wholesale, frozen", every time)."""
    f = q.filter
    low = question.lower()
    if f.spec_type and not re.search(rf"\b{f.spec_type}\b", low):
        notes.append(f"ignored {f.spec_type} (not in the question)")
        f.spec_type = None
    if f.allergens and not re.search(r"allergen", low):
        notes.append("ignored the allergens condition (not in the question)")
        f.allergens = None
    if f.year is not None and str(f.year) not in question:
        notes.append(f"ignored year {f.year} (not in the question)")
        f.year = None
    if f.person:  # last / created / any is decided by the words, not by the model
        wants_last = re.search(r"\b(last|latest|most recent|recently|newest)\b", low)
        wants_created = re.search(
            r"\b(creat\w*|wrote|written|writes|originat\w*|generat\w*|authored?|started)\b", low
        )
        role = "created" if wants_created else "last" if wants_last else "any"
        if role != f.person_role:
            notes.append(f"read as {role} change (from the words in the question)")
            f.person_role = role  # type: ignore[assignment]
    if f.version is not None and not re.search(
        rf"\bversion\s*{f.version}\b|\bv\s*{f.version}\b", low
    ):
        notes.append(f"ignored version {f.version} (not in the question)")
        f.version = None


def _named_words(q: SpecQuery, question: str, vocab: SpecVocab) -> set[str]:
    """Words that belong to a customer, supplier or person in the question (spellings ignored)."""
    names = [
        n
        for kind, known in (
            ("customer", vocab.customers),
            ("supplier", vocab.suppliers),
            ("person", vocab.people),
        )
        for n in _mentioned(question, vocab, kind, known)
    ]
    if q.filter.customer:
        names.append(q.filter.customer)
    names += [c.supplier for c in q.filter.packaging if c.supplier]
    return {_norm_name(w) for n in names for w in re.findall(r"[A-Za-z0-9]+", n)}


def _strip_named_from_product(
    q: SpecQuery, question: str, vocab: SpecVocab, notes: list[str]
) -> None:
    """The model sometimes puts a supplier or customer into the product ("product Smart
    Packaging"): those words are the name, not a product."""
    f = q.filter
    if not f.product:
        return
    named = _named_words(q, question, vocab)
    words = re.findall(r"[A-Za-z0-9]+", f.product)
    kept = [w for w in words if _norm_name(w) not in named]
    if len(kept) < len(words):
        notes.append(f'"{f.product}" is a name, not a product' if not kept else "")
        f.product = " ".join(kept) or None
    notes[:] = [n for n in notes if n]


def _restore_product(q: SpecQuery, question: str, vocab: SpecVocab, notes: list[str]) -> None:
    """Product words in the question that the model left out (it dropped "turkey" from "turkey
    specs for Wholesale"): words that are in product names, in the order asked, and are not a
    customer, supplier or person, a field word or a question word."""
    f = q.filter
    if f.product or q.intent in ("content", "unsupported", "oldest", "newest", "most_revised"):
        return
    named = _named_words(q, question, vocab)
    plain = question
    for rx, _ in _FIELD_PHRASES:  # "cases per pallet", "max life": a field, not a product
        plain = re.sub(rx, " ", plain, flags=re.I)
    plain = _FIELD_CUES.sub(" ", plain)
    product_set = set(vocab.product_words)
    words = [
        w
        for w in re.findall(r"[A-Za-z]+", plain)  # letters only: "100" is a number, not a product
        if len(w) > 2
        and w.lower() not in _STOP
        and _norm_name(w) not in named
        and (w.lower() in product_set or w.lower().rstrip("s") in product_set)
    ]
    if words and q.intent in (
        "count",
        "list",
        "rank",
        "distinct",
        "packaging",
        "compare",
        "common",
    ):
        f.product = " ".join(words)
        notes.append(f'added product "{f.product}" (named in the question)')


def _ground(q: SpecQuery, question: str, notes: list[str]) -> None:
    """Drop what the model invented: names, codes and phrases that are not in the question."""
    _ground_numbers(q, question, notes)
    _ground_flags(q, question, notes)
    _order_by_field(q, question, notes)
    f = q.filter
    if (
        f.customer
        and not _exact_in_question(f.customer, question)
        and _in_question(f.customer, question)
    ):
        typed = difflib.get_close_matches(
            _norm_name(f.customer), re.findall(r"[a-z0-9]+", question.lower()), n=1, cutoff=0.78
        )
        notes.append(f"{typed[0] if typed else 'name'} read as {f.customer}")
    if f.customer and not _in_question(f.customer, question):
        notes.append(f"ignored customer {f.customer} (not in the question)")
        f.customer = None
    if f.product:
        kept = [w for w in re.findall(r"[A-Za-z0-9]+", f.product) if _in_question(w, question)]
        if len(kept) < len(re.findall(r"[A-Za-z0-9]+", f.product)):
            lost = f.product if not kept else None
            if lost:
                notes.append(f'ignored product "{lost}" (not in the question)')
            f.product = " ".join(kept) or None
    for cond in list(f.packaging):
        for attr in ("code", "supplier"):
            val = getattr(cond, attr)
            if val and not _in_question(val, question):
                notes.append(f"ignored {attr} {val} (not in the question)")
                setattr(cond, attr, None)
        if not (cond.code or cond.supplier or cond.kind):
            f.packaging.remove(cond)


def _enrich(q: SpecQuery, question: str, vocab: SpecVocab, notes: list[str]) -> None:
    """Add what the question names but the model left out. The model reads the question into
    the structure, but it is not trusted to be complete: a supplier, customer, code, packaging
    type, section or frozen/fresh that is plainly in the question must be in the query."""
    f = q.filter
    low = question.lower()
    words = set(re.findall(r"[a-z0-9]+", low))
    negated = bool(_NEGATION.search(question))
    if not f.customer:
        found = _mentioned(question, vocab, "customer", vocab.customers)
        name = found[0] if found else None
        suppliers = _mentioned(question, vocab, "supplier", vocab.suppliers)
        used_as_supplier = any(
            c.supplier and name and _norm_name(c.supplier) == _norm_name(name) for c in f.packaging
        )
        if (
            name
            and not used_as_supplier
            and not (name in suppliers and re.search(r"\b(from|by|supplied|supplier)\b", low))
        ):
            f.customer = name
            notes.append(f"added customer {name} (named in the question)")
    if not f.spec_type and not negated:
        if "frozen" in words:
            f.spec_type = "frozen"
            notes.append("added frozen (named in the question)")
        elif "fresh" in words:
            f.spec_type = "fresh"
            notes.append("added fresh (named in the question)")
    if q.intent in ("packaging", "rank", "common"):
        if not q.section and (m := re.search(r"\b(primary|secondary)\b", low)):
            q.section = m.group(1)  # type: ignore[assignment]
        if not q.kind and (hit := sorted(words & set(KIND_WORDS))):
            q.kind = hit[0]
            notes.append(f"added {hit[0]} (named in the question)")
    for code in _CODE.findall(question):
        if not any(c.code == code for c in f.packaging):
            f.packaging.append(PackCond(code=code, negate=negated and "not" in words))
            notes.append(f"added code {code} (named in the question)")
    customer = f.customer
    plain = _FIELD_CUES.sub(" ", question)  # "case tare" is not a supplier called Tare
    for name in _mentioned(plain, vocab, "supplier", vocab.suppliers):
        if customer and _norm_name(name) == _norm_name(customer):
            continue
        if any(c.supplier and _norm_name(c.supplier) == _norm_name(name) for c in f.packaging):
            continue
        bare = next((c for c in f.packaging if not c.supplier and not c.code), None)
        if bare is not None:
            bare.supplier = name
        else:
            kind = q.kind if q.intent in ("list", "count", "distinct") else None
            f.packaging.append(PackCond(supplier=name, kind=kind))
        notes.append(f"added supplier {name} (named in the question)")


def _part_of_a_supplier(name: str, question: str, vocab: SpecVocab) -> bool:
    """The model read a word of a supplier named in the question ("Smart" of "Smart Packaging")
    as a customer."""
    words = {_norm_name(w) for w in re.findall(r"[A-Za-z0-9]+", name)}
    return bool(words) and any(
        words <= {_norm_name(w) for w in re.findall(r"[A-Za-z0-9]+", sup)}
        for sup in _mentioned(question, vocab, "supplier", vocab.suppliers)
    )


def validate(query: SpecQuery, vocab: SpecVocab, question: str = "") -> Understood:
    """Check every name against the data; correct typos and say so; report what is unknown."""
    out = Understood(query=query.model_copy(deep=True))
    q, f = out.query, out.query.filter
    if question:
        _ground(q, question, out.notes)
        _enrich(q, question, vocab, out.notes)
        _strip_named_from_product(q, question, vocab, out.notes)
        _restore_product(q, question, vocab, out.notes)
    product_set = set(vocab.product_words)
    if f.customer:
        f.customer = canonical(vocab, "customer", f.customer) or f.customer
        name, close = match_name(f.customer, vocab.customers)
        if name:
            if _norm_name(f.customer) != _norm_name(name):
                out.notes.append(f"{f.customer} read as {name}")
            f.customer = name
        elif _part_of_a_supplier(f.customer, question, vocab):
            out.notes.append(f'"{f.customer}" is part of a supplier name, not a customer')
            f.customer = None
        else:
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            out.problems.append(f'No customer called "{f.customer}".{hint}')
    if f.product:
        words, notes = fix_product_words(f.product, product_set)
        out.product_words, out.notes = words, [*out.notes, *notes]
        if not words:
            f.product = None
    if q.compare_with:
        name, _ = match_name(
            canonical(vocab, "customer", q.compare_with) or q.compare_with, vocab.customers
        )
        if f.customer and name:
            q.compare_with = name
        else:
            words, notes = fix_product_words(q.compare_with, product_set)
            out.compare_words, out.notes = words, [*out.notes, *notes]
    if f.person:
        f.person = canonical(vocab, "person", f.person) or f.person
        name, close = match_name(f.person, vocab.people)
        if name:
            if _norm_name(f.person) != _norm_name(name):
                out.notes.append(f"{f.person} read as {name}")
            f.person = name
        else:
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            out.problems.append(f'No person called "{f.person}" in the version histories.{hint}')
    for cond in f.packaging:
        if cond.code and cond.code not in vocab.codes:
            out.problems.append(f"No current spec uses packaging code {cond.code}.")
        if cond.supplier:
            cond.supplier = canonical(vocab, "supplier", cond.supplier) or cond.supplier
            name, close = match_name(cond.supplier, vocab.suppliers)
            if name:
                if _norm_name(cond.supplier) != _norm_name(name):
                    out.notes.append(f"{cond.supplier} read as {name}")
                cond.supplier = name
            else:
                hint = f" Did you mean {', '.join(close)}?" if close else ""
                out.problems.append(f'No packaging supplier called "{cond.supplier}".{hint}')
    if q.kind:
        out.kinds = _kind_keywords(q.kind)
    if question:
        out.notes.extend(_unknown_names(question, vocab, out))
    return out


_CHATTER = {
    "what",
    "which",
    "how",
    "who",
    "when",
    "where",
    "list",
    "show",
    "give",
    "tell",
    "please",
    "can",
    "could",
    "would",
    "and",
    "the",
    "are",
    "is",
    "do",
    "does",
    "we",
    "our",
    "all",
    "any",
}


def _unknown_names(question: str, vocab: SpecVocab, u: "Understood") -> list[str]:
    """Capitalised words in the question that match no customer, supplier or product word and
    that the query did not use. The model drops a name it does not recognise and the answer then
    covers every spec, so say so."""
    known = {
        w
        for n in [
            *vocab.customers,
            *vocab.suppliers,
            *vocab.people,
            *(raw for names in vocab.aliases.values() for raw in names),
        ]
        for w in re.findall(r"[a-z0-9]+", n.lower())
    }
    known |= set(vocab.product_words) | _STOP | _CHATTER
    used = {w.lower() for w in u.product_words + u.compare_words}
    out: list[str] = []
    for i, word in enumerate(re.findall(r"[A-Za-z0-9&'\u2019]+", question)):
        low = word.lower().rstrip("s").replace("'", "")
        if i == 0 or not word[0].isupper() or len(word) < 3 or word.isdigit():
            continue
        if low in known or word.lower() in known or low in used or _CODE.fullmatch(word):
            continue
        if difflib.get_close_matches(low, known, n=1, cutoff=0.8):
            continue
        out.append(f'did not recognise "{word}", so it was not applied')
    return out


def describe(u: Understood) -> str:
    """The query in words, so a misreading is visible: 'count of specs: product "whole duck"...'."""
    q, f = u.query, u.query.filter
    parts: list[str] = []
    if f.customer:
        parts.append(f"customer {f.customer}")
    if f.product:
        parts.append(f'product "{" ".join(u.product_words) or f.product}"')
    if f.spec_type:
        parts.append(f.spec_type)
    if f.allergens:
        parts.append("declares allergens" if f.allergens == "yes" else "declares no allergens")
    if f.version is not None:
        parts.append(f"on version {f.version}")
    if f.year:
        parts.append(f"dated {f.year}")
    for num in f.numeric:
        words = {
            "gt": "more than",
            "gte": "at least",
            "lt": "less than",
            "lte": "at most",
            "eq": "",
        }
        label = num.field.replace("_", " ")
        parts.append(f"{label} {words[num.op]} {num.value:g}".replace("  ", " "))
    if f.storage_c is not None:
        parts.append(f"stored at {f.storage_c:g}°C")
    if f.person:
        role = {"any": "changed by", "last": "last changed by", "created": "created by"}
        parts.append(f"{role[f.person_role]} {f.person}")
    for c in f.packaging:
        what = c.code or c.supplier or c.kind or "packaging"
        where = f"{c.section} " if c.section else ""
        parts.append(f"{'does not use' if c.negate else 'uses'} {where}{what}")
    first = " ".join(u.product_words) or f.customer
    second = " ".join(u.compare_words) or q.compare_with
    compared = ""
    if q.compare_with:
        compared = (
            f' ("{first}" and "{second}")' if q.intent == "common" else f' against "{second}"'
        )
    head = {
        "count": "count of specs",
        "list": "list of specs",
        "distinct": f"number of different {q.distinct_of or 'value'}s",
        "rank": f"{q.group_by or 'group'} ranked by number of specs",
        "compare": "comparison" + compared,
        "packaging": "packaging codes used",
        "common": "packaging codes used by both groups" + compared,
        "oldest": "oldest specs",
        "newest": "newest specs",
        "most_revised": "most revised specs",
        "content": "content question",
        "unsupported": "not answerable from the tables",
    }[q.intent]
    extra = []
    if q.section:
        extra.append(f"{q.section} only")
    if q.kind:
        extra.append(f"{q.kind} only")
    text = head + (": " + ", ".join(parts) if parts else " (all current specs)")
    if extra:
        text += f" [{', '.join(extra)}]"
    return text


# --- running it ---------------------------------------------------------------------------------


class _Sql:
    """Collects parameters so every value goes in as a parameter, never into the text."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self._n = 0

    def bind(self, value: Any) -> str:
        self._n += 1
        name = f"p{self._n}"
        self.params[name] = value
        return f":{name}"


def _row_conds(
    sql: _Sql, *, code: str | None, supplier: str | None, kinds: list[str], section: str | None
) -> list[str]:
    conds: list[str] = []
    if code:
        conds.append(f"p.code = {sql.bind(code)}")
    if supplier:
        low = sql.bind(supplier.lower())
        name = f"lower({_canon('supplier', 'p.supplier')})"
        conds.append(f"({name} = {low} OR {name} LIKE {low} || ' %')")
    if kinds:
        likes = " OR ".join(f"p.description ILIKE {sql.bind('%' + k + '%')}" for k in kinds)
        conds.append(f"({likes})")
    if section:
        conds.append(f"p.section = {sql.bind(section.upper())}")
    return conds


async def _spec_conditions(
    repo: Any, sql: _Sql, f: SpecFilter, words: list[str]
) -> tuple[list[str], str]:
    """The WHERE conditions on ``f`` (fact_specs) and ``d`` (documents), and the product mode."""
    conds: list[str] = []
    mode = "none"
    if f.customer:
        who_c = f"lower({_canon('customer', 'f.customer')})"
        conds.append(f"{who_c} = {sql.bind(f.customer.lower())}")
    if words:
        product, params, mode = await repo._product_condition(words)
        for key, value in params.items():  # re-bind under this query's names
            product = [c.replace(f":{key}", sql.bind(value)) for c in product]
        conds += product
    if f.spec_type:
        conds.append(f"f.spec_type ILIKE {sql.bind(f.spec_type + '%')}")
    if f.allergens == "yes":
        conds.append(
            "(f.allergens IS NOT NULL AND lower(f.allergens) NOT IN ('none', 'nil', 'n/a', ''))"
        )
    elif f.allergens == "no":
        conds.append("(f.allergens IS NULL OR lower(f.allergens) IN ('none', 'nil', 'n/a', ''))")
    if f.version is not None:
        conds.append(f"f.version = {sql.bind(f.version)}")
    if f.year:
        conds.append(f"EXTRACT(year FROM f.version_date) = {sql.bind(f.year)}")
    for num in f.numeric:
        conds.append(f"f.{num.field} {_OPS[num.op]} {sql.bind(num.value)}")
    if f.storage_c is not None:
        bound = sql.bind(f.storage_c)
        conds.append(f"(f.temp_min_c <= {bound} AND f.temp_max_c >= {bound})")
    if f.person:
        who = sql.bind(f.person.lower())
        history = "FROM spec_versions v WHERE v.doc_id = f.doc_id"
        if f.person_role == "last":
            conds.append(
                f"(SELECT lower({_canon('person', 'v.person')}) {history} "
                "ORDER BY v.changed_on DESC NULLS LAST, v.id DESC LIMIT 1) = " + who
            )
        elif f.person_role == "created":
            conds.append(
                f"(SELECT lower({_canon('person', 'v.person')}) {history} "
                "ORDER BY v.changed_on ASC NULLS LAST, v.id ASC LIMIT 1) = " + who
            )
        else:
            conds.append(
                f"EXISTS (SELECT 1 {history} AND lower({_canon('person', 'v.person')}) = {who})"
            )
    for cond in f.packaging:
        rows = _row_conds(
            sql,
            code=cond.code,
            supplier=cond.supplier,
            kinds=_kind_keywords(cond.kind) if cond.kind else [],
            section=cond.section,
        )
        exists = (
            "EXISTS (SELECT 1 FROM spec_packaging p WHERE p.doc_id = f.doc_id"
            + "".join(f" AND {c}" for c in rows)
            + ")"
        )
        conds.append(f"NOT {exists}" if cond.negate else exists)
    return conds, mode


def _cte(name: str, conds: list[str]) -> str:
    """A named selection of current specs, for queries that use two groups at once."""
    where = " AND ".join(conds) or "TRUE"
    return (
        f"{name} AS (SELECT f.doc_id FROM fact_specs f JOIN current c ON c.doc_id = f.doc_id "
        f"JOIN documents d ON d.doc_id = f.doc_id WHERE {where})"
    )


def _base(repo: Any, conds: list[str]) -> str:
    where = " AND ".join(conds) or "TRUE"
    return (
        f"WITH {repo.CURRENT_SQL}, base AS ("
        "SELECT f.*, d.filename FROM fact_specs f "
        "JOIN current c ON c.doc_id = f.doc_id "
        f"JOIN documents d ON d.doc_id = f.doc_id WHERE {where})"
    )


_ROWS = " FROM spec_packaging p JOIN base b ON b.doc_id = p.doc_id"  # packaging rows of the base
_USAGE_DESC = (
    " (SELECT x.description FROM spec_packaging x WHERE x.code = p.code AND x.description <> ''"
    " GROUP BY x.description ORDER BY count(*) DESC, x.description LIMIT 1)"
)
_USAGE_SUPPLIER = (
    " (SELECT x.supplier FROM spec_packaging x WHERE x.code = p.code AND x.supplier <> ''"
    " GROUP BY x.supplier ORDER BY count(*) DESC, x.supplier LIMIT 1)"
)


def _more(shown: int, total: int) -> str:
    return f"...and {total - shown} more." if total > shown else ""


def _usage_line(
    i: int, code: str, desc: str | None, supplier: str | None, n: int, kind: str
) -> str:
    detail = ", ".join(x for x in (desc or "description not read", supplier or "") if x)
    return f"{i}. {code}: {detail}{kind}, used by {n} specs"


async def _citations(repo: Any, rows: list[Any]) -> list[Citation]:
    out: list[Citation] = []
    for i, r in enumerate(rows[:5], start=1):
        found = await repo.get_page(r[0], 1)
        out.append(
            Citation(
                n=i,
                doc_id=r[0],
                filename=r[-1],
                page=1,
                score=1.0,
                snippet=(found[1].text.strip()[:SNIPPET_CHARS] if found else r[-1]),
            )
        )
    return out


def _events(
    text: str, citations: list[Citation], via: str, export: ExportRef | None = None
) -> list[AskEvent]:
    return [
        *([SourcesEvent(citations=citations)] if citations else []),
        TokenEvent(text=text),
        DoneEvent(
            answer=text,
            answered=True,
            cited=[c.n for c in citations],
            via=via,  # type: ignore[arg-type]
            export=export,
        ),
    ]


async def execute(repo: Any, u: Understood) -> list[AskEvent]:
    """Run an understood query and write the answer, starting with how it was understood."""
    q, f = u.query, u.query.filter
    head = f"Understood as: {describe(u)}."
    if u.notes:
        head += f" ({'; '.join(u.notes)}.)"
    if u.problems:
        return _events(f"{head}\n" + " ".join(u.problems), [], "fact_spec")
    total_specs = await repo.fact_spec_count()
    sql = _Sql()
    conds, mode = await _spec_conditions(repo, sql, f, u.product_words)
    base = _base(repo, conds)
    note = ""
    if f.product and mode == "words":
        note = (
            f'\nNo spec name contains "{" ".join(u.product_words)}" as a phrase, so this uses '
            "the specs whose names contain all of those words, in any order."
        )
    if f.product and mode == "none":
        return _events(
            f"{head}\nNo current Fact Spec has a product name matching "
            f'"{" ".join(u.product_words)}".',
            [],
            "fact_spec",
        )
    (matched,) = (await repo.query_rows(base + " SELECT count(*) FROM base", sql.params))[0]
    via = "fact_spec"
    export = ExportRef(query=u.query.model_dump(mode="json"), count=matched) if matched else None

    if q.intent == "count":
        text = f"{head}\n{matched} of the {total_specs} current Fact Specs match.{note}"
        return _events(text, [], via, export)

    if q.intent == "compare":
        sql2 = _Sql()
        g = f.model_copy(deep=True)
        words2: list[str] = u.product_words
        label_a = f.customer or " ".join(u.product_words) or "all specs"
        label_b = q.compare_with or ""
        if f.customer and q.compare_with and not u.compare_words:
            g.customer = q.compare_with
        else:
            g.product, words2 = q.compare_with, u.compare_words
        conds2, mode2 = await _spec_conditions(repo, sql2, g, words2)
        (other,) = (
            await repo.query_rows(_base(repo, conds2) + " SELECT count(*) FROM base", sql2.params)
        )[0]
        if matched == other:
            verdict = "They are equal."
        else:
            winner, win_n, lose = (
                (label_a, matched, other) if matched > other else (label_b, other, matched)
            )
            verdict = f'"{winner}" has more ({win_n} against {lose}).'
        text = f'{head}\n"{label_a}": {matched} specs; "{label_b}": {other} specs. {verdict}{note}'
        return _events(text, [], via)

    if q.intent in ("list", "oldest", "newest", "most_revised"):
        order = {
            "list": "nav_code, filename",
            "oldest": "version_date ASC NULLS LAST, nav_code",
            "newest": "version_date DESC NULLS LAST, nav_code",
            "most_revised": "version DESC NULLS LAST, nav_code",
        }[q.intent]
        limit = 10 if q.intent == "most_revised" else 5
        if q.intent == "list":
            limit = min(q.limit or MAX_LISTED, MAX_LISTED)
        if q.limit and q.intent != "list":
            limit = min(q.limit, MAX_LISTED)
        shown_field = q.order_by if q.intent == "list" else None
        if shown_field:  # sort by a number the spec states, and show that number
            order = f"{shown_field} {'DESC' if q.descending else 'ASC'} NULLS LAST, nav_code"
            if not q.limit:
                limit = 10
        extra = f", {shown_field}" if shown_field else ", NULL"
        rows = await repo.query_rows(
            base + " SELECT doc_id, nav_code, product, customer, version, version_date, filename"
            f"{extra} FROM base ORDER BY {order} LIMIT {int(limit)}",
            sql.params,
        )
        if not rows:
            return _events(f"{head}\nNo current Fact Spec matches.", [], via)
        title = {
            "list": (
                f"{matched} of the {total_specs} current Fact Specs match"
                + (
                    f", by {str(shown_field).replace('_', ' ')} "
                    f"({'highest' if q.descending else 'lowest'} first):"
                    if shown_field
                    else ":"
                )
            ),
            "oldest": "The oldest by version date:",
            "newest": "The newest by version date:",
            "most_revised": "The most revised (highest version number):",
        }[q.intent]
        lines = [f"{head}", title]
        for i, r in enumerate(rows, start=1):
            when = f", {r[5]:%d/%m/%Y}" if r[5] else ""
            lines.append(
                f"{i}. {r[1] or '-'} {r[2] or r[6]}"
                + (f" ({r[3]})" if r[3] else "")
                + f", version {r[4]}{when}"
                + (
                    f", {shown_field.replace('_', ' ')}: {r[7]:g}"
                    if shown_field and r[7] is not None
                    else ""
                )
                + f" [{i}]"
            )
        if more := _more(len(rows), matched) if q.intent == "list" and not q.limit else "":
            lines.append(more)
        if q.intent in ("oldest", "newest"):
            missing = (
                await repo.query_rows(
                    base + " SELECT count(*) FROM base WHERE version_date IS NULL", sql.params
                )
            )[0][0]
            if missing:
                lines.append(
                    f"({missing} matching specs have no version date and are left out "
                    "of the order.)"
                )
        if note:
            lines.append(note.strip())
        return _events(
            "\n".join(lines), await _citations(repo, [(r[0], r[6]) for r in rows]), via, export
        )

    if q.intent == "common":
        g = f.model_copy(deep=True)
        words_b = u.compare_words
        if f.customer and q.compare_with and not u.compare_words:
            g.customer = q.compare_with
        else:
            g.product = q.compare_with
        conds_b, _ = await _spec_conditions(repo, sql, g, words_b)
        rows_c = _row_conds(sql, code=None, supplier=None, kinds=u.kinds, section=q.section)
        in_a = "p.doc_id IN (SELECT doc_id FROM a)"
        in_b = "p.doc_id IN (SELECT doc_id FROM b)"
        rows = await repo.query_rows(
            f"WITH {repo.CURRENT_SQL}, {_cte('a', conds)}, {_cte('b', conds_b)}"
            f" SELECT p.code, count(DISTINCT CASE WHEN {in_a} THEN p.doc_id END) na,"
            f" count(DISTINCT CASE WHEN {in_b} THEN p.doc_id END) nb,"
            + _USAGE_DESC
            + ","
            + _USAGE_SUPPLIER
            + " FROM spec_packaging p WHERE (p.doc_id IN (SELECT doc_id FROM a) OR p.doc_id IN"
            " (SELECT doc_id FROM b))"
            + "".join(f" AND {c}" for c in rows_c)
            + " GROUP BY p.code HAVING count(DISTINCT CASE WHEN "
            + in_a
            + " THEN p.doc_id END) > 0 AND count(DISTINCT CASE WHEN "
            + in_b
            + " THEN p.doc_id END) > 0 ORDER BY LEAST(count(DISTINCT CASE WHEN "
            + in_a
            + " THEN p.doc_id END), count(DISTINCT CASE WHEN "
            + in_b
            + " THEN p.doc_id END)) DESC, p.code LIMIT 15",
            sql.params,
        )
        label_a = f.customer or " ".join(u.product_words) or "group A"
        label_b = q.compare_with or "group B"
        if not rows:
            return _events(f"{head}\nNo packaging code is used by both groups.", [], "packaging")
        lines = [head, f'Packaging codes used by both "{label_a}" and "{label_b}":']
        for i, r in enumerate(rows, start=1):
            detail = ", ".join(x for x in (r[3] or "description not read", r[4] or "") if x)
            lines.append(
                f'{i}. {r[0]}: {detail}, {r[1]} "{label_a}" specs and {r[2]} "{label_b}" specs'
            )
        return _events("\n".join(lines), [], "packaging")

    row_conds = _row_conds(sql, code=None, supplier=None, kinds=u.kinds, section=q.section)
    row_where = "".join(f" AND {c}" for c in row_conds)
    limit = min(q.limit or 5, 15)

    if q.intent == "distinct":
        what = q.distinct_of or "customer"
        if what == "customer":
            (n,) = (
                await repo.query_rows(
                    base
                    + f" SELECT count(DISTINCT lower({_canon('customer', 'customer')})) FROM base"
                    + " WHERE customer IS NOT NULL",
                    sql.params,
                )
            )[0]
        elif what == "supplier":
            (n,) = (
                await repo.query_rows(
                    base
                    + " SELECT count(DISTINCT p.supplier)"
                    + _ROWS
                    + " WHERE p.supplier <> ''"
                    + row_where,
                    sql.params,
                )
            )[0]
        else:
            (n,) = (
                await repo.query_rows(
                    base + " SELECT count(DISTINCT p.code)" + _ROWS + " WHERE TRUE" + row_where,
                    sql.params,
                )
            )[0]
        via = "packaging" if what != "customer" else "fact_spec"
        scope = "all current specs" if matched == total_specs else f"the {matched} matching specs"
        return _events(f"{head}\n{n} different {what}s among {scope}.{note}", [], via, export)

    # rank / packaging: group the matching specs
    group = q.group_by if q.intent == "rank" else "code"
    if q.intent == "packaging" and q.group_by in ("supplier",):
        group = "supplier"
    if group == "customer":
        rows = await repo.query_rows(
            base
            + f" SELECT min({_canon('customer', 'customer')}), count(*) n FROM base"
            + " WHERE customer IS NOT NULL"
            + f" GROUP BY lower({_canon('customer', 'customer')}) ORDER BY n DESC, 1 LIMIT {limit}",
            sql.params,
        )
        lines = [head, f"Customers among the {matched} matching specs, most first:"]
        lines += [f"{i}. {r[0]}: {r[1]} specs" for i, r in enumerate(rows, start=1)]
    elif group == "person":
        rows = await repo.query_rows(
            base
            + f" SELECT {_canon('person', 'v.person')}, count(DISTINCT v.doc_id) n"
            + " FROM spec_versions v JOIN base b ON b.doc_id = v.doc_id"
            + f" GROUP BY {_canon('person', 'v.person')}"
            + f" ORDER BY n DESC, 1 LIMIT {limit}",
            sql.params,
        )
        lines = [head, f"People in the version histories of the {matched} matching specs:"]
        lines += [f"{i}. {r[0]}: changed {r[1]} specs" for i, r in enumerate(rows, start=1)]
    elif group in ("spec_type", "version", "year"):
        expr = {
            "spec_type": "lower(spec_type)",
            "version": "version",
            "year": "EXTRACT(year FROM version_date)::int",
        }[group]
        rows = await repo.query_rows(
            base
            + f" SELECT {expr}, count(*) n FROM base WHERE {expr} IS NOT NULL"
            + f" GROUP BY 1 ORDER BY n DESC, 1 LIMIT {limit}",
            sql.params,
        )
        lines = [head, f"By {group} among the {matched} matching specs, most first:"]
        lines += [f"{i}. {r[0]}: {r[1]} specs" for i, r in enumerate(rows, start=1)]
    elif group == "supplier":
        rows = await repo.query_rows(
            base
            + f" SELECT {_canon('supplier', 'p.supplier')}, count(DISTINCT p.doc_id) n,"
            + " count(DISTINCT p.code)"
            + _ROWS
            + " WHERE p.supplier <> ''"
            + row_where
            + f" GROUP BY {_canon('supplier', 'p.supplier')} ORDER BY n DESC, 1 LIMIT {limit}",
            sql.params,
        )
        lines = [head, f"Packaging suppliers among the {matched} matching specs, most first:"]
        lines += [f"{i}. {r[0]}: {r[1]} specs, {r[2]} codes" for i, r in enumerate(rows, start=1)]
        via = "packaging"
    else:  # code
        rows = await repo.query_rows(
            base
            + " SELECT p.code, count(DISTINCT p.doc_id) n,"
            + _USAGE_DESC
            + ","
            + _USAGE_SUPPLIER
            + ", min(lower(p.section))"
            + _ROWS
            + " WHERE TRUE"
            + row_where
            + f" GROUP BY p.code ORDER BY n DESC, p.code LIMIT {max(limit, 1) if q.limit else 15}",
            sql.params,
        )
        lines = [head, f"Packaging codes used by the {matched} matching specs, most used first:"]
        lines += [
            _usage_line(i, r[0], r[2], r[3], r[1], f" ({r[4]})" if r[4] else "")
            for i, r in enumerate(rows, start=1)
        ]
        via = "packaging"
    if not rows:
        return _events(f"{head}\nNothing found among the {matched} matching specs.", [], via)
    if note:
        lines.append(note.strip())
    return _events("\n".join(lines), [], via, export)


# --- follow-up questions ("give me all from Tesco" after "list whole duck specs") ---

_FOLLOW_UP = re.compile(
    r"^\s*(?:and|also|now|then|but|only|just|what about|how about|which of (?:those|them)"
    r"|of (?:those|them)|those|them|from|for|give me (?:all|the)|show (?:me )?(?:all|the)"
    r"|filter|narrow|exclude|excluding|without|with|that (?:are|use|have)|the ones|ones)\b",
    re.I,
)
_CONTINUABLE = {
    "count",
    "list",
    "distinct",
    "rank",
    "packaging",
    "oldest",
    "newest",
    "most_revised",
}


_THOSE = re.compile(r"\b(?:(?:of|from|among|in) )?(?:those|them|these)\b|\bthe above\b", re.I)


def is_follow_up(question: str) -> bool:
    """The question refines the one before: it starts like a refinement ("from Tesco", "only
    frozen") or points back at it ("how many of those use Reflex")."""
    if len(question.split()) > 14:
        return False
    return bool(_FOLLOW_UP.match(question) or _THOSE.search(question))


def strip_follow_up_cue(question: str) -> str:
    """The question without its cue words, so "only" or "those" are not read as product words."""
    plain = _THOSE.sub(" ", _FOLLOW_UP.sub(" ", question, count=1))
    return re.sub(r"\s+", " ", plain).strip() or question


def _set(value: Any) -> bool:
    return value not in (None, "", [], "any")


def merge_with_context(context: SpecQuery, new: SpecQuery) -> SpecQuery:
    """The previous query with this one's conditions laid over it: a condition the new question
    states replaces the old one, the rest carries over. Packaging and number conditions add up."""
    merged = context.model_copy(deep=True)
    merged.intent = new.intent
    for name in SpecFilter.model_fields:
        value = getattr(new.filter, name)
        if name in ("packaging", "numeric"):
            old = getattr(merged.filter, name)
            have = {c.model_dump_json() for c in old}
            setattr(
                merged.filter, name, [*old, *(c for c in value if c.model_dump_json() not in have)]
            )
        elif name == "person_role":
            if new.filter.person:
                merged.filter.person_role = value
        elif _set(value):
            setattr(merged.filter, name, value)
    return merged


async def _continue(
    repo: Any, context: SpecQuery, question: str, query: SpecQuery, vocab: SpecVocab
) -> list[AskEvent] | None:
    """Answer a refinement of the previous answer, or None if it is a question of its own."""
    if query.intent in ("content", "count", "list"):
        # a refinement keeps the kind of answer unless the words change it: "how many" counts,
        # anything else ("give me all", "only frozen", "what about Lidl") lists or keeps the last
        counting = re.search(r"\b(how many|how much|number of|count)\b", question, re.I)
        kept = context.intent if context.intent in ("count", "list") else "list"
        listing = re.search(r"\b(give me|show|list|which)\b", question, re.I)
        intent = "count" if counting else "list" if listing else kept
        query = query.model_copy(update={"intent": intent})
    fresh = validate(query, vocab, question)
    f = fresh.query.filter
    if fresh.problems or query.intent not in _CONTINUABLE or f.product:
        return None  # a problem to report, or a new subject: not a refinement
    if not any(_set(getattr(f, n)) for n in SpecFilter.model_fields if n != "person_role"):
        if re.search(r"\b(show|list|give me|which)\b", question, re.I):
            # "show me all of them": no new condition, so list what the last answer counted
            before = validate(context, vocab, "")
            again = before.model_copy(deep=True)
            again.query.intent = "list"
            again.notes = [f"listing your last question: {describe(before)}"]
            return await execute(repo, again)
        return None
    before = validate(context, vocab, "")
    merged = validate(merge_with_context(before.query, fresh.query), vocab, "")
    merged.notes = [*fresh.notes, f"continuing from your last question: {describe(before)}"]
    return await execute(repo, merged)


async def answer_from_specquery(
    question: str,
    repo: Any,
    llm: BaseLLMProvider | None,
    context: SpecQuery | None = None,
) -> list[AskEvent] | None:
    """Events for a question that combines conditions, else None (single-shape handlers run).

    ``context`` is the previous answer's query, so a refinement ("give me all from Tesco") is
    read together with it."""
    if llm is None or await repo.fact_spec_count() == 0:
        return None
    vocab = await get_vocab(repo)
    follow = context is not None and is_follow_up(question)
    if not follow and not needs_engine(question, vocab):
        return None
    plain = strip_follow_up_cue(question) if follow else question
    query = await read_question(llm, plain, vocab)
    if query is not None and query.intent == "content" and follow and context is not None:
        continued = await _continue(repo, context, plain, query, vocab)
        if continued is not None:
            return continued
    if query is None or query.intent == "content":
        return None  # nothing to compute: what a spec says is read by search
    if query.intent == "unsupported":
        why = query.reason or "the spec tables do not hold that"
        text = f"I can't answer that from the spec tables: {why}."
        return _events(text, [], "fact_spec")
    if follow and context is not None:
        continued = await _continue(repo, context, plain, query, vocab)
        if continued is not None:
            return continued
        if not needs_engine(question, vocab):
            return None
    return await execute(repo, validate(query, vocab, question))


# --- short or vague questions ------------------------------------------------------------------

MENU = (
    "I answer questions about the product specs (Fact Specs) from their tables. For example:\n"
    '- Counts: "how many Tesco duck breast specs are frozen", "how many specs have no '
    'allergens", "how many different customers do we have"\n'
    '- Lists: "which specs are for Lidl", "list the products for Mindful Chef", '
    '"which specs use both 7903818 and 7901619"\n'
    '- Packaging: "what packaging codes are used for whole duck", "what primary packaging '
    'does Ocado use", "what film do we use for duck legs", "top 3 packaging codes"\n'
    '- Rankings and comparisons: "which customers buy whole duck", "which has more specs, '
    'turkey or duck", "what packaging is common to turkey and duck breast"\n'
    '- One spec: "tell me about spec 4056877", "what packaging does 4056877 use", '
    '"what is the max life for 4056877"\n'
    '- Dates: "what is the oldest spec", "which specs were updated in 2025"'
)
_HELP = {
    "help",
    "examples",
    "menu",
    "what can you do",
    "what can i ask",
    "specs",
    "spec",
    "packaging",
}


async def answer_vague(question: str, repo: Any) -> list[AskEvent] | None:
    """A menu for "help" or a bare topic, and a short profile for a bare customer or product
    word, instead of one random passage. Only for questions of one to three words."""
    tokens = re.findall(r"[A-Za-z0-9&'\u2019]+", question)
    if not tokens or len(tokens) > 3 or _NAV.search(question) or _CODE.search(question):
        return None
    if await repo.fact_spec_count() == 0:
        return None
    low = " ".join(t.lower() for t in tokens)
    if low in _HELP:
        return _events(MENU, [], "fact_spec")
    vocab = await get_vocab(repo)
    name, _ = match_name(
        canonical(vocab, "customer", " ".join(tokens)) or " ".join(tokens), vocab.customers
    )
    words = [t for t in tokens if t.lower() not in _STOP]
    if name:
        total = await repo.count_specs(customer=name)
        frozen = await repo.count_specs(customer=name, spec_type="frozen")
        _, usage = await repo.packaging_for_products(name.split(), limit=3)
        top = "; ".join(f"{u.code} ({u.description or 'description not read'})" for u in usage)
        text = (
            f"{name}: {total} current specs ({frozen} frozen, {total - frozen} other). "
            f"Most used packaging: {top}.\n"
            f'Try: "list the {name} specs", "what packaging does {name} use", '
            f'"how many {name} specs are frozen".'
        )
        return _events(text, [], "fact_spec")
    if words and all(w.lower() in set(vocab.product_words) for w in words):
        total = await repo.count_specs(words=words)
        if total:
            group = " ".join(words)
            text = (
                f'"{group}": {total} current specs have it in the product name.\n'
                f'Try: "what packaging codes are used for {group}", "which customers buy '
                f'{group}", "how many {group} specs are frozen".'
            )
            return _events(text, [], "fact_spec")
    return None


# --- download: every row of a table answer ------------------------------------------------------

EXPORT_COLUMNS = [
    "NAV code",
    "Product",
    "Customer",
    "Type",
    "Status",
    "Version",
    "Version date",
    "Allergens",
    "File name",
]
MAX_EXPORT_ROWS = 5000
_EXPORTABLE = {"count", "list", "distinct", "rank", "packaging", "oldest", "newest", "most_revised"}
_EXPORT_ORDER = {
    "oldest": "version_date ASC NULLS LAST, nav_code",
    "newest": "version_date DESC NULLS LAST, nav_code",
    "most_revised": "version DESC NULLS LAST, nav_code",
}


class ExportError(ValueError):
    """The query cannot be exported; the message is safe to show."""


async def export_specs(repo: Any, query: SpecQuery) -> list[tuple[Any, ...]]:
    """Every current spec the query selects (not just the 25 an answer lists).

    The query comes from the browser, so it is validated exactly like one read from a question:
    names are checked against the data and every value goes into the SQL as a parameter.
    """
    if query.intent not in _EXPORTABLE:
        raise ExportError("That kind of answer is a comparison and has no list of specs to export.")
    u = validate(query, await get_vocab(repo))
    if u.problems:
        raise ExportError(" ".join(u.problems))
    sql = _Sql()
    conds, _ = await _spec_conditions(repo, sql, u.query.filter, u.product_words)
    order = _EXPORT_ORDER.get(query.intent, "nav_code, filename")
    rows = await repo.query_rows(
        _base(repo, conds)
        + " SELECT nav_code, product, customer, spec_type, status, version, version_date,"
        " allergens, filename FROM base"
        f" ORDER BY {order} LIMIT {MAX_EXPORT_ROWS}",
        sql.params,
    )
    return [tuple(r) for r in rows]


def _safe(cell: Any) -> str:
    """A cell for a spreadsheet: text that would run as a formula is made plain text."""
    text = "" if cell is None else (f"{cell:%d/%m/%Y}" if hasattr(cell, "strftime") else str(cell))
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def csv_text(rows: list[tuple[Any, ...]]) -> str:
    """The rows as CSV that Excel opens correctly (UTF-8 with a byte-order mark)."""
    out = io.StringIO()
    out.write("\ufeff")
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(EXPORT_COLUMNS)
    for row in rows:
        writer.writerow([_safe(c) for c in row])
    return out.getvalue()
