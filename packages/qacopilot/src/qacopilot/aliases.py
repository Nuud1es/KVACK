"""Suggest which spellings are one customer, supplier or person (nothing is applied by itself).

Specs are typed by hand over years, so one customer appears as "Sainsbury", "Sainsbury's" and
"Sainsbury’s". Counting and listing by customer then splits them. This module only *suggests*
groups; a person reviews the file and approves rows before anything changes
(``python -m qa_api.aliases``).

Two levels of confidence:

- ``safe``: the spellings differ only in case, punctuation, apostrophes or a plural s.
- ``check``: they differ by a word such as "Certification/Retail", "Foods", "Ltd", or look like
  a typo of each other. Probably the same name, but it is a judgement.

A string that joins two names ("Booker/Brakes") is never suggested: it names two customers.
"""

import difflib
import re
from dataclasses import dataclass

_NOISE_WORDS = re.compile(
    r"\b(certification|certificate|retail|customer order|ltd|limited|uk|plc|foods?|the|group)\b",
    re.I,
)


@dataclass(frozen=True)
class Proposal:
    kind: str
    raw: str
    canonical: str
    specs: int
    confidence: str  # safe | check
    reason: str


def safe_key(name: str) -> str:
    """Case, apostrophes, punctuation and a plural s do not make a different name."""
    low = name.lower().replace("’", "").replace("'", "")
    key = re.sub(r"[^a-z0-9]+", "", low)
    return key[:-1] if key.endswith("s") and len(key) > 4 else key


def loose_key(name: str) -> str:
    """Also ignore words like "Certification/Retail", "Foods" and "Ltd"."""
    low = _NOISE_WORDS.sub(" ", name.lower().replace("’", "").replace("'", ""))
    key = re.sub(r"[^a-z0-9]+", "", low)
    return key[:-1] if key.endswith("s") and len(key) > 4 else key


def _joins_two_names(name: str) -> bool:
    return (
        "/" in name
        and not re.search(r"certification/\w*|certificate/", name, re.I)
        or bool(re.search(r"\s/\s", name))
    )


def _tidy(name: str) -> str:
    return name.replace("’", "'").strip()


_TAG = re.compile(r"\b(certification|certificate|retail|customer order)\b", re.I)


def _pick(names: list[str], counts: dict[str, int]) -> str:
    """The name to use: the most used spelling, never a "Certification/Retail" tagged variant."""
    plain = [n for n in names if not _TAG.search(n)] or names
    return _tidy(max(plain, key=lambda n: (counts[n], -len(n))))


def suggest(counts: dict[str, int], kind: str = "customer") -> list[Proposal]:
    """Groups of spellings that are probably one name, most used spelling as the canonical one."""
    names = [n for n in counts if n and n.strip() and not _joins_two_names(n)]
    safe: dict[str, list[str]] = {}
    for n in names:
        safe.setdefault(safe_key(n), []).append(n)
    loose: dict[str, list[str]] = {}
    for key, group in safe.items():
        loose.setdefault(loose_key(group[0]) or key, []).extend(group)
    # typos: two loose keys that are almost the same
    keys = list(loose)
    merged: dict[str, str] = {}
    for a in keys:
        for b in difflib.get_close_matches(a, keys, n=2, cutoff=0.85):
            if a < b and len(a) >= 6:
                merged[b] = merged.get(a, a)
    groups: dict[str, list[str]] = {}
    for key, group in loose.items():
        groups.setdefault(merged.get(key, key), []).extend(group)

    out: list[Proposal] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        canonical = _pick(group, counts)
        canon_safe = safe_key(canonical)
        for raw in sorted(group, key=lambda n: -counts[n]):
            if _tidy(raw) == canonical and raw == canonical:
                continue
            same = safe_key(raw) == canon_safe
            if same and raw == canonical:
                continue
            out.append(
                Proposal(
                    kind=kind,
                    raw=raw,
                    canonical=canonical,
                    specs=counts[raw],
                    confidence="safe" if same else "check",
                    reason=(
                        "same name, spelling or case"
                        if same
                        else "differs by a word such as Certification/Retail, Foods or Ltd, "
                        "or looks like a typo"
                    ),
                )
            )
    return sorted(out, key=lambda p: (p.confidence != "safe", p.canonical.lower(), -p.specs))


def suggest_people(counts: dict[str, int]) -> list[Proposal]:
    """A first name on its own that belongs to exactly one full name ("Peter")."""
    full = [n for n in counts if " " in n.strip()]
    out: list[Proposal] = []
    for n in counts:
        if " " in n.strip():
            continue
        matches = [f for f in full if f.split()[0].lower() == n.lower()]
        if len(matches) == 1:
            out.append(
                Proposal(
                    "person",
                    n,
                    matches[0],
                    counts[n],
                    "check",
                    "a first name that matches one person",
                )
            )
    return out + [
        p for p in suggest({n: c for n, c in counts.items() if " " in n.strip()}, "person")
    ]
