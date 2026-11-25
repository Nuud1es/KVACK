"""Names that need a person to decide: is this spelling an alias, or a genuinely new name?

``aliases.suggest`` finds spellings that look like one name. This adds the ones it cannot be
sure of: a name used by only a few specs that resembles a more common one. Each item says what
looks likely ("probably an alias of X" or "probably a new name") and lists the closest names, so
the person only has to confirm. Nothing is applied here.
"""

import difflib
from typing import Literal

from pydantic import BaseModel

from qacopilot.aliases import Proposal, _joins_two_names, loose_key, suggest, suggest_people

RARE_SPECS = 2  # a name used by this few specs or fewer may be a slip, so it is worth a look
MAYBE_CUTOFF = 0.6  # how alike two names must be to be listed as a possible match
LIKELY_ALIAS = 0.8  # at or above this a rare name is recommended as an alias


class NameCandidate(BaseModel):
    name: str
    specs: int


class NameReview(BaseModel):
    kind: Literal["customer", "supplier", "person"]
    name: str
    specs: int
    recommendation: Literal["alias", "new_name"]
    suggested: str | None  # the name to make it an alias of, when the recommendation is alias
    confidence: Literal["safe", "check", "unclear"]
    reason: str
    candidates: list[NameCandidate]  # the closest existing names, to pick another one


def _settled(kind: str, name: str, done: set[tuple[str, str]]) -> bool:
    return (kind, name.strip().lower()) in done


def _proposals(kind: str, counts: dict[str, int]) -> list[Proposal]:
    return suggest_people(counts) if kind == "person" else suggest(counts, kind)


def review_names(
    counts: dict[str, dict[str, int]],
    aliased: set[tuple[str, str]],
    decided: set[tuple[str, str]],
    targets: set[tuple[str, str]] | None = None,
) -> list[NameReview]:
    """What needs a decision. ``counts``: kind -> spelling -> specs; ``aliased`` and ``decided``
    hold (kind, lower-cased spelling) for what is already settled; ``targets`` the names an
    alias points to (already reviewed)."""
    targets = targets or set()
    out: list[NameReview] = []
    for kind, names in counts.items():
        done = aliased | decided

        listed: set[str] = set()
        for p in _proposals(kind, names):
            if _settled(kind, p.raw, done):
                continue
            listed.add(p.raw)
            listed.add(p.canonical)
            out.append(
                NameReview(
                    kind=kind,  # type: ignore[arg-type]
                    name=p.raw,
                    specs=p.specs,
                    recommendation="alias",
                    suggested=p.canonical,
                    confidence=p.confidence,  # type: ignore[arg-type]
                    reason=p.reason,
                    candidates=[NameCandidate(name=p.canonical, specs=names.get(p.canonical, 0))],
                )
            )
        keys = {n: loose_key(n) or n.lower() for n in names}
        for name, specs in names.items():
            if specs > RARE_SPECS or name in listed or _settled(kind, name, done | targets):
                continue
            if kind == "customer" and _joins_two_names(name):
                continue  # "Booker/Brakes" names two customers: no single answer
            others = [n for n in names if n != name and names[n] > specs]
            by_key = {keys[n]: n for n in others}
            close = difflib.get_close_matches(keys[name], list(by_key), n=3, cutoff=MAYBE_CUTOFF)
            if not close:
                continue  # nothing resembles it: plainly a new name, nobody needs to look
            best = difflib.SequenceMatcher(None, keys[name], close[0]).ratio()
            alias = best >= LIKELY_ALIAS
            out.append(
                NameReview(
                    kind=kind,  # type: ignore[arg-type]
                    name=name,
                    specs=specs,
                    recommendation="alias" if alias else "new_name",
                    suggested=by_key[close[0]] if alias else None,
                    confidence="unclear",
                    reason=(
                        "used by few specs and very like a more common name"
                        if alias
                        else "used by few specs; resembles other names but is probably its own"
                    ),
                    candidates=[
                        NameCandidate(name=by_key[k], specs=names[by_key[k]]) for k in close
                    ],
                )
            )
    order = {"safe": 0, "check": 1, "unclear": 2}
    return sorted(out, key=lambda r: (order[r.confidence], r.kind, r.name.lower()))
