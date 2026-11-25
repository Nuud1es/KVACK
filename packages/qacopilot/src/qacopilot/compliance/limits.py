"""Deterministic limit checking. No LLM: the model extracts text, this code decides.

Handles the formats seen on lab reports and specs: ``<10``, ``>1000``, ``2.20E+03``,
``1.0E+05 cfu/g``, ``< 1.0E+05 cfu/g`` and presence tests such as ``Not detected in 25g``.
Anything it cannot read confidently is ``REVIEW`` for a person, never a guess.
"""

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"  # cannot be decided automatically


class Value(BaseModel):
    """A number with an optional ``<`` / ``>`` qualifier (a censored lab result)."""

    number: float
    qualifier: Literal["<", ">"] | None = None


class Limit(BaseModel):
    """A maximum (``max``), a minimum (``min``) or a presence test (``absent``)."""

    kind: Literal["max", "min", "absent"]
    number: float | None = None
    text: str = ""


# "1,200" is a thousands separator; a comma used any other way ("1,5") is ambiguous, so it is not
# parsed at all and the result goes to a person. The lookahead stops a partial match of "12,5".
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?!\d|[.,]\d)"
_VALUE = re.compile(rf"^\s*(?P<q>[<>]=?|≤|≥)?\s*(?P<n>{_NUMBER})")
_ABSENT = re.compile(
    r"not\s+detected|non[-\s]?detected|\bnd\b|absent|negative|nil|none\s+detected|not\s+found",
    re.IGNORECASE,
)
_PRESENT = re.compile(r"\bdetected\b|present|positive", re.IGNORECASE)


def says_absent(text: str) -> bool:
    """True for "not detected" style wording ("ND", "absent", "negative", ...)."""
    return bool(_ABSENT.search(text))


def parse_numeric_value(text: str) -> Value | None:
    """Parse ``'<10'``, ``'2.20E+03'``, ``'> 1000 cfu/g'``; None if no leading number."""
    m = _VALUE.match(text)
    if not m:
        return None
    q = m.group("q")
    qualifier: Literal["<", ">"] | None = None
    if q in ("<", "<=", "≤"):
        qualifier = "<"
    elif q in (">", ">=", "≥"):
        qualifier = ">"
    return Value(number=float(m.group("n").replace(",", "")), qualifier=qualifier)


def parse_limit(text: str) -> Limit | None:
    """Parse a limit; a bare number is read as a maximum, as on micro specs."""
    t = text.strip()
    if _ABSENT.search(t) and not re.search(r"[<>]\s*\d", t):
        return Limit(kind="absent", text=t)
    m = _VALUE.match(t)
    if not m:
        return None
    q = m.group("q")
    kind: Literal["max", "min"] = "min" if q in (">", ">=", "≥") else "max"
    return Limit(kind=kind, number=float(m.group("n").replace(",", "")), text=t)


def evaluate_result(result: str, limit: Limit) -> tuple[Verdict, str]:
    """Verdict for one result against one limit, with a short reason for the reviewer."""
    if limit.kind == "absent":
        if _ABSENT.search(result):
            return Verdict.PASS, "not detected, as required"
        if _PRESENT.search(result):
            return Verdict.FAIL, "detected where none is allowed"
        value = parse_numeric_value(result)
        if value and value.qualifier == "<":
            return Verdict.REVIEW, "'<' result against a presence test"
        if value and value.number == 0:
            return Verdict.PASS, "zero"
        if value:
            return Verdict.FAIL, "positive count where none is allowed"
        return Verdict.REVIEW, f"cannot read result {result!r}"

    assert limit.number is not None
    value = parse_numeric_value(result)
    if value is None:
        if _ABSENT.search(result):
            return Verdict.PASS if limit.kind == "max" else Verdict.FAIL, "not detected"
        return Verdict.REVIEW, f"cannot read result {result!r}"
    n, lim = value.number, limit.number

    if limit.kind == "max":
        if value.qualifier == "<":  # true value is below n
            return (Verdict.PASS, f"below {n:g}, within {lim:g}") if n <= lim else (
                Verdict.REVIEW, f"below {n:g}, which is above the limit {lim:g}")  # fmt: skip
        if value.qualifier == ">":  # true value is above n
            return (Verdict.FAIL, f"above {n:g}, over {lim:g}") if n >= lim else (
                Verdict.REVIEW, f"above {n:g}; limit {lim:g} may be exceeded")  # fmt: skip
        return (Verdict.PASS, f"{n:g} within {lim:g}") if n <= lim else (
            Verdict.FAIL, f"{n:g} exceeds {lim:g}")  # fmt: skip

    # minimum
    if value.qualifier == ">":
        return (Verdict.PASS, f"above {n:g}, meets {lim:g}") if n >= lim else (
            Verdict.REVIEW, f"above {n:g}; may or may not reach {lim:g}")  # fmt: skip
    if value.qualifier == "<":
        return (Verdict.FAIL, f"below {n:g}, under {lim:g}") if n <= lim else (
            Verdict.REVIEW, f"below {n:g}; may be under {lim:g}")  # fmt: skip
    return (Verdict.PASS, f"{n:g} meets {lim:g}") if n >= lim else (
        Verdict.FAIL, f"{n:g} below {lim:g}")  # fmt: skip
