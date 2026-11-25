"""Score extracted fields against expected values, deterministically.

Comparison is lenient about formatting but not about content: numbers compare as numbers
(``<10`` equals ``< 10 cfu/g``), "not detected" phrasings are equivalent, other strings match if one
contains the other (case-insensitive), dates compare as ISO strings.
"""

from datetime import date
from typing import Any

from pydantic import BaseModel

from qacopilot.compliance.check import normalize_test_name
from qacopilot.compliance.limits import parse_limit, parse_numeric_value, says_absent


class FieldScore(BaseModel):
    matched: int
    total: int
    misses: list[str]

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 1.0


def _text(value: Any) -> str:
    return value.isoformat() if isinstance(value, date) else str(value).strip()


def values_match(expected: Any, actual: Any) -> bool:
    if actual is None:
        return expected is None
    a, e = _text(actual), _text(expected)
    if says_absent(e) and says_absent(a):
        return True
    pe, pa = parse_numeric_value(e), parse_numeric_value(a)
    if pe and pa:
        return pe == pa
    return e.casefold() in a.casefold() or a.casefold() in e.casefold()


def _limits_match(expected: str, actual: str) -> bool:
    le, la = parse_limit(expected), parse_limit(actual)
    if le is None or la is None:
        return values_match(expected, actual)
    return le.kind == la.kind and le.number == la.number


def score_extraction(expected: dict[str, Any], actual: BaseModel) -> FieldScore:
    """``expected`` maps field -> value. Dict values (CoA ``results``, spec ``micro_limits``) are
    {test name: value}; list values (``allergens``) compare as sets."""
    data = actual.model_dump()
    matched, total, misses = 0, 0, []
    for name, want in expected.items():
        got = data.get(name)
        if isinstance(want, dict):
            rows = {
                normalize_test_name(r["test"]): r.get("result", r.get("limit")) for r in got or []
            }
            for test, value in want.items():
                total += 1
                have = rows.get(normalize_test_name(test))
                ok = have is not None and (
                    _limits_match(value, have)
                    if name == "micro_limits"
                    else values_match(value, have)
                )
                matched += ok
                if not ok:
                    misses.append(f"{name}[{test}]: expected {value!r}, got {have!r}")
        elif isinstance(want, list):
            total += 1
            ok = {w.casefold() for w in want} == {str(g).casefold() for g in got or []}
            matched += ok
            if not ok:
                misses.append(f"{name}: expected {want!r}, got {got!r}")
        else:
            total += 1
            ok = values_match(want, got)
            matched += ok
            if not ok:
                misses.append(f"{name}: expected {want!r}, got {got!r}")
    return FieldScore(matched=matched, total=total, misses=misses)
