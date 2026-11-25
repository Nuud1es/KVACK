"""Read a labelled date straight from text, as a safety net under the model.

Used only when the model returned nothing for a date that the document prints with a clear label
("Next calibration due: 2027-04-02"). UK documents are day-first, so 12/05/2025 is 12 May.
Anything that does not parse to a real calendar date is ignored rather than guessed.
"""

import re
from datetime import date

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],  # fmt: skip
        start=1,
    )
    for m in names
}

_ISO = r"(?P<iy>\d{4})-(?P<im>\d{1,2})-(?P<id>\d{1,2})"
_NUMERIC = r"(?P<nd>\d{1,2})[./\-](?P<nm>\d{1,2})[./\-](?P<ny>\d{4}|\d{2})"
_WORDS = r"(?P<wd>\d{1,2})(?:st|nd|rd|th)?[ \-]+(?P<wm>[A-Za-z]{3,9})\.?,?[ \-]+(?P<wy>\d{4}|\d{2})"
_DATE = re.compile(rf"\b(?:{_ISO}|{_NUMERIC}|{_WORDS})\b")

# Labels that mean "the next calibration is due" (or the certificate expires).
NEXT_DUE = re.compile(
    r"next\s+(?:calibration\s+|test\s+)?(?:due|date)|(?:re-?calibration|re-?test)\s+(?:due|date)"
    r"|suggested\s+date\s+for\s+re-?calibration|due\s+date|valid\s+(?:to|until)|expiry\s+date",
    re.IGNORECASE,
)


def _year(text: str) -> int:
    return int(text) + (2000 if len(text) == 2 else 0)


def _build(day: str, month: int, year: str) -> date | None:
    try:
        return date(_year(year), month, int(day))
    except ValueError:  # 31/02/2026 and the like
        return None


def parse_date(match: re.Match[str]) -> date | None:
    g = match.groupdict()
    if g["iy"]:
        return _build(g["id"], int(g["im"]), g["iy"])
    if g["nd"]:
        return _build(g["nd"], int(g["nm"]), g["ny"])
    month = _MONTHS.get(g["wm"].lower())
    return _build(g["wd"], month, g["wy"]) if month else None


def find_labelled_date(text: str, label: re.Pattern[str], window: int = 40) -> date | None:
    """The first valid date within ``window`` characters after the label, or None."""
    for lab in label.finditer(text):
        after = text[lab.end() : lab.end() + window]
        for found in _DATE.finditer(after):
            parsed = parse_date(found)
            if parsed is not None:
                return parsed
    return None


def find_dates(text: str) -> set[date]:
    """Every valid calendar date printed in ``text`` (ISO, day-first numeric, "2 April 2027")."""
    return {d for m in _DATE.finditer(text) if (d := parse_date(m)) is not None}
