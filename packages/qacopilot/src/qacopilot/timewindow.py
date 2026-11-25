"""Reading "next 30 days" / "2 weeks" / "this month" from a request (no heavy imports)."""

import re

# (?<![\d.]) stops "5000 days" matching as "000 days"; any size is accepted and then capped.
_WINDOW = re.compile(r"(?<![\d.])(\d{1,6})\s*(day|week|month)s?", re.IGNORECASE)
_MONTH = re.compile(r"\b(this|next|coming)\s+month\b", re.IGNORECASE)
_WEEK = re.compile(r"\b(this|next|coming)\s+week\b", re.IGNORECASE)

DEFAULT_WINDOW_DAYS = 30
MAX_WINDOW_DAYS = 365


def requested_window_days(query: str) -> int:
    """The window the request asks for, before any cap (always at least 1)."""
    m = _WINDOW.search(query)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
    elif _MONTH.search(query):
        days = 30
    elif _WEEK.search(query):
        days = 7
    else:
        days = DEFAULT_WINDOW_DAYS
    return max(1, days)


def parse_window_days(query: str) -> int:
    """How many days ahead the request means ("next 30 days", "2 weeks", "this month").

    Defaults to 30; capped at a year so a typo cannot ask for the whole database. Use
    ``window_cap_note`` to tell the person when the cap changed what they asked for.
    """
    return min(requested_window_days(query), MAX_WINDOW_DAYS)


def window_cap_note(query: str) -> str:
    """A sentence saying the request was shortened, or "" when it was not."""
    asked = requested_window_days(query)
    if asked <= MAX_WINDOW_DAYS:
        return ""
    return f"You asked for {asked} days; reminders look at most {MAX_WINDOW_DAYS} days ahead."
