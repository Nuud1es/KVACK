"""A small in-memory limit on how often one address may try something (per process)."""

import time
from collections import defaultdict, deque


class Throttle:
    """Allow ``limit`` events per ``window_s`` seconds for each key (here: a client address)."""

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def _recent(self, key: str) -> deque[float]:
        events = self._events[key]
        cutoff = time.monotonic() - self.window_s
        while events and events[0] < cutoff:
            events.popleft()
        if not events:
            self._events.pop(key, None)  # no growth for addresses that went quiet
            return deque()
        return events

    def blocked(self, key: str) -> bool:
        return len(self._recent(key)) >= self.limit

    def record(self, key: str) -> None:
        self._recent(key)
        self._events[key].append(time.monotonic())
