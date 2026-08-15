"""In-memory sliding-window rate limiter.

GPT-2 scoring is CPU-bound and takes hundreds of milliseconds per essay, so a
handful of concurrent callers can saturate the process and make the service
unusable for everyone else. This caps how often any one client can spend that
budget.

**Scope, stated plainly:** the counters live in this process's memory. They are
not shared across workers and do not survive a restart, so running four uvicorn
workers means roughly four times the configured limit in aggregate. That is the
right trade for a single-process demo and the wrong one for production, where
this belongs in Redis or at the ingress. It is a courtesy limit that keeps one
enthusiastic caller from degrading a live demo -- not a defence against a
determined attacker, who can rotate source addresses freely.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome of one rate-limit check."""

    allowed: bool
    remaining: int
    retry_after: float  # seconds until the oldest hit leaves the window


class SlidingWindowRateLimiter:
    """Allow ``limit`` requests per ``window_seconds`` per key.

    A sliding window rather than a fixed one: a fixed window lets a caller send
    ``limit`` requests at 0:59 and ``limit`` more at 1:00, briefly doubling the
    intended rate. Each key keeps a deque of hit timestamps, and anything older
    than the window is discarded on read.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        max_tracked_clients: int = 10_000,
    ) -> None:
        self._limit = limit
        self._window = float(window_seconds)
        self._max_tracked = max_tracked_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> RateLimitDecision:
        """Record a request for ``key`` and say whether it is allowed.

        A rejected request is **not** recorded. Otherwise a client hammering the
        endpoint would keep pushing its own window forward and stay locked out
        indefinitely -- the limit would become a ban.
        """
        now = time.monotonic() if now is None else now
        cutoff = now - self._window

        with self._lock:
            timestamps = self._hits.get(key)
            if timestamps is None:
                timestamps = deque()
                self._hits[key] = timestamps
                if len(self._hits) > self._max_tracked:
                    self._evict_stale(cutoff)

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self._limit:
                retry_after = max(timestamps[0] + self._window - now, 0.0)
                return RateLimitDecision(False, 0, retry_after)

            timestamps.append(now)
            return RateLimitDecision(True, self._limit - len(timestamps), 0.0)

    def _evict_stale(self, cutoff: float) -> None:
        """Drop keys with no activity inside the window. Caller holds the lock.

        Bounds memory under a spray of unique addresses. If every tracked key is
        still active, the map is left alone rather than evicting a live client:
        over-admitting briefly is better than resetting someone's window.
        """
        stale = [k for k, ts in self._hits.items() if not ts or ts[-1] <= cutoff]
        for key in stale:
            del self._hits[key]

    def reset(self) -> None:
        """Clear all counters. Intended for tests."""
        with self._lock:
            self._hits.clear()

    @property
    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._hits)
