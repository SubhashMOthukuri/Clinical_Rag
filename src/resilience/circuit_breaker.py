"""Shared circuit breaker for external service clients.

Used by fda_client, rxnorm_client, embedder, pinecone_store.
Prevents cascade failures when an external dependency is unhealthy.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Trips after N consecutive failures; auto-closes after cooldown.

    States:
        CLOSED — normal operation, calls pass through
        OPEN   — too many failures, calls fail fast for cooldown_s seconds
    """

    def __init__(self, threshold: int = 5, cooldown_s: float = 30):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if cooldown_s <= 0:
            raise ValueError("cooldown_s must be > 0")
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._open_until = 0.0

    def is_open(self) -> bool:
        """Returns True if breaker is open (calls should fail fast)."""
        return time.monotonic() < self._open_until

    def record_success(self) -> None:
        """Call after a successful operation. Resets failure counter."""
        if self._failures > 0:
            logger.info("circuit_breaker.reset", extra={"prev_failures": self._failures})
        self._failures = 0

    def record_failure(self) -> None:
        """Call after a failed operation. Trips breaker at threshold."""
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = time.monotonic() + self._cooldown
            logger.warning(
                "circuit_breaker.tripped",
                extra={
                    "failures": self._failures,
                    "cooldown_s": self._cooldown,
                },
            )


# ============================================================================
# PHASE 2 TODOs
# ============================================================================
# [ ] Half-open state: after cooldown, allow ONE test call before fully closing
# [ ] Configurable: per-error-type counting (don't count rate limits as failures)
# [ ] Decorator interface: @breaker.protect for ergonomic wrapping
# [ ] Async context manager: `async with breaker:` for try/except-free usage
# [ ] Emit metric circuit_breaker_state{name="..."} (0=closed, 1=open)
# [ ] Emit counter circuit_breaker_trips_total{name="..."}
# [ ] Per-instance naming for multi-breaker observability