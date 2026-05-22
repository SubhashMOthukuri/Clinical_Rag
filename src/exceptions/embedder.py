"""Custom exceptions for embedder module. Lets callers dispatch fallback vs retry."""


class EmbedderError(Exception):
    """Base for all embedder failures."""


class EmbedderUnavailable(EmbedderError):
    """OpenAI down or circuit breaker open → caller falls back to FDA only."""


class EmbedderRateLimited(EmbedderError):
    """OpenAI returned 429 → caller retries with backoff. Not a service outage."""

    def __init__(self, retry_after_s: float):
        self.retry_after_s = retry_after_s
        super().__init__(f"Rate limited, retry after {retry_after_s}s")


class EmbedderInvalidInput(EmbedderError):
    """Text empty or exceeds token limit → programmer/data bug, do not swallow."""


class EmbedderTimeout(EmbedderError):
    """Call exceeded latency budget → caller falls back to FDA only."""
    """Call Extended Lantecy budget-> caller fall back to FDA only."""
