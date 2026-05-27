"""Custom exceptions for generator module. Lets callers dispatch fallback vs retry."""


class GeneratorError(Exception):
    """Base for all generator failures."""


class GeneratorUnavailable(GeneratorError):
    """All LLM providers (OpenAI + Groq) down → caller falls back to FDA only."""


class GeneratorTimeout(GeneratorError):
    """LLM call exceeded latency budget → triggers provider fallback or FDA only."""


class GeneratorRateLimited(GeneratorError):
    """LLM provider returned 429 → triggers provider fallback."""

    def __init__(self, retry_after_s: float):
        self.retry_after_s = retry_after_s
        super().__init__(f"Rate limited, retry after {retry_after_s}s")


class MalformedLLMOutput(GeneratorError):
    """LLM returned unparseable JSON after retry → reject answer, FDA only."""


class CitationVerificationFailed(GeneratorError):
    """LLM cited a source not in the provided chunks (hallucination) → reject, FDA only."""