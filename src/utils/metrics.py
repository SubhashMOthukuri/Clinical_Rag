"""In-process metrics for the MedReconcile MVP.

Tracks counters and latency histograms across the full pipeline:
  reconcile → enrich (rxnorm + fda) → retrieve (embed + pinecone + rerank) → generate

Usage:
    from src.utils.metrics import mvp_metrics, M

    mvp_metrics.incr(M.RECONCILE_REQUESTS)
    mvp_metrics.incr(M.GENERATOR_FALLBACKS, provider="gemini")

    with mvp_metrics.time(M.RETRIEVER_EMBED_LATENCY):
        vector = await embedder.embed(query)

    snapshot = mvp_metrics.snapshot()   # dict — log or expose on /metrics
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock


# ---------------------------------------------------------------------------
# Metric name constants — one place to change names, autocomplete in editors
# ---------------------------------------------------------------------------
class M:
    # ---- request / pipeline ------------------------------------------------
    RECONCILE_REQUESTS        = "reconcile.requests.total"
    RECONCILE_LATENCY         = "reconcile.latency_ms"
    RECONCILE_STATUS_SUCCESS  = "reconcile.status.success"
    RECONCILE_STATUS_PARTIAL  = "reconcile.status.partial"
    RECONCILE_STATUS_ERROR    = "reconcile.status.error"

    # ---- drug enrichment ---------------------------------------------------
    RXNORM_REQUESTS           = "rxnorm.requests.total"
    RXNORM_LATENCY            = "rxnorm.latency_ms"
    RXNORM_UNVERIFIED         = "rxnorm.unverified.total"
    FDA_REQUESTS              = "fda.requests.total"
    FDA_LATENCY               = "fda.latency_ms"
    FDA_MISS                  = "fda.miss.total"           # drug not found in FDA

    # ---- retriever ---------------------------------------------------------
    RETRIEVER_EMBED_LATENCY   = "retriever.embed_latency_ms"
    RETRIEVER_PINECONE_LATENCY = "retriever.pinecone_latency_ms"
    RETRIEVER_RERANK_LATENCY  = "retriever.rerank_latency_ms"
    RETRIEVER_CHUNKS_RETURNED = "retriever.chunks_returned"   # histogram
    RETRIEVER_EMPTY_RESULTS   = "retriever.empty_results.total"
    RETRIEVER_RERANK_FAILURES = "retriever.rerank_failures.total"
    RETRIEVER_EMBED_ERRORS    = "retriever.embed_errors.total"
    RETRIEVER_PINECONE_ERRORS = "retriever.pinecone_errors.total"

    # ---- generator ---------------------------------------------------------
    GENERATOR_LLM_LATENCY     = "generator.llm_latency_ms"
    GENERATOR_FALLBACKS       = "generator.fda_fallbacks.total"
    GENERATOR_GEMINI_OK       = "generator.gemini_ok.total"
    GENERATOR_GEMINI_ERRORS   = "generator.gemini_errors.total"
    GENERATOR_GROQ_OK         = "generator.groq_ok.total"
    GENERATOR_GROQ_ERRORS     = "generator.groq_errors.total"
    GENERATOR_ALL_FAILED      = "generator.all_providers_failed.total"

    # ---- business / quality ------------------------------------------------
    WARNINGS_SEVERITY_RED     = "warnings.severity.red"
    WARNINGS_SEVERITY_YELLOW  = "warnings.severity.yellow"
    WARNINGS_SEVERITY_GREEN   = "warnings.severity.green"
    DRUG_PAIRS_PER_REQUEST    = "drug_pairs_per_request"    # histogram
    MEDS_PER_REQUEST          = "meds_per_request"          # histogram

    # ---- circuit breakers --------------------------------------------------
    BREAKER_GEMINI_OPEN       = "breaker.gemini.open.total"
    BREAKER_GROQ_OPEN         = "breaker.groq.open.total"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _hist_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50":  round(_percentile(values, 50), 3),
        "p95":  round(_percentile(values, 95), 3),
        "p99":  round(_percentile(values, 99), 3),
        "max":  round(max(values), 3),
    }


# ---------------------------------------------------------------------------
# Metrics store
# ---------------------------------------------------------------------------
class MvpMetrics:
    def __init__(self) -> None:
        self._counter: dict[str, int] = defaultdict(int)
        self._latency: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    # -- counters ------------------------------------------------------------

    def incr(self, name: str, amount: int = 1, **tags: str) -> None:
        """Increment a counter. Tags are appended to the key as k=v pairs."""
        key = self._key(name, tags)
        with self._lock:
            self._counter[key] += amount

    # -- histograms ----------------------------------------------------------

    def observe(self, name: str, value: float, **tags: str) -> None:
        """Record one observation (e.g. latency in ms, chunk count)."""
        key = self._key(name, tags)
        with self._lock:
            self._latency[key].append(value)

    # -- timing context manager ----------------------------------------------

    @contextmanager
    def time(self, name: str, **tags: str):
        """Context manager that records elapsed wall-clock time in ms."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            self.observe(name, elapsed_ms, **tags)

    # -- snapshot ------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a point-in-time copy of all counters and histogram stats."""
        with self._lock:
            counters = dict(self._counter)
            latency_copy = {k: list(v) for k, v in self._latency.items()}

        return {
            "counters": counters,
            "histograms": {k: _hist_stats(v) for k, v in latency_copy.items()},
        }

    def reset(self) -> None:
        """Clear all data (useful between tests)."""
        with self._lock:
            self._counter.clear()
            self._latency.clear()

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _key(name: str, tags: dict[str, str]) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
mvp_metrics = MvpMetrics()
