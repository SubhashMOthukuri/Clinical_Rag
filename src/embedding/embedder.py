"""Embedder module: text → vectors.

Single responsibility: turn text into 768-dim vectors using OpenAI's API.
Does NOT know about Pinecone, chunks, metadata, or storage.
Caller (ingest script or retriever) handles wiring vectors to downstream systems.

Failure handling:
- Empty/oversized input → EmbedderInvalidInput (programmer bug, do not swallow)
- OpenAI rate limit (429) → EmbedderRateLimited (caller retries with backoff)
- OpenAI timeout → EmbedderTimeout (caller falls back to FDA)
- Circuit breaker open → EmbedderUnavailable (fail fast)
- Generic API failure → EmbedderUnavailable + breaker.record_failure()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, Sequence

import openai
import tiktoken
from openai import AsyncOpenAI

from src.exceptions.embedder import (
    EmbedderInvalidInput,
    EmbedderRateLimited,
    EmbedderTimeout,
    EmbedderUnavailable,
)
from src.resilience.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


# ====== Contract ======
class Embedder(Protocol):
    """Contract every embedder must satisfy. No state, no implementation."""

    @property
    def dimensions(self) -> int:
        """Vector size produced by this embedder."""
        ...

    @property
    def max_input_tokens(self) -> int:
        """Max tokens this embedder accepts per text."""
        ...

    async def embed(
        self, text: str, *, correlation_id: str | None = None
    ) -> list[float]:
        """Embed a single text. Used by retriever for nurse queries."""
        ...

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
        *,
        correlation_id: str | None = None,
    ) -> list[list[float]]:
        """Embed many texts. Used by ingest pipeline. Order preserved."""
        ...


# ====== Implementation ======
class OpenAIEmbedder:
    """OpenAI implementation of Embedder protocol.

    Uses text-embedding-3-small with dimensions=768 to fit Pinecone free tier.
    """

    # TODO(prod): move these to config/env vars, not hardcoded
    DEFAULT_SINGLE_TIMEOUT_S = 2.0
    DEFAULT_BATCH_TIMEOUT_S = 10.0
    DEFAULT_BREAKER_THRESHOLD = 5
    DEFAULT_BREAKER_COOLDOWN_S = 30

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 768,
        single_timeout_s: float = DEFAULT_SINGLE_TIMEOUT_S,
        batch_timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
        breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
        breaker_cooldown_s: int = DEFAULT_BREAKER_COOLDOWN_S,
        client: AsyncOpenAI | None = None,
    ):
        # Inject client for testability; default to real client in production
        self._client = client or AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._max_input_tokens = 8191  # OpenAI text-embedding-3-small limit
        self._single_timeout_s = single_timeout_s
        self._batch_timeout_s = batch_timeout_s
        self._circuit_breaker = CircuitBreaker(
            threshold=breaker_threshold,
            cooldown_s=breaker_cooldown_s,
        )
        self._tokenizer = tiktoken.encoding_for_model(model)

    def __repr__(self) -> str:
        return (
            f"OpenAIEmbedder(model={self._model}, dimensions={self._dimensions})"
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    async def close(self) -> None:
        """Clean up HTTP connection pool. Call on app shutdown."""
        await self._client.close()

    async def embed(
        self, text: str, *, correlation_id: str | None = None
    ) -> list[float]:
        """Embed a single text. Latency-critical (runtime path).

        Args:
            text: Non-empty text within token limit.
            correlation_id: Request tracing ID, propagated to logs.

        Returns:
            768-dim vector.

        Raises:
            EmbedderInvalidInput: Empty or oversized text.
            EmbedderUnavailable: Circuit open or OpenAI failed.
            EmbedderRateLimited: OpenAI 429.
            EmbedderTimeout: Exceeded latency budget.
        """
        log_ctx = {"cid": correlation_id}

        # 1. Input validation
        if not text or not text.strip():
            raise EmbedderInvalidInput("Text is empty")

        token_count = len(self._tokenizer.encode(text))
        if token_count > self._max_input_tokens:
            raise EmbedderInvalidInput(
                f"Text has {token_count} tokens, max is {self._max_input_tokens}"
            )

        # 2. Circuit breaker check
        if self._circuit_breaker.is_open():
            raise EmbedderUnavailable("Circuit breaker is open")

        # 3. Call OpenAI with timeout
        try:
            response = await asyncio.wait_for(
                self._client.embeddings.create(
                    model=self._model,
                    input=text,
                    dimensions=self._dimensions,
                ),
                timeout=self._single_timeout_s,
            )
            vector = response.data[0].embedding
        except asyncio.TimeoutError:
            self._circuit_breaker.record_failure()
            logger.warning(
                "embedder.timeout",
                extra={**log_ctx, "text_len": len(text), "tokens": token_count},
            )
            # TODO(phase 2): increment counter embedder_calls_total{status="timeout"}
            raise EmbedderTimeout(
                f"OpenAI embed call exceeded {self._single_timeout_s}s"
            )
        except openai.RateLimitError as e:
            # Rate limit ≠ service outage — do NOT trip breaker
            retry_after = float(getattr(e, "retry_after", 1.0) or 1.0)
            logger.warning(
                "embedder.rate_limited",
                extra={**log_ctx, "retry_after_s": retry_after},
            )
            # TODO(phase 2): increment counter embedder_calls_total{status="rate_limited"}
            raise EmbedderRateLimited(retry_after_s=retry_after)
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(
                "embedder.failure",
                extra={**log_ctx, "error": str(e), "error_type": type(e).__name__},
            )
            # TODO(phase 2): increment counter embedder_calls_total{status="failed"}
            raise EmbedderUnavailable(f"OpenAI failed: {e}") from e

        # 4. Success
        self._circuit_breaker.record_success()
        # TODO(phase 2): increment counter embedder_calls_total{status="success"}
        # TODO(phase 2): observe histogram embedder_latency_seconds
        # TODO(phase 2): increment counter embedder_tokens_total (for cost tracking)
        return vector

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 100,
        *,
        correlation_id: str | None = None,
    ) -> list[list[float]]:
        """Embed many texts in batches. Throughput-critical (ingest path).

        Contract: output[i] is the vector for input texts[i]. Order preserved.

        Args:
            texts: Non-empty texts within token limit.
            batch_size: Items per OpenAI call. 100 balances blast radius and throughput.
            correlation_id: Trace ID propagated to logs.

        Returns:
            List of 768-dim vectors, same length and order as input.

        Raises:
            EmbedderInvalidInput: Any text empty or oversized.
            EmbedderUnavailable: Circuit open or batch failed.
            EmbedderRateLimited: OpenAI 429.
            EmbedderTimeout: Batch exceeded timeout.
        """
        log_ctx = {"cid": correlation_id}

        if not texts:
            return []

        # 1. Validate all inputs upfront — fail fast before sending anything
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise EmbedderInvalidInput(f"Empty text at index {i}")
            token_count = len(self._tokenizer.encode(text))
            if token_count > self._max_input_tokens:
                raise EmbedderInvalidInput(
                    f"Text at index {i} has {token_count} tokens, "
                    f"max is {self._max_input_tokens}"
                )

        all_vectors: list[list[float]] = []
        total = len(texts)

        # 2. Batch loop
        for batch_start in range(0, total, batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1

            if self._circuit_breaker.is_open():
                # Include progress info so caller can checkpoint
                raise EmbedderUnavailable(
                    f"Circuit breaker open at batch {batch_num} "
                    f"(completed {len(all_vectors)}/{total})"
                )

            try:
                response = await asyncio.wait_for(
                    self._client.embeddings.create(
                        model=self._model,
                        input=list(batch),
                        dimensions=self._dimensions,
                    ),
                    timeout=self._batch_timeout_s,
                )
                vectors = [item.embedding for item in response.data]

                # Defensive: verify OpenAI returned one vector per input
                if len(vectors) != len(batch):
                    raise EmbedderUnavailable(
                        f"Got {len(vectors)} vectors for {len(batch)} inputs "
                        f"at batch {batch_num}"
                    )
            except asyncio.TimeoutError:
                self._circuit_breaker.record_failure()
                logger.warning(
                    "embedder.batch.timeout",
                    extra={
                        **log_ctx,
                        "batch_num": batch_num,
                        "batch_size": len(batch),
                        "completed": len(all_vectors),
                    },
                )
                # TODO(phase 2): counter embedder_batch_total{status="timeout"}
                raise EmbedderTimeout(
                    f"Batch {batch_num} exceeded {self._batch_timeout_s}s "
                    f"(completed {len(all_vectors)}/{total})"
                )
            except openai.RateLimitError as e:
                retry_after = float(getattr(e, "retry_after", 1.0) or 1.0)
                logger.warning(
                    "embedder.batch.rate_limited",
                    extra={
                        **log_ctx,
                        "batch_num": batch_num,
                        "retry_after_s": retry_after,
                        "completed": len(all_vectors),
                    },
                )
                # TODO(phase 2): counter embedder_batch_total{status="rate_limited"}
                raise EmbedderRateLimited(retry_after_s=retry_after)
            except Exception as e:
                self._circuit_breaker.record_failure()
                logger.error(
                    "embedder.batch.failed",
                    extra={
                        **log_ctx,
                        "batch_num": batch_num,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "completed": len(all_vectors),
                    },
                )
                # TODO(phase 2): counter embedder_batch_total{status="failed"}
                raise EmbedderUnavailable(
                    f"Batch {batch_num} failed: {e} "
                    f"(completed {len(all_vectors)}/{total})"
                ) from e
            else:
                self._circuit_breaker.record_success()
                all_vectors.extend(vectors)
                logger.info(
                    "embedder.batch.success",
                    extra={
                        **log_ctx,
                        "batch_num": batch_num,
                        "completed": len(all_vectors),
                        "total": total,
                    },
                )
                # TODO(phase 2): counter embedder_batch_total{status="success"}
                # TODO(phase 2): histogram embedder_batch_latency_seconds

        return all_vectors


# ============================================================================
# PHASE 2 TODOs (post-MVP)
# ============================================================================
# [ ] Wrap OpenAI calls with tenacity retry (3 attempts, exp backoff, jitter)
# [ ] Emit Prometheus metrics:
#     - embedder_calls_total{status="success|timeout|failed|rate_limited"}
#     - embedder_latency_seconds (histogram, p50/p95/p99)
#     - embedder_tokens_total (counter, for cost tracking)
#     - embedder_batch_latency_seconds (histogram)
# [ ] Add cost tracking — sum of tokens × $0.02/1M
# [ ] Split connect_timeout vs read_timeout in OpenAI client
# [ ] Per-batch timeout scaling: max(5.0, 0.1 * len(batch))
# [ ] Callback-based batch progress: on_batch_complete(vectors, offset)
#     so ingest can checkpoint per batch instead of losing work on failure
# [ ] Add MedCPT or PubMedBERT alternative embedder behind same Protocol
# [ ] A/B framework to compare retrieval quality across embedders

# ============================================================================
# PROD-LEVEL TODOs (before production deploy)
# ============================================================================
# [ ] Move hardcoded constants to config (timeouts, breaker params)
# [ ] Add OpenTelemetry tracing spans around each call
# [ ] Add structured cost alert — page if daily spend exceeds threshold
# [ ] Implement graceful shutdown — drain in-flight requests on SIGTERM
# [ ] Add circuit_breaker_state metric (closed=0, open=1) for dashboards
# [ ] Document SLA expectations: p95 < 100ms single, p95 < 2s batch
# [ ] Load test: 50 RPS sustained, verify breaker behavior under failure injection
# [ ] Verify behavior when OpenAI returns partial results (currently raises — confirm desired)
# [ ] Add Pydantic validation if embedder is exposed via API
# [ ] Secret rotation — OpenAI key must be refreshable without restart