"""Pinecone vector store: upsert and query with circuit breaker, rate-limit handling, and structured logging.

Single responsibility: persist and retrieve embeddings. Does not know about
chunking, embedding models, or downstream consumers.

Failure handling:
- Dimension mismatch on upsert → PineconeInvalidInput (programmer bug)
- Circuit breaker open         → PineconeUnavailable (fail fast)
- asyncio.TimeoutError         → PineconeTimeout (caller falls back)
- PineconeApiException 429     → PineconeRateLimited (caller retries, does not trip breaker)
- PineconeApiException other   → PineconeUnavailable + breaker.record_failure()
- Generic Exception            → PineconeUnavailable + breaker.record_failure()
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException

from src.exceptions.pinecone import (
    PineconeIndexNotFound,
    PineconeInvalidInput,
    PineconeRateLimited,
    PineconeTimeout,
    PineconeUnavailable,
)
from src.resilience.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata stored alongside each vector. 8 fields, locked schema."""
    text: str
    title: str
    source: str
    article_id: str
    article_type: str
    token_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class VectorRecord:
    """One Pinecone record: id + vector + typed metadata."""
    id: str
    values: list[float]
    metadata: ChunkMetadata


@dataclass(frozen=True)
class QueryResult:
    """One chunk returned from a vector search."""
    id: str
    score: float
    metadata: ChunkMetadata


class VectorStore(Protocol):
    async def upsert_batch(
        self,
        vector_records: Sequence[VectorRecord],
        namespace: str,
        *,
        correlation_id: str | None = None,
    ) -> None:
        """Returns None. Raises on failure. Caller checkpoints."""
        ...

    async def query(
        self,
        query_vector: list[float],
        top_k: int,
        namespace: str,
        filters: dict | None = None,
        *,
        correlation_id: str | None = None,
    ) -> list[QueryResult]:
        """Return top-k chunks ranked by cosine similarity."""
        ...

    async def close(self) -> None:
        ...


class PineconeStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimensions: int = 768,
        query_timeout_s: float = 2.0,
        upsert_timeout_s: float = 10.0,
        breaker_threshold: int = 5,
        breaker_cooldown_s: int = 30,
        client: Pinecone | None = None,
    ):
        self._index_name = index_name
        self._dimensions = dimensions
        self._query_timeout_s = query_timeout_s
        self._upsert_timeout_s = upsert_timeout_s

        self._client = client or Pinecone(api_key=api_key)

        existing = [idx.name for idx in self._client.list_indexes()]
        if index_name not in existing:
            raise PineconeIndexNotFound(
                f"Index '{index_name}' not found. Existing: {existing}"
            )

        info = self._client.describe_index(index_name)
        if info.dimension != dimensions:
            raise PineconeInvalidInput(
                f"Index dim {info.dimension} != expected {dimensions}"
            )

        self._index = self._client.Index(index_name)

        self._circuit_breaker = CircuitBreaker(
            threshold=breaker_threshold,
            cooldown_s=breaker_cooldown_s,
        )

        logger.info(
            "pinecone_store.initialized",
            extra={"index": index_name, "dimensions": dimensions},
        )

    def __repr__(self) -> str:
        return f"PineconeStore(index={self._index_name!r}, dim={self._dimensions})"

    async def upsert_batch(
        self,
        vector_records: Sequence[VectorRecord],
        namespace: str,
        *,
        correlation_id: str | None = None,
    ) -> None:
        if not vector_records:
            return
        for i, record in enumerate(vector_records):
            if len(record.values) != self._dimensions:
                raise PineconeInvalidInput(
                    f"Record {i} has dim {len(record.values)}, expected {self._dimensions}"
                )
        if self._circuit_breaker.is_open():
            raise PineconeUnavailable("Circuit breaker is open.")

        log_ctx = {"batch_size": len(vector_records), "namespace": namespace, "cid": correlation_id}
        pinecone_records = [
            {"id": r.id, "values": r.values, "metadata": asdict(r.metadata)}
            for r in vector_records
        ]

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._index.upsert,
                    vectors=pinecone_records,
                    namespace=namespace,
                ),
                timeout=self._upsert_timeout_s,
            )
        except asyncio.TimeoutError:
            self._circuit_breaker.record_failure()
            logger.warning("pinecone.upsert.timeout", extra=log_ctx)
            raise PineconeTimeout(f"Upsert exceeded {self._upsert_timeout_s}s")
        except PineconeApiException as e:
            if e.status_code == 429:
                logger.warning("pinecone.upsert.rate_limited", extra=log_ctx)
                raise PineconeRateLimited(retry_after_s=1.0)
            self._circuit_breaker.record_failure()
            logger.error("pinecone.upsert.failed", extra={**log_ctx, "error": str(e)})
            raise PineconeUnavailable(f"Upsert failed: {e}") from e
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error("pinecone.upsert.failed", extra={**log_ctx, "error": str(e)})
            raise PineconeUnavailable(f"Upsert failed: {e}") from e
        else:
            self._circuit_breaker.record_success()
            logger.info("pinecone.upsert.success", extra=log_ctx)

    async def query(
        self,
        query_vector: list[float],
        top_k: int,
        namespace: str,
        filters: dict | None = None,
        *,
        correlation_id: str | None = None,
    ) -> list[QueryResult]:
        if self._circuit_breaker.is_open():
            raise PineconeUnavailable("Circuit breaker is open")

        log_ctx = {"top_k": top_k, "namespace": namespace, "cid": correlation_id}

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._index.query,
                    vector=query_vector,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filters,
                    include_metadata=True,
                ),
                timeout=self._query_timeout_s,
            )
            results = [
                QueryResult(
                    id=match.id,
                    score=match.score,
                    metadata=ChunkMetadata(**match.metadata),
                )
                for match in response.matches
            ]
        except asyncio.TimeoutError:
            self._circuit_breaker.record_failure()
            logger.warning("pinecone.query.timeout", extra=log_ctx)
            raise PineconeTimeout(f"Query exceeded {self._query_timeout_s}s")
        except PineconeApiException as e:
            if e.status_code == 429:
                logger.warning("pinecone.query.rate_limited", extra=log_ctx)
                raise PineconeRateLimited(retry_after_s=1.0)
            self._circuit_breaker.record_failure()
            logger.error("pinecone.query.failed", extra={**log_ctx, "error": str(e)})
            raise PineconeUnavailable(f"Query failed: {e}") from e
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error("pinecone.query.failed", extra={**log_ctx, "error": str(e)})
            raise PineconeUnavailable(f"Query failed: {e}") from e
        else:
            self._circuit_breaker.record_success()
            if not results:
                logger.info("pinecone.query.no_matches", extra=log_ctx)
            return results

    async def close(self) -> None:
        """Clean up Pinecone client. Call on app shutdown."""
        logger.info("pinecone_store.closed")


# ============================================================================
# PHASE 2 TODOs
# ============================================================================
# [ ] Tenacity retry on transient errors (3 attempts, exp backoff, jitter)
# [ ] Batch upsert chunking — split >100 records into sub-batches internally
# [ ] Metrics: pinecone_query_total{status}, pinecone_query_latency_seconds
# [ ] Metrics: pinecone_upsert_total{status}, vectors_upserted_total

# ============================================================================
# PROD TODOs
# ============================================================================
# [ ] __init__ blocking calls (list_indexes, describe_index, Index()) — confirm
#     safe in async lifespan or move to async classmethod factory
# [ ] Move index_name, namespace to config / env vars
# [ ] OpenTelemetry span around query (latency-critical path)
# [ ] Verify query p95 < 80ms under load test
# [ ] Add circuit_breaker_state metric (0=closed, 1=open) for dashboards