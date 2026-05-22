"""
Unit tests for src/retrieval/pinecone_store.py

Covers:
- PineconeStore.__init__: happy path, index not found, dimension mismatch, injected client, __repr__
- upsert_batch: empty list, dimension mismatch, circuit breaker open, happy path,
                timeout, 429 rate limit, non-429 API error, generic error, correlation_id
- query: circuit breaker open, happy path, empty matches + no_matches log,
         metadata mapping, filters passed as filter=, timeout, 429 rate limit,
         non-429 API error, generic error, correlation_id threaded through logs
- close: logs pinecone_store.closed
- Circuit breaker integration: failures trip breaker, success resets, rate limit does not trip
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine.*never awaited:RuntimeWarning"
)

from pinecone.exceptions import PineconeApiException
from src.exceptions.pinecone import (
    PineconeIndexNotFound,
    PineconeInvalidInput,
    PineconeRateLimited,
    PineconeTimeout,
    PineconeUnavailable,
)
from src.retrieval.pinecone_store import (
    ChunkMetadata,
    PineconeStore,
    QueryResult,
    VectorRecord,
)

# ============================================================================
# Helpers
# ============================================================================

DIM = 768


def fake_metadata(**overrides) -> dict:
    base = dict(
        text="some clinical text",
        title="Metformin",
        source="StatPearls",
        article_id="NBK001",
        article_type="drug_interaction",
        token_count=42,
        created_at="2026-01-01",
        updated_at="2026-01-01",
    )
    base.update(overrides)
    return base


def fake_chunk_metadata(**overrides) -> ChunkMetadata:
    return ChunkMetadata(**fake_metadata(**overrides))


def fake_vector_record(record_id: str = "vec-001", dim: int = DIM) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        values=[0.1] * dim,
        metadata=fake_chunk_metadata(),
    )


def make_api_exception(status_code: int = 500) -> PineconeApiException:
    return PineconeApiException(f"API error {status_code}", status_code=status_code)


def make_pinecone_client(index_name: str = "test-index", dim: int = DIM) -> MagicMock:
    """Mock Pinecone client that passes __init__ validation."""
    client = MagicMock()
    idx_obj = MagicMock()
    idx_obj.name = index_name
    client.list_indexes.return_value = [idx_obj]
    info = MagicMock()
    info.dimension = dim
    client.describe_index.return_value = info
    client.Index.return_value = MagicMock()
    return client


def make_store(
    index_name: str = "test-index",
    dim: int = DIM,
    breaker_threshold: int = 5,
    breaker_cooldown_s: int = 60,
    query_timeout_s: float = 2.0,
    upsert_timeout_s: float = 10.0,
) -> PineconeStore:
    client = make_pinecone_client(index_name=index_name, dim=dim)
    return PineconeStore(
        api_key="fake-key",
        index_name=index_name,
        dimensions=dim,
        query_timeout_s=query_timeout_s,
        upsert_timeout_s=upsert_timeout_s,
        breaker_threshold=breaker_threshold,
        breaker_cooldown_s=breaker_cooldown_s,
        client=client,
    )


def make_query_response(n: int = 2) -> MagicMock:
    """Fake Pinecone query response with n matches."""
    response = MagicMock()
    response.matches = [
        MagicMock(id=f"vec-{i:03d}", score=0.9 - i * 0.1, metadata=fake_metadata())
        for i in range(n)
    ]
    return response


# ============================================================================
# __init__
# ============================================================================

class TestInit:

    def test_happy_path_stores_attributes(self):
        store = make_store(index_name="my-index", dim=768)
        assert store._index_name == "my-index"
        assert store._dimensions == 768

    def test_injected_client_is_used(self):
        client = make_pinecone_client()
        store = PineconeStore(api_key="k", index_name="test-index", client=client)
        assert store._client is client

    def test_index_not_found_raises(self):
        client = make_pinecone_client(index_name="other-index")
        with pytest.raises(PineconeIndexNotFound, match="test-index"):
            PineconeStore(api_key="k", index_name="test-index", client=client)

    def test_dimension_mismatch_raises(self):
        client = make_pinecone_client(dim=1536)
        with pytest.raises(PineconeInvalidInput, match="1536"):
            PineconeStore(api_key="k", index_name="test-index", dimensions=768, client=client)

    def test_circuit_breaker_created(self):
        store = make_store()
        assert store._circuit_breaker is not None

    def test_repr_contains_index_and_dim(self):
        store = make_store(index_name="clinical-index", dim=768)
        r = repr(store)
        assert "clinical-index" in r
        assert "768" in r


# ============================================================================
# upsert_batch
# ============================================================================

class TestUpsertBatch:

    @pytest.mark.asyncio
    async def test_empty_list_returns_without_api_call(self):
        store = make_store()
        await store.upsert_batch([], namespace="ns")
        store._index.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_invalid_input(self):
        store = make_store(dim=768)
        bad_record = VectorRecord(id="v1", values=[0.1] * 512, metadata=fake_chunk_metadata())
        with pytest.raises(PineconeInvalidInput, match="dim"):
            await store.upsert_batch([bad_record], namespace="ns")

    @pytest.mark.asyncio
    async def test_dimension_mismatch_does_not_call_api(self):
        store = make_store(dim=768)
        bad = VectorRecord(id="v1", values=[0.0] * 512, metadata=fake_chunk_metadata())
        with pytest.raises(PineconeInvalidInput):
            await store.upsert_batch([bad], namespace="ns")
        store._index.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises_unavailable(self):
        store = make_store()
        with patch.object(store._circuit_breaker, "is_open", return_value=True):
            with pytest.raises(PineconeUnavailable, match="[Cc]ircuit"):
                await store.upsert_batch([fake_vector_record()], namespace="ns")
        store._index.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_calls_index_upsert(self):
        store = make_store()
        store._index.upsert = MagicMock(return_value=None)
        await store.upsert_batch([fake_vector_record()], namespace="ns")
        store._index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_happy_path_records_success(self):
        store = make_store()
        store._index.upsert = MagicMock(return_value=None)
        with patch.object(store._circuit_breaker, "record_success") as mock_ok:
            await store.upsert_batch([fake_vector_record()], namespace="ns")
        mock_ok.assert_called_once()

    @pytest.mark.asyncio
    async def test_happy_path_logs_success(self, caplog):
        store = make_store()
        store._index.upsert = MagicMock(return_value=None)
        with caplog.at_level("INFO"):
            await store.upsert_batch([fake_vector_record()], namespace="ns")
        assert any("upsert.success" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_multiple_records_all_passed_to_upsert(self):
        store = make_store()
        store._index.upsert = MagicMock(return_value=None)
        records = [fake_vector_record(f"v{i}") for i in range(5)]
        await store.upsert_batch(records, namespace="ns")
        call_kwargs = store._index.upsert.call_args
        assert len(call_kwargs.kwargs["vectors"]) == 5

    @pytest.mark.asyncio
    async def test_timeout_raises_pinecone_timeout(self):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(PineconeTimeout):
                await store.upsert_batch([fake_vector_record()], namespace="ns")

    @pytest.mark.asyncio
    async def test_timeout_records_failure(self):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
                with pytest.raises(PineconeTimeout):
                    await store.upsert_batch([fake_vector_record()], namespace="ns")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(self, caplog):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with caplog.at_level("WARNING"):
                with pytest.raises(PineconeTimeout):
                    await store.upsert_batch([fake_vector_record()], namespace="ns")
        assert any("upsert.timeout" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_rate_limit_429_raises_pinecone_rate_limited(self):
        store = make_store()
        store._index.upsert = MagicMock(side_effect=make_api_exception(429))
        with pytest.raises(PineconeRateLimited):
            await store.upsert_batch([fake_vector_record()], namespace="ns")

    @pytest.mark.asyncio
    async def test_rate_limit_429_does_not_record_failure(self):
        store = make_store()
        store._index.upsert = MagicMock(side_effect=make_api_exception(429))
        with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(PineconeRateLimited):
                await store.upsert_batch([fake_vector_record()], namespace="ns")
        mock_fail.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_logs_warning(self, caplog):
        store = make_store()
        store._index.upsert = MagicMock(side_effect=make_api_exception(429))
        with caplog.at_level("WARNING"):
            with pytest.raises(PineconeRateLimited):
                await store.upsert_batch([fake_vector_record()], namespace="ns")
        assert any("rate_limited" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_429_api_error_raises_unavailable(self):
        store = make_store()
        store._index.upsert = MagicMock(side_effect=make_api_exception(503))
        with pytest.raises(PineconeUnavailable):
            await store.upsert_batch([fake_vector_record()], namespace="ns")

    @pytest.mark.asyncio
    async def test_non_429_api_error_records_failure(self):
        store = make_store()
        store._index.upsert = MagicMock(side_effect=make_api_exception(503))
        with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(PineconeUnavailable):
                await store.upsert_batch([fake_vector_record()], namespace="ns")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_exception_raises_unavailable(self):
        store = make_store()
        store._index.upsert = MagicMock(side_effect=RuntimeError("connection reset"))
        with pytest.raises(PineconeUnavailable):
            await store.upsert_batch([fake_vector_record()], namespace="ns")

    @pytest.mark.asyncio
    async def test_generic_exception_records_failure(self):
        store = make_store()
        store._index.upsert = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(PineconeUnavailable):
                await store.upsert_batch([fake_vector_record()], namespace="ns")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_correlation_id_in_success_log(self, caplog):
        store = make_store()
        store._index.upsert = MagicMock(return_value=None)
        with caplog.at_level("INFO"):
            await store.upsert_batch(
                [fake_vector_record()], namespace="ns", correlation_id="req-abc"
            )
        success_records = [r for r in caplog.records if "upsert.success" in r.message]
        assert success_records
        assert success_records[0].__dict__.get("cid") == "req-abc"

    @pytest.mark.asyncio
    async def test_correlation_id_in_timeout_log(self, caplog):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with caplog.at_level("WARNING"):
                with pytest.raises(PineconeTimeout):
                    await store.upsert_batch(
                        [fake_vector_record()], namespace="ns", correlation_id="req-xyz"
                    )
        timeout_records = [r for r in caplog.records if "timeout" in r.message]
        assert timeout_records
        assert timeout_records[0].__dict__.get("cid") == "req-xyz"


# ============================================================================
# query
# ============================================================================

class TestQuery:

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises_unavailable(self):
        store = make_store()
        with patch.object(store._circuit_breaker, "is_open", return_value=True):
            with pytest.raises(PineconeUnavailable, match="[Cc]ircuit"):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")
        store._index.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_returns_query_results(self):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=3))
        results = await store.query([0.1] * DIM, top_k=3, namespace="ns")
        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)

    @pytest.mark.asyncio
    async def test_result_fields_populated_correctly(self):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=1))
        results = await store.query([0.1] * DIM, top_k=1, namespace="ns")
        assert results[0].id == "vec-000"
        assert results[0].score == pytest.approx(0.9)
        assert isinstance(results[0].metadata, ChunkMetadata)

    @pytest.mark.asyncio
    async def test_metadata_fields_mapped_from_match(self):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=1))
        results = await store.query([0.1] * DIM, top_k=1, namespace="ns")
        meta = results[0].metadata
        assert meta.title == "Metformin"
        assert meta.source == "StatPearls"
        assert meta.article_id == "NBK001"

    @pytest.mark.asyncio
    async def test_empty_matches_returns_empty_list(self):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=0))
        results = await store.query([0.1] * DIM, top_k=5, namespace="ns")
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_matches_logs_no_matches(self, caplog):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=0))
        with caplog.at_level("INFO"):
            await store.query([0.1] * DIM, top_k=5, namespace="ns")
        assert any("no_matches" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_filters_passed_as_filter_kwarg_to_sdk(self):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=1))
        filters = {"article_type": {"$eq": "drug_interaction"}}
        await store.query([0.1] * DIM, top_k=5, namespace="ns", filters=filters)
        call_kwargs = store._index.query.call_args.kwargs
        assert call_kwargs["filter"] == filters

    @pytest.mark.asyncio
    async def test_none_filters_passed_as_none_to_sdk(self):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=1))
        await store.query([0.1] * DIM, top_k=5, namespace="ns", filters=None)
        call_kwargs = store._index.query.call_args.kwargs
        assert call_kwargs["filter"] is None

    @pytest.mark.asyncio
    async def test_happy_path_records_success(self):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=1))
        with patch.object(store._circuit_breaker, "record_success") as mock_ok:
            await store.query([0.1] * DIM, top_k=1, namespace="ns")
        mock_ok.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_raises_pinecone_timeout(self):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(PineconeTimeout):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")

    @pytest.mark.asyncio
    async def test_timeout_records_failure(self):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
                with pytest.raises(PineconeTimeout):
                    await store.query([0.1] * DIM, top_k=5, namespace="ns")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(self, caplog):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with caplog.at_level("WARNING"):
                with pytest.raises(PineconeTimeout):
                    await store.query([0.1] * DIM, top_k=5, namespace="ns")
        assert any("query.timeout" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_rate_limit_429_raises_pinecone_rate_limited(self):
        store = make_store()
        store._index.query = MagicMock(side_effect=make_api_exception(429))
        with pytest.raises(PineconeRateLimited):
            await store.query([0.1] * DIM, top_k=5, namespace="ns")

    @pytest.mark.asyncio
    async def test_rate_limit_429_does_not_record_failure(self):
        store = make_store()
        store._index.query = MagicMock(side_effect=make_api_exception(429))
        with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(PineconeRateLimited):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")
        mock_fail.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_logs_warning(self, caplog):
        store = make_store()
        store._index.query = MagicMock(side_effect=make_api_exception(429))
        with caplog.at_level("WARNING"):
            with pytest.raises(PineconeRateLimited):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")
        assert any("rate_limited" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_429_api_error_raises_unavailable(self):
        store = make_store()
        store._index.query = MagicMock(side_effect=make_api_exception(503))
        with pytest.raises(PineconeUnavailable):
            await store.query([0.1] * DIM, top_k=5, namespace="ns")

    @pytest.mark.asyncio
    async def test_non_429_api_error_records_failure(self):
        store = make_store()
        store._index.query = MagicMock(side_effect=make_api_exception(503))
        with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(PineconeUnavailable):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_exception_raises_unavailable(self):
        store = make_store()
        store._index.query = MagicMock(side_effect=RuntimeError("network error"))
        with pytest.raises(PineconeUnavailable):
            await store.query([0.1] * DIM, top_k=5, namespace="ns")

    @pytest.mark.asyncio
    async def test_generic_exception_records_failure(self):
        store = make_store()
        store._index.query = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(store._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(PineconeUnavailable):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_exception_logs_error(self, caplog):
        store = make_store()
        store._index.query = MagicMock(side_effect=RuntimeError("boom"))
        with caplog.at_level("ERROR"):
            with pytest.raises(PineconeUnavailable):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")
        assert any("query.failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_correlation_id_in_no_matches_log(self, caplog):
        store = make_store()
        store._index.query = MagicMock(return_value=make_query_response(n=0))
        with caplog.at_level("INFO"):
            await store.query(
                [0.1] * DIM, top_k=5, namespace="ns", correlation_id="req-999"
            )
        no_match_records = [r for r in caplog.records if "no_matches" in r.message]
        assert no_match_records
        assert no_match_records[0].__dict__.get("cid") == "req-999"

    @pytest.mark.asyncio
    async def test_correlation_id_in_timeout_log(self, caplog):
        store = make_store()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with caplog.at_level("WARNING"):
                with pytest.raises(PineconeTimeout):
                    await store.query(
                        [0.1] * DIM, top_k=5, namespace="ns", correlation_id="req-123"
                    )
        timeout_records = [r for r in caplog.records if "timeout" in r.message]
        assert timeout_records
        assert timeout_records[0].__dict__.get("cid") == "req-123"

    @pytest.mark.asyncio
    async def test_correlation_id_in_error_log(self, caplog):
        store = make_store()
        store._index.query = MagicMock(side_effect=RuntimeError("boom"))
        with caplog.at_level("ERROR"):
            with pytest.raises(PineconeUnavailable):
                await store.query(
                    [0.1] * DIM, top_k=5, namespace="ns", correlation_id="req-456"
                )
        error_records = [r for r in caplog.records if "query.failed" in r.message]
        assert error_records
        assert error_records[0].__dict__.get("cid") == "req-456"


# ============================================================================
# close
# ============================================================================

class TestClose:

    @pytest.mark.asyncio
    async def test_close_logs_closed(self, caplog):
        store = make_store()
        with caplog.at_level("INFO"):
            await store.close()
        assert any("pinecone_store.closed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        store = make_store()
        await store.close()  # should complete without error


# ============================================================================
# Circuit breaker integration
# ============================================================================

class TestCircuitBreakerIntegration:

    @pytest.mark.asyncio
    async def test_consecutive_upsert_failures_trip_breaker(self):
        store = make_store(breaker_threshold=3, breaker_cooldown_s=60)
        store._index.upsert = MagicMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(PineconeUnavailable):
                await store.upsert_batch([fake_vector_record()], namespace="ns")
        assert store._circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_consecutive_query_failures_trip_breaker(self):
        store = make_store(breaker_threshold=3, breaker_cooldown_s=60)
        store._index.query = MagicMock(side_effect=RuntimeError("fail"))
        for _ in range(3):
            with pytest.raises(PineconeUnavailable):
                await store.query([0.1] * DIM, top_k=5, namespace="ns")
        assert store._circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        store = make_store(breaker_threshold=5, breaker_cooldown_s=60)
        store._index.upsert = MagicMock(side_effect=RuntimeError("fail"))
        for _ in range(2):
            with pytest.raises(PineconeUnavailable):
                await store.upsert_batch([fake_vector_record()], namespace="ns")

        store._index.upsert = MagicMock(return_value=None)
        await store.upsert_batch([fake_vector_record()], namespace="ns")
        assert store._circuit_breaker._failures == 0

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_trip_breaker(self):
        store = make_store(breaker_threshold=1, breaker_cooldown_s=60)
        store._index.upsert = MagicMock(side_effect=make_api_exception(429))
        with pytest.raises(PineconeRateLimited):
            await store.upsert_batch([fake_vector_record()], namespace="ns")
        assert not store._circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_open_breaker_makes_upsert_fail_fast(self):
        store = make_store(breaker_threshold=1, breaker_cooldown_s=60)
        store._index.upsert = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(PineconeUnavailable):
            await store.upsert_batch([fake_vector_record()], namespace="ns")

        store._index.upsert.reset_mock()
        with pytest.raises(PineconeUnavailable, match="[Cc]ircuit"):
            await store.upsert_batch([fake_vector_record()], namespace="ns")
        store._index.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_open_breaker_makes_query_fail_fast(self):
        store = make_store(breaker_threshold=1, breaker_cooldown_s=60)
        store._index.query = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(PineconeUnavailable):
            await store.query([0.1] * DIM, top_k=5, namespace="ns")

        store._index.query.reset_mock()
        with pytest.raises(PineconeUnavailable, match="[Cc]ircuit"):
            await store.query([0.1] * DIM, top_k=5, namespace="ns")
        store._index.query.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
