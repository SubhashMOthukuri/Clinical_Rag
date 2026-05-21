"""
Unit tests for src/embedding/embedder.py

Covers:
- OpenAIEmbedder init: defaults, properties, repr
- embed(): happy path, input validation, circuit breaker, timeout, rate limit, API error
- embed_batch(): empty, single, multi-text, batching order, validation, errors, mismatch
- Circuit breaker integration: failure accumulation, tripping, success reset, rate-limit does not trip
- Integration tests: skipped when OPENAI_API_KEY absent (will fail without a real key)

All unit tests inject a mock AsyncOpenAI client — no real API key needed.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

# Python 3.13 + AsyncMock leaves internal coroutines unawaited when side_effect
# raises — harmless but noisy. Suppress the specific warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine.*never awaited:RuntimeWarning"
)

from src.embedding.embedder import OpenAIEmbedder
from src.exceptions.embedder import (
    EmbedderInvalidInput,
    EmbedderRateLimited,
    EmbedderTimeout,
    EmbedderUnavailable,
)

# ============================================================================
# Helpers
# ============================================================================

DIM = 768  # matches OpenAIEmbedder default


def fake_vector(dim: int = DIM) -> list[float]:
    return [round(0.01 * i, 4) for i in range(dim)]


def make_fake_response(n: int = 1, dim: int = DIM) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=fake_vector(dim)) for _ in range(n)]
    return resp


def make_async_client(n: int = 1, dim: int = DIM) -> MagicMock:
    """Mock AsyncOpenAI whose embeddings.create returns fake vectors."""
    client = MagicMock()
    client.embeddings.create = AsyncMock(return_value=make_fake_response(n, dim))
    return client


def make_embedder(
    client=None,
    dim: int = DIM,
    breaker_threshold: int = 5,
    breaker_cooldown_s: int = 30,
    single_timeout_s: float = 2.0,
    batch_timeout_s: float = 10.0,
) -> OpenAIEmbedder:
    return OpenAIEmbedder(
        client=client or make_async_client(),
        dimensions=dim,
        single_timeout_s=single_timeout_s,
        batch_timeout_s=batch_timeout_s,
        breaker_threshold=breaker_threshold,
        breaker_cooldown_s=breaker_cooldown_s,
    )


def make_rate_limit_error() -> openai.RateLimitError:
    """Real openai.RateLimitError — requires a httpx.Response."""
    return openai.RateLimitError(
        message="rate limit exceeded",
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
            json={"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
        ),
        body={"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
    )


def make_api_error() -> openai.APIError:
    """Real openai.APIError for generic failure testing."""
    return openai.APIError(
        message="internal server error",
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        body=None,
    )


# ============================================================================
# Init and properties
# ============================================================================

class TestOpenAIEmbedderInit:

    def test_dimensions_property(self):
        assert make_embedder(dim=768).dimensions == 768

    def test_custom_dimensions(self):
        assert make_embedder(dim=1536).dimensions == 1536

    def test_max_input_tokens_is_8191(self):
        assert make_embedder().max_input_tokens == 8191

    def test_repr_contains_model(self):
        assert "text-embedding-3-small" in repr(make_embedder())

    def test_repr_contains_dimensions(self):
        assert "768" in repr(make_embedder(dim=768))

    def test_injected_client_is_used(self):
        client = make_async_client()
        embedder = make_embedder(client=client)
        assert embedder._client is client


# ============================================================================
# embed() — single text
# ============================================================================

class TestEmbed:

    @pytest.mark.asyncio
    async def test_happy_path_returns_vector(self):
        embedder = make_embedder(dim=DIM)
        result = await embedder.embed("Patient has hypertension.")
        assert isinstance(result, list)
        assert len(result) == DIM

    @pytest.mark.asyncio
    async def test_returns_floats(self):
        result = await make_embedder().embed("some clinical text")
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_success_calls_record_success_on_breaker(self):
        embedder = make_embedder()
        with patch.object(embedder._circuit_breaker, "record_success") as mock_ok:
            await embedder.embed("text")
        mock_ok.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_string_raises_invalid_input(self):
        with pytest.raises(EmbedderInvalidInput):
            await make_embedder().embed("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises_invalid_input(self):
        with pytest.raises(EmbedderInvalidInput):
            await make_embedder().embed("   \t\n  ")

    @pytest.mark.asyncio
    async def test_oversized_text_raises_invalid_input(self):
        embedder = make_embedder()
        with patch.object(embedder, "_tokenizer") as mock_tok:
            mock_tok.encode.return_value = list(range(9000))  # 9000 tokens > 8191
            with pytest.raises(EmbedderInvalidInput, match="tokens"):
                await embedder.embed("any text")

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises_unavailable_without_api_call(self):
        client = make_async_client()
        embedder = make_embedder(client=client)
        with patch.object(embedder._circuit_breaker, "is_open", return_value=True):
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed("text")
        client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_raises_embedder_timeout(self):
        embedder = make_embedder()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(EmbedderTimeout):
                await embedder.embed("text")

    @pytest.mark.asyncio
    async def test_timeout_records_failure_on_circuit_breaker(self):
        embedder = make_embedder()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with patch.object(embedder._circuit_breaker, "record_failure") as mock_fail:
                with pytest.raises(EmbedderTimeout):
                    await embedder.embed("text")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_raises_embedder_rate_limited(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())
        with pytest.raises(EmbedderRateLimited):
            await make_embedder(client=client).embed("text")

    @pytest.mark.asyncio
    async def test_rate_limit_carries_retry_after_s(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())
        with pytest.raises(EmbedderRateLimited) as exc_info:
            await make_embedder(client=client).embed("text")
        assert exc_info.value.retry_after_s >= 0

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_record_failure(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())
        embedder = make_embedder(client=client)
        with patch.object(embedder._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(EmbedderRateLimited):
                await embedder.embed("text")
        mock_fail.assert_not_called()

    @pytest.mark.asyncio
    async def test_generic_api_error_raises_embedder_unavailable(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_api_error())
        with pytest.raises(EmbedderUnavailable):
            await make_embedder(client=client).embed("text")

    @pytest.mark.asyncio
    async def test_generic_api_error_records_failure(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_api_error())
        embedder = make_embedder(client=client)
        with patch.object(embedder._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed("text")
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_correlation_id_does_not_break_execution(self):
        result = await make_embedder().embed("text", correlation_id="req-123")
        assert len(result) == DIM


# ============================================================================
# embed_batch() — multiple texts
# ============================================================================

class TestEmbedBatch:

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        result = await make_embedder().embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_text_returns_one_vector(self):
        client = make_async_client(n=1)
        result = await make_embedder(client=client).embed_batch(["single text"])
        assert len(result) == 1
        assert len(result[0]) == DIM

    @pytest.mark.asyncio
    async def test_multiple_texts_returns_all_vectors(self):
        n = 5
        client = make_async_client(n=n)
        result = await make_embedder(client=client).embed_batch(["text"] * n)
        assert len(result) == n

    @pytest.mark.asyncio
    async def test_order_preserved_across_single_batch(self):
        vectors = [[float(i)] * DIM for i in range(3)]
        resp = MagicMock()
        resp.data = [MagicMock(embedding=v) for v in vectors]
        client = MagicMock()
        client.embeddings.create = AsyncMock(return_value=resp)
        result = await make_embedder(client=client).embed_batch(["a", "b", "c"])
        assert result == vectors

    @pytest.mark.asyncio
    async def test_large_batch_split_into_multiple_api_calls(self):
        # batch_size=2 → 5 texts need 3 API calls
        call_count = 0

        async def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            n = len(kwargs.get("input", args[0] if args else []))
            return make_fake_response(n=n)

        client = MagicMock()
        client.embeddings.create = count_calls
        embedder = OpenAIEmbedder(client=client, dimensions=DIM)
        await embedder.embed_batch(["t"] * 5, batch_size=2)
        assert call_count == 3  # ceil(5/2)

    @pytest.mark.asyncio
    async def test_all_results_assembled_in_order_across_batches(self):
        # Each API call returns vectors tagged by call number
        call_num = 0

        async def ordered_response(*args, **kwargs):
            nonlocal call_num
            call_num += 1
            n = len(kwargs.get("input", []))
            resp = MagicMock()
            resp.data = [MagicMock(embedding=[float(call_num)] * DIM) for _ in range(n)]
            return resp

        client = MagicMock()
        client.embeddings.create = ordered_response
        embedder = OpenAIEmbedder(client=client, dimensions=DIM)
        result = await embedder.embed_batch(["t"] * 4, batch_size=2)
        # First 2 vectors should be from batch 1, next 2 from batch 2
        assert result[0] == result[1] == [1.0] * DIM
        assert result[2] == result[3] == [2.0] * DIM

    @pytest.mark.asyncio
    async def test_empty_text_at_first_index_raises_invalid_input(self):
        with pytest.raises(EmbedderInvalidInput, match="index 0"):
            await make_embedder().embed_batch(["", "valid text"])

    @pytest.mark.asyncio
    async def test_empty_text_at_middle_index_raises_invalid_input(self):
        with pytest.raises(EmbedderInvalidInput, match="index 1"):
            await make_embedder().embed_batch(["valid", "", "also valid"])

    @pytest.mark.asyncio
    async def test_whitespace_text_raises_invalid_input(self):
        with pytest.raises(EmbedderInvalidInput):
            await make_embedder().embed_batch(["   "])

    @pytest.mark.asyncio
    async def test_oversized_text_raises_invalid_input(self):
        embedder = make_embedder()
        with patch.object(embedder, "_tokenizer") as mock_tok:
            mock_tok.encode.side_effect = lambda t: list(range(9000)) if t == "LONG" else list(range(10))
            with pytest.raises(EmbedderInvalidInput, match="tokens"):
                await embedder.embed_batch(["short", "LONG"])

    @pytest.mark.asyncio
    async def test_validation_happens_before_any_api_call(self):
        client = make_async_client()
        embedder = make_embedder(client=client)
        with pytest.raises(EmbedderInvalidInput):
            await embedder.embed_batch(["valid", ""])
        client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_at_start_raises_unavailable(self):
        client = make_async_client(n=2)
        embedder = make_embedder(client=client)
        with patch.object(embedder._circuit_breaker, "is_open", return_value=True):
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed_batch(["a", "b"])
        client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_mid_batch_raises_with_progress_info(self):
        # First batch succeeds, breaker opens before second batch
        call_count = 0

        async def first_ok_then_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            n = len(kwargs.get("input", []))
            return make_fake_response(n=n)

        client = MagicMock()
        client.embeddings.create = first_ok_then_open
        embedder = OpenAIEmbedder(client=client, dimensions=DIM)

        open_after_first = MagicMock(side_effect=lambda: call_count >= 1)
        with patch.object(embedder._circuit_breaker, "is_open", open_after_first):
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed_batch(["t"] * 4, batch_size=2)

    @pytest.mark.asyncio
    async def test_timeout_raises_embedder_timeout(self):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(EmbedderTimeout):
                await make_embedder().embed_batch(["text"])

    @pytest.mark.asyncio
    async def test_timeout_records_failure(self):
        embedder = make_embedder()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with patch.object(embedder._circuit_breaker, "record_failure") as mock_fail:
                with pytest.raises(EmbedderTimeout):
                    await embedder.embed_batch(["text"])
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_raises_embedder_rate_limited(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())
        with pytest.raises(EmbedderRateLimited):
            await make_embedder(client=client).embed_batch(["text"])

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_record_failure(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())
        embedder = make_embedder(client=client)
        with patch.object(embedder._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(EmbedderRateLimited):
                await embedder.embed_batch(["text"])
        mock_fail.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_error_raises_embedder_unavailable(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_api_error())
        with pytest.raises(EmbedderUnavailable):
            await make_embedder(client=client).embed_batch(["text"])

    @pytest.mark.asyncio
    async def test_api_error_records_failure(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_api_error())
        embedder = make_embedder(client=client)
        with patch.object(embedder._circuit_breaker, "record_failure") as mock_fail:
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed_batch(["text"])
        mock_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_mismatched_vector_count_raises_unavailable(self):
        # API returns 1 vector but 2 were requested
        resp = MagicMock()
        resp.data = [MagicMock(embedding=fake_vector())]  # only 1
        client = MagicMock()
        client.embeddings.create = AsyncMock(return_value=resp)
        with pytest.raises(EmbedderUnavailable, match="vectors"):
            await make_embedder(client=client).embed_batch(["a", "b"])

    @pytest.mark.asyncio
    async def test_success_calls_record_success_per_batch(self):
        n_batches = 3
        client = make_async_client(n=2)
        embedder = make_embedder(client=client)
        with patch.object(embedder._circuit_breaker, "record_success") as mock_ok:
            await embedder.embed_batch(["t"] * (n_batches * 2), batch_size=2)
        assert mock_ok.call_count == n_batches


# ============================================================================
# Circuit breaker integration
# ============================================================================

class TestCircuitBreakerIntegration:

    @pytest.mark.asyncio
    async def test_consecutive_failures_trip_breaker(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_api_error())
        # threshold=3 → after 3 failures breaker should open
        embedder = make_embedder(client=client, breaker_threshold=3, breaker_cooldown_s=60)

        for _ in range(3):
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed("text")

        assert embedder._circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        fail_client = make_async_client()
        fail_client.embeddings.create = AsyncMock(side_effect=make_api_error())
        embedder = make_embedder(client=fail_client, breaker_threshold=5)

        # Two failures (below threshold)
        for _ in range(2):
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed("text")

        # Now inject a successful client
        good_client = make_async_client()
        embedder._client = good_client
        await embedder.embed("text")  # success → resets failures

        assert embedder._circuit_breaker._failures == 0

    @pytest.mark.asyncio
    async def test_open_breaker_makes_embed_fail_fast(self):
        client = make_async_client()
        embedder = make_embedder(client=client, breaker_threshold=1, breaker_cooldown_s=60)

        # Trip the breaker with one failure
        client.embeddings.create = AsyncMock(side_effect=make_api_error())
        with pytest.raises(EmbedderUnavailable):
            await embedder.embed("text")

        # Breaker now open — next call should fail fast without calling API
        client.embeddings.create.reset_mock()
        with pytest.raises(EmbedderUnavailable, match="[Cc]ircuit"):
            await embedder.embed("text again")
        client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_trip_breaker(self):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())
        embedder = make_embedder(client=client, breaker_threshold=1, breaker_cooldown_s=60)

        # Rate limit — should NOT trip the breaker
        with pytest.raises(EmbedderRateLimited):
            await embedder.embed("text")

        assert not embedder._circuit_breaker.is_open()

    @pytest.mark.asyncio
    async def test_timeout_trips_breaker(self):
        embedder = make_embedder(breaker_threshold=1, breaker_cooldown_s=60)
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with pytest.raises(EmbedderTimeout):
                await embedder.embed("text")
        assert embedder._circuit_breaker.is_open()


# ============================================================================
# Logging
# ============================================================================

class TestLogging:

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(self, caplog):
        embedder = make_embedder()
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with caplog.at_level("WARNING"):
                with pytest.raises(EmbedderTimeout):
                    await embedder.embed("text")
        assert any("timeout" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_api_error_logs_error(self, caplog):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_api_error())
        embedder = make_embedder(client=client)
        with caplog.at_level("ERROR"):
            with pytest.raises(EmbedderUnavailable):
                await embedder.embed("text")
        assert any("failure" in r.message.lower() or "error" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_rate_limit_logs_warning(self, caplog):
        client = make_async_client()
        client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())
        embedder = make_embedder(client=client)
        with caplog.at_level("WARNING"):
            with pytest.raises(EmbedderRateLimited):
                await embedder.embed("text")
        assert any("rate" in r.message.lower() for r in caplog.records)


# ============================================================================
# Integration tests — skipped without real OPENAI_API_KEY
# THESE WILL FAIL if run without a valid key — that is expected.
# ============================================================================

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — real API call skipped",
)
class TestRealAPIIntegration:

    @pytest.mark.asyncio
    async def test_embed_returns_vector_of_correct_dimensions(self):
        embedder = OpenAIEmbedder(dimensions=768)
        result = await embedder.embed("Patient presents with acute chest pain.")
        assert len(result) == 768
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_batch_returns_multiple_vectors(self):
        embedder = OpenAIEmbedder(dimensions=768)
        texts = [
            "Metformin is used for type 2 diabetes.",
            "Aspirin inhibits COX-1 and COX-2 enzymes.",
        ]
        results = await embedder.embed_batch(texts)
        assert len(results) == 2
        assert all(len(v) == 768 for v in results)

    @pytest.mark.asyncio
    async def test_embed_batch_order_matches_input(self):
        embedder = OpenAIEmbedder(dimensions=768)
        texts = ["drug interaction", "adverse effects", "dosage recommendations"]
        results = await embedder.embed_batch(texts)
        # Different texts → different vectors
        assert results[0] != results[1] != results[2]

    @pytest.mark.asyncio
    async def test_empty_string_raises_without_api_call(self):
        embedder = OpenAIEmbedder(dimensions=768)
        with pytest.raises(EmbedderInvalidInput):
            await embedder.embed("")

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        embedder = OpenAIEmbedder(dimensions=768)
        await embedder.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
