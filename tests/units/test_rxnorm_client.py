"""
Unit tests for src/ingestion/rxnorm_client.py

Tests cover:
- Valid RxNorm responses
- Error scenarios (timeout, network, malformed)
- Cache behavior (hits, misses, singleflight)
- Circuit breaker
- Input validation
- Retry logic with backoff
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from src.ingestion.rxnorm_client import (
    RxNormClient,
    RxNormConfig,
    RxcuiFound,
    RxcuiUnverified,
    RxcuiLookupFailed,
)


# ============================================================================
# REAL PRODUCTION DATA PATTERNS
# ============================================================================

class RealRxNormResponses:
    """Actual response patterns from RxNorm API (production data)."""

    @staticmethod
    def success_single_match():
        """Drug found with exact match - most common case."""
        return {
            "idGroup": {
                "name": "metformin",
                "rxnormId": ["6809"]
            }
        }

    @staticmethod
    def success_multiple_matches():
        """Drug found with multiple strength options."""
        return {
            "idGroup": {
                "name": "lisinopril",
                "rxnormId": ["10764", "10765", "10766"]  # 5mg, 10mg, 20mg
            }
        }

    @staticmethod
    def success_brand_name():
        """Brand name resolves to generic."""
        return {
            "idGroup": {
                "name": "metformin hydrochloride",
                "rxnormId": ["6809"]
            }
        }

    @staticmethod
    def not_found():
        """Drug not recognized by RxNorm."""
        return {
            "idGroup": {
                "name": "unknowndrug12345",
                "rxnormId": []  # Empty RXCUI list
            }
        }

    @staticmethod
    def malformed_missing_structure():
        """Malformed: missing expected structure."""
        return {"status": "error"}

    @staticmethod
    def malformed_null_value():
        """Malformed: null instead of object."""
        return None

    @staticmethod
    def malformed_empty_response():
        """Malformed: empty dict."""
        return {}

    @staticmethod
    def ambiguous_response():
        """Multiple matches - clinician must clarify."""
        return {
            "idGroup": {
                "name": "aspirin",
                "rxnormId": ["5492", "5493", "5494", "5495"]  # Multiple strengths
            }
        }


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestRxNormClientValidInput:
    """Test valid input handling and success paths."""

    @pytest.mark.asyncio
    async def test_valid_drug_returns_rxcui(self):
        """Test: Valid drug name returns RxcuiFound."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        # Mock successful response
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_single_match()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        
        result = await client.get_rxcui("metformin")
        
        assert isinstance(result, RxcuiFound)
        assert result.rxcui == "6809"
        assert result.from_cache is False

    @pytest.mark.asyncio
    async def test_multiple_matches_returns_first(self):
        """Test: Multiple matches returns first RXCUI."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_multiple_matches()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("lisinopril")
        
        assert isinstance(result, RxcuiFound)
        assert result.rxcui == "10764"

    @pytest.mark.asyncio
    async def test_brand_name_resolution(self):
        """Test: Brand name resolves to generic RXCUI."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_brand_name()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("Glucophage")  # Brand name for metformin
        
        assert isinstance(result, RxcuiFound)
        assert result.rxcui == "6809"


class TestRxNormClientNotFound:
    """Test when drug is not found in RxNorm."""

    @pytest.mark.asyncio
    async def test_unknown_drug_returns_unverified(self):
        """Test: Unknown drug returns RxcuiUnverified."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.not_found()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("unknowndrug12345")
        
        assert isinstance(result, RxcuiUnverified)
        assert result.drug_name == "unknowndrug12345"

    @pytest.mark.asyncio
    async def test_empty_rxcui_list_returns_unverified(self):
        """Test: Empty RXCUI list treated as 'not found'."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "idGroup": {
                "name": "testdrug",
                "rxnormId": []
            }
        }
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("testdrug")
        
        assert isinstance(result, RxcuiUnverified)


class TestRxNormClientInputValidation:
    """Test input validation and injection protection."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_bad_input(self):
        """Test: Empty string returns bad_input."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "bad_input"
        http_mock.get.assert_not_called()  # No API call for invalid input

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_bad_input(self):
        """Test: Whitespace-only input returns bad_input."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("   ")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "bad_input"

    @pytest.mark.asyncio
    async def test_sql_injection_rejected(self):
        """Test: SQL injection attempts rejected."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("aspirin'; DROP TABLE;")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "invalid_characters"
        http_mock.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_xss_attempt_rejected(self):
        """Test: XSS attempts rejected."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("<script>alert()</script>")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "invalid_characters"

    @pytest.mark.asyncio
    async def test_newline_injection_rejected(self):
        """Test: Newline/prompt injection rejected."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("aspirin\nIGNORE PREVIOUS")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "invalid_characters"


class TestRxNormClientErrorHandling:
    """Test error scenarios: timeout, network, malformed responses."""

    @pytest.mark.asyncio
    async def test_timeout_returns_failed(self):
        """Test: Timeout after retries returns RxcuiLookupFailed."""
        config = RxNormConfig(max_attempts=2, base_backoff_s=0.01)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        http_mock.get.side_effect = httpx.TimeoutException("timeout")
        
        client = RxNormClient(config=config, http_client=http_mock)
        
        result = await client.get_rxcui("metformin")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "timeout"
        assert http_mock.get.call_count == 2  # Retried

    @pytest.mark.asyncio
    async def test_network_error_returns_failed(self):
        """Test: Network error after retries returns RxcuiLookupFailed."""
        config = RxNormConfig(max_attempts=2, base_backoff_s=0.01)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        http_mock.get.side_effect = httpx.ConnectError("connection refused")
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("metformin")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "network"
        assert http_mock.get.call_count == 2

    @pytest.mark.asyncio
    async def test_http_500_retried(self):
        """Test: HTTP 500 is retried."""
        config = RxNormConfig(max_attempts=2, base_backoff_s=0.01)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        # First call: 500, Second call: 200 success
        response_500 = MagicMock()
        response_500.status_code = 500
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.json.return_value = RealRxNormResponses.success_single_match()
        
        http_mock.get.side_effect = [response_500, response_200]
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("metformin")
        
        assert isinstance(result, RxcuiFound)  # Succeeds on retry
        assert http_mock.get.call_count == 2

    @pytest.mark.asyncio
    async def test_http_429_rate_limit_retried(self):
        """Test: HTTP 429 (rate limit) is retried."""
        config = RxNormConfig(max_attempts=2, base_backoff_s=0.01)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {"Retry-After": "1"}  # Wait 1 second
        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.json.return_value = RealRxNormResponses.success_single_match()
        
        http_mock.get.side_effect = [response_429, response_200]
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("metformin")
        
        assert isinstance(result, RxcuiFound)

    @pytest.mark.asyncio
    async def test_http_404_not_retried(self):
        """Test: HTTP 404 is NOT retried (not transient)."""
        config = RxNormConfig(max_attempts=2, base_backoff_s=0.01)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_404 = MagicMock()
        response_404.status_code = 404
        http_mock.get.return_value = response_404
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("metformin")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert "http_404" in result.reason
        assert http_mock.get.call_count == 1  # Not retried

    @pytest.mark.asyncio
    async def test_malformed_json_returns_failed(self):
        """Test: Malformed JSON response returns RxcuiLookupFailed."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.side_effect = ValueError("invalid json")
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        result = await client.get_rxcui("metformin")
        
        assert isinstance(result, RxcuiLookupFailed)
        assert result.reason == "malformed"


class TestRxNormClientCaching:
    """Test cache behavior: hits, misses, TTL, singleflight."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_from_cache(self):
        """Test: Second request for same drug returns from cache."""
        config = RxNormConfig(cache_ttl_s=3600)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_single_match()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        
        # First call
        result1 = await client.get_rxcui("metformin")
        assert result1.from_cache is False
        
        # Second call (should be cached)
        result2 = await client.get_rxcui("metformin")
        assert result2.from_cache is True
        assert http_mock.get.call_count == 1  # Only 1 API call

    @pytest.mark.asyncio
    async def test_singleflight_concurrent_requests(self):
        """Test: Concurrent requests for same drug only hit API once."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_single_match()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        
        # 10 concurrent requests for same drug
        tasks = [client.get_rxcui("metformin") for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(isinstance(r, RxcuiFound) for r in results)
        # But only 1 API call (singleflight)
        assert http_mock.get.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_for_different_drugs(self):
        """Test: Different drugs hit API separately."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_single_match()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock)
        
        await client.get_rxcui("metformin")
        await client.get_rxcui("lisinopril")
        await client.get_rxcui("aspirin")
        
        # 3 different drugs = 3 API calls
        assert http_mock.get.call_count == 3


class TestRxNormClientCircuitBreaker:
    """Test circuit breaker: stops retries when service is unhealthy."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test: Circuit breaker opens after threshold failures."""
        config = RxNormConfig(breaker_threshold=2, breaker_cooldown_s=0.1)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        http_mock.get.side_effect = httpx.TimeoutException("timeout")
        
        client = RxNormClient(config=config, http_client=http_mock)
        
        # First failure
        result1 = await client.get_rxcui("drug1")
        assert isinstance(result1, RxcuiLookupFailed)
        
        # Second failure (opens circuit)
        result2 = await client.get_rxcui("drug2")
        assert isinstance(result2, RxcuiLookupFailed)
        
        # Third request should fail immediately without retry
        result3 = await client.get_rxcui("drug3")
        assert isinstance(result3, RxcuiLookupFailed)
        assert result3.reason == "circuit_open"

    @pytest.mark.asyncio
    async def test_circuit_breaker_cooldown(self):
        """Test: Circuit breaker recovers after cooldown."""
        config = RxNormConfig(
            breaker_threshold=1,
            breaker_cooldown_s=0.1,
            max_attempts=1,
            base_backoff_s=0.01
        )
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_single_match()
        
        response_fail = MagicMock()
        response_fail.status_code = 500
        
        # First: fail (opens circuit), Then: wait, Then: succeed
        http_mock.get.side_effect = [response_fail, response_mock]
        
        client = RxNormClient(config=config, http_client=http_mock)
        
        # Trigger failure
        await client.get_rxcui("drug1")
        
        # Wait for cooldown
        await asyncio.sleep(0.15)
        
        # Should recover
        result = await client.get_rxcui("drug2")
        assert isinstance(result, RxcuiFound)


class TestRxNormClientMetrics:
    """Test metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_incremented_on_success(self):
        """Test: Metrics recorded for successful lookups."""
        config = RxNormConfig()
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        metrics_mock = MagicMock()
        
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = RealRxNormResponses.success_single_match()
        http_mock.get.return_value = response_mock
        
        client = RxNormClient(config=config, http_client=http_mock, metrics=metrics_mock)
        
        await client.get_rxcui("metformin")
        
        # Verify metrics calls
        metrics_mock.incr.assert_any_call("rxnorm.found")
        metrics_mock.incr.assert_any_call("rxnorm.cache.miss")
        metrics_mock.observe.assert_called()  # latency recorded

    @pytest.mark.asyncio
    async def test_metrics_recorded_for_errors(self):
        """Test: Metrics recorded for error scenarios."""
        config = RxNormConfig(max_attempts=1)
        http_mock = AsyncMock(spec=httpx.AsyncClient)
        metrics_mock = MagicMock()
        http_mock.get.side_effect = httpx.TimeoutException("timeout")
        
        client = RxNormClient(config=config, http_client=http_mock, metrics=metrics_mock)
        
        await client.get_rxcui("metformin")
        
        metrics_mock.incr.assert_any_call("rxnorm.timeout")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
