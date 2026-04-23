import pytest
import respx
import httpx
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from src.ingestion.fda_client import FDAClient, FDAConfig, FDADrugData

@pytest.fixture
def sample_fda_response():
    return {
        "results": [{
            "id": "test-label-id-123",
            "warnings": ["Stomach bleeding warning"],
            "drug_interactions": ["Interacts with warfarin"],
            "do_not_use": ["Do not use after heart surgery"],
            "ask_doctor": ["Ask doctor if kidney disease"],
            "openfda": {
                "generic_name": ["ibuprofen"],
                "rxcui": ["310965"],
                "pharm_class_epc": ["Nonsteroidal Anti-inflammatory Drug [EPC]"]
            }
        }]
    }

@pytest.mark.asyncio
async def test_get_drug_data_success(sample_fda_response):
    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(return_value=httpx.Response(
            200, json=sample_fda_response
        ))

        client = FDAClient()
        result = await client.get_drug_data("ibuprofen")

        assert result is not None
        assert result.generic_name == "ibuprofen"
        assert result.rxcui == "310965"
        assert result.source == "FRESH_FDA"
        assert len(result.warnings) == 1
        await client.aclose()
@pytest.mark.asyncio
async def test_get_drug_data_fda_error():
    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(return_value=httpx.Response(
            200, json={"error": {"code": "NOT_FOUND"}}
        ))

        client = FDAClient()
        result = await client.get_drug_data("unknowndrug")

        assert result is None
        await client.aclose()
@pytest.mark.asyncio
async def test_get_drug_data_timeout():
    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(side_effect=httpx.TimeoutException("timeout"))

        client = FDAClient()
        result = await client.get_drug_data("ibuprofen")

        assert result is None
        await client.aclose()
@pytest.mark.asyncio
async def test_get_drug_data_redis_cache_hit():
    mock_redis = AsyncMock()

    cached_data = json.dumps({
        "generic_name": "ibuprofen",
        "rxcui": "310965",
        "drug_class": "NSAID",
        "warnings": ["Stomach bleeding"],
        "drug_interactions": [],
        "do_not_use": [],
        "ask_doctor": [],
        "source": "REDIS_CACHE",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "fda_label_id": "test-id",
    })

    mock_redis.get = AsyncMock(return_value=cached_data.encode())

    client = FDAClient(redis_client=mock_redis)
    result = await client.get_drug_data("ibuprofen")

    assert result is not None
    assert result.source == "REDIS_CACHE"
    assert result.generic_name == "ibuprofen"
    mock_redis.get.assert_called_once_with("fda:ibuprofen")
    await client.aclose()
@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    config = FDAConfig(breaker_threshold=2, breaker_cooldown_s=60.0)

    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(side_effect=httpx.TimeoutException("timeout"))

        client = FDAClient(config=config)

        # First call — fails, breaker failure count = 1
        result1 = await client.get_drug_data("ibuprofen")
        # Second call — fails, breaker opens
        result2 = await client.get_drug_data("ibuprofen")
        # Third call — circuit open, no API call made
        result3 = await client.get_drug_data("ibuprofen")

        assert result1 is None
        assert result2 is None
        assert result3 is None
        assert client._breaker.is_open()
        await client.aclose()
@pytest.mark.asyncio
async def test_redis_cache_written_after_fresh_fda(sample_fda_response):
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)  # cache miss
    mock_redis.setex = AsyncMock()

    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(return_value=httpx.Response(
            200, json=sample_fda_response
        ))

        client = FDAClient(redis_client=mock_redis)
        result = await client.get_drug_data("ibuprofen")

        # Verify Redis was written after fresh FDA call
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "fda:ibuprofen"  # correct key
        assert call_args[1] == 604_800           # correct TTL
        await client.aclose()
@pytest.mark.asyncio
async def test_get_drug_data_network_error():
    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(side_effect=httpx.ConnectError("connection refused"))

        client = FDAClient()
        result = await client.get_drug_data("ibuprofen")

        assert result is None
        assert client._breaker._failures == 1
        await client.aclose()
@pytest.mark.asyncio
async def test_get_drug_data_http_500():
    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(return_value=httpx.Response(500))

        client = FDAClient()
        result = await client.get_drug_data("ibuprofen")

        assert result is None
        assert client._breaker._failures == 1
        await client.aclose()
@pytest.mark.asyncio
async def test_get_drug_data_empty_results():
    with respx.mock:
        respx.get(
            "https://api.fda.gov/drug/label.json"
        ).mock(return_value=httpx.Response(
            200, json={"results": []}
        ))

        client = FDAClient()
        result = await client.get_drug_data("ibuprofen")

        assert result is None
        await client.aclose()
