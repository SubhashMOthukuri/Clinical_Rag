"""
Unit tests for src/ingestion/drug_cache_store.py

Tests cover:
- save_drug_data: happy path, upsert, DB failure
- get_drug_by_name: Redis hit, DB hit, DB miss, DB failure fallback
- refresh_rxnorm_cache: stale rows, batch processing, partial failures, empty
- _refresh_single: success, rxcui not found, exception handling
"""

import pytest
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.ingestion.drug_cache_store import DrugCacheStore, DrugRecord
from src.ingestion.rxnorm_client import RxcuiFound, RxcuiUnverified


# ============================================================================
# FIXTURES
# ============================================================================

def make_store(conn=None, redis=None):
    """Build a DrugCacheStore with a mocked db_pool and optional redis."""
    db_pool = MagicMock()
    mock_conn = conn or AsyncMock()

    @asynccontextmanager
    async def fake_acquire():
        yield mock_conn

    db_pool.acquire = fake_acquire
    return DrugCacheStore(db_pool=db_pool, redis_client=redis), mock_conn


def make_drug_row(
    drug_name="metformin",
    rxcui="6809",
    normalized_name="metformin",
    verified=True,
    lookup_count=1,
):
    now = datetime.now(timezone.utc)
    row = {
        "drug_name": drug_name,
        "rxcui": rxcui,
        "normalized_name": normalized_name,
        "verified": verified,
        "lookup_count": lookup_count,
        "last_verified_at": now,
        "created_at": now,
    }
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: row[key]
    return mock


# ============================================================================
# save_drug_data
# ============================================================================

class TestSaveDrugData:

    @pytest.mark.asyncio
    async def test_saves_new_drug_successfully(self):
        store, conn = make_store()
        conn.execute = AsyncMock()

        await store.save_drug_data("Metformin", "6809", "metformin")

        conn.execute.assert_called_once()
        sql, *args = conn.execute.call_args.args
        assert "INSERT INTO drug_master" in sql
        assert args[0] == "metformin"  # lowercased
        assert args[1] == "6809"

    @pytest.mark.asyncio
    async def test_drug_name_lowercased_before_save(self):
        store, conn = make_store()
        conn.execute = AsyncMock()

        await store.save_drug_data("ASPIRIN", "1191", "aspirin")

        _, drug_name_arg, *_ = conn.execute.call_args.args
        assert drug_name_arg == "aspirin"

    @pytest.mark.asyncio
    async def test_upsert_on_conflict(self):
        store, conn = make_store()
        conn.execute = AsyncMock()

        await store.save_drug_data("metformin", "6809", "metformin")

        sql = conn.execute.call_args.args[0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE SET" in sql
        assert "lookup_count = drug_master.lookup_count + 1" in sql

    @pytest.mark.asyncio
    async def test_db_failure_raises_exception(self):
        store, conn = make_store()
        conn.execute = AsyncMock(side_effect=Exception("DB connection lost"))

        with pytest.raises(Exception, match="DB connection lost"):
            await store.save_drug_data("metformin", "6809", "metformin")

    @pytest.mark.asyncio
    async def test_db_failure_logs_error(self, caplog):
        store, conn = make_store()
        conn.execute = AsyncMock(side_effect=Exception("timeout"))

        with pytest.raises(Exception):
            with caplog.at_level("ERROR"):
                await store.save_drug_data("metformin", "6809", "metformin")

        assert "drug_cache.save_failed" in caplog.text
        assert "metformin" in caplog.text


# ============================================================================
# get_drug_by_name
# ============================================================================

class TestGetDrugByName:

    @pytest.mark.asyncio
    async def test_returns_drug_record_from_db(self):
        store, conn = make_store()
        conn.fetchrow = AsyncMock(return_value=make_drug_row())

        result = await store.get_drug_by_name("metformin")

        assert isinstance(result, DrugRecord)
        assert result.drug_name == "metformin"
        assert result.rxcui == "6809"
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_returns_none_on_db_miss(self):
        store, conn = make_store()
        conn.fetchrow = AsyncMock(return_value=None)

        result = await store.get_drug_by_name("unknowndrug")

        assert result is None

    @pytest.mark.asyncio
    async def test_drug_name_lowercased_for_lookup(self):
        store, conn = make_store()
        conn.fetchrow = AsyncMock(return_value=None)

        await store.get_drug_by_name("METFORMIN")

        _, name_arg = conn.fetchrow.call_args.args
        assert name_arg == "metformin"

    @pytest.mark.asyncio
    async def test_db_failure_returns_none_not_raises(self):
        store, conn = make_store()
        conn.fetchrow = AsyncMock(side_effect=Exception("DB down"))

        result = await store.get_drug_by_name("metformin")

        assert result is None

    @pytest.mark.asyncio
    async def test_db_failure_logs_error(self, caplog):
        store, conn = make_store()
        conn.fetchrow = AsyncMock(side_effect=Exception("connection refused"))

        with caplog.at_level("ERROR"):
            await store.get_drug_by_name("metformin")

        assert "drug_cache.get_failed" in caplog.text

    @pytest.mark.asyncio
    async def test_redis_hit_skips_db(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=b"6809")
        store, conn = make_store(redis=redis)
        conn.fetchrow = AsyncMock()

        result = await store.get_drug_by_name("metformin")

        assert isinstance(result, DrugRecord)
        assert result.rxcui == "6809"
        conn.fetchrow.assert_not_called()

    @pytest.mark.asyncio
    async def test_redis_miss_falls_through_to_db(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        store, conn = make_store(redis=redis)
        conn.fetchrow = AsyncMock(return_value=make_drug_row())

        result = await store.get_drug_by_name("metformin")

        assert isinstance(result, DrugRecord)
        conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_redis_goes_straight_to_db(self):
        store, conn = make_store(redis=None)
        conn.fetchrow = AsyncMock(return_value=make_drug_row())

        result = await store.get_drug_by_name("metformin")

        assert isinstance(result, DrugRecord)
        conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_hit_returns_verified_true(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=b"6809")
        store, _ = make_store(redis=redis)

        result = await store.get_drug_by_name("metformin")

        assert result.verified is True

    @pytest.mark.asyncio
    async def test_logs_cache_hit(self, caplog):
        store, conn = make_store()
        conn.fetchrow = AsyncMock(return_value=make_drug_row())

        with caplog.at_level("INFO"):
            await store.get_drug_by_name("metformin")

        assert "drug_cache.hit" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_cache_miss(self, caplog):
        store, conn = make_store()
        conn.fetchrow = AsyncMock(return_value=None)

        with caplog.at_level("INFO"):
            await store.get_drug_by_name("unknowndrug")

        assert "drug_cache.miss" in caplog.text


# ============================================================================
# refresh_rxnorm_cache
# ============================================================================

def make_stale_row(drug_name):
    row = MagicMock()
    row.__getitem__ = lambda self, key: drug_name if key == "drug_name" else None
    return row


class TestRefreshRxnormCache:

    @pytest.mark.asyncio
    async def test_empty_stale_list_returns_zero_counts(self):
        store, conn = make_store()
        conn.fetch = AsyncMock(return_value=[])
        rxnorm = AsyncMock()

        result = await store.refresh_rxnorm_cache(rxnorm)

        assert result == {"refreshed": 0, "failed": 0}
        rxnorm.get_rxcui.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_drugs_refreshed_successfully(self):
        store, conn = make_store()
        stale = [make_stale_row("metformin"), make_stale_row("aspirin")]
        conn.fetch = AsyncMock(return_value=stale)
        conn.execute = AsyncMock()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiFound(rxcui="6809", from_cache=False))

        result = await store.refresh_rxnorm_cache(rxnorm)

        assert result["refreshed"] == 2
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_partial_failure_counted_correctly(self):
        store, conn = make_store()
        stale = [make_stale_row("metformin"), make_stale_row("baddrug")]
        conn.fetch = AsyncMock(return_value=stale)
        conn.execute = AsyncMock()

        async def mock_get_rxcui(drug_name):
            if drug_name == "metformin":
                return RxcuiFound(rxcui="6809", from_cache=False)
            raise Exception("RxNorm unreachable")

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = mock_get_rxcui

        result = await store.refresh_rxnorm_cache(rxnorm)

        assert result["refreshed"] == 1
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_unverified_result_counted_as_failed(self):
        store, conn = make_store()
        stale = [make_stale_row("unknowndrug")]
        conn.fetch = AsyncMock(return_value=stale)

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiUnverified(drug_name="unknowndrug"))

        result = await store.refresh_rxnorm_cache(rxnorm)

        assert result["refreshed"] == 0
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_batch_size_10_processes_all(self):
        store, conn = make_store()
        stale = [make_stale_row(f"drug{i}") for i in range(25)]
        conn.fetch = AsyncMock(return_value=stale)
        conn.execute = AsyncMock()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiFound(rxcui="0000", from_cache=False))

        result = await store.refresh_rxnorm_cache(rxnorm)

        assert result["refreshed"] == 25
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_redis_cache_cleared_after_refresh(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.delete = AsyncMock()
        store, conn = make_store(redis=redis)
        stale = [make_stale_row("metformin")]
        conn.fetch = AsyncMock(return_value=stale)
        conn.execute = AsyncMock()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiFound(rxcui="6809", from_cache=False))

        await store.refresh_rxnorm_cache(rxnorm)

        redis.delete.assert_called_once_with("fda:metformin")

    @pytest.mark.asyncio
    async def test_logs_completion_summary(self, caplog):
        store, conn = make_store()
        conn.fetch = AsyncMock(return_value=[])
        rxnorm = AsyncMock()

        with caplog.at_level("INFO"):
            await store.refresh_rxnorm_cache(rxnorm)

        assert "refresh.complete" in caplog.text


# ============================================================================
# _refresh_single
# ============================================================================

class TestRefreshSingle:

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self):
        store, conn = make_store()
        conn.execute = AsyncMock()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiFound(rxcui="6809", from_cache=False))

        result = await store._refresh_single("metformin", rxnorm)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_rxcui_not_found(self):
        store, conn = make_store()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiUnverified(drug_name="unknowndrug"))

        result = await store._refresh_single("unknowndrug", rxnorm)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        store, conn = make_store()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(side_effect=Exception("network error"))

        result = await store._refresh_single("metformin", rxnorm)

        assert result is False

    @pytest.mark.asyncio
    async def test_logs_failure_on_exception(self, caplog):
        store, conn = make_store()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(side_effect=Exception("timeout"))

        with caplog.at_level("ERROR"):
            await store._refresh_single("metformin", rxnorm)

        assert "refresh.single_failed" in caplog.text
        assert "metformin" in caplog.text

    @pytest.mark.asyncio
    async def test_db_updated_with_new_rxcui(self):
        store, conn = make_store()
        conn.execute = AsyncMock()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiFound(rxcui="9999", from_cache=False))

        await store._refresh_single("aspirin", rxnorm)

        sql, rxcui_arg, _, drug_name_arg = conn.execute.call_args.args
        assert "UPDATE drug_master" in sql
        assert rxcui_arg == "9999"
        assert drug_name_arg == "aspirin"

    @pytest.mark.asyncio
    async def test_redis_not_required(self):
        store, conn = make_store(redis=None)
        conn.execute = AsyncMock()

        rxnorm = AsyncMock()
        rxnorm.get_rxcui = AsyncMock(return_value=RxcuiFound(rxcui="6809", from_cache=False))

        result = await store._refresh_single("metformin", rxnorm)

        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])