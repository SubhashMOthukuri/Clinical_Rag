from __future__ import annotations

from datetime import datetime, timezone
import logging
import asyncio
from dataclasses import dataclass

from src.ingestion.rxnorm_client import RxcuiFound

logger = logging.getLogger(__name__)


@dataclass
class DrugRecord:
    drug_name: str
    rxcui: str
    normalized_name: str
    verified: bool
    lookup_count: int
    last_verified_at: datetime
    created_at: datetime


class DrugCacheStore:
    def __init__(self, db_pool, redis_client=None):
        self._db = db_pool
        self._redis = redis_client

    async def save_drug_data(
        self,
        drug_name: str,
        rxcui: str,
        normalized_name: str,
    ) -> None:
        try:
            async with self._db.acquire() as conn:
                await conn.execute("""
                    INSERT INTO drug_master (
                        drug_name, rxcui, normalized_name,
                        verified, lookup_count, last_verified_at, created_at
                    ) VALUES ($1, $2, $3, TRUE, 1, $4, $4)
                    ON CONFLICT (drug_name) DO UPDATE SET
                        rxcui = EXCLUDED.rxcui,
                        normalized_name = EXCLUDED.normalized_name,
                        verified = TRUE,
                        lookup_count = drug_master.lookup_count + 1,
                        last_verified_at = EXCLUDED.last_verified_at
                """, drug_name.lower(), rxcui, normalized_name,
                     datetime.now(timezone.utc))
                logger.info("drug_cache.saved drug=%s rxcui=%s", drug_name, rxcui)
        except Exception as e:
            logger.error("drug_cache.save_failed drug=%s error=%s", drug_name, e)
            raise

    async def get_drug_by_name(
        self,
        drug_name: str,
    ) -> DrugRecord | None:
        if self._redis:
            cached = await self._redis.get(f"rxcui:{drug_name}")
            if cached:
                logger.info("drug_cache.redis_hit drug=%s", drug_name)
                return DrugRecord(
                    drug_name=drug_name,
                    rxcui=cached.decode(),
                    normalized_name=drug_name,
                    verified=True,
                    lookup_count=0,
                    last_verified_at=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                )

        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT drug_name, rxcui, normalized_name,
                           verified, lookup_count,
                           last_verified_at, created_at
                    FROM drug_master
                    WHERE drug_name = $1
                """, drug_name.lower())

                if row is None:
                    logger.info("drug_cache.miss drug=%s", drug_name)
                    return None

                logger.info("drug_cache.hit drug=%s rxcui=%s", drug_name, row["rxcui"])
                return DrugRecord(
                    drug_name=row["drug_name"],
                    rxcui=row["rxcui"],
                    normalized_name=row["normalized_name"],
                    verified=row["verified"],
                    lookup_count=row["lookup_count"],
                    last_verified_at=row["last_verified_at"],
                    created_at=row["created_at"],
                )
        except Exception as e:
            logger.error("drug_cache.get_failed drug=%s error=%s", drug_name, e)
            return None

    async def refresh_rxnorm_cache(
        self,
        rxnorm_client,
    ) -> dict:
        refreshed = 0
        failed = 0

        async with self._db.acquire() as conn:
            stale_rows = await conn.fetch("""
                SELECT drug_name FROM drug_master
                WHERE last_verified_at < NOW() - INTERVAL '7 days'
            """)

        batch_size = 10
        rows = list(stale_rows)

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            results = await asyncio.gather(
                *[self._refresh_single(r["drug_name"], rxnorm_client) for r in batch],
                return_exceptions=True,
            )
            for r in results:
                if r is True:
                    refreshed += 1
                else:
                    failed += 1

        logger.info("refresh.complete refreshed=%d failed=%d", refreshed, failed)
        return {"refreshed": refreshed, "failed": failed}

    async def _refresh_single(self, drug_name: str, rxnorm_client) -> bool:
        try:
            result = await rxnorm_client.get_rxcui(drug_name)
            if isinstance(result, RxcuiFound):
                async with self._db.acquire() as conn:
                    await conn.execute("""
                        UPDATE drug_master
                        SET rxcui = $1,
                            last_verified_at = $2,
                            verified = TRUE
                        WHERE drug_name = $3
                    """, result.rxcui, datetime.now(timezone.utc), drug_name)

                if self._redis:
                    await self._redis.delete(f"fda:{drug_name}")

                return True
            return False
        except Exception as e:
            logger.error("refresh.single_failed drug=%s error=%s", drug_name, e)
            return False